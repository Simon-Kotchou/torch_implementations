import math
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_audio_spectrogram_transformer import ASTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SSASTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "Simon-Kotchou/ssast-base-patch-audioset-16-16"
_EXPECTED_OUTPUT_SHAPE = [1, 1214, 768]

# Audio classification docstring
_SEQ_CLASS_CHECKPOINT = "Simon-Kotchou/ssast-base-frame-audioset-128-2"
_SEQ_CLASS_EXPECTED_OUTPUT = "'Speech'"
_SEQ_CLASS_EXPECTED_LOSS = 0.17


AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Simon-Kotchou/ssast-base-frame-audioset-128-1",
    # See all Audio Spectrogram Transformer models at https://huggingface.co/models?filter=ast
]


class SSASTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config: SSASTConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = SSASTPatchEmbeddings(config)

        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        num_patches = frequency_out_dimension * time_out_dimension
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def get_shape(self, config):
        # see Karpathy's cs231n blog on how to calculate the output dimensions
        # https://cs231n.github.io/convolutional-networks/#conv
        patch_size = config.patch_freq_size * config.patch_time_size
        frequency_out_dimension = (config.num_mel_bins - patch_size) // config.frequency_stride + 1
        time_out_dimension = (config.max_length - patch_size) // config.time_stride + 1

        return frequency_out_dimension, time_out_dimension

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        batch_size = input_values.shape[0]
        embeddings = self.patch_embeddings(input_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class SSASTPatchEmbeddings(nn.Module):
    """
    This class turns `input_values` into the initial `hidden_states` (patch embeddings) of shape `(batch_size,
    seq_length, hidden_size)` to be consumed by a Transformer.
    """

    def __init__(self, config):
        super().__init__()

        patch_freq_size = config.patch_freq_size
        patch_time_size = config.patch_time_size
        patch_size = config.patch_freq_size * config.patch_time_size
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        self.projection = nn.Conv2d(
            1, config.hidden_size, kernel_size=(patch_freq_size, patch_time_size), stride=(frequency_stride, time_stride)
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = input_values.unsqueeze(1)
        input_values = input_values.transpose(2, 3)
        embeddings = self.projection(input_values).flatten(2).transpose(1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->AST
class ASTSelfAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->AST
class ASTSelfOutput(nn.Module):
    """
    The residual connection is defined in ASTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->AST
class ASTAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.attention = ASTSelfAttention(config)
        self.output = ASTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->AST
class ASTIntermediate(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->AST
class ASTOutput(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->AST
class ASTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ASTAttention(config)
        self.intermediate = ASTIntermediate(config)
        self.output = ASTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in AST, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in AST, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->AST
class ASTEncoder(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ASTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class ASTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ASTConfig
    base_model_prefix = "audio_spectrogram_transformer"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.deit.modeling_deit.DeiTPreTrainedModel._init_weights
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SSASTBaseModel(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = SSASTEmbeddings(config)
        self.encoder = ASTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_values, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_values is None:
            raise ValueError("You have to specify input_values")

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class SSASTForPretraining(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__(config)
        self.config = config

        self.ast = SSASTBaseModel(config)

        self.cls_token_num = 2
        self.original_embedding_dim = config.hidden_size
        self.num_patches = self.ast.embeddings.patch_embeddings.num_patches

        self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
        self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

        self.cpredlayer = nn.Sequential(
            nn.Linear(self.original_embedding_dim, self.original_embedding_dim), 
            nn.ReLU(), 
            nn.Linear(self.original_embedding_dim, 256)
        )
        self.gpredlayer = nn.Sequential(
            nn.Linear(self.original_embedding_dim, self.original_embedding_dim), 
            nn.ReLU(), 
            nn.Linear(self.original_embedding_dim, 256)
        )
        self.unfold = torch.nn.Unfold(
            kernel_size=(config.patch_freq_size, config.patch_time_size), 
            stride=(config.frequency_stride, config.time_stride)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_values, task, cluster=True, mask_patch=400):
        input_values = input_values.unsqueeze(1).transpose(2, 3)

        if task == 'pretrain_mpc':
            return self.mpc(input_values, mask_patch=mask_patch, cluster=cluster)
        elif task == 'pretrain_mpg':
            return self.mpg(input_values, mask_patch=mask_patch, cluster=cluster)
        else:
            raise Exception('Task unrecognized.')

    def mpc(self, input_values, mask_patch, cluster):
        input = self.unfold(input_values).transpose(1, 2)
        B = input_values.shape[0]
        x = self.ast.embeddings.patch_embeddings(input_values)

        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        for i in range(B):
            if cluster:
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        cls_tokens = self.ast.embeddings.cls_token.expand(B, -1, -1)
        dist_token = self.ast.embeddings.distillation_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.ast.embeddings.position_embeddings
        x = self.ast.embeddings.dropout(x)

        for blk in self.ast.encoder.layer:
            x = blk(x)
        x = self.ast.layernorm(x)

        pred = torch.empty((B, mask_patch, 256), device=x.device).float()
        for i in range(B):
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])

        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in range(B):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            correct += torch.sum(torch.eq(torch.argmax(total, dim=0), torch.arange(0, mask_patch, device=x.device)))
            nce += torch.sum(torch.diag(torch.log_softmax(total, dim=-1)))
        acc = 1.0 * correct / (B * mask_patch)
        nce = nce / (-1.0 * B * mask_patch)

        return acc, nce

    def mpg(self, input_values, mask_patch, cluster):
        B = input_values.shape[0]
        x = self.ast.embeddings.patch_embeddings(input_values)
        input = self.unfold(input_values).transpose(1, 2)

        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        for i in range(B):
            if cluster:
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        cls_tokens = self.ast.embeddings.cls_token.expand(B, -1, -1)
        dist_token = self.ast.embeddings.distillation_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.ast.embeddings.position_embeddings
        x = self.ast.embeddings.dropout(x)

        for blk in self.ast.encoder.layer:
            x = blk(x)
        x = self.ast.layernorm(x)

        pred = torch.empty((B, mask_patch, self.config.patch_freq_size * self.config.patch_time_size), device=x.device).float()
        target = torch.empty((B, mask_patch, self.config.patch_freq_size * self.config.patch_time_size), device=x.device).float()

        for i in range(B):
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]

        mse = torch.mean((pred - target) ** 2)

        return mse

    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []
        cur_clus = np.random.randint(low=3, high=cluster+3)

        while len(set(mask_id)) <= mask_size:
            start_id = np.random.randint(sequence_len)
            cur_mask = []
            for i in range(cur_clus):
                for j in range(cur_clus):
                    mask_cand = start_id + self.config.frequency_stride * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = np.random.choice(range(sequence_len), mask_size, replace=False)
        return torch.tensor(mask_id)
    
class SSASTForAudioClassification(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ast = SSASTBaseModel(config)
        self.classifier = ASTMLPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ast(
            input_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )