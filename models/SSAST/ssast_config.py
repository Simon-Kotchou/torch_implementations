""" Self-Supervised Audio Spectogram Transformer (SSAST) model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Simon-Kotchou/ssast-base-patch-audioset-16-16": (
        "https://huggingface.co/Simon-Kotchou/ssast-base-patch-audioset-16-16/resolve/main/config.json"
    ),
}

class SSASTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SSASTModel`]. It is used to instantiate an SSAST
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        patch_freq_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch in the frequency dimension.
        patch_time_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch in the time dimension.
        frequency_stride (`int`, *optional*, defaults to 16):
            Frequency stride to use when patchifying the spectrograms.
        time_stride (`int`, *optional*, defaults to 16):
            Temporal stride to use when patchifying the spectrograms.
        max_length (`int`, *optional*, defaults to 1024):
            Temporal dimension of the spectrograms.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Frequency dimension of the spectrograms (number of Mel-frequency bins).
        num_labels (`int`, *optional*, defaults to 527):
            Number of labels for audio classification.
        mask_patch_prob (`float`, *optional*, defaults to 0.75):
            Probability of masking a spectrogram patch in masked patch modeling (MPM).
        mask_time_prob (`float`, *optional*, defaults to 0.5):
            Probability of masking a spectrogram time step in masked patch modeling (MPM).
        mask_patch_num (`int`, *optional*, defaults to 400):
            Number of spectrogram patches to mask in masked patch modeling (MPM).
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to use mean pooling for audio representation.
        mask_token_val (`float`, *optional*, defaults to 0.0):
            The value of masked spectrogram patches.
        cluster (`bool`, *optional*, defaults to `True`):
            Whether to cluster the masked patches or randomly select them.
        model_size (`str`, *optional*, defaults to `"base"`):
            The model size. Can be `"tiny"`, `"small"`, or `"base"`.

    Example:

    ```python
    >>> from transformers import SSASTConfig, SSASTModel

    >>> # Initializing a SSAST configuration
    >>> configuration = SSASTConfig()

    >>> # Initializing a model from the configuration
    >>> model = SSASTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ssast"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        patch_freq_size=16,
        patch_time_size=16,
        qkv_bias=True,
        frequency_stride=16,
        time_stride=16,
        max_length=1024,
        num_mel_bins=128,
        num_labels=527,
        mask_patch_prob=0.75,
        mask_time_prob=0.5,
        mask_patch_num=400,
        use_mean_pooling=True,
        mask_token_val=0.0,
        cluster=True,
        model_size="base",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.patch_freq_size = patch_freq_size
        self.patch_time_size = patch_time_size
        self.qkv_bias = qkv_bias
        self.frequency_stride = frequency_stride
        self.time_stride = time_stride
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins
        self.num_labels = num_labels
        self.mask_patch_prob = mask_patch_prob
        self.mask_time_prob = mask_time_prob
        self.mask_patch_num = mask_patch_num
        self.use_mean_pooling = use_mean_pooling
        self.mask_token_val = mask_token_val
        self.cluster = cluster
        self.model_size = model_size