"""Convert Audio Spectrogram Transformer checkpoints from the original repository. URL: https://github.com/YuanGongND/ast"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_audio_spectrogram_transformer_config(model_name):
    config = ASTConfig()

    if "16-16" in model_name:
        config.time_stride = 16
        config.frequency_stride = 16
        config.patch_size = 16
        if "small" in model_name:
            config.hidden_size = 384
            config.num_attention_heads = 6
            config.intermediate_size = 1536
        elif "tiny" in model_name:
            config.hidden_size = 192
            config.num_attention_heads = 3
            config.intermediate_size = 768
    else:
        raise ValueError("Model not supported")

    repo_id = "huggingface/label-files"
    if "speech-commands" in model_name:
        config.num_labels = 35
        filename = "speech-commands-v2-id2label.json"
    else:
        config.num_labels = 527
        filename = "audioset-id2label.json"

    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name):
    if "module.v" in name:
        name = name.replace("module.v", "audio_spectrogram_transformer")
    if "cls_token" in name:
        name = name.replace("cls_token", "embeddings.cls_token")
    if "dist_token" in name:
        name = name.replace("dist_token", "embeddings.distillation_token")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    # transformer blocks
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # final layernorm
    if "audio_spectrogram_transformer.norm" in name:
        name = name.replace("audio_spectrogram_transformer.norm", "audio_spectrogram_transformer.layernorm")
    # classifier head
    if "module.mlp_head.0" in name:
        name = name.replace("module.mlp_head.0", "classifier.layernorm")
    if "module.mlp_head.1" in name:
        name = name.replace("module.mlp_head.1", "classifier.dense")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[3])
            dim = config.hidden_size
            if "weight" in key:
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.value.bias"
                ] = val[-dim:]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def remove_keys(state_dict):
    ignore_keys = [
        "module.v.head.weight",
        "module.v.head.bias",
        "module.v.head_dist.weight",
        "module.v.head_dist.bias",
        "module.p_input_fdim", 
        "module.p_input_tdim", 
        "module.mask_embed", 
        "module.cpredlayer.0.weight", 
        "module.cpredlayer.0.bias", 
        "module.cpredlayer.2.weight", 
        "module.cpredlayer.2.bias", 
        "module.gpredlayer.0.weight", 
        "module.gpredlayer.0.bias", 
        "module.gpredlayer.2.weight", 
        "module.gpredlayer.2.bias"
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


@torch.no_grad()
def convert_audio_spectrogram_transformer_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Audio Spectrogram Transformer structure.
    """
    config = get_audio_spectrogram_transformer_config(model_name)

    model_name_to_url = {
        "ssast-base-patch-audioset-16-16": (
            "https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1"
        ),
        "ssast-base-frame-audioset-128-2": (
            "https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1"
        ),
        "ssast-small-patch-audioset-16-16": (
            "https://www.dropbox.com/s/i24w446rl9pkf05/SSAST-Small-Patch-400.pth?dl=1"
        ),
        "ssast-tiny-patch-audioset-16-16": (
            "https://www.dropbox.com/s/fkbtf78y94113wz/SSAST-Tiny-Patch-400.pth?dl=1"
        ),
        "ssast-tiny-frame-audioset-128-2": (
            "https://www.dropbox.com/s/rx7g60ruzawffzv/SSAST-Tiny-Frame-400.pth?dl=1"
        ),
    }

    # load original state_dict
    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    mlp_head = torch.nn.Sequential(torch.nn.LayerNorm(config.hidden_size),
                                          torch.nn.Linear(config.hidden_size, config.num_labels))
    key_mapping = {
    '0.weight': 'classifier.layernorm.weight',
    '0.bias': 'classifier.layernorm.bias',
    '1.weight': 'classifier.dense.weight',
    '1.bias': 'classifier.dense.bias'
    }

    # Update the existing state dictionary
    for key, value in mlp_head.state_dict().items():
        mapped_key = key_mapping[key]
        state_dict[mapped_key] = value

    # remove some keys
    remove_keys(state_dict)
    # rename some keys
    new_state_dict = convert_state_dict(state_dict, config)

    # load 🤗 model
    model = ASTForAudioClassification(config)
    #model = ASTModel(config)
    model.eval()

    model.load_state_dict(new_state_dict)

    # verify outputs on dummy input
    # source: https://github.com/YuanGongND/ast/blob/79e873b8a54d0a3b330dd522584ff2b9926cd581/src/run.py#L62
    mean = -4.2677393 if "speech-commands" not in model_name else -6.845978
    std = 4.5689974 if "speech-commands" not in model_name else 5.5654526
    max_length = 1024 if "speech-commands" not in model_name else 128
    feature_extractor = ASTFeatureExtractor(mean=mean, std=std, max_length=max_length)

    if "speech-commands" in model_name:
        dataset = load_dataset("speech_commands", "v0.02", split="validation")
        waveform = dataset[0]["audio"]["array"]
    else:
        filepath = hf_hub_download(
            repo_id="nielsr/audio-spectogram-transformer-checkpoint",
            filename="sample_audio.flac",
            repo_type="dataset",
        )

        waveform, _ = torchaudio.load(filepath)
        waveform = waveform.squeeze().numpy()

    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

    # forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    print(logits[0, :3])
    # if model_name == "ast-finetuned-audioset-10-10-0.4593":
    #     expected_slice = torch.tensor([-0.8760, -7.0042, -8.6602])
    # elif model_name == "ast-finetuned-audioset-10-10-0.450":
    #     expected_slice = torch.tensor([-1.1986, -7.0903, -8.2718])
    # elif model_name == "ast-finetuned-audioset-10-10-0.448":
    #     expected_slice = torch.tensor([-2.6128, -8.0080, -9.4344])
    # elif model_name == "ast-finetuned-audioset-10-10-0.448-v2":
    #     expected_slice = torch.tensor([-1.5080, -7.4534, -8.8917])
    # elif model_name == "ast-finetuned-audioset-12-12-0.447":
    #     expected_slice = torch.tensor([-0.5050, -6.5833, -8.0843])
    # elif model_name == "ast-finetuned-audioset-14-14-0.443":
    #     expected_slice = torch.tensor([-0.3826, -7.0336, -8.2413])
    # elif model_name == "ast-finetuned-audioset-16-16-0.442":
    #     expected_slice = torch.tensor([-1.2113, -6.9101, -8.3470])
    # elif model_name == "ast-finetuned-speech-commands-v2":
    #     expected_slice = torch.tensor([6.1589, -8.0566, -8.7984])
    # else:
    #     raise ValueError("Unknown model name")
    # if not torch.allclose(logits[0, :3], expected_slice, atol=1e-4):
    #     raise ValueError("Logits don't match")
    # print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and feature extractor to the hub...")
        model.push_to_hub(f"Simon-Kotchou/{model_name}")
        feature_extractor.push_to_hub(f"Simon-Kotchou/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="ssast-base-patch-audioset-16-16",
        type=str,
        help="Name of the Audio Spectrogram Transformer model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_audio_spectrogram_transformer_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)