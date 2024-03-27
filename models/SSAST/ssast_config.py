""" Self-Supervised Audio Spectogram Transformer (SSAST) model configuration"""


from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Simon-Kotchou/ssast-base-patch-audioset-16-16": (
        "https://huggingface.co/Simon-Kotchou/ssast-base-patch-audioset-16-16/resolve/main/config.json"
    ),
}


class SSASTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SSASTModel`]. It is used to instantiate an SSAST
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SSAST
    [Simon-Kotchou/ssast-base-patch-audioset-16-16](https://huggingface.co/Simon-Kotchou/ssast-base-patch-audioset-16-16)
    architecture.

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
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        patch_freq_size (`int`, *optional*, defaults to 16):
            The number of frequency steps for each patch.
        patch_time_size (`int`, *optional*, defaults to 16):
            The number of time steps for each patch.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        frequency_stride (`int`, *optional*, defaults to 10):
            Frequency stride to use when patchifying the spectrograms.
        time_stride (`int`, *optional*, defaults to 10):
            Temporal stride to use when patchifying the spectrograms.
        max_length (`int`, *optional*, defaults to 1024):
            Temporal dimension of the spectrograms.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Frequency dimension of the spectrograms (number of Mel-frequency bins).

    Example:

    ```python
    >>> from transformers import SSASTConfig, SSASTModel

    >>> # Initializing a SSAST Simon-Kotchou/ssast-base-patch-audioset-16-16 style configuration
    >>> configuration = SSASTConfig()

    >>> # Initializing a model (with random weights) from the Simon-Kotchou/ssast-base-patch-audioset-16-16 style configuration
    >>> model = SSASTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "self-supervised-audio-spectrogram-transformer"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        patch_freq_size=16,
        patch_time_size=16,
        qkv_bias=True,
        frequency_stride=10,
        time_stride=10,
        max_length=1024,
        num_mel_bins=128,
        num_labels=527,
        mask_patch=400,  # Add this parameter for pretraining
        cluster=True,  # Add this parameter for pretraining
        **kwargs
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
        self.mask_patch = mask_patch
        self.cluster = cluster