import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union

@dataclass
class CelebaMetadata:
    """CelebA dataset metadata configuration"""
    use_attributes: bool = True
    attribute_embedding_dim: int = 512
    attributes_as_condition: bool = True
    selected_attributes: List[str] = field(default_factory=lambda: [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        'Wearing_Necktie', 'Young'
    ])
    landmark_points: bool = False
    identity_conditioning: bool = False
    num_identities: int = 10177  # Number of unique identities in CelebA

@dataclass
class AutoencoderConfig:
    """Continuous autoencoder configuration"""
    input_resolution: int = 256  # Higher resolution for CelebA-HQ if available
    latent_resolution: int = 32  # 8x downsampling
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16  # Increased for detailed face features
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    embed_dim: int = 512
    dropout: float = 0.0
    use_residual: bool = True
    normalization: str = "group"  # "batch", "group", "layer", or "instance"
    activation: str = "silu"  # "relu", "leaky_relu", "silu", "gelu"
    latent_regularization: bool = True
    regularization_weight: float = 0.01

@dataclass
class MMDiTBlockConfig:
    """MM-DiT block configuration"""
    dim: int = 1280
    heads: int = 8
    dim_head: int = 64
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    qkv_bias: bool = True
    use_rotary_embedding: bool = True
    sandwich_norm: bool = True
    use_flash_attention: bool = True
    cross_attention: bool = True
    modulation_features: List[str] = field(default_factory=lambda: ["alpha", "beta", "delta", "epsilon"])

@dataclass
class FluxModelConfig:
    """Rectified flow transformer model configuration"""
    image_size: int = 256
    patch_size: int = 2
    input_channels: int = 16  # Matches autoencoder latent channels
    hidden_size: int = 1280
    depth: int = 24
    num_heads: int = 20
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    block_config: MMDiTBlockConfig = field(default_factory=MMDiTBlockConfig)
    
    # Time embedding
    time_embed_dim: int = 1280
    time_embed_type: str = "sinusoidal"  # "sinusoidal", "fourier", "learnable"
    
    # Conditional inputs
    use_metadata: bool = True
    condition_drop_prob: float = 0.1  # For classifier-free guidance
    metadata_config: CelebaMetadata = field(default_factory=CelebaMetadata)
    
    # Architecture scaling for different resolutions
    model_scaling: Dict[str, Dict] = field(default_factory=lambda: {
        "small": {
            "hidden_size": 768,
            "depth": 16,
            "num_heads": 12
        },
        "base": {
            "hidden_size": 1280,
            "depth": 24,
            "num_heads": 20
        },
        "large": {
            "hidden_size": 1536,
            "depth": 32,
            "num_heads": 24
        }
    })

@dataclass
class ODESolverConfig:
    """ODE solver configuration for sampling"""
    method: str = "dopri5"  # "dopri5", "rk4", "euler", "heun"
    rtol: float = 1e-5
    atol: float = 1e-5
    safety_factor: float = 0.9
    min_step_size: float = 1e-5
    max_step_size: float = 0.1
    max_steps: int = 1000
    scheduler: str = "cosine"  # "linear", "cosine", "sigmoid"
    steps_for_fixed: int = 50  # Used only for fixed-step methods
    step_size_controller: str = "standard"  # "standard", "proportional", "pi"

@dataclass
class OptimizationConfig:
    """Training optimization configuration"""
    optimizer: str = "adamw"  # "adam", "adamw", "lion"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    clip_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant", "warmup_cosine" 
    warmup_steps: int = 5000
    min_lr_ratio: float = 0.1
    
    # Exponential Moving Average
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_interval: int = 10

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    total_steps: int = 400000
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 10000
    
    # Early stopping
    patience: int = 20
    metric_for_early_stopping: str = "val_loss"
    
    # Loss configuration
    loss_type: str = "mse"  # "mse", "l1", "huber", "smooth_l1"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "velocity": 1.0,
        "velocity_reg": 0.001,  # Regularization on velocity field magnitude
        "latent_reg": 0.01,     # Regularization on autoencoder latents
        "attribute_pred": 0.1   # Auxiliary loss for attribute prediction
    })
    
    # Sampling during training
    sample_interval: int = 5000
    num_samples: int = 25
    
    # Data augmentation
    flip_prob: float = 0.5
    brightness_jitter: float = 0.01
    contrast_jitter: float = 0.01
    saturation_jitter: float = 0.01
    hue_jitter: float = 0.01
    
    # Advanced configurations
    curriculum_learning: bool = True
    curriculum_schedule: Dict[str, Any] = field(default_factory=lambda: {
        "start_resolution": 64,
        "end_resolution": 256,
        "resolution_steps": 100000,
        "attribute_difficulty": {
            "start_step": 0,
            "end_step": 100000,
            "easy_attributes": ["Smiling", "Male", "Young"],
            "hard_attributes": ["Narrow_Eyes", "Bags_Under_Eyes", "Receding_Hairline"]
        }
    })

@dataclass
class DataConfig:
    """Dataset configuration"""
    data_root: str = "data/celeba"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    
    # Higher resolution options if using CelebA-HQ
    use_celeba_hq: bool = False
    hq_root: str = "data/celeba-hq"
    
    # Dataset preprocessing
    img_size: int = 256
    center_crop: bool = True
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    
    # Metadata handling
    metadata_path: str = "data/celeba/list_attr_celeba.txt"
    identity_path: str = "data/celeba/identity_CelebA.txt"
    landmark_path: str = "data/celeba/list_landmarks_align_celeba.txt"
    
    # Sample validation subsets
    val_num_samples: int = 5000
    val_identities: int = 100
    
    # Caching options
    cache_latents: bool = True
    cache_dir: str = "cache/latents"
    num_workers: int = 8
    pin_memory: bool = True

@dataclass
class LoggingConfig:
    """Logging and tracking configuration"""
    project_name: str = "celeba-flux"
    experiment_name: str = "base-run"
    use_wandb: bool = True
    use_tensorboard: bool = True
    log_dir: str = "logs"
    sample_dir: str = "samples"
    checkpoint_dir: str = "checkpoints"
    
    # Evaluation metrics to track
    metrics: List[str] = field(default_factory=lambda: [
        "fid", "is", "attribute_accuracy", "lpips", "psnr", "face_identity_consistency"
    ])
    
    # Face-specific metrics
    face_landmarks_tracking: bool = True
    face_recognition_model: str = "facenet"  # "facenet", "arcface", "sphereface"
    
    # Logging details
    log_histograms: bool = True
    log_gradients: bool = False
    log_memory: bool = True
    log_system_metrics: bool = True

@dataclass
class InferenceConfig:
    """Inference configuration"""
    sample_batch_size: int = 25
    guidance_scale: float = 3.5
    attribute_conditioning: bool = True
    identity_morphing: bool = False
    interpolation_steps: int = 10
    save_intermediates: bool = True
    output_format: str = "png"  # "png", "jpg", "webp"
    visualization_grid_size: Tuple[int, int] = (5, 5)

@dataclass
class FluxConfig:
    """Complete configuration for Flux rectified flow model"""
    # Component configurations
    model: FluxModelConfig = field(default_factory=FluxModelConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    optimizer: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    ode_solver: ODESolverConfig = field(default_factory=ODESolverConfig)
    
    # Global settings
    seed: int = 42
    device: str = "cuda"
    precision: str = "bf16"
    compile_model: bool = True
    distributed_training: bool = True
    model_size: str = "base"  # "small", "base", "large"
    checkpoint_path: Optional[str] = None
    resume_training: bool = False
    
    def get_model_config_by_size(self) -> FluxModelConfig:
        """Get model configuration based on selected size"""
        model_config = self.model
        scaling = self.model.model_scaling[self.model_size]
        
        for key, value in scaling.items():
            setattr(model_config, key, value)
            
        return model_config
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> 'FluxConfig':
        """Create configuration from nested dictionary"""
        # Implement parsing logic here if needed
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FluxConfig':
        """Load configuration from file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.create_from_dict(config_dict)


def get_default_celeba_config(model_size: str = "base") -> FluxConfig:
    """Get default configuration for CelebA dataset with specified model size"""
    config = FluxConfig(model_size=model_size)
    
    # Adjust settings based on model size
    if model_size == "small":
        config.training.batch_size = 128
        config.training.total_steps = 300000
    elif model_size == "large":
        config.training.batch_size = 32
        config.training.gradient_accumulation_steps = 2
        config.training.total_steps = 500000
        config.optimizer.learning_rate = 8e-5
    
    return config
