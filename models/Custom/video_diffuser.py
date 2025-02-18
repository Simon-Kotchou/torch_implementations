mport torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from typing import List, Tuple, Optional, Union, Dict, Callable

# Import our previously defined components
from temporal_memory_integration import DiffusionMemory, TemporalAttention
from depth_aware_convolutions import DepthAwareConv, DepthAwareResBlock, DepthAwareTemporalBlock


class HunyuanVideoDiffusionModel(nn.Module):
    """
    HunyuanVideo diffusion model implementation.
    
    This model implements the dual-stream to single-stream architecture with:
    1. Causal 3D VAE for video compression
    2. Dual-stream processing for text and video
    3. Single-stream fusion layers
    4. Depth-aware temporal processing
    5. 3D Rotary position embeddings
    6. Flow matching training objective
    """
    def __init__(
        self,
        latent_channels: int = 16,
        model_channels: int = 3072,
        time_embed_dim: int = 1024,
        text_embed_dim: int = 4096,
        context_dim: int = 3072,
        num_res_blocks: int = 3,
        attention_resolutions: List[int] = [8, 16, 32],
        dropout: float = 0.1,
        channel_mult: List[int] = [1, 2, 3, 4],
        num_heads: int = 24,
        num_head_channels: int = 128,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        use_temporal_memory: bool = True,
        use_depth_aware_conv: bool = True,
        num_dual_stream_blocks: int = 20,
        num_single_stream_blocks: int = 40,
        max_frames: int = 32,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.time_embed_dim = time_embed_dim
        self.text_embed_dim = text_embed_dim
        self.context_dim = context_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_temporal_memory = use_temporal_memory
        self.use_depth_aware_conv = use_depth_aware_conv
        self.num_dual_stream_blocks = num_dual_stream_blocks
        self.num_single_stream_blocks = num_single_stream_blocks
        self.max_frames = max_frames
        self.use_flash_attention = use_flash_attention
        
        # Define model architecture components
        
        # Timestep embedding for diffusion model
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        # Initial projection from latent space to model dimension
        self.input_proj = nn.Conv3d(
            latent_channels,
            model_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )
        
        # 3D Positional Encoding
        self.pos_encoding = PositionalEncoding3D(
            d_model=model_channels,
            max_seq_length=max_frames * 128 * 128  # Typical max sequence length for coords
        )
        
        # ===== DUAL STREAM PATHWAY =====
        # Video stream processing
        self.video_dual_blocks = nn.ModuleList([
            DualStreamBlock(
                dim=model_channels,
                num_heads=num_heads,
                dim_head=num_head_channels,
                dropout=dropout,
                use_temporal_memory=use_temporal_memory,
                use_depth_aware_conv=use_depth_aware_conv and (i % 4 == 0),
                use_cross_attention=False
            )
            for i in range(num_dual_stream_blocks)
        ])
        
        # Text stream processing
        self.text_dual_blocks = nn.ModuleList([
            TextEncoderBlock(
                dim=context_dim,
                num_heads=num_heads // 2,
                dim_head=num_head_channels,
                dropout=dropout
            )
            for _ in range(num_dual_stream_blocks)
        ])
        
        # CLIP global feature projection
        self.global_text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        )
        
        # ===== SINGLE STREAM PATHWAY =====
        self.single_stream_blocks = nn.ModuleList([
            SingleStreamBlock(
                dim=model_channels,
                context_dim=context_dim,
                num_heads=num_heads,
                dim_head=num_head_channels,
                dropout=dropout,
                use_temporal_memory=use_temporal_memory and (i % 4 == 0),
                use_depth_aware_conv=use_depth_aware_conv and (i % 4 == 0)
            )
            for i in range(num_single_stream_blocks)
        ])
        
        # Output projection back to latent space
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv3d(
                model_channels, 
                latent_channels, 
                kernel_size=(1, 3, 3), 
                padding=(0, 1, 1)
            )
        )
        
        # Activation function
        self.act = nn.SiLU()
        
    def get_position_indices(self, batch_size, time_steps, height, width):
        """
        Generate position indices for 3D RoPE.
        
        Args:
            batch_size: Batch size
            time_steps: Number of time steps
            height, width: Spatial dimensions
            
        Returns:
            t_idx, h_idx, w_idx: Position indices for each dimension
        """
        # Create coordinate tensors for each dimension
        t_coords = torch.arange(time_steps, device=self.device)
        h_coords = torch.arange(height, device=self.device)
        w_coords = torch.arange(width, device=self.device)
        
        # Create meshgrid of coordinates
        t_idx, h_idx, w_idx = torch.meshgrid(t_coords, h_coords, w_coords, indexing='ij')
        
        # Reshape to sequence format
        t_idx = t_idx.reshape(time_steps * height * width).unsqueeze(0).expand(batch_size, -1)
        h_idx = h_idx.reshape(time_steps * height * width).unsqueeze(0).expand(batch_size, -1)
        w_idx = w_idx.reshape(time_steps * height * width).unsqueeze(0).expand(batch_size, -1)
        
        return t_idx, h_idx, w_idx
    
    def patchify(self, x):
        """
        Convert 5D tensor [B, C, T, H, W] to sequence of patches [B, T*H'*W', C*p*p]
        where p is the patch size.
        """
        batch, channels, time, height, width = x.shape
        
        # Use 3D convolution with patch_size×patch_size kernel to create patches
        # For now, we use the input_proj directly
        x_patched = self.input_proj(x)
        
        # Reshape to sequence format
        x_seq = einops.rearrange(
            x_patched, 
            'b c t h w -> b (t h w) c'
        )
        
        return x_seq
    
    def unpatchify(self, x_seq, time, height, width):
        """
        Convert sequence of patches back to 5D tensor
        """
        batch, seq_len, channels = x_seq.shape
        
        # Reshape back to spatial-temporal format
        x_spatial = einops.rearrange(
            x_seq,
            'b (t h w) c -> b c t h w',
            t=time, h=height, w=width
        )
        
        return x_spatial
    
    @property
    def device(self):
        return next(self.parameters()).device