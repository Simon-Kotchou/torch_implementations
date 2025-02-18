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
    
    def forward(
        self, 
        x, 
        timesteps, 
        text_embeddings, 
        clip_text_embeddings=None, 
        depth_maps=None,
        frame_indices=None
    ):
        """
        Forward pass of HunyuanVideo model.
        
        Args:
            x: Input latent video [B, C, T, H, W]
            timesteps: Diffusion timesteps [B]
            text_embeddings: Text embeddings from LLM [B, L, D]
            clip_text_embeddings: Optional global CLIP text features [B, D_clip]
            depth_maps: Optional depth maps [B, T, 1, H, W]
            frame_indices: Optional frame position indices [B, T]
            
        Returns:
            Predicted velocity field
        """
        batch_size, channels, time_steps, height, width = x.shape
        
        # Time embedding
        t_emb = self.time_embed(timesteps)  # [B, D_t]
        
        # Patchify input video
        x_seq = self.patchify(x)  # [B, T*H*W, C]
        
        # Get position indices for 3D RoPE
        t_idx, h_idx, w_idx = self.get_position_indices(
            batch_size, time_steps, height, width
        )
        
        # Process global text features if provided
        global_context = None
        if clip_text_embeddings is not None:
            global_context = self.global_text_proj(clip_text_embeddings)  # [B, D_c]
        
        # ===== DUAL STREAM PROCESSING =====
        video_features = x_seq
        text_features = text_embeddings
        
        for i in range(self.num_dual_stream_blocks):
            # Process video stream
            video_features = self.video_dual_blocks[i](
                video_features,
                t_emb,
                t_idx, h_idx, w_idx,
                depth_maps=depth_maps if self.use_depth_aware_conv else None,
                global_context=global_context
            )
            
            # Process text stream
            text_features = self.text_dual_blocks[i](text_features)
        
        # ===== SINGLE STREAM PROCESSING =====
        # Concatenate video and text features
        concat_features = torch.cat([video_features, text_features], dim=1)
        
        # Update position indices for concatenated sequence
        text_seq_len = text_features.shape[1]
        t_idx_extended = torch.cat([
            t_idx,
            torch.zeros(batch_size, text_seq_len, device=self.device)
        ], dim=1)
        h_idx_extended = torch.cat([
            h_idx,
            torch.zeros(batch_size, text_seq_len, device=self.device)
        ], dim=1)
        w_idx_extended = torch.cat([
            w_idx,
            torch.zeros(batch_size, text_seq_len, device=self.device)
        ], dim=1)
        
        # Process through single stream blocks
        for i in range(self.num_single_stream_blocks):
            concat_features = self.single_stream_blocks[i](
                concat_features,
                t_emb,
                t_idx_extended, h_idx_extended, w_idx_extended,
                depth_maps=depth_maps if self.use_depth_aware_conv else None,
                global_context=global_context
            )
        
        # Extract video features from concatenated sequence
        video_seq_len = time_steps * height * width
        video_output = concat_features[:, :video_seq_len]
        
        # Reshape back to spatial-temporal format
        video_spatial = self.unpatchify(video_output, time_steps, height, width)
        
        # Final projection to output space
        output = self.output_proj(video_spatial)
        
        return output


class DualStreamBlock(nn.Module):
    """
    Dual stream block for processing video features independently from text.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        dim_head=64,
        dropout=0.0,
        use_temporal_memory=True,
        use_depth_aware_conv=False,
        use_cross_attention=False,
        context_dim=None
    ):
        super().__init__()
        self.use_temporal_memory = use_temporal_memory
        self.use_depth_aware_conv = use_depth_aware_conv
        self.use_cross_attention = use_cross_attention
        
        # Self-attention with RoPE
        self.attention = SelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        # Optional cross-attention
        if use_cross_attention and context_dim is not None:
            self.cross_attention = CrossAttentionBlock(
                query_dim=dim,
                context_dim=context_dim,
                heads=num_heads,
                dim_head=dim_head,
                dropout=dropout
            )
            self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Optional temporal memory
        if use_temporal_memory:
            self.temporal_memory = DiffusionMemory(
                channels=dim,
                memory_length=16
            )
        
        # Optional depth-aware processing
        if use_depth_aware_conv:
            self.depth_block = DepthAwareConv(
                in_channels=dim,
                out_channels=dim
            )
        
        # Global context conditioning
        self.global_cond = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        ) if context_dim is not None else None
        
    def forward(
        self,
        x,
        time_emb,
        t_idx, h_idx, w_idx,
        depth_maps=None,
        global_context=None
    ):
        # Self-attention with RoPE
        h = self.norm1(x)
        h = self.attention(h, t_idx, h_idx, w_idx)
        x = x + h
        
        # Cross-attention (if enabled)
        if self.use_cross_attention and hasattr(self, 'cross_attention'):
            h = self.norm2(x)
            h = self.cross_attention(h, global_context.unsqueeze(1) if global_context is not None else None)
            x = x + h
        
        # Feed-forward network
        h = self.norm3(x)
        h = self.ffn(h)
        x = x + h
        
        # Apply global conditioning (if provided)
        if global_context is not None and self.global_cond is not None:
            global_features = self.global_cond(global_context).unsqueeze(1)
            x = x + global_features
        
        # Reshape for temporal memory and depth-aware processing
        batch_size, seq_len, channels = x.shape
        
        if self.use_temporal_memory or self.use_depth_aware_conv:
            # Inference time: need to infer dimensions
            time_steps = int(seq_len ** (1/3))
            height = width = int((seq_len / time_steps) ** 0.5)
            
            # Reshape to spatial-temporal format
            x_spatial = einops.rearrange(
                x,
                'b (t h w) c -> b c t h w',
                t=time_steps, h=height, w=width
            )
            
            # Apply temporal memory
            if self.use_temporal_memory:
                x_spatial = self.temporal_memory(x_spatial)
            
            # Apply depth-aware processing
            if self.use_depth_aware_conv and depth_maps is not None:
                # Process each frame with depth awareness
                outputs = []
                for t in range(time_steps):
                    frame = x_spatial[:, :, t]  # [B, C, H, W]
                    depth = depth_maps[:, t]    # [B, 1, H, W]
                    
                    # Apply depth-aware processing
                    out_t = self.depth_block(frame, depth)
                    outputs.append(out_t)
                
                # Stack processed frames
                x_spatial = torch.stack(outputs, dim=2)  # [B, C, T, H, W]
            
            # Reshape back to sequence format
            x = einops.rearrange(
                x_spatial,
                'b c t h w -> b (t h w) c'
            )
            
        return x