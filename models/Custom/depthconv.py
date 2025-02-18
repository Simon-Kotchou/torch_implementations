import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class DepthAwareConv(nn.Module):
    """
    Depth-aware convolution module that incorporates geometric information
    from Depth Pro outputs through depth-modulated convolutions.
    
    This module enhances standard convolutions by conditioning them on depth maps,
    allowing the network to reason about spatial geometry during diffusion.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Standard spatial convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Depth feature extraction
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        # Feature modulation
        self.modulation = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Adaptive instance normalization parameters
        self.adain_scale = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.adain_bias = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x, depth_map):
        """
        Forward pass applying depth-aware convolution.
        
        Args:
            x: Input feature maps [B, C, H, W]
            depth_map: Depth information [B, 1, H, W]
            
        Returns:
            Depth-modulated feature maps
        """
        batch_size = x.shape[0]
        
        # Ensure depth map dimensions match input
        if depth_map.shape[-2:] != x.shape[-2:]:
            depth_map = F.interpolate(
                depth_map, 
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
        # Get depth embeddings
        depth_features = self.depth_encoder(depth_map)
        
        # Apply standard convolution
        conv_features = self.conv(x)
        
        # Compute modulation factors
        mod_factors = self.modulation(depth_features)
        
        # Apply AdaIN-style conditioning
        scales = self.adain_scale(depth_features)
        biases = self.adain_bias(depth_features)
        
        # Normalize features
        mean = conv_features.mean(dim=(2, 3), keepdim=True)
        std = conv_features.std(dim=(2, 3), keepdim=True) + 1e-5
        normalized = (conv_features - mean) / std
        
        # Apply depth-based modulation
        output = normalized * scales + biases
        
        # Apply multiplicative gating
        output = output * mod_factors
        
        return output
    
class DepthAwareResBlock(nn.Module):
    """
    Residual block with depth-aware convolutions for video diffusion models.
    Incorporates geometric information for improved spatial reasoning.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None, use_scale_shift_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # First depth-aware convolution with normalization
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = DepthAwareConv(in_channels, out_channels)
        
        # Optional timestep embedding for diffusion models
        self.has_time_emb = time_emb_dim is not None
        if self.has_time_emb:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2 if use_scale_shift_norm else out_channels)
            )
            
        # Second depth-aware convolution with normalization  
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = DepthAwareConv(out_channels, out_channels)
        
        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x, depth_map, time_emb=None):
        """
        Forward pass through the depth-aware residual block.
        
        Args:
            x: Input tensor [B, C, H, W]
            depth_map: Depth maps [B, 1, H, W]
            time_emb: Optional timestep embeddings for diffusion [B, D]
            
        Returns:
            Updated feature maps with depth awareness
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h, depth_map)
        
        # Apply timestep embedding if provided
        if self.has_time_emb:
            assert time_emb is not None
            time_emb = self.time_mlp(time_emb)
            
            if self.use_scale_shift_norm:
                scale, shift = torch.chunk(time_emb, 2, dim=1)
                scale = scale.unsqueeze(-1).unsqueeze(-1)
                shift = shift.unsqueeze(-1).unsqueeze(-1)
                h = self.norm2(h) * (1 + scale) + shift
                h = F.silu(h)
            else:
                h = F.silu(self.norm2(h))
                h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        else:
            h = F.silu(self.norm2(h))
            
        # Second convolution
        h = self.conv2(h, depth_map)
        
        # Skip connection
        return h + self.skip(x)


class DepthAwareTemporalBlock(nn.Module):
    """
    Combined depth-aware and temporal block for video diffusion,
    integrating both geometric and temporal consistency.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        
        # Depth-aware processing
        self.depth_block = DepthAwareResBlock(in_channels, out_channels, time_emb_dim)
        
        # Temporal consistency processing
        self.temporal_conv = nn.Conv3d(
            out_channels, 
            out_channels, 
            kernel_size=(3, 1, 1), 
            padding=(1, 0, 0)
        )
        
        self.temporal_norm = nn.GroupNorm(32, out_channels)
        
    def forward(self, x, depth_maps, time_emb=None):
        """
        Forward pass integrating both depth awareness and temporal consistency.
        
        Args:
            x: Input features [B, C, T, H, W]
            depth_maps: Depth information [B, T, 1, H, W]
            time_emb: Optional timestep embeddings [B, D]
            
        Returns:
            Processed features with both geometric and temporal consistency
        """
        batch, channels, time, height, width = x.shape
        
        # Process each frame with depth awareness
        outputs = []
        for t in range(time):
            frame = x[:, :, t]
            depth = depth_maps[:, t]
            
            # Apply depth-aware processing
            out_t = self.depth_block(frame, depth, time_emb)
            outputs.append(out_t)
            
        # Stack outputs along temporal dimension
        stacked = torch.stack(outputs, dim=2)  # [B, C, T, H, W]
        
        # Apply temporal convolution
        temp_features = F.silu(self.temporal_norm(stacked.transpose(1, 2)).transpose(1, 2))
        temp_out = self.temporal_conv(temp_features)
        
        return temp_out