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