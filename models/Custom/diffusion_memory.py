import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class DiffusionMemory(nn.Module):
    """
    Temporal memory module adapted from SAM 2's memory bank architecture for
    ensuring frame coherence in video diffusion models.
    
    This module maintains temporal consistency by propagating features across frames
    using a 3D convolutional memory mechanism.
    """
    def __init__(self, channels=256, memory_length=8, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.memory_length = memory_length
        
        # 3D convolution for temporal feature propagation
        self.temporal_conv = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(kernel_size, 1, 1), 
            padding=(kernel_size//2, 0, 0)
        )
        
        # Attention mechanism for cross-frame feature propagation
        self.temporal_attn = TemporalAttention(channels)
        
        # Feature projection for memory bank
        self.memory_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(32, channels),
            nn.SiLU()
        )
        
        # Memory bank for storing past frame features
        self.register_buffer('memory_bank', torch.zeros(1, channels, memory_length, 1, 1))
        self.memory_idx = 0
        
    def forward(self, x):
        """
        Forward pass through the memory module.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        
        Returns:
            Enhanced features with temporal consistency
        """
        batch, channels, time, height, width = x.shape
        assert channels == self.channels, f"Channel dimension mismatch: got {channels}, expected {self.channels}"
        
        # Apply temporal convolution for local coherence
        local_features = self.temporal_conv(x)
        
        # Use attention for global temporal coherence
        attn_features = self.temporal_attn(x)
        
        # Combine local and global temporal features
        enhanced_features = local_features + attn_features
        
        # For inference mode with single frame input, we use the memory bank
        if time == 1 and self.training == False:
            return self._single_frame_inference(enhanced_features, height, width)
            
        return enhanced_features
    
    def _single_frame_inference(self, x, height, width):
        """
        Special handling for single-frame inference using memory bank.
        """
        # Get current frame features
        current_frame = x[:, :, 0]  # [B, C, H, W]
        
        # Project features for memory
        projected_features = self.memory_proj(current_frame)
        
        # Update memory bank (FIFO queue)
        if self.memory_idx < self.memory_length:
            self.memory_bank[:, :, self.memory_idx] = projected_features.mean(dim=[2, 3], keepdim=True)
        else:
            # Shift memory bank and add new features
            self.memory_bank = torch.roll(self.memory_bank, -1, dims=2)
            self.memory_bank[:, :, -1] = projected_features.mean(dim=[2, 3], keepdim=True)
        
        self.memory_idx = (self.memory_idx + 1) % self.memory_length
        
        # Apply memory features to current frame
        memory_features = einops.repeat(
            self.memory_bank, 
            'b c t 1 1 -> b c t h w', 
            h=height, 
            w=width
        )
        
        # Create artificial temporal dimension for compatibility
        x_expanded = einops.repeat(x, 'b c 1 h w -> b c t h w', t=self.memory_length)
        
        # Combine current frame with memory features
        memory_enhanced = self.temporal_conv(x_expanded * memory_features)
        
        # Return the latest frame
        return memory_enhanced[:, :, -1:, :, :]
    
    def reset_memory(self):
        """Reset the memory bank - useful between video sequences"""
        self.memory_bank.zero_()
        self.memory_idx = 0