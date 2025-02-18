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

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for global feature propagation across frames.
    Implements a simplified version of the transformer attention blocks from SAM 2.
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        # QKV projections
        self.q_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)
        
        # Learnable temporal positional embedding
        self.time_pos_emb = nn.Parameter(torch.zeros(1, channels, 16, 1, 1))
        
    def forward(self, x):
        """
        Forward pass through temporal attention.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        
        Returns:
            Attention-enhanced features
        """
        batch, channels, time, height, width = x.shape
        
        # Add positional embeddings (truncate or expand as needed)
        pos_emb = self.time_pos_emb[:, :, :time]
        if time > pos_emb.shape[2]:
            # Handle longer sequences through interpolation
            pos_emb = F.interpolate(
                pos_emb, 
                size=(time, 1, 1), 
                mode='trilinear', 
                align_corners=False
            )
        
        x_pos = x + pos_emb
        
        # Project to queries, keys, values
        q = self.q_proj(x_pos)
        k = self.k_proj(x_pos)
        v = self.v_proj(x_pos)
        
        # Reshape for multi-head attention
        q = einops.rearrange(q, 'b (h d) t h w -> b h t (h w) d', h=self.num_heads)
        k = einops.rearrange(k, 'b (h d) t h w -> b h t (h w) d', h=self.num_heads)
        v = einops.rearrange(v, 'b (h d) t h w -> b h t (h w) d', h=self.num_heads)
        
        # Compute attention scores (scaled dot-product attention)
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhtpd,bhtqd->bhtpq', q, k) * scale
        
        # Attention weights
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhtpq,bhtqd->bhtpd', attn, v)
        
        # Reshape back to original format
        out = einops.rearrange(out, 'b h t (h w) d -> b (h d) t h w', h=height, w=width)
        
        # Final projection
        out = self.out_proj(out)
        
        return out


# Full integration with U-Net for video diffusion
class VideoUNetBlock(nn.Module):
    """
    U-Net block with integrated temporal memory for video diffusion models.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Main convolution blocks
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()
        
        # Temporal memory integration
        self.memory = DiffusionMemory(channels=out_channels)
        
        # Skip connection if needed
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)
        
    def forward(self, x, t_emb):
        """
        Forward pass incorporating temporal memory and timestep embeddings.
        
        Args:
            x: Input feature maps [B, C, T, H, W]
            t_emb: Timestep embeddings [B, D]
        """
        # Main path
        h = self.act(self.norm1(self.conv1(x)))
        
        # Add time embeddings
        t = self.time_mlp(t_emb)
        t = einops.rearrange(t, 'b c -> b c 1 1 1')
        h = h + t
        
        # Second conv
        h = self.act(self.norm2(self.conv2(h)))
        
        # Apply temporal memory
        h = self.memory(h)
        
        # Skip connection
        return h + self.residual(x)


# Example usage
def test_memory_module():
    # Create sample input: [batch, channels, time, height, width]
    x = torch.randn(2, 256, 8, 64, 64)
    
    # Initialize memory module
    memory = DiffusionMemory(channels=256)
    
    # Forward pass
    output = memory(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test single frame inference
    single_frame = torch.randn(2, 256, 1, 64, 64)
    memory.eval()
    output_single = memory(single_frame)
    print(f"Single frame output shape: {output_single.shape}")
    
    return output


if __name__ == "__main__":
    test_memory_module()