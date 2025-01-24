"""
Finite Scalar Quantization (FSQ) Implementation in PyTorch.

As described in "Finite Scalar Quantization: VQ-VAE Made Simple" (Mentzer et al., 2023)
https://arxiv.org/abs/2309.15505

This implementation provides a drop-in replacement for Vector Quantization (VQ) in VQ-VAE
architectures. FSQ projects the VAE representation down to a few dimensions (typically < 10)
and quantizes each dimension to a small set of fixed values.

Example:
    >>> fsq = FSQ(levels=[8, 5, 5, 5])  # 4 dimensions with different quantization levels
    >>> x = torch.randn(32, 4)  # batch_size=32, d=4
    >>> quantized = fsq(x)
    >>> codes = fsq.codes_to_indices(quantized)
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQ(nn.Module):
    """Finite Scalar Quantization module.
    
    Args:
        levels (List[int]): Number of quantization levels for each dimension.
            Example: [8, 5, 5, 5] means 4 dimensions with 8 levels for first dim
            and 5 levels for remaining dims.
            
    Attributes:
        dim (int): Number of dimensions (length of levels list)
        codebook_size (int): Total number of possible codes (product of levels)
        register_buffer is used for non-trainable tensors that should move with the model
    """
    
    def __init__(self, levels: List[int]) -> None:
        super().__init__()
        
        self.dim = len(levels)
        self.levels = torch.tensor(levels)
        self.codebook_size = math.prod(levels)
        
        # Compute basis for converting between codes and indices
        basis = torch.cat([
            torch.tensor([1]),
            torch.cumprod(torch.tensor(levels[:-1]), dim=0)
        ]).to(torch.int32)
        
        # Register buffers for tensors that should be saved and moved with model
        self.register_buffer("_levels", self.levels)
        self.register_buffer("_basis", basis)
        
        # Compute half width for each dimension for normalization
        self.register_buffer("_half_width", self.levels // 2)
        
    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound input tensor to prepare for quantization.
        
        Args:
            z: Input tensor of shape (..., d) where d is number of dimensions
            
        Returns:
            Bounded tensor of same shape as input
        """
        eps = 1e-3
        half_l = (self._levels - 1) * (1 - eps) / 2
        
        # Handle odd/even number of levels
        offset = torch.where(
            self._levels % 2 == 1,
            torch.tensor(0.0, device=z.device),
            torch.tensor(0.5, device=z.device)
        )
        
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift[None]) * half_l[None] - offset[None]
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor.
        
        Uses straight-through estimator for gradients through rounding operation.
        
        Args:
            z: Input tensor of shape (..., d)
            
        Returns:
            Quantized tensor of same shape as input, normalized to [-1, 1]
        """
        # Bound and quantize
        bounded = self.bound(z)
        quantized = bounded.round()
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        # Normalize to [-1, 1]
        return quantized / self._half_width[None]
        
    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert quantized codes to indices in the implicit codebook.
        
        Args:
            codes: Tensor of quantized codes shape (..., d)
            
        Returns:
            Tensor of indices shape (...) with values in [0, codebook_size)
        """
        # Scale codes from [-1, 1] to [0, L-1]
        scaled = (codes * self._half_width[None]) + self._half_width[None]
        
        # Convert to indices using basis vectors
        return (scaled * self._basis[None]).sum(dim=-1).to(torch.int32)
        
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices back to quantized codes.
        
        Args:
            indices: Tensor of indices shape (...)
            
        Returns:
            Tensor of codes shape (..., d) with values in [-1, 1]
        """
        # Add channel dim for division
        indices = indices[..., None]
        
        # Convert to codes using integer division and modulo
        codes = torch.div(indices, self._basis[None], rounding_mode='floor') 
        codes = torch.remainder(codes, self._levels[None])
        
        # Scale back to [-1, 1]
        return (codes - self._half_width[None]) / self._half_width[None]


class FSQImageEncoder(nn.Module):
    """Example encoder network that uses FSQ.
    
    Args:
        in_channels (int): Number of input image channels
        hidden_dim (int): Hidden dimension size
        fsq_levels (List[int]): Quantization levels for FSQ
    """
    
    def __init__(
        self, 
        in_channels: int = 3,
        hidden_dim: int = 256,
        fsq_levels: List[int] = [8, 5, 5, 5]
    ) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, len(fsq_levels), kernel_size=1)
        )
        
        self.fsq = FSQ(levels=fsq_levels)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Tuple of:
                - Quantized codes [B, D, H//4, W//4] 
                - Indices [B, H//4, W//4]
        """
        z = self.encoder(x)
        quantized = self.fsq(z)
        indices = self.fsq.codes_to_indices(quantized)
        return quantized, indices


def test_fsq():
    """Simple test function to verify FSQ implementation."""
    # Create FSQ module
    levels = [8, 5, 5, 5]
    fsq = FSQ(levels)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, len(levels))
    quantized = fsq(x)
    assert quantized.shape == x.shape
    assert torch.all(quantized >= -1) and torch.all(quantized <= 1)
    
    # Test index conversion
    indices = fsq.codes_to_indices(quantized)
    assert indices.shape == (batch_size,)
    assert torch.all(indices >= 0) and torch.all(indices < fsq.codebook_size)
    
    # Test roundtrip
    codes = fsq.indices_to_codes(indices)
    assert torch.allclose(codes, quantized)
    
    print("All tests passed!")


if __name__ == "__main__":
    test_fsq()