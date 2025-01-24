"""
Comprehensive testing suite for Finite Scalar Quantization (FSQ).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
from typing import List, Tuple, Dict

from fsq import FSQ, FSQImageEncoder  # Import from previous implementation


class FSQTester:
    """Test suite for FSQ implementation."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    def test_basic_properties(self, levels: List[int]) -> None:
        """Test basic properties of FSQ module."""
        fsq = FSQ(levels).to(self.device)
        print(f"\n=== Testing FSQ with levels {levels} ===")
        
        # Test dimensions
        x = torch.randn(100, len(levels)).to(self.device)
        quantized = fsq(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {quantized.shape}")
        print(f"Codebook size: {fsq.codebook_size}")
        
        # Test value ranges
        print(f"Min value: {quantized.min().item():.3f}")
        print(f"Max value: {quantized.max().item():.3f}")
        
        # Test unique codes
        indices = fsq.codes_to_indices(quantized)
        unique_codes = len(torch.unique(indices))
        print(f"Unique codes used: {unique_codes}")
        print(f"Codebook utilization: {100 * unique_codes / fsq.codebook_size:.1f}%")
    
    def visualize_quantization(self, levels: List[int]) -> None:
        """Visualize how FSQ quantizes 2D data."""
        if len(levels) != 2:
            raise ValueError("This visualization only works for 2D FSQ")
            
        fsq = FSQ(levels).to(self.device)
        
        # Create 2D grid of points
        x = torch.linspace(-2, 2, 100)
        y = torch.linspace(-2, 2, 100)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(self.device)
        
        # Quantize points
        with torch.no_grad():
            quantized = fsq(points)
            
        # Plot original and quantized points
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.scatter(points[:, 0].cpu(), points[:, 1].cpu(), 
                   alpha=0.1, label='Original')
        plt.title('Original Points')
        plt.grid(True)
        plt.axis('equal')
        
        plt.subplot(122)
        plt.scatter(quantized[:, 0].cpu(), quantized[:, 1].cpu(),
                   alpha=0.1, label='Quantized')
        plt.title('Quantized Points')
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def test_gradients(self, levels: List[int]) -> None:
        """Test gradient flow through FSQ."""
        fsq = FSQ(levels).to(self.device)
        
        # Create input that requires gradient
        x = torch.randn(100, len(levels), requires_grad=True).to(self.device)
        
        # Forward pass
        quantized = fsq(x)
        loss = quantized.pow(2).mean()
        
        # Backward pass
        loss.backward()
        
        print("\n=== Gradient Analysis ===")
        print(f"Input gradient shape: {x.grad.shape}")
        print(f"Input gradient mean: {x.grad.abs().mean().item():.3f}")
        print(f"Input gradient std: {x.grad.std().item():.3f}")
        
    def test_reconstruction(self, levels: List[int], data_dim: int = 28) -> None:
        """Test FSQ in an autoencoder setup using MNIST."""
        # Create simple autoencoder
        encoder = FSQImageEncoder(
            in_channels=1,
            hidden_dim=64,
            fsq_levels=levels
        ).to(self.device)
        
        decoder = nn.Sequential(
            nn.ConvTranspose2d(len(levels), 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transform)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        # Training loop
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=1e-3
        )
        
        print("\n=== Testing Reconstruction ===")
        for epoch in range(5):
            total_loss = 0
            for batch, _ in loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                quantized, _ = encoder(batch)
                recon = decoder(quantized)
                loss = F.mse_loss(recon, batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(loader):.4f}")
            
        # Visualize results
        with torch.no_grad():
            batch = next(iter(loader))[0][:8].to(self.device)
            quantized, _ = encoder(batch)
            recon = decoder(quantized)
            
            plt.figure(figsize=(12, 4))
            for i in range(8):
                # Original
                plt.subplot(2, 8, i + 1)
                plt.imshow(batch[i][0].cpu(), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Original')
                    
                # Reconstruction
                plt.subplot(2, 8, i + 9)
                plt.imshow(recon[i][0].cpu(), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Reconstructed')
                    
            plt.tight_layout()
            plt.show()


def run_all_tests():
    """Run complete test suite with various configurations."""
    tester = FSQTester()
    
    # Test different configurations
    configs = [
        [8, 6, 5],      # 2^8 codes
        [8, 5, 5, 5],   # 2^10 codes
        [7, 5, 5, 5, 5] # 2^12 codes
    ]
    
    for levels in configs:
        tester.test_basic_properties(levels)
        tester.test_gradients(levels)
    
    # Visualize 2D quantization
    tester.visualize_quantization([8, 8])
    
    # Test reconstruction
    tester.test_reconstruction([8, 5, 5, 5])


if __name__ == "__main__":
    run_all_tests()