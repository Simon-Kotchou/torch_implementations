import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tqdm.notebook import tqdm
import einops
from IPython.display import clear_output
import math
from typing import List, Tuple, Optional, Union, Dict, Callable

class FlowMatchingScheduler:
    """
    Scheduler for Flow Matching based diffusion process.
    Implements time-step scheduling strategies for video diffusion.
    """
    def __init__(
        self,
        num_inference_steps=50,
        scheduler_type='shifted',
        shifting_factor=7.0,
        min_t=0.002,
        max_t=0.998
    ):
        self.num_inference_steps = num_inference_steps
        self.scheduler_type = scheduler_type
        self.shifting_factor = shifting_factor
        self.min_t = min_t
        self.max_t = max_t
        
        # Create timestep schedule based on selected strategy
        self.timesteps = self._get_schedule()
        
    def _get_schedule(self):
        """
        Generate timestep schedule based on chosen strategy.
        """
        if self.scheduler_type == 'linear':
            # Linear schedule from max_t to min_t
            return torch.linspace(
                self.max_t, self.min_t, self.num_inference_steps
            )
        
        elif self.scheduler_type == 'quadratic':
            # Quadratic schedule giving more steps to early diffusion
            steps = np.linspace(0, 1, self.num_inference_steps)
            # Convert to quadratic curve
            steps = 1 - np.square(steps)
            # Scale to desired range
            steps = steps * (self.max_t - self.min_t) + self.min_t
            return torch.from_numpy(steps).float()
        
        elif self.scheduler_type == 'cosine':
            # Cosine schedule
            steps = torch.arange(self.num_inference_steps + 1).float() / self.num_inference_steps
            alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            timesteps = torch.clamp(alpha_cumprod, self.min_t, self.max_t)
            return timesteps
        
        elif self.scheduler_type == 'shifted':
            # Shifted schedule using the shifting factor
            # This focuses more steps on early diffusion (crucial for fewer steps)
            steps = np.linspace(0, 1, self.num_inference_steps)
            # Apply shifting function t' = s*t/(1+(s-1)*t)
            s = self.shifting_factor
            steps = s * steps / (1 + (s - 1) * steps)
            # Scale to desired range
            steps = (1 - steps) * (self.max_t - self.min_t) + self.min_t
            return torch.from_numpy(steps).float()
            
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def get_timesteps(self):
        """
        Get all timesteps for inference.
        """
        return self.timesteps
    
    def adjust_shifting_factor(self, num_steps):
        """
        Dynamically adjust shifting factor based on number of steps.
        Lower inference steps require higher shifting factor.
        """
        if num_steps <= 10:
            self.shifting_factor = 17.0
        elif num_steps <= 20:
            self.shifting_factor = 15.0
        elif num_steps <= 30:
            self.shifting_factor = 12.0
        elif num_steps <= 40:
            self.shifting_factor = 9.0
        else:
            self.shifting_factor = 7.0
            
        # Update timesteps with new shifting factor
        if self.scheduler_type == 'shifted':
            self.timesteps = self._get_schedule()
        
        return self.timesteps


class NoiseUtils:
    """
    Utilities for managing noise in diffusion models.
    Provides methods for adding/removing noise and computing diffusion targets.
    """
    @staticmethod
    def q_sample(x_start, x_noise, t):
        """
        Diffuse data to timestep t by interpolating between
        data and noise according to schedule.
        
        Args:
            x_start: Starting clean data
            x_noise: Random noise
            t: Diffusion timesteps [batch_size]
            
        Returns:
            Noisy samples at timestep t
        """
        # Reshape t for broadcasting
        if len(x_start.shape) == 2:  # For 2D data
            t = t.view(-1, 1)
        else:  # For video data
            t = t.view(-1, 1, 1, 1, 1)
            
        # Linear interpolation between start and noise
        # t=1 is all signal, t=0 is all noise
        return t * x_start + (1 - t) * x_noise
    
    @staticmethod
    def compute_flow_velocity(x_start, x_noise):
        """
        Compute the ground-truth velocity field for Flow Matching.
        
        Args:
            x_start: Clean data
            x_noise: Noise data
            
        Returns:
            Velocity field (direction from noise to data)
        """
        # For linear interpolation, velocity is x_target - x_noise
        return x_start - x_noise
    
    @staticmethod
    def get_noise_level_embedding(timesteps, embedding_dim=256):
        """
        Sinusoidal embedding for noise levels/timesteps.
        
        Args:
            timesteps: Timesteps to embed [batch_size]
            embedding_dim: Dimension of embedding
            
        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:  # Zero pad if dim is odd
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
            
        return emb
    
    @staticmethod
    def noise_like(shape, device, repeat=False):
        """
        Create new noise tensor or repeat noise for video frames.
        
        Args:
            shape: Shape of noise tensor
            device: Device to create tensor on
            repeat: If True, generates one noise and repeats across frames
            
        Returns:
            Noise tensor
        """
        if repeat:
            # Create one noise and repeat for all frames
            # Useful for frame-consistent noise
            batch, channel, time, height, width = shape
            noise = torch.randn((batch, channel, 1, height, width), device=device)
            return noise.expand(batch, channel, time, height, width)
        else:
            # Different noise for each frame
            return torch.randn(shape, device=device)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a simple video UNet model for testing
class SimpleVideoUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=32, feature_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Downsampling
        self.down1 = nn.Conv3d(in_channels, feature_dim, kernel_size=3, padding=1)
        self.down2 = nn.Conv3d(feature_dim, feature_dim*2, kernel_size=3, stride=2, padding=1)
        
        # Middle
        self.mid1 = nn.Conv3d(feature_dim*2, feature_dim*2, kernel_size=3, padding=1)
        self.mid2 = nn.Conv3d(feature_dim*2, feature_dim*2, kernel_size=3, padding=1)
        
        # Upsampling
        self.up1 = nn.ConvTranspose3d(feature_dim*2, feature_dim, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.Conv3d(feature_dim*2, feature_dim, kernel_size=3, padding=1)
        
        # Output
        self.out = nn.Conv3d(feature_dim, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, timesteps):
        # Embed time
        t_emb = NoiseUtils.get_noise_level_embedding(timesteps, 32)
        t_emb = self.time_embedding(t_emb)
        
        # Downsample
        x1 = F.silu(self.down1(x))
        x2 = F.silu(self.down2(x1))
        
        # Middle with time embedding
        x2 = self.mid1(x2)
        # Add time embedding to each spatial location
        t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1, 1)
        x2 = x2 + t_emb
        x2 = F.silu(x2)
        x2 = F.silu(self.mid2(x2))
        
        # Upsample
        x = F.silu(self.up1(x2))
        # Skip connection
        x = torch.cat([x, x1], dim=1)
        x = F.silu(self.up2(x))
        
        # Output
        return self.out(x)

def generate_2d_flow_visualization():
    """Generate a visualization of 2D flow matching with make_moons dataset"""
    print("Generating 2D flow visualization...")
    
    # Generate target distribution from make_moons
    n_samples = 2000
    X_target, _ = make_moons(n_samples=n_samples, noise=0.05)
    X_target = torch.tensor(X_target, dtype=torch.float32)
    
    # Generate source distribution (Gaussian noise)
    X_source = torch.randn_like(X_target)
    
    # Create scheduler
    scheduler = FlowMatchingScheduler(num_inference_steps=100, scheduler_type='shifted')
    timesteps = scheduler.get_timesteps()
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Number of steps to visualize
    vis_steps = 10
    step_indices = np.linspace(0, len(timesteps)-1, vis_steps, dtype=int)
    
    for i, idx in enumerate(step_indices):
        t = timesteps[idx].item()
        
        # Interpolate between noise and target based on timestep
        X_t = NoiseUtils.q_sample(X_target, X_source, torch.tensor([t]))
        
        # Compute the velocity at this timestep
        velocity = NoiseUtils.compute_flow_velocity(X_target, X_source)
        
        # Subsample vectors for cleaner visualization
        subsample = 100
        indices = np.random.choice(n_samples, subsample, replace=False)
        
        plt.subplot(2, 5, i+1)
        plt.scatter(X_t[:, 0], X_t[:, 1], s=10, alpha=0.6, c=torch.linspace(0, 1, n_samples))
        
        # Plot velocity vectors
        if i < vis_steps - 1:  # Don't plot vectors for final state
            scale = 0.1  # Scale factor for vectors
            plt.quiver(
                X_t[indices, 0].numpy(), 
                X_t[indices, 1].numpy(),
                velocity[indices, 0].numpy(), 
                velocity[indices, 1].numpy(),
                color='red', alpha=0.8, scale=1/scale, width=0.003
            )
        
        plt.title(f't = {t:.2f}')
        plt.grid(alpha=0.3)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('flow_matching_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_video_flow_matching():
    """Test the flow matching components on simple video data"""
    print("Testing video flow matching...")
    
    # Create a simple video shape: [batch, channels, time, height, width]
    batch_size = 2
    channels = 3
    frames = 8
    height = 32
    width = 32
    
    # Create model
    model = SimpleVideoUNet().to(device)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create scheduler
    scheduler = FlowMatchingScheduler(
        num_inference_steps=50,
        scheduler_type='shifted',
        shifting_factor=9.0
    )
    timesteps = scheduler.get_timesteps().to(device)
    
    # Visualize the different scheduler types
    plt.figure(figsize=(12, 6))
    
    scheduler_types = ['linear', 'quadratic', 'cosine', 'shifted']
    for i, sched_type in enumerate(scheduler_types):
        sched = FlowMatchingScheduler(num_inference_steps=50, scheduler_type=sched_type)
        ts = sched.get_timesteps().cpu().numpy()
        
        plt.subplot(2, 2, i+1)
        plt.plot(np.arange(len(ts)), ts, 'o-', linewidth=2)
        plt.title(f'{sched_type.capitalize()} Scheduler')
        plt.xlabel('Step')
        plt.ylabel('Timestep t')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scheduler_types.png', dpi=150)
    plt.show()
    
    # Generate target and noise samples
    # Target: Use sin waves along time dimension for interesting patterns
    t_vals = torch.linspace(0, 2*np.pi, frames)
    h_vals = torch.linspace(0, 2*np.pi, height)
    w_vals = torch.linspace(0, 2*np.pi, width)
    
    # Create wave patterns
    t_grid = einops.rearrange(t_vals, 't -> t 1 1')
    h_grid = einops.rearrange(h_vals, 'h -> 1 h 1')
    w_grid = einops.rearrange(w_vals, 'w -> 1 1 w')
    
    # Generate patterns with phase shifts for each channel
    patterns = []
    for phase in [0, 2*np.pi/3, 4*np.pi/3]:  # 120° phase shifts
        pattern = torch.sin(t_grid + phase) * torch.sin(h_grid) * torch.sin(w_grid)
        patterns.append(pattern)
    
    # Stack channels
    target_video = torch.stack(patterns, dim=0)
    # Add batch dimension and normalize to [-1, 1]
    target_video = target_video.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    target_video = target_video.to(device)
    
    # Generate noise video
    noise_video = torch.randn_like(target_video).to(device)
    
    # Testing inference steps
    print("Running inference steps...")
    
    frames_to_show = 4
    inference_video = []
    
    with torch.no_grad():
        # Start from pure noise
        x_t = noise_video.clone()
        
        # Visualize initial state
        inference_video.append(x_t.detach().cpu())
        
        # Run inference steps
        for i, t in enumerate(tqdm(timesteps)):
            # Expand timestep for batch
            timestep_batch = t.expand(batch_size)
            
            # Get model prediction (velocity)
            velocity = model(x_t, timestep_batch)
            
            # Euler step
            dt = 1.0 / len(timesteps)
            x_t = x_t + velocity * dt
            
            # Save every few steps for visualization
            if i % (len(timesteps) // frames_to_show) == 0 or i == len(timesteps) - 1:
                inference_video.append(x_t.detach().cpu())
    
    # Visualize the inference process
    plt.figure(figsize=(15, 10))
    
    # Plot the first video in the batch
    steps_to_viz = len(inference_video)
    for i, video in enumerate(inference_video):
        video_data = video[0].numpy()  # First batch
        
        # Plot each channel separately
        for c in range(channels):
            plt.subplot(steps_to_viz, channels, i*channels + c + 1)
            
            # Show middle frame
            middle_frame = video_data[c, frames//2]
            plt.imshow(middle_frame, cmap='viridis')
            plt.title(f'Step {i}, Ch {c}, Frame {frames//2}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('video_flow_inference.png', dpi=150)
    plt.show()
    
    # Animated visualization of a single channel over time
    from matplotlib.animation import FuncAnimation
    
    def create_animation(video_frames, channel=0):
        """Create animation of one channel from a list of video frames"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get min/max for consistent colormap scaling
        all_data = torch.cat(video_frames, dim=0)
        vmin = all_data.min().item()
        vmax = all_data.max().item()
        
        # First frame
        im = ax.imshow(
            video_frames[0][0, channel, frames//2].numpy(),
            cmap='plasma',
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(im, ax=ax)
        title = ax.set_title('Initial State')
        
        def update(frame_idx):
            """Update function for animation"""
            data = video_frames[frame_idx][0, channel, frames//2].numpy()
            im.set_array(data)
            progress = frame_idx / (len(video_frames) - 1)
            title.set_text(f'Step {frame_idx}, Progress: {progress:.1%}')
            return [im, title]
        
        ani = FuncAnimation(
            fig, update, frames=len(video_frames), interval=300, blit=False
        )
        return ani
    
    # Create and save animation
    ani = create_animation(inference_video)
    ani.save('flow_animation.gif', writer='pillow', fps=4, dpi=100)
    
    print("Testing completed. Visualizations saved.")

if __name__ == "__main__":
    # Run 2D flow matching visualization
    generate_2d_flow_visualization()
    
    # Run video flow matching test
    test_video_flow_matching()