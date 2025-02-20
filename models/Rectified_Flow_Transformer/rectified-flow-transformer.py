import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm

class RectifiedFlowTransformer(nn.Module):
    def __init__(self, input_dim, spatial_dims, embed_dim=768, depth=12, 
                num_heads=12, mlp_ratio=4.0, time_embed_dim=512,
                dropout=0.1, attention_dropout=0.1):
        """
        Rectified Flow Transformer for learning the velocity field
        input_dim: number of channels in latent representation
        spatial_dims: tuple (h, w) of spatial dimensions in latent space
        """
        super().__init__()
        self.input_dim = input_dim
        self.h, self.w = spatial_dims
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # Enhanced time embedding with sinusoidal frequencies
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, embed_dim),
        )
        
        # Positional embedding for spatial dimensions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.h * self.w, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Input projection to embedding dimension
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, embed_dim // 2),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection back to input dimension
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, input_dim * self.h * self.w),
        )
        
    def forward(self, x, t):
        """
        Forward pass of the rectified flow transformer
        x: [B, C, H, W] latent features to predict velocity for
        t: [B] timesteps in [0, 1]
        """
        batch_size = x.shape[0]
        
        # Project input to embedding dimension
        h = self.input_proj(x)                           # [B, embed_dim, H, W]
        h = h.flatten(2).permute(0, 2, 1)                # [B, H*W, embed_dim]
        
        # Add positional embeddings
        h = h + self.pos_embed
        
        # Create time embeddings and add to all tokens
        t_embed = self.time_embed(t)                     # [B, embed_dim]
        t_embed = t_embed.unsqueeze(1).expand(-1, self.h * self.w, -1)
        h = h + t_embed
        
        # Process with transformer blocks
        for block in self.blocks:
            h = block(h)
        
        h = self.norm(h)                                 # [B, H*W, embed_dim]
        
        # Project to velocity field
        h = self.output_proj(h)                          # [B, H*W, C*H*W]
        velocity = h.view(batch_size, self.h * self.w, self.input_dim, self.h, self.w)
        
        # Average over token dimension - each token predicts full velocity field
        velocity = velocity.mean(dim=1)                  # [B, C, H, W]
        
        return velocity

class SinusoidalTimeEmbedding(nn.Module):
    """Enhanced sinusoidal embeddings for time"""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        t: [B] float timesteps in [0, 1]
        returns: [B, dim] time embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=t.device) / half_dim
        )
        
        # Create sinusoidal embeddings
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # For odd dimensions, add extra sine feature
        if self.dim % 2 == 1:
            embedding = torch.cat([
                embedding, 
                torch.sin(t.unsqueeze(1) * freqs[-1].unsqueeze(0))
            ], dim=-1)
            
        return embedding

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization design"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, 
                dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=dropout
        )
        
    def forward(self, x):
        # Self attention with residual connection
        x = x + self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x)
        )[0]
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class MLP(nn.Module):
    """Multi-layer perceptron with SiLU activation"""
    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RectifiedFlowODESolver:
    """ODE solver for rectified flow generation"""
    def __init__(self, flow_model, device='cuda'):
        self.flow_model = flow_model
        self.device = device
        
    def sample(self, batch_size, latent_shape, steps=100, 
              scheduler='cosine', denoise_strength=0.0, guidance_scale=1.0):
        """
        Generate samples using the rectified flow model
        batch_size: number of samples to generate
        latent_shape: shape of latent vectors [C, H, W]
        steps: number of solver steps
        scheduler: timestep scheduler ('linear', 'cosine', 'sigmoid')
        denoise_strength: adding noise during generation for diversity (0.0-1.0)
        """
        # Create time schedule from t=0 to t=1 with specified scheduler
        ts = self._get_time_schedule(steps, scheduler)
        
        # Sample Gaussian noise as starting point at t=0
        xt = torch.randn(batch_size, *latent_shape, device=self.device)
        
        # Store trajectory for visualization
        trajectory = [xt.detach().cpu()]
        
        # Rectified flow integration from t=0 to t=1
        for i in range(steps - 1):
            t_now = ts[i]
            t_next = ts[i+1]
            dt = t_next - t_now
            
            # Get current timestep for batch
            t_batch = torch.ones(batch_size, device=self.device) * t_now
            
            # Use 5th order Runge-Kutta integration (more accurate than original)
            xt = self._runge_kutta_step(xt, t_batch, dt, guidance_scale)
            
            # Optional: add small noise for diversity
            if denoise_strength > 0 and i < steps - 10:
                noise_scale = denoise_strength * (1.0 - t_next) * 0.1
                xt = xt + torch.randn_like(xt) * noise_scale
            
            trajectory.append(xt.detach().cpu())
            
            if (i + 1) % 10 == 0 or i == steps - 2:
                print(f"Generation step {i+1}/{steps-1}, t={t_next:.4f}")
        
        return xt, trajectory
    
    def _get_time_schedule(self, steps, scheduler_type):
        """Create time schedule from t=0 to t=1"""
        if scheduler_type == 'linear':
            ts = torch.linspace(0, 1, steps, device=self.device)
        elif scheduler_type == 'cosine':
            # Cosine schedule as in improved diffusion
            s = 0.008
            steps_ = steps + 1
            x = torch.linspace(0, steps_, steps, device=self.device)
            alphas = torch.cos(((x / steps_) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas = alphas / alphas[0]
            ts = 1 - alphas
        elif scheduler_type == 'sigmoid':
            # Sigmoid schedule for more even spacing
            x = torch.linspace(-6, 6, steps, device=self.device)
            ts = torch.sigmoid(x)
            # Normalize to [0, 1]
            ts = (ts - ts[0]) / (ts[-1] - ts[0])
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return ts
    
    def _runge_kutta_step(self, x, t, dt, guidance_scale=1.0):
        """
        Adaptive 5th order Runge-Kutta integration step
        x: current state
        t: current time
        dt: time step
        """
        # Standard RK45 coefficients
        k1 = self._get_velocity(x, t, guidance_scale)
        k2 = self._get_velocity(x + dt * (k1 / 5), t + dt / 5, guidance_scale)
        k3 = self._get_velocity(x + dt * (3 * k1 / 40 + 9 * k2 / 40), t + 3 * dt / 10, guidance_scale)
        k4 = self._get_velocity(x + dt * (3 * k1 / 10 - 9 * k2 / 10 + 6 * k3 / 5), t + 3 * dt / 5, guidance_scale)
        k5 = self._get_velocity(x + dt * (-11 * k1 / 54 + 5 * k2 / 2 - 70 * k3 / 27 + 35 * k4 / 27), t + dt, guidance_scale)
        k6 = self._get_velocity(
            x + dt * (1631 * k1 / 55296 + 175 * k2 / 512 + 575 * k3 / 13824 + 44275 * k4 / 110592 + 253 * k5 / 4096),
            t + 7 * dt / 8,
            guidance_scale
        )
        
        # 5th order update
        x_next = x + dt * (
            37 * k1 / 378 + 250 * k3 / 621 + 125 * k4 / 594 + 512 * k6 / 1771
        )
        
        return x_next
    
    def _get_velocity(self, x, t, guidance_scale=1.0):
        """Get velocity prediction from model"""
        with torch.no_grad():
            v = self.flow_model(x, t)
            
            # Apply classifier-free guidance if scale != 1.0
            if guidance_scale != 1.0 and guidance_scale > 0:
                # Get unconditional prediction (using empty conditioning)
                t_null = torch.zeros_like(t)
                v_null = self.flow_model(x, t_null)
                
                # Apply guidance
                v = v_null + guidance_scale * (v - v_null)
                
            return v

def train_rectified_flow(continuous_ae, data, batch_size=32, steps=50000, 
                       lr=1e-4, device='cuda'):
    """
    Train the rectified flow transformer
    continuous_ae: trained continuous autoencoder model
    data: training data
    """
    continuous_ae.eval()  # Set autoencoder to eval mode
    
    # Get latent dimensions from autoencoder output
    with torch.no_grad():
        # Use a small batch to determine dimensions
        sample_batch = torch.tensor(data[:2], dtype=torch.float32).to(device)
        latent = continuous_ae.encode(sample_batch)
        latent_shape = latent.shape[1:]  # [C, H, W]
        
    # Create model
    flow_model = RectifiedFlowTransformer(
        input_dim=latent_shape[0],
        spatial_dims=(latent_shape[1], latent_shape[2]),
        embed_dim=768,
        depth=12,
        num_heads=12
    ).to(device)
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        betas=(0.9, 0.99)
    )
    
    # Cosine learning rate scheduler with linear warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=10,
        anneal_strategy='cos'
    )
    
    # Setup data loader
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    
    # Training metrics
    losses = {
        'total': [],
        'velocity_mse': [],
        'velocity_norm': [],
        'reg_loss': []
    }
    
    # Create checkpoint directory
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    # Start training
    flow_model.train()
    global_step = 0
    epoch = 0
    running_loss = 0.0
    
    progress_bar = tqdm(total=steps, desc="Training rectified flow")
    
    try:
        while global_step < steps:
            epoch += 1
            for batch_idx, (images,) in enumerate(dataloader):
                if global_step >= steps:
                    break
                    
                images = images.to(device)
                batch_size = images.shape[0]
                
                # Get continuous latent vectors from autoencoder
                with torch.no_grad():
                    z1 = continuous_ae.encode(images)
                
                # Sample random noise
                z0 = torch.randn_like(z1)
                
                # Sample random timesteps in (0,1)
                t = torch.rand(batch_size, device=device)
                
                # Construct point along the trajectory with linear interpolation
                zt = t.view(-1, 1, 1, 1) * z1 + (1 - t.view(-1, 1, 1, 1)) * z0
                
                # True velocity field
                v_target = z1 - z0
                
                # Predict velocity field
                v_pred = flow_model(zt, t)
                
                # Compute Loss
                velocity_mse = F.mse_loss(v_pred, v_target)
                
                # Regularization losses
                velocity_norm = torch.mean(torch.norm(v_pred.view(batch_size, -1), dim=1))
                reg_scale = min(1.0, global_step / (steps * 0.3)) * 1e-4
                reg_loss = reg_scale * velocity_norm
                
                # Total loss
                loss = velocity_mse + reg_loss
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                running_loss += loss.item()
                losses['total'].append(loss.item())
                losses['velocity_mse'].append(velocity_mse.item())
                losses['velocity_norm'].append(velocity_norm.item())
                losses['reg_loss'].append(reg_loss.item())
                
                # Update progress
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'v_mse': f"{velocity_mse.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })
                
                # Save checkpoint and visualize periodically
                if global_step % 1000 == 0 or global_step == steps:
                    # Save checkpoint
                    torch.save({
                        'step': global_step,
                        'model_state_dict': flow_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                    }, f'checkpoints/rectified_flow_step_{global_step}.pt')
                    
                    # Save latest checkpoint
                    torch.save({
                        'step': global_step,
                        'model_state_dict': flow_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                    }, 'checkpoints/rectified_flow_latest.pt')
                    
                    # Report metrics
                    avg_loss = running_loss / 1000
                    print(f"\nStep {global_step}: Average loss = {avg_loss:.6f}")
                    running_loss = 0.0
                    
                    # Visualize loss curves if we have enough data
                    if len(losses['total']) > 100:
                        plot_training_curves(losses, global_step)
        
        # Training finished
        print(f"Training completed after {epoch} epochs, {global_step} steps")
        
        # Final checkpoint
        torch.save({
            'step': global_step,
            'model_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / (global_step % 1000),
            'losses': losses,
        }, 'checkpoints/rectified_flow_final.pt')
        
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        torch.save({
            'step': global_step,
            'model_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / (global_step % 1000 + 1),
            'losses': losses,
        }, 'checkpoints/rectified_flow_interrupted.pt')
    
    return flow_model, losses

def plot_training_curves(losses, step):
    """Plot training curves for the rectified flow model"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses['total'][-5000:])
    plt.title('Total Loss')
    plt.xlabel('Step')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(losses['velocity_mse'][-5000:])
    plt.title('Velocity MSE')
    plt.xlabel('Step')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(losses['velocity_norm'][-5000:])
    plt.title('Velocity Norm')
    plt.xlabel('Step')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'checkpoints/rectified_flow_loss_step_{step}.png', dpi=150)
    plt.close()