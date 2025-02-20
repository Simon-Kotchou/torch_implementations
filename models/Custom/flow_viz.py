import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math
import os
from matplotlib.animation import FuncAnimation
from PIL import Image

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load LFW dataset with exact dimensions
print("Loading LFW dataset...")
lfw_people = fetch_lfw_people(resize=None)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
print(f"Image size: {h}x{w}")

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Flow Matching Scheduler
class FlowMatchingScheduler:
    """Scheduler for Flow Matching process."""
    def __init__(
        self,
        num_inference_steps=50,
        scheduler_type='cosine',
        min_t=0.002,
        max_t=0.998
    ):
        self.num_inference_steps = num_inference_steps
        self.scheduler_type = scheduler_type
        self.min_t = min_t
        self.max_t = max_t
        self.timesteps = self._get_schedule()
        
    def _get_schedule(self):
        if self.scheduler_type == 'linear':
            return torch.linspace(self.max_t, self.min_t, self.num_inference_steps)
        
        elif self.scheduler_type == 'cosine':
            steps = torch.arange(self.num_inference_steps + 1).float() / self.num_inference_steps
            alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            timesteps = torch.clamp(alpha_cumprod, self.min_t, self.max_t)
            return timesteps[:-1]  # Remove the last timestep
            
        elif self.scheduler_type == 'shifted':
            # Shifted schedule with a shifting factor
            shifting_factor = 7.0
            steps = np.linspace(0, 1, self.num_inference_steps)
            # Apply shifting function t' = s*t/(1+(s-1)*t)
            s = shifting_factor
            steps = s * steps / (1 + (s - 1) * steps)
            # Scale to desired range
            steps = (1 - steps) * (self.max_t - self.min_t) + self.min_t
            return torch.from_numpy(steps).float()
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def get_timesteps(self):
        return self.timesteps

# Attention block
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x):
        x_norm = self.norm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        return x + attn_output

# Feed-forward block
class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.net(self.norm(x))

# DiT-like transformer block
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=64, hidden_dim=None, dropout=0.0):
        super().__init__()
        self.attn = AttentionBlock(dim, num_heads, dim_head, dropout)
        self.ff = FeedForwardBlock(dim, hidden_dim, dropout)
        
    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x

# Advanced time embedding
def advanced_time_embedding(t, dim=256):
    """Creates an advanced timestep embedding with learnable projection."""
    half_dim = dim // 2
    freqs = torch.exp(-torch.arange(half_dim, device=t.device) * math.log(10000) / half_dim)
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding

# Flow Matching DiT Network
class FlowMatchingDiT(nn.Module):
    def __init__(
        self, 
        input_dim,  # Use actual input dimension
        embed_dim=512, 
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        time_embed_dim=256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=int(embed_dim * mlp_ratio),
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, input_dim)
        )
        
    def forward(self, x, t):
        """Forward pass with timestep conditioning"""
        # Time embedding
        t_emb = advanced_time_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Project input
        h = self.input_proj(x)
        
        # Add time embedding
        h = h + t_emb.unsqueeze(1)
        
        # Apply DiT blocks
        for block in self.blocks:
            h = block(h)
        
        # Apply final normalization
        h = self.norm(h)
        
        # Project back to input dimension
        velocity = self.output_proj(h).squeeze(1)
        
        return velocity

# Linear interpolation for flow matching
def linear_interpolation(x_0, x_1, t):
    """Linear interpolation between x_0 and x_1 at time t"""
    if t.dim() == 1:
        t = t.unsqueeze(1)
    return (1-t) * x_0 + t * x_1

# Train the flow matching model
def train_flow_model(data, steps=10000, batch_size=128, lr=1e-4, checkpoint_freq=1000):
    """Train a flow matching model on image data."""
    data = data.to(device)
    n_samples, n_features = data.shape
    
    print(f"Training with input dimension: {n_features}")
    
    # Create model with correct input dimension
    model = FlowMatchingDiT(input_dim=n_features).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/10)
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Resume from checkpoint if exists
    start_step = 0
    checkpoint_path = "checkpoints/flow_model_latest.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"Resuming from step {start_step}")
    
    # Training loop
    losses = []
    running_loss = 0.0
    pbar = tqdm(range(start_step, steps), desc="Training flow model")
    
    for step in pbar:
        optimizer.zero_grad()
        
        # Sample batch
        batch_indices = torch.randint(0, n_samples, (batch_size,))
        x_1 = data[batch_indices]
        
        # Sample noise
        x_0 = torch.randn_like(x_1)
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=device)
        
        # Linear interpolation at time t
        x_t = linear_interpolation(x_0, x_1, t)
        
        # Ground truth velocity
        v_t = x_1 - x_0
        
        # Get model prediction
        x_t_reshaped = x_t.unsqueeze(1)  # Add sequence dimension for transformer
        v_pred = model(x_t_reshaped, t)
        
        # Compute loss
        loss = F.mse_loss(v_pred, v_t)
        running_loss += loss.item()
        losses.append(loss.item())
        
        # Update progress bar
        if step % 50 == 0:
            pbar.set_postfix({"loss": loss.item()})
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Save checkpoint
        if (step + 1) % checkpoint_freq == 0 or step == steps - 1:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item()
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, f"checkpoints/flow_model_step_{step+1}.pt")
            
            # Log average loss
            avg_loss = running_loss / checkpoint_freq
            print(f"Step {step+1}, Average Loss: {avg_loss:.6f}")
            running_loss = 0.0
    
    return model, losses

# Generate samples using trained flow model
def generate_samples(model, n_samples=16, n_features=None, h=62, w=47, steps=100):
    """Generate face samples using the trained flow model."""
    model.eval()
    
    # Use actual feature dimension from model
    if n_features is None:
        n_features = model.input_dim
    
    print(f"Generating samples with dimension: {n_features}")
    
    # Setup scheduler
    scheduler = FlowMatchingScheduler(num_inference_steps=steps, scheduler_type='cosine')
    timesteps = scheduler.get_timesteps().to(device)
    
    # Start from random noise
    x = torch.randn(n_samples, n_features).to(device)
    
    # Store all intermediate states for visualization
    all_states = [x.detach().cpu()]
    
    # Inference steps
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            # Expand timestep for batch
            t_batch = t.expand(n_samples)
            
            # Get velocity prediction
            x_reshaped = x.unsqueeze(1)  # Add sequence dimension
            velocity = model(x_reshaped, t_batch)
            
            # Euler step
            dt = 1.0 / len(timesteps) if i < len(timesteps) - 1 else 0
            x = x + velocity * dt
            
            # Store intermediate state
            all_states.append(x.detach().cpu())
    
    # Convert to image space
    final_samples = scaler.inverse_transform(all_states[-1].numpy())
    
    # Clip values and reshape
    final_images = np.clip(final_samples, 0, 255)
    final_images = final_images.reshape(n_samples, h, w)
    
    # Process all intermediate states
    all_images = []
    for state in all_states:
        images = scaler.inverse_transform(state.numpy())
        images = np.clip(images, 0, 255)
        images = images.reshape(n_samples, h, w)
        all_images.append(images)
    
    return final_images, all_images

# Visualize generated samples
def visualize_samples(images, save_path="generated_faces.png"):
    """Visualize generated samples in a grid."""
    n_samples = len(images)
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))
    
    plt.figure(figsize=(cols*2, rows*2))
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()

# Plot training loss curve
def visualize_training(losses, save_path="training_loss.png"):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    window_size = 100
    if len(losses) > window_size:
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(losses, alpha=0.3, color='blue')
        plt.plot(np.arange(window_size-1, len(losses)), smoothed_losses, color='blue')
    else:
        plt.plot(losses, color='blue')
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('Flow Matching Training Loss')
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150)
    print(f"Saved loss curve to {save_path}")
    plt.close()

# Create animation of generation process
def create_grid_animation(all_images, save_path="face_generation_grid.gif", fps=15):
    """Create a grid animation showing multiple samples evolving."""
    n_steps = len(all_images)
    n_samples = min(16, len(all_images[0]))
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))
    
    # Prepare directory for frames
    os.makedirs("frames", exist_ok=True)
    frame_paths = []
    
    # Create individual frames
    for t in tqdm(range(n_steps), desc="Creating animation frames"):
        plt.figure(figsize=(cols*2, rows*2))
        
        for i in range(n_samples):
            plt.subplot(rows, cols, i+1)
            plt.imshow(all_images[t][i], cmap='gray')
            plt.axis('off')
        
        plt.suptitle(f"Generation Progress: {t/(n_steps-1)*100:.1f}%", fontsize=16)
        plt.tight_layout()
        
        # Save frame
        frame_path = f"frames/frame_{t:04d}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        frame_paths.append(frame_path)
        plt.close()
    
    # Combine frames into GIF
    frames = [Image.open(f) for f in frame_paths]
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=1000//fps,
        loop=0
    )
    print(f"Saved grid animation to {save_path}")

# Main function
def main():
    """Main function to train model and generate samples."""
    try:
        print("Starting face image generation with Flow Matching DiT...")
        
        # Train the model with correct dimensions
        model, losses = train_flow_model(X_tensor, steps=5000)
        
        # Plot training loss
        visualize_training(losses)
        
        # Generate samples (passing actual dimensions)
        final_images, all_images = generate_samples(
            model, 
            n_samples=50, 
            n_features=n_features,
            h=h, 
            w=w, 
            steps=50
        )
        
        # Visualize generated samples
        visualize_samples(final_images)
        
        # Create animation of generation process
        create_grid_animation(all_images)
        
        print("Flow Matching DiT experiment completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()