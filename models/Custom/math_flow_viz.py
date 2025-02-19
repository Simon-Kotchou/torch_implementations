import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons
import matplotlib.cm as cm
from tqdm import tqdm
import math
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sinusoidal_embedding(t, dim=64):
    """
    Sinusoidal embedding for timestep t
    """
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * math.log(10000) / half)
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class FlowNetwork(torch.nn.Module):
    """
    Neural network for modeling flow fields
    """
    def __init__(self, input_dim=2, hidden_dim=128, time_dim=64):
        super().__init__()
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(time_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, t):
        # Embed time
        t_emb = sinusoidal_embedding(t, 64)
        t_emb = self.time_embed(t_emb)
        
        # Concatenate input and time embedding
        x_input = torch.cat([x, t_emb], dim=1)
        
        # Get velocity prediction
        velocity = self.net(x_input)
        return velocity

def linear_interpolation(x_0, x_1, t):
    """
    Linear interpolation between x_0 and x_1 at time t
    """
    return (1-t) * x_0 + t * x_1

def compute_vector_field_grid(flow_network, t, x_range, y_range, resolution=20):
    """
    Compute vector field on a grid for visualization
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Convert to tensor
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                              dtype=torch.float32).to(device)
    t_batch = torch.ones(len(grid_points), device=device) * t
    
    # Get velocities
    with torch.no_grad():
        velocities = flow_network(grid_points, t_batch).cpu().numpy()
    
    # Reshape for plotting
    U = velocities[:, 0].reshape(resolution, resolution)
    V = velocities[:, 1].reshape(resolution, resolution)
    
    # Calculate vector magnitudes
    magnitudes = np.sqrt(U**2 + V**2)
    
    return X, Y, U, V, magnitudes

def visualize_mathematical_flow(source, target, flow_network, save_path='mathematical_flow.gif'):
    """
    Create detailed visualization showing mathematical aspects of flow matching
    """
    # Parameters
    num_frames = 40
    num_trajectories = 500
    resolution = 25  # Grid resolution for vector field
    
    # Sample trajectories
    indices = np.random.choice(len(source), num_trajectories, replace=False)
    source_sub = source[indices].to(device)
    target_sub = target[indices].to(device)
    
    # Calculate boundaries with padding
    all_points = torch.cat([source, target], dim=0)
    x_min, y_min = all_points.min(dim=0)[0].numpy() - 0.5
    x_max, y_max = all_points.max(dim=0)[0].numpy() + 0.5
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 1])
    
    # Main flow plot
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_aspect('equal')
    ax_main.set_title('Flow Matching Transformation', fontsize=14)
    
    # Vector field plot
    ax_vector = fig.add_subplot(gs[1, 0])
    ax_vector.set_xlim(x_min, x_max)
    ax_vector.set_ylim(y_min, y_max)
    ax_vector.set_aspect('equal')
    ax_vector.set_title('Vector Field', fontsize=14)
    
    # Flow magnitude plot
    ax_magnitude = fig.add_subplot(gs[1, 1])
    ax_magnitude.set_xlim(x_min, x_max)
    ax_magnitude.set_ylim(y_min, y_max)
    ax_magnitude.set_aspect('equal')
    ax_magnitude.set_title('Flow Magnitude', fontsize=14)
    
    # Probability density evolution
    ax_density = fig.add_subplot(gs[1, 2])
    ax_density.set_xlim(x_min, x_max)
    ax_density.set_ylim(y_min, y_max)
    ax_density.set_aspect('equal')
    ax_density.set_title('Probability Density', fontsize=14)
    
    # Initialize plots
    scatter_main = ax_main.scatter([], [], s=10, c=[], cmap='plasma', alpha=0.7)
    quiver_main = ax_main.quiver([], [], [], [], color='red', alpha=0.6, scale=20)
    
    # Vector field visualization
    quiver_field = ax_vector.quiver(
        np.zeros((resolution, resolution)),
        np.zeros((resolution, resolution)),
        np.zeros((resolution, resolution)),
        np.zeros((resolution, resolution)),
        np.zeros((resolution, resolution)),
        cmap='viridis',
        scale=20
    )
    
    # Magnitude contour
    contour_plot = ax_magnitude.contourf(
        np.zeros((resolution, resolution)),
        np.zeros((resolution, resolution)),
        np.zeros((resolution, resolution)),
        50,
        cmap='inferno'
    )
    magnitude_colorbar = fig.colorbar(contour_plot, ax=ax_magnitude)
    magnitude_colorbar.set_label('Vector Magnitude')
    
    # Density plot
    density_plot = ax_density.hexbin(
        [], [], gridsize=20, cmap='Blues', 
        extent=[x_min, x_max, y_min, y_max]
    )
    density_colorbar = fig.colorbar(density_plot, ax=ax_density)
    density_colorbar.set_label('Point Density')
    
    # Time indicator
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14)
    
    # Mathematical annotation
    eq_text = fig.text(
        0.02, 0.95, 
        r"$\frac{dx}{dt} = v_\theta(x, t)$",
        ha='left', fontsize=16
    )
    
    plt.tight_layout()
    
    # Initialize trajectories
    trajectories = [[] for _ in range(num_trajectories)]
    
    def init():
        scatter_main.set_offsets(np.empty((0, 2)))
        quiver_main.set_offsets(np.empty((0, 2)))
        quiver_main.set_UVC(np.empty(0), np.empty(0))
        density_plot.set_array(np.array([]))
        return (scatter_main, quiver_main, quiver_field, 
                contour_plot.collections[0], density_plot, time_text)
    
    def update(frame):
        nonlocal contour_plot
        
        # Current timestep
        t = frame / (num_frames - 1)
        t_tensor = torch.tensor([t], device=device)
        
        # Integration step for trajectories
        if frame > 0:
            # Previous positions
            prev_positions = torch.stack([torch.tensor(traj[-1], device=device) 
                                         for traj in trajectories])
            
            # Get velocities at previous positions
            t_batch = torch.ones(len(prev_positions), device=device) * (t - 1/(num_frames-1))
            with torch.no_grad():
                velocities = flow_network(prev_positions, t_batch)
            
            # Euler integration
            dt = 1.0 / (num_frames - 1)
            new_positions = prev_positions + dt * velocities
            
            # Store new positions
            for i, pos in enumerate(new_positions.cpu().numpy()):
                trajectories[i].append(pos)
                
            current_positions = new_positions
        else:
            # Initial positions
            for i, pos in enumerate(source_sub.cpu().numpy()):
                trajectories[i].append(pos)
            current_positions = source_sub
        
        # Get current velocities for quiver plot
        t_batch = torch.ones(len(current_positions), device=device) * t
        with torch.no_grad():
            current_velocities = flow_network(current_positions, t_batch)
        
        # Update main scatter plot
        current_np = current_positions.cpu().numpy()
        scatter_main.set_offsets(current_np)
        scatter_main.set_array(np.linspace(0, 1, len(current_np)))
        
        # Update trajectory lines
        for i, traj in enumerate(trajectories):
            if len(traj) > 1:
                traj_array = np.array(traj)
                # Only plot if we have multiple points
                if i < 50:  # Limit number of lines for clarity
                    if not hasattr(update, f'line_{i}'):
                        line, = ax_main.plot([], [], alpha=0.3, c='gray', linewidth=0.5)
                        setattr(update, f'line_{i}', line)
                    getattr(update, f'line_{i}').set_data(traj_array[:, 0], traj_array[:, 1])
        
        # Update main quiver plot (show subset for clarity)
        subsample = min(50, len(current_np))
        idx = np.random.choice(len(current_np), subsample, replace=False)
        quiver_main.set_offsets(current_np[idx])
        velocities_np = current_velocities.cpu().numpy()
        quiver_main.set_UVC(velocities_np[idx, 0], velocities_np[idx, 1])
        
        # Update vector field visualization
        X, Y, U, V, magnitudes = compute_vector_field_grid(
            flow_network, t, [x_min, x_max], [y_min, y_max], resolution
        )
        quiver_field.set_UVC(U, V)
        quiver_field.set_offsets(np.stack([X.flatten(), Y.flatten()], axis=1))
        
        # Update magnitude contour plot
        for coll in contour_plot.collections:
            coll.remove()
        contour_plot = ax_magnitude.contourf(X, Y, magnitudes, 50, cmap='inferno')
        
        # Update density plot
        density_plot.set_offsets(current_np)
        density_plot.update_scalarmappable()
        
        # Update time text
        time_text.set_text(f'Time: t = {t:.2f}')
        
        return_elements = [scatter_main, quiver_main]
        for i in range(min(50, len(trajectories))):
            if hasattr(update, f'line_{i}'):
                return_elements.append(getattr(update, f'line_{i}'))
        
        return_elements.extend([quiver_field, contour_plot.collections[0], 
                              density_plot, time_text])
        return tuple(return_elements)
    
    # Create animation
    animation = FuncAnimation(
        fig, update, frames=num_frames, init_func=init, 
        blit=True, interval=100
    )
    
    # Save animation
    animation.save(save_path, writer='pillow', fps=8, dpi=120)
    print(f"Saved mathematical flow visualization to {save_path}")
    
    return animation

def train_flow_model(source, target, steps=1000, lr=1e-3):
    """
    Train a flow matching model to transform source to target
    """
    flow_network = FlowNetwork().to(device)
    optimizer = torch.optim.Adam(flow_network.parameters(), lr=lr)
    
    source = source.to(device)
    target = target.to(device)
    
    losses = []
    
    # Training loop
    for step in tqdm(range(steps), desc="Training flow model"):
        optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.rand(len(source), device=device)
        
        # Linear interpolation at time t
        x_t = linear_interpolation(source, target, t.view(-1, 1))
        
        # Ground truth velocity (for straight-line ODE)
        v_t = target - source  # constant velocity field
        
        # Get model prediction
        v_pred = flow_network(x_t, t)
        
        # Compute loss
        loss = torch.mean((v_pred - v_t) ** 2)
        losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
    return flow_network, losses

def create_advanced_flow_visualization():
    """Create advanced mathematical flow visualization"""
    print(f"Using device: {device}")
    
    # Generate moon dataset
    n_samples = 2000
    X, _ = make_moons(n_samples=n_samples, noise=0.05)
    target = torch.tensor(X, dtype=torch.float32)
    
    # Generate source distribution (Gaussian)
    source = torch.randn(n_samples, 2)
    
    # Scale to similar ranges
    source = source * 0.5
    
    print("Training flow model...")
    flow_network, losses = train_flow_model(source, target, steps=1000)
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('Flow Matching Training Loss')
    plt.grid(alpha=0.3)
    plt.savefig('flow_training_loss.png', dpi=150, bbox_inches='tight')
    
    print("Creating advanced mathematical visualization...")
    visualize_mathematical_flow(source, target, flow_network)
    
if __name__ == "__main__":
    create_advanced_flow_visualization()