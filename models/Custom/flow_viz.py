import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
import matplotlib.cm as cm
from tqdm import tqdm
import math

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
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * math.log(10000) / half).to(device)
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
    return t * x_1 + (1 - t) * x_0

def create_flow_matching_animation(source, target, flow_network=None, num_frames=100, num_trajectories=1000):
    """
    Create animation of flow matching between source and target distributions
    
    Args:
        source: Source distribution points [N, 2]
        target: Target distribution points [N, 2]
        flow_network: Optional trained flow network
        num_frames: Number of frames in animation
        num_trajectories: Number of trajectories to show
    
    Returns:
        Animation object
    """
    # Subsample points for clearer visualization
    indices = np.random.choice(len(source), num_trajectories, replace=False)
    source_sub = source[indices]
    target_sub = target[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], s=5, c=[], cmap='viridis', alpha=0.8)
    quiver = ax.quiver([], [], [], [], color='red', alpha=0.5, scale=30)
    title = ax.set_title('')
    
    # Set axes limits with some padding
    all_points = torch.cat([source, target], dim=0)
    x_min, y_min = all_points.min(dim=0)[0].numpy() - 0.5
    x_max, y_max = all_points.max(dim=0)[0].numpy() + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    # Create colormap for time
    colors = cm.viridis(np.linspace(0, 1, num_trajectories))
    
    # Function to initialize animation
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        quiver.set_offsets(np.empty((0, 2)))
        quiver.set_UVC(np.empty(0), np.empty(0))
        return scatter, quiver, title
    
    # Function to update animation for each frame
    def update(frame):
        t = frame / (num_frames - 1)
        t_tensor = torch.tensor([t], device=device)
        
        # Get current positions through interpolation or flow
        if flow_network is None:
            # Simple linear interpolation
            current = linear_interpolation(source_sub, target_sub, t)
            # Compute analytical velocity (target - source)
            velocity = target_sub - source_sub
        else:
            # Use flow network for velocity
            with torch.no_grad():
                t_batch = t_tensor.repeat(len(source_sub))
                current = source_sub.clone().to(device)
                
                # Get velocity from flow network
                velocity = flow_network(current, t_batch)
                velocity = velocity.cpu()
                current = current.cpu()
        
        # Update scatter plot
        scatter.set_offsets(current.numpy())
        scatter.set_array(np.arange(len(current)))
        
        # Update velocity vectors (show a subset for clarity)
        if t < 0.99:  # Don't show vectors at final state
            subsample = min(100, len(current))
            idx = np.random.choice(len(current), subsample, replace=False)
            
            quiver.set_offsets(current[idx].numpy())
            quiver.set_UVC(velocity[idx, 0].numpy(), velocity[idx, 1].numpy())
        else:
            quiver.set_offsets(np.empty((0, 2)))
            quiver.set_UVC(np.empty(0), np.empty(0))
            
        title.set_text(f't = {t:.2f}')
        return scatter, quiver, title
    
    # Create animation
    animation = FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=50
    )
    
    return animation

def generate_spirals(n_samples=1000, noise=0.1):
    """Generate two interlocking spirals"""
    n = np.sqrt(np.random.rand(n_samples)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_samples) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples) * noise
    spiral1 = np.vstack([d1x, d1y]).T
    
    n = np.sqrt(np.random.rand(n_samples)) * 780 * (2 * np.pi) / 360
    d2x = np.cos(n) * n + np.random.rand(n_samples) * noise
    d2y = -np.sin(n) * n + np.random.rand(n_samples) * noise
    spiral2 = np.vstack([d2x, d2y]).T
    
    spiral = np.vstack([spiral1, spiral2])
    spiral = spiral / (spiral.max() / 2) - 1
    return torch.tensor(spiral, dtype=torch.float32)

def generate_distributions(dataset='spirals', n_samples=2000):
    """Generate source and target distributions"""
    # Source: Gaussian noise
    source = torch.randn(n_samples, 2)
    
    # Target: based on dataset choice
    if dataset == 'moons':
        target_np, _ = make_moons(n_samples=n_samples, noise=0.05)
        target = torch.tensor(target_np, dtype=torch.float32)
    elif dataset == 'circles':
        target_np, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5)
        target = torch.tensor(target_np, dtype=torch.float32)
    elif dataset == 'spirals':
        target = generate_spirals(n_samples=n_samples)
    elif dataset == 'swiss_roll':
        target_np, _ = make_swiss_roll(n_samples=n_samples, noise=0.05)
        target = torch.tensor(target_np[:, [0, 2]], dtype=torch.float32)
        # Normalize swiss roll
        target = target / 5.0
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return source, target

def train_flow_model(source, target, steps=1000, lr=1e-3):
    """
    Train a flow matching model to transform source to target
    """
    flow_network = FlowNetwork().to(device)
    optimizer = torch.optim.Adam(flow_network.parameters(), lr=lr)
    
    source = source.to(device)
    target = target.to(device)
    
    # Training loop
    for step in tqdm(range(steps), desc="Training flow model"):
        optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.rand(len(source), device=device)
        
        # Linear interpolation at time t
        x_t = linear_interpolation(source, target, t.view(-1, 1))
        
        # Ground truth velocity
        v_t = target - source  # Constant for linear path
        
        # Get model prediction
        v_pred = flow_network(x_t, t)
        
        # Compute loss
        loss = torch.mean((v_pred - v_t) ** 2)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
    return flow_network

def visualize_transformations():
    """Create flow matching visualizations for different datasets"""
    datasets = ['moons', 'circles', 'spirals']
    for dataset in datasets:
        print(f"Creating visualization for {dataset} dataset...")
        
        # Generate distributions
        source, target = generate_distributions(dataset=dataset)
        
        # Train flow model
        flow_network = train_flow_model(source, target, steps=500)
        
        # Create animation
        animation = create_flow_matching_animation(
            source, target, flow_network=flow_network, num_frames=50
        )
        
        # Save animation
        animation.save(f'flow_matching_{dataset}.gif', writer='pillow', fps=10, dpi=100)
        print(f"Saved animation to flow_matching_{dataset}.gif")
        
        # Also create a linear interpolation animation for comparison
        animation_linear = create_flow_matching_animation(
            source, target, flow_network=None, num_frames=50
        )
        animation_linear.save(f'linear_interpolation_{dataset}.gif', writer='pillow', fps=10, dpi=100)
        print(f"Saved linear interpolation to linear_interpolation_{dataset}.gif")
        
if __name__ == "__main__":
    print(f"Using device: {device}")
    visualize_transformations()