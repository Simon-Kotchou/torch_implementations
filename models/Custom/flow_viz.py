import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def sinusoidal_embedding(t, dim=64):
    """Sinusoidal embedding for timestep t"""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * np.log(10000) / half)
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class FlowNetwork(torch.nn.Module):
    """Neural network for modeling flow fields"""
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
    """Linear interpolation between x_0 and x_1 at time t"""
    return (1-t) * x_0 + t * x_1

def generate_distributions(dataset='moons', n_samples=2000):
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
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return source, target

def train_flow_model(source, target, steps=500, lr=1e-3):
    """Train a flow matching model to transform source to target"""
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
        
        # Ground truth velocity
        v_t = target - source  # Constant for linear path
        
        # Get model prediction
        v_pred = flow_network(x_t, t)
        
        # Compute loss
        loss = torch.mean((v_pred - v_t) ** 2)
        losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
    return flow_network, losses

def visualize_flow_field(flow_network, source, target, num_steps=10, save_path=None):
    """Visualize the flow field and sample trajectories"""
    # Set up the plot
    plt.figure(figsize=(15, 15))
    
    # Determine plot boundaries
    all_points = torch.cat([source, target], dim=0)
    x_min, y_min = all_points.min(dim=0)[0].numpy() - 0.5
    x_max, y_max = all_points.max(dim=0)[0].numpy() + 0.5
    
    # Time steps to visualize
    timesteps = np.linspace(0, 1, num_steps)
    
    # Number of rows and columns in the grid
    n_rows = int(np.ceil(np.sqrt(num_steps)))
    n_cols = int(np.ceil(num_steps / n_rows))
    
    # Sample a subset of points for clearer visualization
    n_viz_points = 500
    indices = np.random.choice(len(source), n_viz_points, replace=False)
    source_viz = source[indices].to(device)
    target_viz = target[indices].to(device)
    
    # Track positions through time
    current_positions = source_viz.clone()
    
    # Grid for computing vector field
    resolution = 20
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    
    for i, t in enumerate(timesteps):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        # Plot source and target distributions
        if i == 0:
            plt.scatter(source[:, 0].cpu().numpy(), source[:, 1].cpu().numpy(), 
                      s=1, alpha=0.3, color='blue', label='Source')
        elif i == num_steps-1:
            plt.scatter(target[:, 0].cpu().numpy(), target[:, 1].cpu().numpy(), 
                      s=1, alpha=0.3, color='red', label='Target')
        
        # Compute vector field at this timestep
        t_tensor = torch.ones(len(grid_tensor), device=device) * t
        with torch.no_grad():
            grid_velocities = flow_network(grid_tensor, t_tensor).cpu().numpy()
        
        # Reshape velocities for quiver plot
        U = grid_velocities[:, 0].reshape(resolution, resolution)
        V = grid_velocities[:, 1].reshape(resolution, resolution)
        
        # Plot vector field
        plt.quiver(X, Y, U, V, alpha=0.5, color='gray')
        
        # If not the first step, advance particles
        if i > 0:
            # Get velocities at current positions
            with torch.no_grad():
                velocities = flow_network(current_positions, torch.ones_like(indices, device=device).float() * (t-timesteps[1]+timesteps[0]))
            
            # Euler integration step
            current_positions = current_positions + velocities * (timesteps[1]-timesteps[0])
        
        # Plot current positions
        position_np = current_positions.cpu().numpy()
        plt.scatter(position_np[:, 0], position_np[:, 1], s=5, color='green', alpha=0.7)
        
        # Set title and axis limits
        plt.title(f't = {t:.2f}')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        if i == 0 or i == num_steps-1:
            plt.legend()
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

def main():
    datasets = ['moons', 'circles']
    
    for dataset in datasets:
        print(f"\nVisualizing flow matching for {dataset} dataset...")
        
        # Generate distributions
        source, target = generate_distributions(dataset=dataset)
        
        # Train flow model
        flow_network, losses = train_flow_model(source, target)
        
        # Plot loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.yscale('log')
        plt.xlabel('Training Steps')
        plt.ylabel('MSE Loss (log scale)')
        plt.title(f'Flow Matching Training Loss - {dataset.capitalize()}')
        plt.grid(alpha=0.3)
        plt.savefig(f'flow_loss_{dataset}.png', dpi=150, bbox_inches='tight')
        
        # Visualize flow field
        visualize_flow_field(flow_network, source, target, num_steps=12, 
                           save_path=f'flow_matching_{dataset}.png')
        
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main()