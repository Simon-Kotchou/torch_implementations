import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import math
import scipy.stats as stats

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleFlow1D(torch.nn.Module):
    """
    Simple 1D flow matching model
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),  # x and t as inputs
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1)   # 1D output
        )
        
    def forward(self, x, t):
        """
        Forward pass for vector field prediction
        Args:
            x: spatial location [batch_size, 1]
            t: time [batch_size]
        """
        # Ensure t has the right shape
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)
        
        # Get velocity
        velocity = self.net(inputs)
        return velocity

def linear_interpolation(p_0, p_1, t):
    """
    Linear interpolation between distributions
    
    Args:
        p_0: Source distribution parameters
        p_1: Target distribution parameters
        t: Time point (0->1)
    
    Returns:
        Interpolated distribution parameters
    """
    return (1-t) * p_0 + t * p_1

def exact_velocity_field(x, t, source_params, target_params):
    """
    Compute exact velocity field for 1D Gaussian transition
    
    The optimal transport velocity field between Gaussians is:
    v(x, t) = (μ_1 - μ_0) + (σ_1/σ_t - σ_0/σ_t) * (x - μ_t)
    
    where μ_t and σ_t are the interpolated means and standard deviations
    """
    μ_0, σ_0 = source_params
    μ_1, σ_1 = target_params
    
    # Interpolate mean and variance
    μ_t = linear_interpolation(μ_0, μ_1, t)
    σ_t_sq = linear_interpolation(σ_0**2, σ_1**2, t) 
    σ_t = np.sqrt(σ_t_sq)
    
    # Compute velocity
    velocity = (μ_1 - μ_0) + (x - μ_t) * (σ_1 - σ_0) / σ_t
    return velocity

def gaussian_pdf(x, mean, std):
    """Compute Gaussian PDF values"""
    return 1/(std * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mean)/std)**2)

def train_flow_model(source_params, target_params, num_steps=2000, batch_size=100, lr=1e-3):
    """Train 1D flow model to match Gaussian transitions"""
    model = SimpleFlow1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr/10)
    
    μ_0, σ_0 = source_params
    μ_1, σ_1 = target_params
    
    # Range for sampling points
    x_min = min(μ_0 - 3*σ_0, μ_1 - 3*σ_1)
    x_max = max(μ_0 + 3*σ_0, μ_1 + 3*σ_1)
    
    losses = []
    
    # Training loop
    for step in tqdm(range(num_steps), desc="Training flow model"):
        optimizer.zero_grad()
        
        # Sample random points from the domain
        x = torch.rand(batch_size, 1, device=device) * (x_max - x_min) + x_min
        
        # Sample random times
        t = torch.rand(batch_size, 1, device=device)
        
        # Compute exact velocity field (ground truth)
        v_exact = exact_velocity_field(x.cpu().numpy(), t.cpu().numpy(), 
                                      source_params, target_params)
        v_exact = torch.tensor(v_exact, dtype=torch.float32).to(device)
        
        # Get model prediction
        v_pred = model(x, t)
        
        # Compute loss
        loss = torch.mean((v_pred - v_exact) ** 2)
        losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return model, losses

def visualize_1d_flow(model, source_params, target_params, save_path="1d_flow.gif"):
    """Create animated visualization of 1D flow matching"""
    μ_0, σ_0 = source_params
    μ_1, σ_1 = target_params
    
    # Setup plot domain
    x_min = min(μ_0 - 4*σ_0, μ_1 - 4*σ_1)
    x_max = max(μ_0 + 4*σ_0, μ_1 + 4*σ_1)
    x_range = np.linspace(x_min, x_max, 1000)
    
    # Setup for visualization
    num_frames = 50
    num_particles = 100
    
    # Sample particles from source distribution
    particles = np.random.normal(μ_0, σ_0, num_particles)
    trajectories = [particles.copy()]
    
    # Euler integration to generate trajectories
    dt = 1.0 / (num_frames - 1)
    for frame in range(1, num_frames):
        t = frame * dt
        
        # Convert to tensor
        x_tensor = torch.tensor(particles.reshape(-1, 1), dtype=torch.float32).to(device)
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        
        # Get velocity from model
        with torch.no_grad():
            velocities = model(x_tensor, t_tensor).cpu().numpy().flatten()
        
        # Update positions
        particles = particles + velocities * dt
        trajectories.append(particles.copy())
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(3, 3)
    
    # Distribution plot
    ax_dist = fig.add_subplot(gs[0, :])
    ax_dist.set_xlim(x_min, x_max)
    ax_dist.set_ylim(0, max(1/(σ_0*np.sqrt(2*np.pi)), 1/(σ_1*np.sqrt(2*np.pi))) * 1.2)
    ax_dist.set_title("Probability Distribution Evolution", fontsize=14)
    ax_dist.set_xlabel("x", fontsize=12)
    ax_dist.set_ylabel("Probability Density", fontsize=12)
    
    # Source and target distributions
    x_plot = np.linspace(x_min, x_max, 1000)
    source_pdf = gaussian_pdf(x_plot, μ_0, σ_0)
    target_pdf = gaussian_pdf(x_plot, μ_1, σ_1)
    
    ax_dist.plot(x_plot, source_pdf, 'b--', alpha=0.5, label="Source")
    ax_dist.plot(x_plot, target_pdf, 'r--', alpha=0.5, label="Target")
    ax_dist.legend()
    
    # Current distribution
    dist_line, = ax_dist.plot([], [], 'g-', linewidth=2, label="Current")
    hist = ax_dist.hist([], bins=30, range=(x_min, x_max), density=True, 
                      alpha=0.3, color='green')[2]
    
    # Vector field plot
    ax_vector = fig.add_subplot(gs[1, :])
    ax_vector.set_xlim(x_min, x_max)
    ax_vector.set_ylim(-max(abs(μ_1 - μ_0), abs(σ_1 - σ_0))*3, 
                      max(abs(μ_1 - μ_0), abs(σ_1 - σ_0))*3)
    ax_vector.set_title("Vector Field: v(x, t)", fontsize=14)
    ax_vector.set_xlabel("x", fontsize=12)
    ax_vector.set_ylabel("Velocity", fontsize=12)
    ax_vector.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Model vector field line
    vector_line, = ax_vector.plot([], [], 'b-', linewidth=2, label="Model v(x, t)")
    
    # Ground truth vector field
    gt_vector_line, = ax_vector.plot([], [], 'r--', linewidth=1.5, alpha=0.7, 
                                   label="Exact v(x, t)")
    ax_vector.legend()
    
    # Particle trajectories plot
    ax_traj = fig.add_subplot(gs[2, :])
    ax_traj.set_xlim(0, 1)
    ax_traj.set_ylim(x_min, x_max)
    ax_traj.set_title("Particle Trajectories", fontsize=14)
    ax_traj.set_xlabel("Time", fontsize=12)
    ax_traj.set_ylabel("Position", fontsize=12)
    
    # Initialize trajectory lines
    traj_lines = []
    for i in range(min(30, num_particles)):  # Plot a subset for clarity
        line, = ax_traj.plot([], [], alpha=0.5)
        traj_lines.append(line)
    
    # Mean and std dev lines
    mean_line, = ax_traj.plot([], [], 'g-', linewidth=2, label="Mean")
    std_upper, = ax_traj.plot([], [], 'g--', linewidth=1.5, alpha=0.7)
    std_lower, = ax_traj.plot([], [], 'g--', linewidth=1.5, alpha=0.7)
    ax_traj.legend()
    
    # Time indicator
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14)
    
    # Mathematical annotation
    eq_text = fig.text(
        0.02, 0.95, 
        r"$\frac{dx}{dt} = v_\theta(x, t)$   where   " + 
        r"$v(x, t) = (\mu_1 - \mu_0) + \frac{(\sigma_1 - \sigma_0)}{\sigma_t}(x - \mu_t)$",
        ha='left', fontsize=14
    )
    
    plt.tight_layout()
    
    def update(frame):
        t = frame / (num_frames - 1)
        
        # Update time text
        time_text.set_text(f'Time: t = {t:.2f}')
        
        # Current distribution parameters
        μ_t = linear_interpolation(μ_0, μ_1, t)
        σ_t_sq = linear_interpolation(σ_0**2, σ_1**2, t)
        σ_t = np.sqrt(σ_t_sq)
        
        # Update distribution plot
        current_pdf = gaussian_pdf(x_plot, μ_t, σ_t)
        dist_line.set_data(x_plot, current_pdf)
        
        # Update histogram
        for rect, height in zip(hist, np.histogram(trajectories[frame], bins=30, 
                                                range=(x_min, x_max), density=True)[0]):
            rect.set_height(height)
        
        # Update vector field plot
        x_tensor = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float32).to(device)
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            model_vectors = model(x_tensor, t_tensor).cpu().numpy().flatten()
        
        # Exact vector field
        exact_vectors = exact_velocity_field(x_range.reshape(-1, 1), t, source_params, target_params)
        
        vector_line.set_data(x_range, model_vectors)
        gt_vector_line.set_data(x_range, exact_vectors)
        
        # Update trajectory lines
        times = np.linspace(0, 1, num_frames)[:frame+1]
        
        for i, line in enumerate(traj_lines):
            if i < len(trajectories[0]):
                traj_data = np.array([traj[i] for traj in trajectories[:frame+1]])
                line.set_data(times, traj_data)
        
        # Update mean and std dev lines
        traj_array = np.array(trajectories[:frame+1])
        means = traj_array.mean(axis=1)
        stds = traj_array.std(axis=1)
        
        mean_line.set_data(times, means)
        std_upper.set_data(times, means + stds)
        std_lower.set_data(times, means - stds)
        
        return [dist_line] + list(hist) + [vector_line, gt_vector_line, 
                                         mean_line, std_upper, std_lower] + traj_lines
    
    # Create animation
    animation = FuncAnimation(
        fig, update, frames=num_frames, interval=100, blit=True
    )
    
    animation.save(save_path, writer='pillow', fps=10, dpi=120)
    print(f"Saved 1D flow visualization to {save_path}")
    
    return animation

def main():
    print(f"Using device: {device}")
    
    # Define source and target distributions
    source_params = (-2.0, 0.5)  # (mean, std)
    target_params = (3.0, 1.5)   # (mean, std)
    
    print(f"Source distribution: N({source_params[0]}, {source_params[1]}²)")
    print(f"Target distribution: N({target_params[0]}, {target_params[1]}²)")
    
    # Train model
    model, losses = train_flow_model(source_params, target_params)
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('1D Flow Matching Training Loss')
    plt.grid(alpha=0.3)
    plt.savefig('1d_flow_loss.png', dpi=150, bbox_inches='tight')
    
    # Visualize flow
    visualize_1d_flow(model, source_params, target_params)

if __name__ == "__main__":
    main()