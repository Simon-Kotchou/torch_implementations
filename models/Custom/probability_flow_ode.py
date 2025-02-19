import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_swiss_roll
import seaborn as sns
from tqdm import tqdm
from scipy.integrate import solve_ivp
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
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * math.log(10000) / half)
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class VectorField(torch.nn.Module):
    """
    Neural network for modeling vector fields
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
        """Forward pass: x is spatial location, t is time"""
        # Embed time
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device)
        if t.dim() == 0:
            t = t.view(1)
            
        # Ensure t is properly shaped
        if len(x) != len(t):
            t = t.repeat(len(x))
            
        t_emb = sinusoidal_embedding(t, 64)
        t_emb = self.time_embed(t_emb)
        
        # Concatenate input and time embedding
        x_input = torch.cat([x, t_emb], dim=1)
        
        # Get velocity prediction
        velocity = self.net(x_input)
        return velocity
        
    def numpy_forward(self, x_np, t_np):
        """Numpy interface for ODE solvers"""
        x = torch.tensor(x_np, dtype=torch.float32).to(device)
        t = torch.tensor(t_np, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            v = self.forward(x.unsqueeze(0), t).squeeze(0).cpu().numpy()
        return v

def create_swiss_roll():
    """Generate swiss roll dataset"""
    n_samples = 2000
    X, _ = make_swiss_roll(n_samples=n_samples, noise=0.1)
    # Use only two dimensions
    X = X[:, [0, 2]]
    # Scale to reasonable range
    X = X / 8.0
    return torch.tensor(X, dtype=torch.float32)

def train_vector_field(data, steps=1000, batch_size=500, lr=1e-3):
    """
    Train a vector field to model probability flow
    """
    model = VectorField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    data = data.to(device)
    n_samples = len(data)
    
    losses = []
    
    # Training loop
    for step in tqdm(range(steps), desc="Training vector field"):
        optimizer.zero_grad()
        
        # Sample batch
        idx = torch.randint(0, n_samples, (batch_size,))
        x_batch = data[idx]
        
        # Sample random noise for perturbation
        noise_scale = 0.1
        noise = torch.randn_like(x_batch) * noise_scale
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=device)
        
        # Perturb data points based on time
        x_t = x_batch + t.view(-1, 1) * noise
        
        # Target is towards the original point
        v_target = -noise
        
        # Get model prediction
        v_pred = model(x_t, t)
        
        # Compute loss
        loss = torch.mean((v_pred - v_target) ** 2)
        losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
    
    return model, losses

def solve_flow_ode(vector_field, x_start, t_span, t_eval):
    """
    Solve probability flow ODE: dx/dt = v(x, t)
    """
    def ode_func(t, x):
        x_reshaped = x.reshape(-1, 2)
        # Compute vector field
        dx = vector_field.numpy_forward(x_reshaped, t)
        return dx.flatten()
    
    # Solve ODE
    solution = solve_ivp(
        ode_func,
        t_span,
        x_start.flatten(),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-3,
        atol=1e-3
    )
    
    # Reshape results
    trajectories = solution.y.reshape(-1, len(x_start), 2)
    return trajectories, solution.t

def visualize_flow_ode(data, vector_field, num_particles=50, save_path="flow_ode.gif"):
    """
    Visualize probability flow ODE
    """
    # Sample starting points from Gaussian
    x_start = torch.randn(num_particles, 2) * 0.5
    
    # Time settings for ODE
    t_span = (1.0, 0.0)  # Reverse time integration (1 → 0)
    num_frames = 40
    t_eval = np.linspace(1.0, 0.0, num_frames)
    
    # Solve ODE system
    print("Solving probability flow ODE...")
    trajectories, t_eval_actual = solve_flow_ode(vector_field, x_start.numpy(), t_span, t_eval)
    
    # Create plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # Target distribution plot
    ax1.set_title("Probability Flow ODE Trajectories", fontsize=16)
    ax1.set_xlabel("x", fontsize=14)
    ax1.set_ylabel("y", fontsize=14)
    
    # Determine plot limits
    all_points = torch.cat([data, x_start], dim=0).numpy()
    x_min, y_min = all_points.min(axis=0) - 0.5
    x_max, y_max = all_points.max(axis=0) + 0.5
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Plot target distribution
    ax1.scatter(data[:, 0].numpy(), data[:, 1].numpy(), 
              s=5, alpha=0.1, color='mediumseagreen')
    
    # Initialize trajectory plots
    lines = []
    for i in range(num_particles):
        line, = ax1.plot([], [], 'r-', linewidth=1, alpha=0.5)
        lines.append(line)
    
    # Scatter plot for current positions
    scat = ax1.scatter([], [], s=30, c='gold', zorder=3)
    
    # Time indicator
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                        fontsize=12, color='white')
    
    # Mathematical annotation
    eq_text = ax1.text(0.5, 0.05, 
                      r"$\frac{dx}{dt} = v_\theta(x, t)$", 
                      transform=ax1.transAxes,
                      fontsize=16, ha='center', color='white')
    
    # Plot for vector field at current time
    vector_plot_res = 20
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, vector_plot_res),
        np.linspace(y_min, y_max, vector_plot_res)
    )
    positions = np.column_stack([X.ravel(), Y.ravel()])
    U = np.zeros((vector_plot_res, vector_plot_res))
    V = np.zeros((vector_plot_res, vector_plot_res))
    
    # Vector field quiver plot
    quiver = ax1.quiver(X, Y, U, V, alpha=0.6, color='dodgerblue', scale=30)
    
    # Panel 2: Log probability plot
    ax2.set_title("Log Probability Evolution", fontsize=16)
    ax2.set_xlabel("Time", fontsize=14)
    ax2.set_ylabel("Log Probability", fontsize=14)
    ax2.set_ylim(-10, 1)
    
    # Placeholder for probability lines
    prob_lines = []
    for i in range(min(10, num_particles)):
        prob_line, = ax2.plot([], [], '-', linewidth=1.5, alpha=0.7)
        prob_lines.append(prob_line)
    
    # Function to compute approximate log probability (simplified)
    def approx_log_prob(points, dataset, sigma=0.1):
        """Simple kernel density log probability"""
        log_probs = []
        for point in points:
            # Compute distances to all data points
            dists = np.sum((dataset - point)**2, axis=1)
            # Kernel density estimate (simplified)
            density = np.sum(np.exp(-dists / (2 * sigma**2)))
            log_prob = np.log(density + 1e-10) - np.log(len(dataset))
            log_probs.append(log_prob)
        return np.array(log_probs)
    
    # Precompute log probabilities for each particle at each step
    log_probs = np.zeros((num_frames, num_particles))
    data_np = data.cpu().numpy()
    for i in range(num_frames):
        log_probs[i] = approx_log_prob(trajectories[i], data_np)
    
    # Animation update function
    def update(frame):
        # Update trajectory lines
        for i, line in enumerate(lines):
            line.set_data(
                trajectories[:frame+1, i, 0],
                trajectories[:frame+1, i, 1]
            )
        
        # Update scatter positions
        scat.set_offsets(trajectories[frame])
        
        # Update time text
        time_text.set_text(f't = {t_eval_actual[frame]:.2f}')
        
        # Update vector field
        # Get vector field at current time
        points_tensor = torch.tensor(positions, dtype=torch.float32).to(device)
        t_tensor = torch.tensor(t_eval_actual[frame], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            vectors = vector_field(points_tensor, t_tensor).cpu().numpy()
        
        U = vectors[:, 0].reshape(vector_plot_res, vector_plot_res)
        V = vectors[:, 1].reshape(vector_plot_res, vector_plot_res)
        
        # Update quiver plot
        quiver.set_UVC(U, V)
        
        # Update probability lines
        for i, line in enumerate(prob_lines):
            if i < len(prob_lines):
                line.set_data(
                    t_eval_actual[:frame+1],
                    log_probs[:frame+1, i]
                )
        
        return lines + [scat, time_text, quiver] + prob_lines
    
    # Create animation
    animation = FuncAnimation(
        fig, update, frames=num_frames, interval=100, blit=True
    )
    
    animation.save(save_path, writer='pillow', fps=10, dpi=100)
    print(f"Saved ODE visualization to {save_path}")
    
    return animation

def main():
    print(f"Using device: {device}")
    
    # Create data
    print("Generating Swiss Roll dataset...")
    data = create_swiss_roll()
    
    # Train vector field model
    print("Training vector field model...")
    vector_field, losses = train_vector_field(data, steps=1500)
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('Vector Field Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('vector_field_loss.png', dpi=150, bbox_inches='tight')
    print("Saved loss curve to vector_field_loss.png")
    
    # Visualize ODE flow
    visualize_flow_ode(data, vector_field)

if __name__ == "__main__":
    main()