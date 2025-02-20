import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import math
from matplotlib.animation import FuncAnimation

def generate_samples(continuous_ae, flow_model, n_samples=36, 
                    steps=100, scheduler='cosine', device='cuda',
                    denoise_strength=0.0, guidance_scale=1.0):
    """
    Generate samples using the rectified flow model and decode them
    
    Args:
        continuous_ae: Trained continuous autoencoder
        flow_model: Trained rectified flow model
        n_samples: Number of samples to generate
        steps: Number of ODE solver steps
        scheduler: Timestep scheduler ('linear', 'cosine', 'sigmoid')
        device: Computing device
        denoise_strength: Noise added during sampling (0.0-1.0)
        guidance_scale: Guidance scale for classifier-free guidance
        
    Returns:
        final_images: Final generated images
        all_images: List of all intermediate images for animation
    """
    continuous_ae.eval()
    flow_model.eval()
    
    # Get latent shape
    with torch.no_grad():
        # Create dummy input to determine shapes
        dummy = torch.zeros(1, continuous_ae.in_channels, 
                          continuous_ae.h, continuous_ae.w, device=device)
        latent = continuous_ae.encode(dummy)
        latent_shape = latent.shape[1:]  # [C, H, W]
    
    # Create solver
    solver = RectifiedFlowODESolver(flow_model, device)
    
    # Generate samples
    print(f"Generating {n_samples} samples with {steps} solver steps...")
    with torch.no_grad():
        # Generate latent samples
        latents, latent_trajectory = solver.sample(
            batch_size=n_samples,
            latent_shape=latent_shape,
            steps=steps,
            scheduler=scheduler,
            denoise_strength=denoise_strength,
            guidance_scale=guidance_scale
        )
        
        # Decode all latents from trajectory for visualization
        all_decoded_images = []
        print("Decoding trajectory images...")
        for i, latent_batch in enumerate(tqdm(latent_trajectory)):
            # Skip some frames to make animation more efficient
            if len(latent_trajectory) > 60 and i % (len(latent_trajectory) // 60) != 0 and i != len(latent_trajectory) - 1:
                continue
                
            latent_batch = latent_batch.to(device)
            decoded = continuous_ae.decode(latent_batch)
            # Convert to numpy
            decoded_np = (decoded.detach().cpu().numpy() * 255).astype(np.uint8)
            
            # Format based on channel count
            if continuous_ae.in_channels == 1:  # Grayscale
                imgs = decoded_np.squeeze(1)
            else:  # RGB
                imgs = decoded_np.transpose(0, 2, 3, 1)
                
            all_decoded_images.append(imgs)
        
        # Decode final latents
        final_decoded = continuous_ae.decode(latents)
        
    # Convert to numpy for visualization
    final_decoded_np = (final_decoded.detach().cpu().numpy() * 255).astype(np.uint8)
    
    if continuous_ae.in_channels == 1:  # Grayscale
        final_images = final_decoded_np.squeeze(1)
    else:  # RGB
        final_images = final_decoded_np.transpose(0, 2, 3, 1)
    
    return final_images, all_decoded_images

def visualize_samples(images, save_path="generated_images.png", is_color=True):
    """Visualize generated samples in a grid"""
    n_samples = len(images)
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))
    
    plt.figure(figsize=(cols * 2, rows * 2))
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        plt.subplot(rows, cols, i + 1)
        if is_color:
            plt.imshow(img.astype(np.uint8))
        else:
            plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved visualization to {save_path}")
    plt.close()

def create_generation_animation(all_images, save_path="generation_animation.gif", 
                               fps=15, dpi=150, is_color=True):
    """Create an animation of the generation process"""
    n_steps = len(all_images)
    n_samples = min(36, len(all_images[0]))
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))
    
    os.makedirs("frames", exist_ok=True)
    frame_paths = []
    
    for t in tqdm(range(n_steps), desc="Creating animation frames"):
        plt.figure(figsize=(cols * 2, rows * 2))
        for i in range(n_samples):
            plt.subplot(rows, cols, i + 1)
            if is_color:
                plt.imshow(all_images[t][i].astype(np.uint8))
            else:
                plt.imshow(all_images[t][i], cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Generation Progress: {t/(n_steps-1)*100:.1f}%", fontsize=16)
        plt.tight_layout()
        
        frame_path = f"frames/frame_{t:04d}.png"
        plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        frame_paths.append(frame_path)
        plt.close()
    
    # Load frames and create GIF
    frames = [Image.open(fp) for fp in frame_paths]
    
    # Resize to reduce file size if needed
    if dpi > 100:
        size = frames[0].size
        new_size = (size[0] // 2, size[1] // 2)
        frames = [f.resize(new_size, Image.LANCZOS) for f in frames]
    
    # Save as GIF    
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                  optimize=True, duration=1000//fps, loop=0)
    print(f"Saved animation to {save_path}")
    
    # Clean up frame files
    for fp in frame_paths:
        os.remove(fp)

def visualize_autoencoder_reconstructions(model, data, save_path="autoencoder_reconstructions.png"):
    """Visualize original and reconstructed images using the autoencoder"""
    model.eval()
    with torch.no_grad():
        recon_batch, _ = model(data)
    
    # Convert to numpy
    original = data.detach().cpu().numpy() * 255
    recon = recon_batch.detach().cpu().numpy() * 255
    
    # Get number of samples and channels
    n_samples = original.shape[0]
    n_channels = original.shape[1]
    
    plt.figure(figsize=(n_samples*2, 4))
    for i in range(n_samples):
        # Original
        plt.subplot(2, n_samples, i+1)
        if n_channels == 1:
            plt.imshow(original[i, 0], cmap='gray')
        else:
            plt.imshow(original[i].transpose(1, 2, 0).astype(np.uint8))
        plt.title("Original")
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(2, n_samples, n_samples+i+1)
        if n_channels == 1:
            plt.imshow(recon[i, 0], cmap='gray')
        else:
            plt.imshow(recon[i].transpose(1, 2, 0).astype(np.uint8))
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def create_latent_space_visualization(continuous_ae, flow_model, data, 
                                    n_samples=100, save_path="latent_manifold.png"):
    """
    Visualize samples from the latent space of the continuous autoencoder
    with flow trajectory
    """
    continuous_ae.eval()
    flow_model.eval()
    
    # Get a subset of data
    data_subset = torch.tensor(data[:n_samples], dtype=torch.float32)
    data_subset = data_subset.to(next(continuous_ae.parameters()).device)
    
    # Encode data to get latent vectors
    with torch.no_grad():
        latents = continuous_ae.encode(data_subset)
        
    # Use PCA to reduce dimensions for visualization
    from sklearn.decomposition import PCA
    
    # Flatten latents for PCA
    latents_flat = latents.detach().cpu().numpy()
    latents_flat = latents_flat.reshape(latents_flat.shape[0], -1)
    
    # Apply PCA
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_flat)
    
    # Sample some random points for flow visualization
    n_flow_samples = 10
    random_indices = np.random.choice(len(latents_2d), n_flow_samples, replace=False)
    solver = RectifiedFlowODESolver(flow_model, data_subset.device)
    
    # Generate flow trajectories for selected points
    flow_trajectories = []
    original_latents = latents[random_indices]
    
    # Generate trajectories
    steps = 20
    ts = torch.linspace(0, 1, steps).to(data_subset.device)
    
    for i, idx in enumerate(random_indices):
        # Start from random noise
        z0 = torch.randn_like(original_latents[0:1])
        trajectory = [z0]
        
        # Generate trajectory points
        for j in range(1, len(ts)):
            t_now = ts[j-1]
            t_next = ts[j]
            dt = t_next - t_now
            
            # Use RK4 step
            t_batch = torch.ones(1, device=data_subset.device) * t_now
            z_next = solver._runge_kutta_step(trajectory[-1], t_batch, dt)
            trajectory.append(z_next)
        
        # Convert to PCA space
        trajectory_flat = [t.detach().cpu().reshape(1, -1).numpy() for t in trajectory]
        trajectory_flat = np.vstack(trajectory_flat)
        trajectory_2d = pca.transform(trajectory_flat)
        flow_trajectories.append(trajectory_2d)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot all latent points
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6, s=30, c='lightblue', label='Encoded Data')
    
    # Plot selected points and their trajectories
    for i, (idx, traj) in enumerate(zip(random_indices, flow_trajectories)):
        plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1], s=100, c='red')
        
        # Plot trajectories with color gradient
        for j in range(len(traj)-1):
            progress = j / (len(traj)-1)
            color = plt.cm.viridis(progress)
            plt.plot(traj[j:j+2, 0], traj[j:j+2, 1], '-', color=color, linewidth=2)
            
            # Add arrow at middle point
            if j == len(traj) // 2:
                dx = traj[j+1, 0] - traj[j, 0]
                dy = traj[j+1, 1] - traj[j, 1]
                plt.arrow(traj[j, 0], traj[j, 1], dx, dy, 
                         head_width=0.3, head_length=0.3, fc='black', ec='black')
    
    plt.title('Latent Space PCA Projection with Flow Trajectories', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Create custom legend for trajectory gradient
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightblue', marker='o', linestyle='None', 
              markersize=10, alpha=0.6, label='Encoded Data Points'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', 
              markersize=10, label='Selected Points'),
        Line2D([0], [0], color=plt.cm.viridis(0), linewidth=2, label='t=0 (noise)'),
        Line2D([0], [0], color=plt.cm.viridis(0.5), linewidth=2, label='t=0.5'),
        Line2D([0], [0], color=plt.cm.viridis(1.0), linewidth=2, label='t=1 (data)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return latents_2d, flow_trajectories

def evaluate_sample_quality(generated_images, reference_images=None, 
                          save_path="sample_quality_metrics.txt"):
    """
    Evaluate generated sample quality using:
    1. Inception Score (IS)
    2. Fréchet Inception Distance (FID) if reference images are provided
    """
    try:
        from torch_fidelity import calculate_metrics
        
        # Prepare directory for temporary image saving
        os.makedirs("temp_gen_images", exist_ok=True)
        os.makedirs("temp_ref_images", exist_ok=True)
        
        # Save generated images
        for i, img in enumerate(generated_images):
            if len(img.shape) == 2:  # Convert grayscale to RGB
                img_rgb = np.stack([img, img, img], axis=2)
            else:
                img_rgb = img
                
            # Ensure uint8 format
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)
                
            Image.fromarray(img_rgb).save(f"temp_gen_images/gen_{i:04d}.png")
        
        # Save reference images if provided
        metrics_dict = {}
        if reference_images is not None:
            for i, img in enumerate(reference_images):
                if len(img.shape) == 2:  # Convert grayscale to RGB
                    img_rgb = np.stack([img, img, img], axis=2)
                else:
                    img_rgb = img
                    
                # Ensure uint8 format
                if img_rgb.dtype != np.uint8:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                    
                Image.fromarray(img_rgb).save(f"temp_ref_images/ref_{i:04d}.png")
            
            # Calculate both IS and FID
            metrics_dict = calculate_metrics(
                input1="temp_gen_images",
                input2="temp_ref_images",
                metrics=["inception_score", "fid"],
                batch_size=32,
                verbose=True
            )
        else:
            # Calculate just IS when no reference provided
            metrics_dict = calculate_metrics(
                input1="temp_gen_images",
                metrics=["inception_score"],
                batch_size=32,
                verbose=True
            )
        
        # Save metrics to file
        with open(save_path, 'w') as f:
            for metric, value in metrics_dict.items():
                f.write(f"{metric}: {value}\n")
                print(f"{metric}: {value}")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree("temp_gen_images")
        if reference_images is not None:
            shutil.rmtree("temp_ref_images")
            
        return metrics_dict
        
    except ImportError:
        print("torch-fidelity package not found. Install with: pip install torch-fidelity")
        return None

def sample_with_conditional_guidance(continuous_ae, flow_model, conditioning_latent,
                                    strength=0.5, steps=100, scheduler='cosine',
                                    device='cuda'):
    """
    Generate samples with guidance toward a specific image
    
    Args:
        continuous_ae: Continuous autoencoder model
        flow_model: Rectified flow model
        conditioning_latent: Latent vector to guide generation toward
        strength: Guidance strength (0.0-1.0)
        steps: Number of ODE solver steps
        scheduler: Timestep scheduler
        device: Computing device
        
    Returns:
        Final image and trajectory
    """
    continuous_ae.eval()
    flow_model.eval()
    
    # Get latent shape
    latent_shape = conditioning_latent.shape[1:]
    
    # Create solver
    solver = RectifiedFlowODESolver(flow_model, device)
    
    # Modified sampling function with conditioning
    def guided_velocity(x, t):
        """Get velocity with conditioning guidance"""
        with torch.no_grad():
            # Get model prediction
            v_pred = flow_model(x, t)
            
            # Add guidance toward conditioning_latent
            # Calculate guidance vector (points toward conditioning_latent)
            guidance_direction = (conditioning_latent - x)
            guidance_strength = strength * (1.0 - t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            
            # Combine model prediction with guidance
            v_guided = v_pred + guidance_strength * guidance_direction
            
            return v_guided
    
    # Override the velocity function in solver
    original_velocity_fn = solver._get_velocity
    solver._get_velocity = guided_velocity
    
    # Generate sample
    with torch.no_grad():
        # Generate latent sample 
        latent, latent_trajectory = solver.sample(
            batch_size=1,
            latent_shape=latent_shape,
            steps=steps,
            scheduler=scheduler
        )
        
        # Decode trajectory
        decoded_trajectory = []
        for latent_t in latent_trajectory:
            decoded = continuous_ae.decode(latent_t.to(device))
            decoded_np = (decoded.detach().cpu().numpy() * 255).astype(np.uint8)
            
            if continuous_ae.in_channels == 1:
                imgs = decoded_np.squeeze(1)
            else:
                imgs = decoded_np.transpose(0, 2, 3, 1)
                
            decoded_trajectory.append(imgs[0])  # Just keep the first (only) image
    
    # Restore original velocity function
    solver._get_velocity = original_velocity_fn
    
    # Decode final latent
    final_decoded = continuous_ae.decode(latent)
    final_np = (final_decoded.detach().cpu().numpy() * 255).astype(np.uint8)
    
    if continuous_ae.in_channels == 1:
        final_image = final_np.squeeze(1)[0]
    else:
        final_image = final_np.transpose(0, 2, 3, 1)[0]
        
    return final_image, decoded_trajectory
