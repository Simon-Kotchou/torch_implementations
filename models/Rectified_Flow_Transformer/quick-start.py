import torch
import numpy as np
import os
from sklearn.datasets import fetch_lfw_people

# Import the key components
from continuous_autoencoder import ContinuousAutoencoder, train_continuous_autoencoder
from rectified_flow_transformer import RectifiedFlowTransformer, train_rectified_flow
from sampling_and_evaluation import generate_samples, visualize_samples, create_generation_animation

def quick_start():
    """Quick start example with minimal configuration"""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 128
    in_channels = 3  # RGB
    latent_dim = 768
    batch_size = 16
    color = True
    
    # Create output directory
    os.makedirs('quick_start_results', exist_ok=True)
    
    # 1. Load dataset - we'll use LFW faces
    print("Loading dataset...")
    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=img_size, color=color)
    
    if color:
        n_samples, h, w, c = lfw_people.images.shape
        X = lfw_people.images.transpose(0, 3, 1, 2)  # [n_samples, 3, h, w]
    else:
        n_samples, h, w = lfw_people.images.shape
        X = lfw_people.data.reshape(n_samples, 1, h, w)
    
    # Normalize to [0, 1]
    X = X / 255.0
    print(f"Loaded {len(X)} images with shape {X.shape}")
    
    # 2. Create and train continuous autoencoder (reduced configuration)
    print("Creating and training autoencoder...")
    autoencoder = ContinuousAutoencoder(
        h=img_size,
        w=img_size,
        in_channels=in_channels,
        latent_dim=latent_dim,
        hidden_dims=[64, 128, 256, 512]
    ).to(device)
    
    # Train for just a few epochs for demonstration
    autoencoder, _ = train_continuous_autoencoder(
        data=X,
        h=img_size,
        w=img_size,
        in_channels=in_channels,
        batch_size=batch_size,
        epochs=5,  # Reduced for quick demonstration
        lr=3e-4,
        device=device
    )
    
    # 3. Create and train rectified flow model (reduced configuration)
    print("Creating and training rectified flow model...")
    # Get latent dimensions from autoencoder output
    with torch.no_grad():
        sample_batch = torch.tensor(X[:2], dtype=torch.float32).to(device)
        latent = autoencoder.encode(sample_batch)
        latent_shape = latent.shape[1:]  # [C, H, W]
    
    flow_model = RectifiedFlowTransformer(
        input_dim=latent_shape[0],
        spatial_dims=(latent_shape[1], latent_shape[2]),
        embed_dim=512,  # Reduced for quick demonstration
        depth=8,        # Reduced for quick demonstration
        num_heads=8
    ).to(device)
    
    # Train for just a few steps for demonstration
    flow_model, _ = train_rectified_flow(
        continuous_ae=autoencoder,
        data=X,
        batch_size=batch_size,
        steps=2000,  # Reduced for quick demonstration
        lr=1e-4,
        device=device
    )
    
    # 4. Generate samples
    print("Generating samples...")
    final_images, all_images = generate_samples(
        continuous_ae=autoencoder,
        flow_model=flow_model,
        n_samples=16,
        steps=50,  # Reduced for quick demonstration
        scheduler='cosine',
        device=device,
        denoise_strength=0.02
    )
    
    # 5. Visualize results
    visualize_samples(
        final_images,
        save_path='quick_start_results/generated_samples.png',
        is_color=color
    )
    
    create_generation_animation(
        all_images,
        save_path='quick_start_results/generation_animation.gif',
        fps=10,
        is_color=color
    )
    
    print("Quick start completed! Check the 'quick_start_results' directory for outputs.")
    return autoencoder, flow_model

if __name__ == "__main__":
    quick_start()
