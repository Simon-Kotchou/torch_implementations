import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math
import os
from matplotlib.animation import FuncAnimation
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

##########################################
# Load LFW Dataset with Color Option
##########################################
def load_lfw_dataset(color=False, resize=None):
    print(f"Loading LFW dataset (color={color})...")
    lfw_people = fetch_lfw_people(resize=resize, color=color)
    
    if color:
        # Color images have shape [n_samples, h, w, 3]
        n_samples, h, w, c = lfw_people.images.shape
        # Reshape to [n_samples, h, w, 3] → [n_samples, 3, h, w]
        X = lfw_people.images.transpose(0, 3, 1, 2)
        in_channels = 3
    else:
        # Grayscale images have shape [n_samples, h, w]
        n_samples, h, w = lfw_people.images.shape
        X = lfw_people.data.reshape(n_samples, 1, h, w)
        in_channels = 1
    
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    
    print(f"Dataset: {n_samples} samples, {n_classes} classes")
    print(f"Image size: {h}x{w}, channels: {in_channels}")
    
    # Normalize to [0, 1]
    X = X / 255.0
        
    return X, y, h, w, in_channels, n_samples, n_classes

##########################################
# Vector Quantization Layer
##########################################
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, diversity_weight=0.1):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.diversity_weight = diversity_weight
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, inputs):
        # inputs: [B, C, H, W] -> convert to BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape  # [B, H, W, C]
        
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = self.commitment_cost * e_latent_loss
        
        # Calculate codebook usage/diversity metrics
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Simple diversity loss - encourage uniform codebook usage
        # Higher perplexity is better (max value is num_embeddings)
        diversity_loss = self.diversity_weight * (self.num_embeddings - perplexity) / self.num_embeddings
        
        # Total loss
        loss = q_latent_loss + commitment_loss + diversity_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices

##########################################
# Residual Block for Higher Quality
##########################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if not self.same_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if not self.same_channels:
            identity = self.shortcut(x)
            
        out += identity
        out = F.relu(out)
        return out

##########################################
# Improved VQ-VAE Architecture
##########################################
class VQVAE(nn.Module):
    def __init__(self, h=125, w=94, in_channels=1, hidden_dims=[16, 32, 64], 
                 embedding_dim=64, num_embeddings=256, commitment_cost=0.25,
                 diversity_weight=0.1):
        super(VQVAE, self).__init__()
        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        
        # Calculate encoded dimensions
        encoder_layers = len(hidden_dims)
        self.h_encoded = h // (2 ** encoder_layers)
        self.w_encoded = w // (2 ** encoder_layers)
        
        # Calculate padded dimensions for exact reconstruction
        self.h_padded = self.h_encoded * (2 ** encoder_layers)
        self.w_padded = self.w_encoded * (2 ** encoder_layers)
        
        # Encoder
        encoder_modules = []
        in_ch = in_channels
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    ResidualBlock(h_dim, h_dim)
                )
            )
            in_ch = h_dim
            
        encoder_modules.append(
            nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1)
        )
        
        self.encoder = nn.Sequential(*encoder_modules)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, diversity_weight)
        
        # Decoder
        decoder_modules = []
        decoder_modules.append(
            nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=1)
        )
        
        for i in range(len(hidden_dims)-1, 0, -1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 
                                      kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i-1]),
                    nn.LeakyReLU(),
                    ResidualBlock(hidden_dims[i-1], hidden_dims[i-1])
                )
            )
            
        # Final layer - use Sigmoid for grayscale, Tanh for RGB
        final_activation = nn.Sigmoid() if in_channels == 1 else nn.Tanh()
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0], in_channels, 
                                  kernel_size=4, stride=2, padding=1),
                final_activation
            )
        )
        
        self.decoder = nn.Sequential(*decoder_modules)
        
    def encode(self, x):
        """Encode input to latent representation before quantization"""
        x = self._ensure_dimensions(x)
        encoded = self.encoder(x)
        return encoded
    
    def encode_and_quantize(self, x):
        """Encode input and apply vector quantization"""
        x = self._ensure_dimensions(x)
        encoded = self.encoder(x)
        _, quantized, perplexity, _, indices = self.vq(encoded)
        return quantized, perplexity, indices
    
    def _ensure_dimensions(self, x):
        """Ensure input has the correct dimensions"""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale
        if x.shape[2:] != (self.h, self.w):
            x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=False)
        return x
        
    def decode(self, z):
        """Decode latent representation"""
        reconstructed = self.decoder(z)
        # Handle RGB post-processing if using tanh
        if self.in_channels == 3 and isinstance(self.decoder[-1][-1], nn.Tanh):
            reconstructed = (reconstructed + 1) / 2  # Convert from [-1,1] to [0,1]
        return reconstructed
    
    def encode_to_indices(self, x):
        """Encode input to discrete indices"""
        x = self._ensure_dimensions(x)
        encoded = self.encoder(x)
        _, _, _, _, indices = self.vq(encoded)
        batch_size = x.shape[0]
        indices = indices.view(batch_size, self.h_encoded, self.w_encoded)
        return indices
    
    def decode_from_indices(self, indices, batch_size):
        """Decode from discrete indices"""
        # Flatten spatial dimensions: [batch_size, h_encoded*w_encoded]
        flat_indices = indices.reshape(batch_size, -1)
        one_hot = F.one_hot(flat_indices, num_classes=self.vq.num_embeddings).float()
        quantized = torch.matmul(one_hot, self.vq.embedding.weight)
        quantized = quantized.reshape(batch_size, self.h_encoded, self.w_encoded, self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decode(quantized)
        
    def forward(self, x):
        """Full forward pass: encode, quantize, decode"""
        x = self._ensure_dimensions(x)
        encoded = self.encoder(x)
        vq_loss, quantized, perplexity, _, _ = self.vq(encoded)
        reconstructed = self.decoder(quantized)
        
        # Ensure output matches input dimensions
        if reconstructed.shape[2:] != x.shape[2:]:
            reconstructed = F.interpolate(reconstructed, size=x.shape[2:], 
                                         mode='bilinear', align_corners=False)
        
        return reconstructed, vq_loss, perplexity
    
##########################################
# Train VQ-VAE
##########################################
def train_vqvae(data, h, w, in_channels, batch_size=64, epochs=50, lr=3e-4,
               hidden_dims=[64, 128, 256], num_embeddings=512, 
               embedding_dim=64, device='cuda'):
    """
    Train the VQ-VAE on images.
    Assumes data is a numpy array of shape [n_samples, channels, h, w].
    """
    n_samples = len(data)
    
    # Create dataset and dataloader
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create VQ-VAE model
    vqvae = VQVAE(h, w, in_channels=in_channels, 
                 hidden_dims=hidden_dims,
                 num_embeddings=num_embeddings,
                 embedding_dim=embedding_dim).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(vqvae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training metrics
    losses = []
    recon_losses = []
    vq_losses = []
    best_loss = float('inf')
    
    # Train loop
    for epoch in range(epochs):
        vqvae.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        pbar = tqdm(dataloader, desc=f"VQ-VAE Epoch {epoch+1}/{epochs}")
        
        for (data,) in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, vq_loss, _ = vqvae(data)
            
            # Calculate reconstruction loss
            recon_loss = F.mse_loss(recon_batch, data)
            
            # Total loss
            loss = recon_loss + vq_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'vq': vq_loss.item()
            })
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_vq_loss = epoch_vq_loss / len(dataloader)
        
        losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        vq_losses.append(avg_vq_loss)
        
        print(f"Epoch {epoch+1}: Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, VQ: {avg_vq_loss:.6f}")
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vqvae.state_dict(), "vqvae_best_model.pt")
            print(f"Saved new best model with loss: {best_loss:.6f}")
            
            # Generate some reconstruction samples at best loss
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                visualize_vqvae_reconstructions(vqvae, data_tensor[:10].to(device), 
                                             save_path=f"vqvae_recon_epoch_{epoch+1}.png")
    
    # Save final model
    torch.save(vqvae.state_dict(), "vqvae_final_model.pt")
    
    # Plot training loss
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(vq_losses)
    plt.title('VQ Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vqvae_training_loss.png', dpi=150)
    plt.close()
    
    return vqvae, losses, recon_losses, vq_losses

##########################################
# Improved Flow Matching DiT (for VQ-VAE latent space)
##########################################
class EnhancedFlowMatchingDiT(nn.Module):
    def __init__(self, input_dim, spatial_dims, embed_dim=512, depth=8, num_heads=8, 
                mlp_ratio=4.0, dropout=0.1, time_embed_dim=512):
        """
        Enhanced Flow Matching model for VQ-VAE latent space
        input_dim: embedding dimension from VQ-VAE
        spatial_dims: tuple (h, w) of the latent spatial dimensions
        """
        super().__init__()
        self.input_dim = input_dim
        self.spatial_dims = spatial_dims
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        
        # Improved time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Project input to embedding dimension
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.LayerNorm([embed_dim, *spatial_dims])
        )
        
        # Reshape for transformer blocks
        self.spatial_tokens = spatial_dims[0] * spatial_dims[1]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim=embed_dim, num_heads=num_heads,
                    hidden_dim=int(embed_dim * mlp_ratio), dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project back to input dimension
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 2, input_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t):
        """
        x: [B, C, H, W] - latent feature maps from VQ-VAE encoder
        t: [B] - timesteps
        """
        batch_size = x.shape[0]
        
        # Embed time
        t_emb = advanced_time_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # [B, embed_dim]
        
        # Project input to embedding space
        h = self.input_proj(x)  # [B, embed_dim, H, W]
        
        # Reshape to sequence for transformer
        h_seq = h.flatten(2).permute(0, 2, 1)  # [B, H*W, embed_dim]
        
        # Add time embedding to each token
        t_emb = t_emb.unsqueeze(1).expand(-1, self.spatial_tokens, -1)
        h_seq = h_seq + t_emb
        
        # Process through transformer blocks
        for block in self.blocks:
            h_seq = block(h_seq)
        
        h_seq = self.norm(h_seq)
        
        # Reshape back to spatial
        h = h_seq.permute(0, 2, 1).reshape(batch_size, self.embed_dim, 
                                         self.spatial_dims[0], self.spatial_dims[1])
        
        # Project back to input dimension
        velocity = self.output_proj(h)
        
        return velocity

##########################################
# Advanced Time Embedding
##########################################
def advanced_time_embedding(t, dim=512):
    half_dim = dim // 2
    freqs = torch.exp(-torch.arange(half_dim, device=t.device) * math.log(10000) / half_dim)
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    # Make embedding larger and add non-linearity
    embedding = nn.Linear(dim, dim).to(t.device)(embedding)
    embedding = F.silu(embedding)
    embedding = nn.Linear(dim, dim).to(t.device)(embedding)
    return embedding

##########################################
# Attention & Feed-Forward Blocks (DiT)
##########################################
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                              dropout=dropout, batch_first=True)
        
    def forward(self, x):
        x_norm = self.norm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        return x + attn_output

class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
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

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.attn = AttentionBlock(dim, num_heads, dim_head, dropout)
        self.ff = FeedForwardBlock(dim, hidden_dim, dropout)
        
    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x

##########################################
# IMPROVED: RK4 ODE Solver for Generation
##########################################
def rk4_step(model, x, t_curr, t_next, device):
    """
    Runge-Kutta 4th order integration step.
    model: the flow model predicting velocity
    x: current state [B, C, H, W]
    t_curr: current timestep
    t_next: next timestep
    """
    dt = t_next - t_curr
    
    # Expand t for batched inputs
    batch_size = x.shape[0]
    t_curr_batch = t_curr.expand(batch_size).to(device)
    t_half = t_curr + dt/2
    t_half_batch = t_half.expand(batch_size).to(device)
    t_next_batch = t_next.expand(batch_size).to(device)
    
    # RK4 integration steps
    k1 = model(x, t_curr_batch)
    k2 = model(x + dt * k1 / 2, t_half_batch)
    k3 = model(x + dt * k2 / 2, t_half_batch)
    k4 = model(x + dt * k3, t_next_batch)
    
    # Update
    x_next = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x_next

##########################################
# Flow Matching Scheduler
##########################################
class FlowMatchingScheduler:
    def __init__(self, num_inference_steps=100, scheduler_type='cosine', min_t=0.001, max_t=0.999):
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
            alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            timesteps = torch.clamp(alpha_cumprod, self.min_t, self.max_t)
            return timesteps[:-1]
        elif self.scheduler_type == 'shifted':
            shifting_factor = 7.0
            steps = np.linspace(0, 1, self.num_inference_steps)
            s = shifting_factor
            steps = s * steps / (1 + (s - 1) * steps)
            steps = (1 - steps) * (self.max_t - self.min_t) + self.min_t
            return torch.from_numpy(steps).float()
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def get_timesteps(self):
        """Returns timesteps in decreasing order (suitable for generation)"""
        return self.timesteps

##########################################
# Generate Samples from VQ-VAE + Flow Model
##########################################
def generate_samples_vqvae_flow(flow_model, vqvae_model, n_samples=36, steps=100, 
                               scheduler_type='cosine', device='cuda'):
    """
    Generate samples using the flow model and decode them with VQ-VAE.
    """
    flow_model.eval()
    vqvae_model.eval()
    
    # Get VQ-VAE latent shape
    with torch.no_grad():
        dummy = torch.zeros(1, vqvae_model.in_channels, 
                           vqvae_model.h, vqvae_model.w, device=device)
        latent, _, _ = vqvae_model.encode_and_quantize(dummy)
        latent_shape = latent.shape
    
    # Create scheduler with properly ordered timesteps (t=1 → t=0)
    scheduler = FlowMatchingScheduler(num_inference_steps=steps, scheduler_type=scheduler_type)
    timesteps = scheduler.get_timesteps().to(device)
    # Reverse timesteps for generation (from t=1 to t=0)
    timesteps = torch.flip(timesteps, [0])
    
    # Sample initial latent from standard normal
    x = torch.randn(n_samples, *latent_shape[1:], device=device)
    
    # Store all states for animation
    all_states = [x.detach().cpu()]
    
    # Integration using RK4 method
    with torch.no_grad():
        for i in range(len(timesteps) - 1):
            # Current and next timestep
            t_curr = timesteps[i]
            t_next = timesteps[i+1]
            
            # Improved ODE integration with RK4
            x = rk4_step(flow_model, x, t_curr, t_next, device)
            
            # Add a small amount of noise for stability (optional)
            if i % 10 == 0 and i < len(timesteps) - 5:
                noise_scale = 0.01 * (1.0 - t_next.item())
                x = x + torch.randn_like(x) * noise_scale
            
            # Save current state for visualization
            all_states.append(x.detach().cpu())
            
            # Log progress
            if i % 20 == 0 or i == len(timesteps) - 2:
                print(f"Generation progress: {i+1}/{len(timesteps)-1} steps, t={t_next.item():.3f}")
    
    # Decode final latents
    with torch.no_grad():
        final_decoded = vqvae_model.decode(all_states[-1].to(device))
    
    # Convert to numpy for visualization
    final_decoded_np = (final_decoded.detach().cpu().numpy() * 255).astype(np.uint8)
    
    if vqvae_model.in_channels == 1:  # If grayscale
        final_images = final_decoded_np.squeeze(1)
    else:  # If RGB
        final_images = final_decoded_np.transpose(0, 2, 3, 1)
    
    # Generate animation frames by decoding all intermediate states
    all_decoded_images = []
    with torch.no_grad():
        for i, state in enumerate(all_states):
            # Skip some frames for efficiency in longer sequences
            if len(all_states) > 60 and i % (len(all_states) // 60) != 0 and i != len(all_states) - 1:
                continue
                
            decoded = vqvae_model.decode(state.to(device))
            decoded_np = (decoded.detach().cpu().numpy() * 255).astype(np.uint8)
            
            # Format for visualization
            if vqvae_model.in_channels == 1:  # If grayscale
                imgs = decoded_np.squeeze(1)
            else:  # If RGB
                imgs = decoded_np.transpose(0, 2, 3, 1)
                
            all_decoded_images.append(imgs)
    
    return final_images, all_decoded_images

##########################################
# Visualization Helpers
##########################################
def visualize_vqvae_reconstructions(vqvae_model, data, save_path="vqvae_reconstructions.png"):
    """
    Visualize original and reconstructed images using the VQ-VAE.
    """
    vqvae_model.eval()
    with torch.no_grad():
        recon_batch, _, _ = vqvae_model(data)
    
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

def visualize_samples(images, save_path="generated_faces.png", is_color=False):
    """
    Visualize generated samples in a grid.
    """
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

def create_grid_animation(all_images, save_path="face_generation_grid.gif", fps=15, dpi=150, is_color=False):
    """
    Create an animation of the generation process.
    """
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
    
    # Load frames with PIL and create GIF
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

##########################################
# Main Function to Run Everything
##########################################
def main_vqvae_flow(color=False, embedding_dim=64, num_embeddings=512,
                   vae_epochs=5, flow_steps=2500):
    # Load dataset
    X, y, h, w, in_channels, n_samples, n_classes = load_lfw_dataset(color=color)
    
    # Print info
    print(f"Running VQ-VAE + Flow Matching on LFW dataset")
    print(f"Color mode: {color}, Image size: {h}x{w}, Channels: {in_channels}")
    
    # Setup directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Train VQ-VAE or load pretrained
    vqvae_path = f"checkpoints/vqvae_best_model_{'color' if color else 'gray'}.pt"
    
    if os.path.exists(vqvae_path):
        print(f"Loading pretrained VQ-VAE from {vqvae_path}")
        vqvae = VQVAE(h, w, in_channels=in_channels,
                     hidden_dims=[64, 128, 256],
                     num_embeddings=num_embeddings,
                     embedding_dim=embedding_dim).to(device)
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    else:
        print(f"Training VQ-VAE for {vae_epochs} epochs...")
        vqvae, vae_losses, recon_losses, vq_losses = train_vqvae(
            data=X,
            h=h,
            w=w,
            in_channels=in_channels,
            batch_size=64,
            epochs=vae_epochs,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device
        )
    
    # 2. Visualize VQ-VAE reconstructions
    print("Generating VQ-VAE reconstructions...")
    sample_data = torch.tensor(X[:10], dtype=torch.float32).to(device)
    visualize_vqvae_reconstructions(
        vqvae_model=vqvae,
        data=sample_data,
        save_path=f"results/vqvae_recon_{'color' if color else 'gray'}.png"
    )
    
    # 3. Train flow model or load pretrained
    flow_path = f"checkpoints/flow_model_vqvae_{'color' if color else 'gray'}_final.pt"
    
    if os.path.exists(flow_path):
        print(f"Loading pretrained flow model from {flow_path}")
        # Get VQ-VAE latent dimensions first
        with torch.no_grad():
            sample_batch = torch.tensor(X[:2], dtype=torch.float32).to(device)
            latent, _, _ = vqvae.encode_and_quantize(sample_batch)
            latent_shape = latent.shape
        
        input_dim = latent_shape[1]
        spatial_dims = (latent_shape[2], latent_shape[3])
        
        flow_model = EnhancedFlowMatchingDiT(
            input_dim=input_dim,
            spatial_dims=spatial_dims,
            embed_dim=512,
            num_heads=8,
            depth=8
        ).to(device)
        flow_model.load_state_dict(torch.load(flow_path, map_location=device))
    else:
        print(f"Training flow model for {flow_steps} steps...")
        flow_model, flow_losses, mse_losses, vel_norms = train_flow_model_with_vqvae(
            data=X,
            vqvae_model=vqvae,
            steps=flow_steps,
            batch_size=64,
            embed_dim=512,
            num_heads=8,
            depth=8,
            device=device
        )
        # Save final model
        with torch.no_grad():
            sample_batch = torch.tensor(X[:2], dtype=torch.float32).to(device)
            latent, _, _ = vqvae.encode_and_quantize(sample_batch)
            latent_shape = latent.shape
        
        input_dim = latent_shape[1]
        spatial_dims = (latent_shape[2], latent_shape[3])
            
        torch.save(flow_model.state_dict(), flow_path)
    
    # 4. Generate samples with different step counts
    for sample_steps in [50, 100, 200]:
        print(f"Generating samples with {sample_steps} integration steps...")
        final_images, all_images = generate_samples_vqvae_flow(
            flow_model=flow_model,
            vqvae_model=vqvae,
            n_samples=36,
            steps=sample_steps,
            device=device
        )
        
        visualize_samples(
            final_images,
            save_path=f"results/vqvae_flow_{'color' if color else 'gray'}_{sample_steps}steps.png",
            is_color=color
        )
        
        # Create animation for 100-step generation
        if sample_steps == 100:
            create_grid_animation(
                all_images,
                save_path=f"results/vqvae_flow_generation_{'color' if color else 'gray'}.gif",
                fps=20,
                is_color=color
            )
    
    print("VQ-VAE + Flow Matching experiment completed successfully!")
    return vqvae, flow_model

##########################################
# Train Flow Model with VQ-VAE Latent Space
##########################################
def train_flow_model_with_vqvae(data, vqvae_model, steps=20000, batch_size=64, 
                               lr=2e-4, embed_dim=512, num_heads=8, depth=8,
                               device='cuda', checkpoint_freq=1000):
    """
    Train the flow matching model in the VQ-VAE latent space.
    """
    n_samples = len(data)
    
    # Setup directories
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/flow_model_vqvae_latest.pt"
    start_step = 0
    
    # Prepare dataset
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get VQ-VAE encoder output shape
    vqvae_model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))[0][:2].to(device)
        latent, _, _ = vqvae_model.encode_and_quantize(sample_batch)
        latent_shape = latent.shape
        
    print(f"VQ-VAE latent shape: {latent_shape}")
    input_dim = latent_shape[1]  # Channels in latent space
    spatial_dims = (latent_shape[2], latent_shape[3])  # H, W in latent space
    
    # Create flow model
    flow_model = EnhancedFlowMatchingDiT(
        input_dim=input_dim,
        spatial_dims=spatial_dims,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth
    ).to(device)
    
    # Check for checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        flow_model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint['step'] + 1
        losses = checkpoint.get('losses', [])
        mse_losses = checkpoint.get('mse_losses', [])
        velocity_norms = checkpoint.get('velocity_norms', [])
        optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=lr/20,
            last_epoch=start_step-1 if start_step > 0 else -1
        )
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming from step {start_step}")
    else:
        optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/20)
        losses = []
        mse_losses = []
        velocity_norms = []
    
    # Training loop
    running_loss = 0.0
    reg_losses = []
    
    pbar = tqdm(range(start_step, steps), desc="Training Flow model with VQ-VAE latents")
    
    try:
        data_iterator = iter(dataloader)
        for step in pbar:
            # Get batch of data
            try:
                batch_data, = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch_data, = next(data_iterator)
            
            batch_data = batch_data.to(device)
            B = batch_data.shape[0]
            
            # Get latent features from VQ-VAE encoder
            with torch.no_grad():
                z_q, _, _ = vqvae_model.encode(batch_data)
            
            # Sample t uniformly for time conditioning - use more points near t=0
            t_raw = torch.rand(B, device=device)
            t = t_raw ** 1.5  # More samples near t=0
            
            # Sample x0 from standard normal with same shape as z_q
            x0 = torch.randn_like(z_q)
            
            # Linear interpolation for training
            x_t = (1 - t.unsqueeze(1).unsqueeze(2).unsqueeze(3)) * x0 + t.unsqueeze(1).unsqueeze(2).unsqueeze(3) * z_q
            
            # Target velocity is from noise to data
            v_true = z_q - x0
            
            # Predict velocity
            v_pred = flow_model(x_t, t)
            
            # Compute loss with regularization
            mse_loss = F.mse_loss(v_pred, v_true)
            
            # Regularization: penalize large velocity norms
            velocity_norm = torch.mean(torch.norm(v_pred.reshape(B, -1), dim=1))
            
            # Compute smoothness regularization (occasionally)
            if step % 10 == 0 and step > 0:
                # Sample new time points close to original ones
                t_perturbed = torch.clamp(t + 0.01 * torch.randn_like(t), min=0, max=1)
                
                # Compute interpolated points at perturbed times
                x_t_perturbed = (1 - t_perturbed.unsqueeze(1).unsqueeze(2).unsqueeze(3)) * x0 + \
                               t_perturbed.unsqueeze(1).unsqueeze(2).unsqueeze(3) * z_q
                
                # Get predictions at perturbed points
                v_pred_perturbed = flow_model(x_t_perturbed, t_perturbed)
                
                # Compute smoothness penalty
                smoothness_loss = F.mse_loss(v_pred, v_pred_perturbed) / (
                    F.mse_loss(t.unsqueeze(1), t_perturbed.unsqueeze(1)) + 1e-6
                )
            else:
                smoothness_loss = torch.tensor(0.0, device=device)
            
            # Total loss with annealed regularization
            reg_weight = min(1.0, step / (steps * 0.3)) * 1e-4
            smoothness_weight = min(1.0, step / (steps * 0.5)) * 1e-3
            
            loss = mse_loss + reg_weight * velocity_norm + smoothness_weight * smoothness_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            running_loss += loss.item()
            losses.append(loss.item())
            mse_losses.append(mse_loss.item())
            velocity_norms.append(velocity_norm.item())
            reg_losses.append((reg_weight * velocity_norm + smoothness_weight * smoothness_loss).item())
            
            pbar.set_postfix({
                'loss': loss.item(),
                'mse': mse_loss.item(),
                'v_norm': velocity_norm.item(),
                'lr': scheduler.get_last_lr()[0]
            })
            
            # Checkpointing
            if (step + 1) % checkpoint_freq == 0 or step == steps - 1:
                checkpoint = {
                    'step': step,
                    'model_state_dict': flow_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': losses,
                    'mse_losses': mse_losses,
                    'velocity_norms': velocity_norms,
                    'reg_losses': reg_losses,
                    'loss': loss.item()
                }
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, f"checkpoints/flow_model_vqvae_step_{step+1}.pt")
                
                # Calculate and report statistics
                avg_loss = running_loss / min(checkpoint_freq, step - start_step + 1)
                print(f"Step {step+1}, Average Loss: {avg_loss:.6f}")
                running_loss = 0.0
                
                # Plot progress at checkpoints
                if len(losses) > 100:
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 3, 1)
                    plt.plot(losses[-5000:])
                    plt.title('Total Loss')
                    plt.yscale('log')
                    
                    plt.subplot(1, 3, 2)
                    plt.plot(mse_losses[-5000:])
                    plt.title('MSE Loss')
                    plt.yscale('log')
                    
                    plt.subplot(1, 3, 3)
                    plt.plot(velocity_norms[-5000:])
                    plt.title('Velocity Norm')
                    
                    plt.tight_layout()
                    plt.savefig(f"checkpoints/vqvae_flow_training_progress_step_{step+1}.png", dpi=150)
                    plt.close()
                
    except Exception as e:
        print(f"Exception during training: {e}")
        # Save emergency checkpoint
        torch.save({
            'step': step if 'step' in locals() else 0,
            'model_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'losses': losses
        }, "checkpoints/flow_model_vqvae_emergency.pt")
        raise
        
    return flow_model, losses, mse_losses, velocity_norms

##########################################
# Generate Samples from VQ-VAE + Flow Model
##########################################
def generate_samples_vqvae_flow(flow_model, vqvae_model, n_samples=36, steps=100, 
                               scheduler_type='cosine', device='cuda'):
    """
    Generate samples using the flow model and decode them with VQ-VAE.
    """
    flow_model.eval()
    vqvae_model.eval()
    
    # Get VQ-VAE latent shape
    with torch.no_grad():
        dummy = torch.zeros(1, vqvae_model.in_channels, 
                           vqvae_model.h, vqvae_model.w, device=device)
        latent, _, _ = vqvae_model.encode_and_quantize(dummy)
        latent_shape = latent.shape
    
    # Create scheduler with properly ordered timesteps (t=1 → t=0)
    scheduler = FlowMatchingScheduler(num_inference_steps=steps, scheduler_type=scheduler_type)
    timesteps = scheduler.get_timesteps().to(device)
    # Reverse timesteps for generation (from t=1 to t=0)
    timesteps = torch.flip(timesteps, [0])
    
    # Sample initial latent from standard normal
    x = torch.randn(n_samples, *latent_shape[1:], device=device)
    
    # Store all states for animation
    all_states = [x.detach().cpu()]
    
    # Integration using RK4 method
    with torch.no_grad():
        for i in range(len(timesteps) - 1):
            # Current and next timestep
            t_curr = timesteps[i]
            t_next = timesteps[i+1]
            
            # Improved ODE integration with RK4
            x = rk4_step(flow_model, x, t_curr, t_next, device)
            
            # Add a small amount of noise for stability (optional)
            if i % 10 == 0 and i < len(timesteps) - 5:
                noise_scale = 0.01 * (1.0 - t_next.item())
                x = x + torch.randn_like(x) * noise_scale
            
            # Save current state for visualization
            all_states.append(x.detach().cpu())
            
            # Log progress
            if i % 20 == 0 or i == len(timesteps) - 2:
                print(f"Generation progress: {i+1}/{len(timesteps)-1} steps, t={t_next.item():.3f}")
    
    # Decode final latents
    with torch.no_grad():
        final_decoded = vqvae_model.decode(all_states[-1].to(device))
    
    # Convert to numpy for visualization
    final_decoded_np = (final_decoded.detach().cpu().numpy() * 255).astype(np.uint8)
    
    if vqvae_model.in_channels == 1:  # If grayscale
        final_images = final_decoded_np.squeeze(1)
    else:  # If RGB
        final_images = final_decoded_np.transpose(0, 2, 3, 1)
    
    # Generate animation frames by decoding all intermediate states
    all_decoded_images = []
    with torch.no_grad():
        for i, state in enumerate(all_states):
            # Skip some frames for efficiency in longer sequences
            if len(all_states) > 60 and i % (len(all_states) // 60) != 0 and i != len(all_states) - 1:
                continue
                
            decoded = vqvae_model.decode(state.to(device))
            decoded_np = (decoded.detach().cpu().numpy() * 255).astype(np.uint8)
            
            # Format for visualization
            if vqvae_model.in_channels == 1:  # If grayscale
                imgs = decoded_np.squeeze(1)
            else:  # If RGB
                imgs = decoded_np.transpose(0, 2, 3, 1)
                
            all_decoded_images.append(imgs)
    
    return final_images, all_decoded_images

##########################################
# Visualization Helpers
##########################################
def visualize_vqvae_reconstructions(vqvae_model, data, save_path="vqvae_reconstructions.png"):
    """
    Visualize original and reconstructed images using the VQ-VAE.
    """
    vqvae_model.eval()
    with torch.no_grad():
        recon_batch, _, _ = vqvae_model(data)
    
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

def visualize_samples(images, save_path="generated_faces.png", is_color=False):
    """
    Visualize generated samples in a grid.
    """
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

def create_grid_animation(all_images, save_path="face_generation_grid.gif", fps=15, dpi=150, is_color=False):
    """
    Create an animation of the generation process.
    """
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
    
    # Load frames with PIL and create GIF
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

##########################################
# Main Function to Run Everything
##########################################
def main_vqvae_flow(color=False, embedding_dim=64, num_embeddings=512,
                   vae_epochs=50, flow_steps=20000):
    # Load dataset
    X, y, h, w, in_channels, n_samples, n_classes = load_lfw_dataset(color=color)
    
    # Print info
    print(f"Running VQ-VAE + Flow Matching on LFW dataset")
    print(f"Color mode: {color}, Image size: {h}x{w}, Channels: {in_channels}")
    
    # Setup directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Train VQ-VAE or load pretrained
    vqvae_path = f"checkpoints/vqvae_best_model_{'color' if color else 'gray'}.pt"
    
    if os.path.exists(vqvae_path):
        print(f"Loading pretrained VQ-VAE from {vqvae_path}")
        vqvae = VQVAE(h, w, in_channels=in_channels,
                     hidden_dims=[64, 128, 256],
                     num_embeddings=num_embeddings,
                     embedding_dim=embedding_dim).to(device)
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    else:
        print(f"Training VQ-VAE for {vae_epochs} epochs...")
        vqvae, vae_losses, recon_losses, vq_losses = train_vqvae(
            data=X,
            h=h,
            w=w,
            in_channels=in_channels,
            batch_size=64,
            epochs=vae_epochs,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device
        )
    
    # 2. Visualize VQ-VAE reconstructions
    print("Generating VQ-VAE reconstructions...")
    sample_data = torch.tensor(X[:10], dtype=torch.float32).to(device)
    visualize_vqvae_reconstructions(
        vqvae_model=vqvae,
        data=sample_data,
        save_path=f"results/vqvae_recon_{'color' if color else 'gray'}.png"
    )
    
    # 3. Train flow model or load pretrained
    flow_path = f"checkpoints/flow_model_vqvae_{'color' if color else 'gray'}_final.pt"
    
    if os.path.exists(flow_path):
        print(f"Loading pretrained flow model from {flow_path}")
        # Get VQ-VAE latent dimensions first
        with torch.no_grad():
            sample_batch = torch.tensor(X[:2], dtype=torch.float32).to(device)
            latent, _, _ = vqvae.encode_and_quantize(sample_batch)
            latent_shape = latent.shape
        
        input_dim = latent_shape[1]
        spatial_dims = (latent_shape[2], latent_shape[3])
        
        flow_model = EnhancedFlowMatchingDiT(
            input_dim=input_dim,
            spatial_dims=spatial_dims,
            embed_dim=128,
            num_heads=4,
            depth=4
        ).to(device)
        flow_model.load_state_dict(torch.load(flow_path, map_location=device))
    else:
        print(f"Training flow model for {flow_steps} steps...")
        flow_model, flow_losses, mse_losses, vel_norms = train_flow_model_with_vqvae(
            data=X,
            vqvae_model=vqvae,
            steps=flow_steps,
            batch_size=64,
            embed_dim=128,
            num_heads=4,
            depth=4,
            device=device
        )
        # Save final model
        with torch.no_grad():
            sample_batch = torch.tensor(X[:2], dtype=torch.float32).to(device)
            latent, _, _ = vqvae.encode_and_quantize(sample_batch)
            latent_shape = latent.shape
        
        input_dim = latent_shape[1]
        spatial_dims = (latent_shape[2], latent_shape[3])
            
        torch.save(flow_model.state_dict(), flow_path)
    
    # 4. Generate samples with different step counts
    for sample_steps in [50, 100, 200]:
        print(f"Generating samples with {sample_steps} integration steps...")
        final_images, all_images = generate_samples_vqvae_flow(
            flow_model=flow_model,
            vqvae_model=vqvae,
            n_samples=36,
            steps=sample_steps,
            device=device
        )
        
        visualize_samples(
            final_images,
            save_path=f"results/vqvae_flow_{'color' if color else 'gray'}_{sample_steps}steps.png",
            is_color=color
        )
        
        # Create animation for 100-step generation
        if sample_steps == 100:
            create_grid_animation(
                all_images,
                save_path=f"results/vqvae_flow_generation_{'color' if color else 'gray'}.gif",
                fps=20,
                is_color=color
            )
    
    print("VQ-VAE + Flow Matching experiment completed successfully!")
    return vqvae, flow_model

if __name__ == "__main__":
    # Run with grayscale first
    vqvae_gray, flow_gray = main_vqvae_flow(
        color=False,
        embedding_dim=128,
        num_embeddings=256,
        vae_epochs=10,  # Reduced from 50
        flow_steps=5000  # Reduced from 20000
    )
    
    # Then try with color if desired
    # vqvae_color, flow_color = main_vqvae_flow(
    #     color=True,
    #     embedding_dim=64,
    #     num_embeddings=256,
    #     vae_epochs=5,  # Reduced from 50
    #     flow_steps=5000  # Reduced from 20000
    # )