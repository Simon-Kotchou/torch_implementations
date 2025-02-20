# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_lfw_people
# from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm
# import math
# import os
# from matplotlib.animation import FuncAnimation
# from PIL import Image

# # Set random seeds for reproducibility
# np.random.seed(42)
# torch.manual_seed(42)

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# ##########################################
# # Load LFW Dataset
# ##########################################
# print("Loading LFW dataset...")
# lfw_people = fetch_lfw_people(resize=None)
# n_samples, h, w = lfw_people.images.shape
# X = lfw_people.data
# n_features = X.shape[1]
# y = lfw_people.target
# target_names = lfw_people.target_names
# n_classes = target_names.shape[0]

# print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
# print(f"Image size: {h}x{w}")

# # Scale the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ##########################################
# # Improved Flow Matching Scheduler
# ##########################################
# class FlowMatchingScheduler:
#     def __init__(self, num_inference_steps=50, scheduler_type='cosine', min_t=0.002, max_t=0.998):
#         self.num_inference_steps = num_inference_steps
#         self.scheduler_type = scheduler_type
#         self.min_t = min_t
#         self.max_t = max_t
#         self.timesteps = self._get_schedule()
        
#     def _get_schedule(self):
#         if self.scheduler_type == 'linear':
#             return torch.linspace(self.max_t, self.min_t, self.num_inference_steps)
#         elif self.scheduler_type == 'cosine':
#             steps = torch.arange(self.num_inference_steps + 1).float() / self.num_inference_steps
#             alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
#             timesteps = torch.clamp(alpha_cumprod, self.min_t, self.max_t)
#             return timesteps[:-1]
#         elif self.scheduler_type == 'shifted':
#             shifting_factor = 7.0
#             steps = np.linspace(0, 1, self.num_inference_steps)
#             s = shifting_factor
#             steps = s * steps / (1 + (s - 1) * steps)
#             steps = (1 - steps) * (self.max_t - self.min_t) + self.min_t
#             return torch.from_numpy(steps).float()
#         else:
#             raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
#     def get_timesteps(self):
#         """Returns timesteps in decreasing order (suitable for generation)"""
#         return self.timesteps

# ##########################################
# # Attention & Feed-Forward Blocks (DiT)
# ##########################################
# class AttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, dim_head=64, dropout=0.0):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
#                                                dropout=dropout, batch_first=True)
        
#     def forward(self, x):
#         x_norm = self.norm(x)
#         attn_output, _ = self.attention(x_norm, x_norm, x_norm)
#         return x + attn_output

# class FeedForwardBlock(nn.Module):
#     def __init__(self, dim, hidden_dim=None, dropout=0.0):
#         super().__init__()
#         hidden_dim = hidden_dim or dim * 4
#         self.norm = nn.LayerNorm(dim)
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
        
#     def forward(self, x):
#         return x + self.net(self.norm(x))

# class DiTBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, dim_head=64, hidden_dim=None, dropout=0.0):
#         super().__init__()
#         self.attn = AttentionBlock(dim, num_heads, dim_head, dropout)
#         self.ff = FeedForwardBlock(dim, hidden_dim, dropout)
        
#     def forward(self, x):
#         x = self.attn(x)
#         x = self.ff(x)
#         return x

# ##########################################
# # Advanced Time Embedding
# ##########################################
# def advanced_time_embedding(t, dim=256):
#     half_dim = dim // 2
#     freqs = torch.exp(-torch.arange(half_dim, device=t.device) * math.log(10000) / half_dim)
#     args = t.unsqueeze(1) * freqs.unsqueeze(0)
#     embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
#     return embedding

# ##########################################
# # Enhanced Flow Matching DiT (for continuous latents)
# ##########################################
# class EnhancedFlowMatchingDiT(nn.Module):
#     def __init__(self, input_dim, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0,
#                  dropout=0.1, time_embed_dim=256):
#         """
#         input_dim: dimension of the continuous latent (e.g. latent_dim)
#         """
#         super().__init__()
#         self.input_dim = input_dim  # here, this is the latent dimension (e.g. 64)
#         self.embed_dim = embed_dim
#         self.time_embed_dim = time_embed_dim
        
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_embed_dim, embed_dim),
#             nn.SiLU(),
#             nn.Linear(embed_dim, embed_dim)
#         )
        
#         self.input_proj = nn.Linear(input_dim, embed_dim)
        
#         self.blocks = nn.ModuleList([
#             DiTBlock(dim=embed_dim, num_heads=num_heads,
#                      hidden_dim=int(embed_dim * mlp_ratio), dropout=dropout)
#             for _ in range(depth)
#         ])
        
#         self.norm = nn.LayerNorm(embed_dim)
        
#         self.output_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.SiLU(),
#             nn.Linear(embed_dim, input_dim)
#         )
        
#     def forward(self, x, t):
#         # x can be either:
#         #   - [batch, n_tokens, input_dim] (already flattened continuous tokens)
#         #   - [batch, H, W, input_dim] (spatial latent; we flatten to sequence)
#         if x.dim() == 4:
#             # [B, H, W, C] -> flatten to [B, H*W, C]
#             b, H, W, C = x.shape
#             x = x.view(b, H * W, C)
#         # Else if already [B, n_tokens, C], then do nothing.
        
#         t_emb = advanced_time_embedding(t, self.time_embed_dim)
#         t_emb = self.time_mlp(t_emb)
#         h_tokens = self.input_proj(x)
#         h_tokens = h_tokens + t_emb.unsqueeze(1)
        
#         for block in self.blocks:
#             h_tokens = block(h_tokens)
        
#         h_tokens = self.norm(h_tokens)
#         velocity = self.output_proj(h_tokens)
#         return velocity

# ##########################################
# # Continuous VAE
# ##########################################
# class ContinuousVAE(nn.Module):
#     def __init__(self, h, w, in_channels=1, hidden_dims=[32, 64, 128, 256], latent_dim=64):
#         """
#         A convolutional VAE which produces a spatial latent (of shape [B, latent_dim, H_enc, W_enc]).
#         """
#         super().__init__()
#         self.h = h
#         self.w = w
#         self.in_channels = in_channels
#         self.latent_dim = latent_dim
        
#         # Encoder
#         encoder_modules = []
#         in_ch = in_channels
#         for h_dim in hidden_dims:
#             encoder_modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU()
#                 )
#             )
#             in_ch = h_dim
#         self.encoder = nn.Sequential(*encoder_modules)
#         # Final conv layers to get mu and logvar (kept spatial)
#         self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)
#         self.fc_logvar = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)
        
#         # Decoder
#         # First, a conv to map latent_dim to the deepest hidden dim
#         self.decoder_input = nn.Conv2d(latent_dim, hidden_dims[-1], kernel_size=1)
#         decoder_modules = []
#         rev_hidden = list(reversed(hidden_dims))
#         for i in range(len(rev_hidden) - 1):
#             decoder_modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(rev_hidden[i], rev_hidden[i+1], kernel_size=4, stride=2, padding=1),
#                     nn.BatchNorm2d(rev_hidden[i+1]),
#                     nn.LeakyReLU()
#                 )
#             )
#         decoder_modules.append(
#             nn.Sequential(
#                 nn.ConvTranspose2d(rev_hidden[-1], in_channels, kernel_size=4, stride=2, padding=1),
#                 nn.Sigmoid()
#             )
#         )
#         self.decoder = nn.Sequential(*decoder_modules)
        
#         # Save encoder output shape by doing a dummy forward pass
#         with torch.no_grad():
#             dummy = torch.zeros(1, in_channels, h, w)
#             enc_out = self.encoder(dummy)
#             self.enc_out_shape = enc_out.shape[2:]  # (H_enc, W_enc)
#             # print("Encoded latent shape:", self.enc_out_shape)
        
#     def encode(self, x):
#         """
#         Encodes input x into spatial latent distribution parameters.
#         x: [B, in_channels, h, w]
#         Returns: mu, logvar (both of shape [B, latent_dim, H_enc, W_enc])
#         """
#         h_enc = self.encoder(x)
#         mu = self.fc_mu(h_enc)
#         logvar = self.fc_logvar(h_enc)
#         return mu, logvar
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def decode(self, z):
#         """
#         z: [B, latent_dim, H_enc, W_enc]
#         Returns: reconstructed x: [B, in_channels, h, w]
#         """
#         z = self.decoder_input(z)
#         x_recon = self.decoder(z)
#         return x_recon
    
#     def forward(self, x):
#         """
#         Returns: x_recon, mu, logvar, z
#         """
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         x_recon = self.decode(z)
#         return x_recon, mu, logvar, z

# ##########################################
# # Training Continuous VAE
# ##########################################
# def train_vae(images, h, w, batch_size=64, epochs=20, lr=1e-4, device='cuda'):
#     """
#     Train the continuous VAE on images.
#     Assumes images is a numpy array of shape [n_samples, h, w] (grayscale).
#     """
#     n_samples = len(images)
#     # Reshape to [n_samples, 1, h, w]
#     images_reshaped = images.reshape(-1, 1, h, w)
#     images_tensor = torch.tensor(images_reshaped, dtype=torch.float32)
    
#     dataset = torch.utils.data.TensorDataset(images_tensor)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     vae = ContinuousVAE(h, w, in_channels=1).to(device)
#     optimizer = optim.Adam(vae.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
#     losses = []
#     best_loss = float('inf')
    
#     for epoch in range(epochs):
#         vae.train()
#         epoch_loss = 0.0
#         pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{epochs}")
#         for (data,) in pbar:
#             data = data.to(device)
#             optimizer.zero_grad()
#             x_recon, mu, logvar, _ = vae(data)
#             # If the reconstructed image does not match the input size, interpolate to match
#             if x_recon.shape[-2:] != data.shape[-2:]:
#                 x_recon = F.interpolate(x_recon, size=data.shape[-2:], mode='bilinear', align_corners=False)
#             recon_loss = F.mse_loss(x_recon, data)
#             # KL divergence loss
#             kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#             loss = recon_loss + kl_loss * 1e-1  # weighting the KL term (adjust as needed)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#             pbar.set_postfix({'loss': loss.item(), 'recon': recon_loss.item(), 'kl': kl_loss.item()})
#         avg_loss = epoch_loss / len(dataloader)
#         losses.append(avg_loss)
#         print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.6f}")
#         scheduler.step(avg_loss)
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(vae.state_dict(), "continuous_vae_best.pt")
#     return vae, losses

# ##########################################
# # REFACTORED: Training Enhanced Flow Model (continuous latent space)
# ##########################################
# def train_enhanced_flow_model(data, vae_model, h, w, steps=5000, batch_size=64, lr=1e-4,
#                               checkpoint_freq=1000, embed_dim=256, num_heads=4, depth=4):
#     """
#     Train the flow matching DiT in the continuous latent space.
#     Flow Matching learns the vector field that maps noise to data.
#     Key differences in this refactored version:
#     1. Proper flow direction (noise → data)
#     2. Consistent time conditioning
#     3. Fixed velocity target (v_true = x1 - x0)
#     """
#     n_samples = len(data)
#     device = next(vae_model.parameters()).device
    
#     checkpoint_path = "checkpoints/enhanced_flow_model_latest.pt"
#     os.makedirs("checkpoints", exist_ok=True)
#     start_step = 0
    
#     # Model initialization/loading logic
#     if os.path.exists(checkpoint_path):
#         print(f"Resuming from checkpoint: {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         flow_model = EnhancedFlowMatchingDiT(
#             input_dim=vae_model.latent_dim,
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             depth=depth
#         ).to(device)
#         flow_model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-5)
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/10)
#         if 'scheduler_state_dict' in checkpoint:
#             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         start_step = checkpoint['step'] + 1
#         losses = checkpoint.get('losses', [])
#         print(f"Resuming from step {start_step}")
#     else:
#         flow_model = EnhancedFlowMatchingDiT(
#             input_dim=vae_model.latent_dim,
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             depth=depth
#         ).to(device)
#         optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-5)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/10)
#         losses = []
    
#     # Prepare dataset for VAE encoding
#     images_reshaped = data.reshape(-1, 1, h, w)
#     dataset = torch.utils.data.TensorDataset(torch.tensor(images_reshaped, dtype=torch.float32))
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     running_loss = 0.0
#     pbar = tqdm(range(start_step, steps), desc="Training Rectified Flow model")
    
#     try:
#         data_iterator = iter(dataloader)
#         for step in pbar:
#             # Get batch of data
#             try:
#                 batch_data, = next(data_iterator)
#             except StopIteration:
#                 data_iterator = iter(dataloader)
#                 batch_data, = next(data_iterator)
            
#             batch_data = batch_data.to(device)
#             B = batch_data.shape[0]  # Batch size
            
#             # Get latent x1 from VAE (target data point)
#             with torch.no_grad():
#                 mu, logvar = vae_model.encode(batch_data)
#                 x1 = vae_model.reparameterize(mu, logvar)
            
#             # Flatten spatial latent: [B, latent_dim, H_enc, W_enc] → [B, n_tokens, latent_dim]
#             B, C, H_enc, W_enc = x1.shape
#             n_tokens = H_enc * W_enc
#             x1_tokens = x1.view(B, C, n_tokens).permute(0, 2, 1)
            
#             # Sample x0 from standard normal (noise)
#             x0_tokens = torch.randn_like(x1_tokens)
            
#             # Sample t uniformly for time conditioning
#             t = torch.rand(B, device=device)
#             t_reshaped = t.view(B, 1, 1)
            
#             # Linear interpolation between noise and data
#             x_t_tokens = (1 - t_reshaped) * x0_tokens + t_reshaped * x1_tokens
            
#             # FIXED: The rectified flow target is the direction from noise to data
#             # This is a key concept in flow matching - we learn a vector field 
#             # that points from random noise to data samples
#             v_true = x1_tokens - x0_tokens
            
#             # Predict velocity at the current point
#             v_pred = flow_model(x_t_tokens, t)
            
#             # Compute loss
#             loss = F.mse_loss(v_pred, v_true)
            
#             # Optimization step
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
            
#             # Logging and checkpointing
#             running_loss += loss.item()
#             losses.append(loss.item())
#             pbar.set_postfix({'loss': loss.item()})
            
#             if (step + 1) % checkpoint_freq == 0 or step == steps - 1:
#                 checkpoint = {
#                     'step': step,
#                     'model_state_dict': flow_model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scheduler_state_dict': scheduler.state_dict(),
#                     'losses': losses,
#                     'loss': loss.item()
#                 }
#                 torch.save(checkpoint, checkpoint_path)
#                 torch.save(checkpoint, f"checkpoints/enhanced_flow_model_step_{step+1}.pt")
#                 avg_loss = running_loss / checkpoint_freq
#                 print(f"Step {step+1}, Average Loss: {avg_loss:.6f}")
#                 running_loss = 0.0
                
#     except Exception as e:
#         print(f"Exception during training: {e}")
#         torch.save({
#             'step': step if 'step' in locals() else 0,
#             'model_state_dict': flow_model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'losses': losses
#         }, "checkpoints/enhanced_flow_model_emergency.pt")
#         raise
        
#     return flow_model, losses

# ##########################################
# # REFACTORED: Generate Enhanced Samples (continuous latent space)
# ##########################################
# def generate_enhanced_samples(flow_model, vae_model, n_samples=16, h=125, w=94, steps=50, 
#                              scheduler_type='cosine'):
#     """
#     Generate samples by properly integrating the ODE from noise to data.
#     Key improvements:
#     1. Proper time integration (t=1 → t=0)
#     2. Correct ODE solving steps
#     3. Higher quality sampling process
#     """
#     flow_model.eval()
#     vae_model.eval()
#     device = next(vae_model.parameters()).device
    
#     # Create scheduler with properly ordered timesteps (t=1 → t=0)
#     scheduler = FlowMatchingScheduler(num_inference_steps=steps, scheduler_type=scheduler_type)
#     timesteps = scheduler.get_timesteps().to(device)
#     # ⚠️ IMPORTANT: We need to reverse the timesteps for generation
#     # This ensures we go from t=1 (noise) to t=0 (data)
#     timesteps = torch.flip(timesteps, [0])
    
#     # Determine latent shape from VAE
#     H_enc, W_enc = vae_model.enc_out_shape
#     latent_dim = vae_model.latent_dim
#     n_tokens = H_enc * W_enc
    
#     # Sample initial latent from standard normal distribution (pure noise at t=1)
#     x = torch.randn(n_samples, latent_dim, H_enc, W_enc, device=device)
#     # Flatten to tokens for the flow model
#     x_tokens = x.view(n_samples, latent_dim, n_tokens).permute(0, 2, 1)
    
#     # Store all states for animation
#     all_states = [x_tokens.detach().cpu()]
    
#     # Numerical ODE integration using Euler method
#     # We integrate from t=1 to t=0 following the learned vector field
#     with torch.no_grad():
#         for i in range(len(timesteps) - 1):
#             # Current and next timestep
#             t_curr = timesteps[i]
#             t_next = timesteps[i+1]
#             dt = t_next - t_curr  # Will be negative as we go from t=1 to t=0
            
#             # Get time condition as batch
#             t_batch = t_curr.expand(n_samples)
            
#             # Predict velocity at current point
#             velocity = flow_model(x_tokens, t_batch)
            
#             # Update using Euler method
#             # dx/dt = v(x,t) → Δx = v(x,t)·Δt
#             # Since dt is negative (going from t=1 to t=0), we negate it
#             x_tokens = x_tokens - velocity * dt
            
#             # Save current state for visualization
#             all_states.append(x_tokens.detach().cpu())
    
#     # Reshape final tokens back to latent spatial shape for VAE decoding
#     final_tokens = all_states[-1].to(device)
#     final_tokens = final_tokens.permute(0, 2, 1).view(n_samples, latent_dim, H_enc, W_enc)
    
#     # Decode using VAE
#     with torch.no_grad():
#         final_decoded = vae_model.decode(final_tokens)
    
#     # Convert to numpy for visualization
#     final_decoded_np = (final_decoded.detach().cpu().numpy() * 255).astype(np.uint8)
#     if final_decoded_np.shape[1] == 1:  # If grayscale
#         final_images = final_decoded_np.squeeze(1)
#     else:
#         final_images = final_decoded_np.transpose(0, 2, 3, 1)
    
#     # Generate animation frames by decoding all intermediate states
#     all_decoded_images = []
#     with torch.no_grad():
#         for state in all_states:
#             # Reshape token sequence back to spatial latent
#             state = state.to(device)
#             state_spatial = state.permute(0, 2, 1).view(n_samples, latent_dim, H_enc, W_enc)
            
#             # Decode
#             decoded = vae_model.decode(state_spatial)
#             decoded_np = (decoded.detach().cpu().numpy() * 255).astype(np.uint8)
            
#             # Format for visualization
#             if decoded_np.shape[1] == 1:
#                 imgs = decoded_np.squeeze(1)
#             else:
#                 imgs = decoded_np.transpose(0, 2, 3, 1)
                
#             all_decoded_images.append(imgs)
    
#     return final_images, all_decoded_images

# ##########################################
# # Visualization and Animation Helpers
# ##########################################
# def visualize_samples(images, save_path="generated_faces.png"):
#     n_samples = len(images)
#     rows = int(np.sqrt(n_samples))
#     cols = int(np.ceil(n_samples / rows))
    
#     plt.figure(figsize=(cols * 2, rows * 2))
#     for i, img in enumerate(images):
#         if i >= rows * cols:
#             break
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(img, cmap='gray')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     print(f"Saved visualization to {save_path}")
#     plt.close()

# def visualize_vae_reconstructions(vae_model, data, n_samples=10, h=125, w=94, save_path="vae_reconstructions.png"):
#     vae_model.eval()
#     device = next(vae_model.parameters()).device
#     indices = np.random.choice(len(data), n_samples, replace=False)
#     samples = data[indices]
#     samples_tensor = torch.tensor(samples.reshape(-1, 1, h, w), dtype=torch.float32).to(device)
#     with torch.no_grad():
#         reconstructions, _, _, _ = vae_model(samples_tensor)
#     reconstructions = reconstructions.detach().cpu().numpy() * 255
#     reconstructions = reconstructions.reshape(-1, h, w)
#     original_samples = samples.reshape(-1, h, w)
    
#     plt.figure(figsize=(2 * n_samples, 4))
#     for i in range(n_samples):
#         plt.subplot(2, n_samples, i + 1)
#         plt.imshow(original_samples[i], cmap='gray')
#         plt.title("Original")
#         plt.axis('off')
#     for i in range(n_samples):
#         plt.subplot(2, n_samples, n_samples + i + 1)
#         plt.imshow(reconstructions[i], cmap='gray')
#         plt.title("Reconstructed")
#         plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     print(f"Saved VAE reconstructions to {save_path}")
#     plt.close()

# def visualize_training(losses, save_path="training_loss.png"):
#     plt.figure(figsize=(10, 6))
#     window_size = min(100, max(5, len(losses) // 50))
#     if len(losses) > window_size:
#         smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
#         plt.plot(losses, alpha=0.3, color='blue')
#         plt.plot(np.arange(window_size-1, len(losses)), smoothed_losses, color='blue')
#     else:
#         plt.plot(losses, color='blue')
#     plt.yscale('log')
#     plt.xlabel('Training Step')
#     plt.ylabel('MSE Loss (log scale)')
#     plt.title('Flow Matching Training Loss')
#     plt.grid(alpha=0.3)
#     plt.savefig(save_path, dpi=150)
#     print(f"Saved loss curve to {save_path}")
#     plt.close()

# def create_grid_animation(all_images, save_path="face_generation_grid.gif", fps=15):
#     n_steps = len(all_images)
#     n_samples = min(16, len(all_images[0]))
#     rows = int(np.sqrt(n_samples))
#     cols = int(np.ceil(n_samples / rows))
#     os.makedirs("frames", exist_ok=True)
#     frame_paths = []
#     for t in tqdm(range(n_steps), desc="Creating animation frames"):
#         plt.figure(figsize=(cols * 2, rows * 2))
#         for i in range(n_samples):
#             plt.subplot(rows, cols, i + 1)
#             plt.imshow(all_images[t][i], cmap='gray')
#             plt.axis('off')
#         plt.suptitle(f"Generation Progress: {t/(n_steps-1)*100:.1f}%", fontsize=16)
#         plt.tight_layout()
#         frame_path = f"frames/frame_{t:04d}.png"
#         plt.savefig(frame_path, dpi=150, bbox_inches='tight')
#         frame_paths.append(frame_path)
#         plt.close()
#     frames = [Image.open(fp) for fp in frame_paths]
#     frames[0].save(save_path, save_all=True, append_images=frames[1:],
#                    optimize=True, duration=1000//fps, loop=0)
#     print(f"Saved grid animation to {save_path}")

# ##########################################
# # Main function: VAE + Enhanced Flow Matching DiT
# ##########################################
# def main_with_vae():
#     try:
#         print("Starting enhanced face image generation with Continuous VAE + Rectified Flow...")
#         images_array = X.reshape(n_samples, h, w)
#         os.makedirs("checkpoints", exist_ok=True)
        
#         vae_cache_path = "checkpoints/continuous_vae_model.pt"
#         if os.path.exists(vae_cache_path):
#             print("Loading cached Continuous VAE model...")
#             vae_model = ContinuousVAE(h, w, in_channels=1).to(device)
#             vae_model.load_state_dict(torch.load(vae_cache_path, map_location=device))
#             vae_losses = []
#         else:
#             print("Training Continuous VAE...")
#             vae_model, vae_losses = train_vae(images=images_array, h=h, w=w,
#                                               batch_size=64, epochs=15, device=device)
#             torch.save(vae_model.state_dict(), vae_cache_path)
            
#         recon_path = "vae_reconstructions.png"
#         if not os.path.exists(recon_path):
#             visualize_vae_reconstructions(vae_model=vae_model, data=images_array, h=h, w=w,
#                                           save_path=recon_path)
#         loss_path = "vae_training_loss.png"
#         if len(vae_losses) > 0 and not os.path.exists(loss_path):
#             visualize_training(vae_losses, save_path=loss_path)
        
#         print("Training rectified flow model in latent space...")
#         flow_model, flow_losses = train_enhanced_flow_model(data=images_array, vae_model=vae_model,
#                                                             h=h, w=w, steps=5000, batch_size=64)
#         visualize_training(flow_losses, save_path="rectified_flow_training_loss.png")
        
#         # Sample with different numbers of integration steps to compare quality
#         for sample_steps in [20, 50, 100]:
#             print(f"Generating samples with {sample_steps} integration steps...")
#             final_images, all_images = generate_enhanced_samples(
#                 flow_model=flow_model, 
#                 vae_model=vae_model,
#                 n_samples=36, 
#                 h=h, 
#                 w=w, 
#                 steps=sample_steps
#             )
#             visualize_samples(final_images, save_path=f"rectified_flow_faces_{sample_steps}steps.png")
            
#             # Only create animation for the medium step count to save time
#             if sample_steps == 50:
#                 create_grid_animation(all_images, save_path="rectified_flow_generation.gif")
        
#         print("Continuous VAE + Rectified Flow experiment completed successfully!")
#     except Exception as e:
#         print(f"Error in main execution: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main_with_vae()

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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

##########################################
# Load LFW Dataset
##########################################
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

##########################################
# Improved Flow Matching Scheduler
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
# Enhanced Flow Matching DiT (for vector latents)
##########################################
class EnhancedFlowMatchingDiT(nn.Module):
    def __init__(self, input_dim, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0,
                 dropout=0.1, time_embed_dim=512):
        """
        input_dim: dimension of the vector latent (e.g. latent_dim)
        """
        super().__init__()
        self.input_dim = input_dim
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
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim=embed_dim, num_heads=num_heads,
                     hidden_dim=int(embed_dim * mlp_ratio), dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project back to input dimension with skip connection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, input_dim)
        )
        
    def forward(self, x, t):
        # x is [batch, input_dim]
        batch_size = x.shape[0]
        
        # Embed time
        t_emb = advanced_time_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Project input to embedding space
        h = self.input_proj(x)
        
        # Add time embedding
        h = h + t_emb
        
        # Add positional information by reshaping to sequence
        # This helps the transformer treat the vector as a sequence
        h = h.unsqueeze(1)  # [B, 1, embed_dim]
        
        # Process through transformer blocks
        for block in self.blocks:
            h = block(h)
        
        h = self.norm(h)
        h = h.squeeze(1)  # Remove sequence dimension
        
        # Project back to input dimension
        velocity = self.output_proj(h)
        
        return velocity

##########################################
# Vector VAE (simplified architecture)
##########################################
# Modify the ImprovedVectorVAE class's decoder to ensure correct output dimensions
class ImprovedVectorVAE(nn.Module):
    def __init__(self, h, w, in_channels=1, hidden_dims=[32, 64, 128, 256, 512], latent_dim=128):
        """
        A convolutional VAE which produces a vector latent (of shape [B, latent_dim]).
        """
        super().__init__()
        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_modules = []
        in_ch = in_channels
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_ch = h_dim
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Calculate output dimensions of encoder
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            enc_out = self.encoder(dummy)
            self.enc_out_shape = enc_out.shape
            self.flatten_dim = np.prod(enc_out.shape[1:]).item()
        
        # FC layers for mu and logvar (vector latent)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder input layer (from latent to flattened conv features)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            nn.LeakyReLU()
        )
        
        # Reshape to match encoder output shape
        self.unflatten = lambda x: x.view(-1, *self.enc_out_shape[1:])
        
        # Decoder (transposed convolutions)
        decoder_modules = []
        hidden_dims = list(reversed(hidden_dims))
        for i in range(len(hidden_dims)-1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final layer with exact output size calculation
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], in_channels,
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*decoder_modules)
        
    def encode(self, x):
        """Encodes input x into latent distribution parameters."""
        h_enc = self.encoder(x)
        h_flat = torch.flatten(h_enc, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterize with improved numerical stability."""
        std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=10))
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decodes latent z to reconstructed image with exact dimensions."""
        h = self.decoder_input(z)
        h = self.unflatten(h)
        x_recon_raw = self.decoder(h)
        
        # Resize to exact input dimensions if needed
        if x_recon_raw.shape[-2:] != (self.h, self.w):
            x_recon = F.interpolate(x_recon_raw, size=(self.h, self.w), mode='bilinear', align_corners=False)
        else:
            x_recon = x_recon_raw
            
        return x_recon
    
    def forward(self, x):
        """Returns: x_recon, mu, logvar, z"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def get_latent_stats(self, dataloader, device):
        """Calculate statistics of the latent space to aid in generation."""
        means, stds = [], []
        with torch.no_grad():
            for (data,) in dataloader:
                data = data.to(device)
                mu, logvar = self.encode(data)
                var = torch.exp(logvar)
                means.append(mu)
                stds.append(torch.sqrt(var))
                
        all_means = torch.cat(means, dim=0)
        all_stds = torch.cat(stds, dim=0)
        
        # Calculate average
        mean_latent = torch.mean(all_means, dim=0)
        std_latent = torch.mean(all_stds, dim=0)
        
        return mean_latent, std_latent

##########################################
# Training Improved Vector VAE
##########################################
def train_improved_vae(images, h, w, batch_size=64, epochs=30, lr=2e-4, device='cuda', 
                       kl_weight=0.01, latent_dim=128):
    """
    Train the improved vector VAE on images.
    Assumes images is a numpy array of shape [n_samples, h, w] (grayscale).
    """
    n_samples = len(images)
    # Reshape to [n_samples, 1, h, w]
    images_reshaped = images.reshape(-1, 1, h, w)
    images_tensor = torch.tensor(images_reshaped, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(images_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    vae = ImprovedVectorVAE(h, w, in_channels=1, latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    
    losses = []
    kl_losses = []
    recon_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{epochs}")
        for (data,) in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = vae(data)
            
            # Reconstruction loss with improved stability
            recon_loss = F.mse_loss(x_recon, data)
            
            # KL divergence loss with improved numerical stability
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Annealed KL weight (increase over time)
            annealed_kl_weight = min(1.0, epoch / (epochs * 0.8)) * kl_weight
            
            # Total loss
            loss = recon_loss + annealed_kl_weight * kl_loss
            
            loss.backward()
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(), 
                'recon': recon_loss.item(), 
                'kl': kl_loss.item(),
                'kl_weight': annealed_kl_weight
            })
            
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        print(f"Epoch {epoch+1}: Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
        
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vae.state_dict(), "improved_vector_vae_best.pt")
            print(f"Saved new best model with loss: {best_loss:.6f}")
    
    # Save final model
    torch.save(vae.state_dict(), "improved_vector_vae_final.pt")
    
    # Plot training losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Total Loss')
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(kl_losses, label='KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('KL Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('vae_training_losses.png', dpi=150)
    plt.close()
    
    return vae, losses

##########################################
# IMPROVED: RK4 ODE Solver for Generation
##########################################
def rk4_step(model, x, t_curr, t_next, device):
    """
    Runge-Kutta 4th order integration step.
    model: the flow model predicting velocity
    x: current state [B, latent_dim]
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
# IMPROVED: Flow Model Training with Better Regularization
##########################################
def train_improved_flow_model(data, vae_model, h, w, steps=20000, batch_size=64, lr=2e-4,
                              checkpoint_freq=1000, embed_dim=512, num_heads=8, depth=8,
                              latent_dim=128):
    """
    Train the improved flow matching model with vector latents.
    """
    n_samples = len(data)
    device = next(vae_model.parameters()).device
    
    # Setup model save directory
    checkpoint_path = "checkpoints/improved_flow_model_latest.pt"
    os.makedirs("checkpoints", exist_ok=True)
    start_step = 0
    
    # Prepare dataset for VAE encoding
    images_reshaped = data.reshape(-1, 1, h, w)
    image_dataset = torch.utils.data.TensorDataset(torch.tensor(images_reshaped, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate latent space statistics to help with training
    latent_mean, latent_std = vae_model.get_latent_stats(dataloader, device)
    print(f"Latent space statistics: Mean norm: {torch.norm(latent_mean):.4f}, Std mean: {torch.mean(latent_std):.4f}")
    
    # Model initialization
    flow_model = EnhancedFlowMatchingDiT(
        input_dim=latent_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth
    ).to(device)
    
    # Check if we're resuming from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        flow_model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint['step'] + 1
        losses = checkpoint.get('losses', [])
        print(f"Resuming from step {start_step}")
        optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=lr/20,
            last_epoch=start_step-1 if start_step > 0 else -1
        )
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/20)
        losses = []
    
    running_loss = 0.0
    velocity_norms = []
    mse_losses = []
    reg_losses = []
    
    pbar = tqdm(range(start_step, steps), desc="Training Improved Flow model")
    
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
            
            # Get vector latent x1 from VAE (target data point)
            with torch.no_grad():
                mu, logvar = vae_model.encode(batch_data)
                x1 = vae_model.reparameterize(mu, logvar)
            
            # Sample t uniformly for time conditioning - use more points near 0
            # This helps with better generation quality at the end of the trajectory
            t_raw = torch.rand(B, device=device)
            t = t_raw ** 1.5  # More samples near t=0
            
            # Sample x0 from normal distribution matching the latent space statistics
            x0 = torch.randn_like(x1) * latent_std.unsqueeze(0) + latent_mean.unsqueeze(0)
            
            # Linear interpolation for training
            x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
            
            # Target velocity is from noise to data
            v_true = x1 - x0
            
            # Predict velocity
            v_pred = flow_model(x_t, t)
            
            # Compute loss with regularization
            mse_loss = F.mse_loss(v_pred, v_true)
            
            # Regularization: penalize large velocity norms
            velocity_norm = torch.mean(torch.norm(v_pred, dim=1))
            
            # Compute smoothness regularization
            if step % 10 == 0 and step > 0:
                # Sample new time points close to original ones
                t_perturbed = torch.clamp(t + 0.01 * torch.randn_like(t), min=0, max=1)
                
                # Compute interpolated points at perturbed times
                x_t_perturbed = (1 - t_perturbed.unsqueeze(1)) * x0 + t_perturbed.unsqueeze(1) * x1
                
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
                torch.save(checkpoint, f"checkpoints/improved_flow_model_step_{step+1}.pt")
                
                # Calculate and report statistics
                avg_loss = running_loss / checkpoint_freq
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
                    plt.savefig(f"checkpoints/training_progress_step_{step+1}.png", dpi=150)
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
        }, "checkpoints/improved_flow_model_emergency.pt")
        raise
        
    return flow_model, losses, mse_losses, velocity_norms

##########################################
# Visualization and Animation Helpers
##########################################
def visualize_samples(images, save_path="generated_faces.png"):
    n_samples = len(images)
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))
    
    plt.figure(figsize=(cols * 2, rows * 2))
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()

def visualize_vae_reconstructions(vae_model, data, n_samples=10, h=125, w=94, save_path="vae_reconstructions.png"):
    vae_model.eval()
    device = next(vae_model.parameters()).device
    indices = np.random.choice(len(data), n_samples, replace=False)
    samples = data[indices]
    samples_tensor = torch.tensor(samples.reshape(-1, 1, h, w), dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructions, _, _, _ = vae_model(samples_tensor)
    reconstructions = reconstructions.detach().cpu().numpy() * 255
    reconstructions = reconstructions.reshape(-1, h, w)
    original_samples = samples.reshape(-1, h, w)
    
    plt.figure(figsize=(2 * n_samples, 4))
    for i in range(n_samples):
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(original_samples[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')
    for i in range(n_samples):
        plt.subplot(2, n_samples, n_samples + i + 1)
        plt.imshow(reconstructions[i], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved VAE reconstructions to {save_path}")
    plt.close()

def visualize_training(losses, save_path="training_loss.png", window_size=100):
    plt.figure(figsize=(12, 8))
    
    # Total loss
    plt.subplot(2, 2, 1)
    plt.plot(losses, alpha=0.3, color='blue', label='Raw')
    if len(losses) > window_size:
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(losses)), smoothed_losses, color='blue', label='Smoothed')
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.title('Flow Matching Training Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Plot last 20% of training
    last_fifth = max(1, len(losses) // 5)
    plt.subplot(2, 2, 2)
    plt.plot(losses[-last_fifth:], color='green')
    plt.xlabel('Last Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Final {last_fifth} Steps')
    plt.grid(alpha=0.3)
    
    # Learning rate if available
    if len(losses) > 1000:
        checkpoint = torch.load("checkpoints/improved_flow_model_latest.pt", map_location='cpu')
        if 'scheduler_state_dict' in checkpoint:
            plt.subplot(2, 2, 3)
            # Extract LR from scheduler state
            lr_lambda = lambda step: max(0.05, 0.5 * (1 + np.cos(step / 20000 * np.pi)))
            lr_schedule = [lr_lambda(i) * 2e-4 for i in range(len(losses))]
            plt.plot(lr_schedule)
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved loss curve to {save_path}")
    plt.close()

def create_grid_animation(all_images, save_path="face_generation_grid.gif", fps=15, dpi=150):
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
            plt.imshow(all_images[t][i], cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Generation Progress: {t/(n_steps-1)*100:.1f}%", fontsize=16)
        plt.tight_layout()
        frame_path = f"frames/frame_{t:04d}.png"
        plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        frame_paths.append(frame_path)
        plt.close()
    
    # Load frames with PIL and optimize
    frames = [Image.open(fp) for fp in frame_paths]
    # Resize to reduce file size if needed
    if dpi > 100:
        size = frames[0].size
        new_size = (size[0] // 2, size[1] // 2)
        frames = [f.resize(new_size, Image.LANCZOS) for f in frames]
        
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   optimize=True, duration=1000//fps, loop=0)
    print(f"Saved grid animation to {save_path}")
    
    # Optional: Remove frame files to save space
    for fp in frame_paths:
        os.remove(fp)

##########################################
# IMPROVED: Generate Samples with RK4 Integration
##########################################
def generate_improved_samples(flow_model, vae_model, n_samples=36, h=125, w=94, steps=100, 
                             scheduler_type='cosine'):
    """
    Generate samples using RK4 integration for improved accuracy.
    """
    flow_model.eval()
    vae_model.eval()
    device = next(vae_model.parameters()).device
    
    # Create scheduler with properly ordered timesteps (t=1 → t=0)
    scheduler = FlowMatchingScheduler(num_inference_steps=steps, scheduler_type=scheduler_type)
    timesteps = scheduler.get_timesteps().to(device)
    # Reverse timesteps for generation (from t=1 to t=0)
    timesteps = torch.flip(timesteps, [0])
    
    # Get latent statistics for better initial sampling
    # Prepare a small dataloader to calculate statistics
    sample_size = min(200, n_samples * 4)  # Use more samples for better statistics
    indices = np.random.choice(len(X), sample_size, replace=False)
    sample_data = X[indices].reshape(-1, 1, h, w)
    sample_dataset = torch.utils.data.TensorDataset(torch.tensor(sample_data, dtype=torch.float32))
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=64, shuffle=False)
    
    # Get latent statistics
    latent_mean, latent_std = vae_model.get_latent_stats(sample_loader, device)
    
    # Sample initial latent from distribution matching encoded data
    x = torch.randn(n_samples, vae_model.latent_dim, device=device) * latent_std.unsqueeze(0) + latent_mean.unsqueeze(0)
    
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
            
            # Normalize if using a normalizing flow
            if i % 10 == 0 or i == len(timesteps) - 2:
                # Optional: apply mild noise regularization to prevent overshooting
                if i < len(timesteps) - 2:
                    noise_scale = 0.01 * (1.0 - t_next.item()) 
                    x = x + torch.randn_like(x) * noise_scale
            
            # Save current state for visualization
            all_states.append(x.detach().cpu())
            
            # Log progress
            if i % 20 == 0 or i == len(timesteps) - 2:
                print(f"Generation progress: {i+1}/{len(timesteps)-1} steps, t={t_next.item():.3f}")
    
    # Decode final latents
    with torch.no_grad():
        final_decoded = vae_model.decode(all_states[-1].to(device))
    
    # Convert to numpy for visualization
    final_decoded_np = (final_decoded.detach().cpu().numpy() * 255).astype(np.uint8)
    if final_decoded_np.shape[1] == 1:  # If grayscale
        final_images = final_decoded_np.squeeze(1)
    else:
        final_images = final_decoded_np.transpose(0, 2, 3, 1)
    
    # Generate animation frames by decoding all intermediate states
    all_decoded_images = []
    with torch.no_grad():
        for i, state in enumerate(all_states):
            # Skip some frames for efficiency
            if len(all_states) > 60 and i % (len(all_states) // 60) != 0 and i != len(all_states) - 1:
                continue
                
            decoded = vae_model.decode(state.to(device))
            decoded_np = (decoded.detach().cpu().numpy() * 255).astype(np.uint8)
            
            # Format for visualization
            if decoded_np.shape[1] == 1:
                imgs = decoded_np.squeeze(1)
            else:
                imgs = decoded_np.transpose(0, 2, 3, 1)
                
            all_decoded_images.append(imgs)
    
    return final_images, all_decoded_images

##########################################
# Main function: Improved Vector VAE + Flow Matching
##########################################
def main_improved():
    try:
        print("Starting improved face image generation with Vector VAE + Flow Matching...")
        images_array = X.reshape(n_samples, h, w)
        os.makedirs("checkpoints", exist_ok=True)
        
        # Hyperparameters
        latent_dim = 128
        vae_epochs = 30
        flow_steps = 20000
        batch_size = 64
        
        vae_cache_path = "checkpoints/improved_vector_vae_best.pt"
        if os.path.exists(vae_cache_path):
            print("Loading cached Vector VAE model...")
            vae_model = ImprovedVectorVAE(h, w, in_channels=1, latent_dim=latent_dim).to(device)
            vae_model.load_state_dict(torch.load(vae_cache_path, map_location=device))
            vae_losses = []
        else:
            print(f"Training Vector VAE for {vae_epochs} epochs...")
            vae_model, vae_losses = train_improved_vae(
                images=images_array, 
                h=h, 
                w=w,
                batch_size=batch_size, 
                epochs=vae_epochs, 
                device=device,
                kl_weight=0.01,  # Higher KL weight for better latent structure
                latent_dim=latent_dim
            )
        
        # Visualize VAE results
        print("Generating VAE reconstructions...")
        recon_path = "improved_vae_reconstructions.png"
        visualize_vae_reconstructions(
            vae_model=vae_model, 
            data=images_array, 
            h=h, 
            w=w,
            n_samples=10,
            save_path=recon_path
        )
        
        if len(vae_losses) > 0:
            print("Plotting VAE training curves...")
            visualize_training(vae_losses, save_path="improved_vae_training_loss.png")
        
        # Prepare dataset for analysis
        images_reshaped = images_array.reshape(-1, 1, h, w)
        dataset = torch.utils.data.TensorDataset(torch.tensor(images_reshaped, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Get latent space statistics
        print("Analyzing latent space statistics...")
        latent_mean, latent_std = vae_model.get_latent_stats(dataloader, device)
        print(f"Latent space: Mean norm: {torch.norm(latent_mean):.4f}, Std mean: {torch.mean(latent_std):.4f}")
        
        print(f"Training improved flow model for {flow_steps} steps...")
        flow_model, flow_losses, mse_losses, vel_norms = train_improved_flow_model(
            data=images_array, 
            vae_model=vae_model,
            h=h, 
            w=w, 
            steps=flow_steps, 
            batch_size=batch_size,
            embed_dim=512,  # Larger model
            num_heads=8,    # More attention heads
            depth=8,        # Deeper network
            latent_dim=latent_dim
        )
        
        # Visualize training results
        print("Plotting flow training curves...")
        visualize_training(
            flow_losses, 
            save_path="improved_flow_training_loss.png",
            window_size=200
        )
        
        # Generate samples with different integration steps
        for sample_steps in [50, 100, 200]:
            print(f"Generating samples with {sample_steps} integration steps...")
            final_images, all_images = generate_improved_samples(
                flow_model=flow_model, 
                vae_model=vae_model,
                n_samples=36, 
                h=h, 
                w=w, 
                steps=sample_steps
            )
            visualize_samples(
                final_images, 
                save_path=f"improved_flow_faces_{sample_steps}steps.png"
            )
            
            # Only create animation for the medium step count to save time
            if sample_steps == 100:
                create_grid_animation(
                    all_images, 
                    save_path="improved_flow_generation.gif",
                    fps=20,
                    dpi=200
                )
        
        print("Improved Vector VAE + Flow Matching experiment completed successfully!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_improved()