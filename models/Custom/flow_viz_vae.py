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

# # Load LFW dataset with exact dimensions
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

# # Scale the data (for the flow model on raw pixels)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ##########################################
# # Flow Matching Scheduler
# ##########################################
# class FlowMatchingScheduler:
#     """Scheduler for Flow Matching process."""
#     def __init__(
#         self,
#         num_inference_steps=50,
#         scheduler_type='cosine',
#         min_t=0.002,
#         max_t=0.998
#     ):
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
#             alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
#             timesteps = torch.clamp(alpha_cumprod, self.min_t, self.max_t)
#             return timesteps[:-1]  # Remove the last timestep
            
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
#         return self.timesteps

# ##########################################
# # Attention & Feed-Forward Blocks
# ##########################################
# class AttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, dim_head=64, dropout=0.0):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.attention = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
        
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
# # Flow Matching DiT Network (Original)
# ##########################################
# class FlowMatchingDiT(nn.Module):
#     def __init__(
#         self, 
#         input_dim,  
#         embed_dim=512, 
#         depth=6,
#         num_heads=8,
#         mlp_ratio=4.0,
#         dropout=0.1,
#         time_embed_dim=256
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.embed_dim = embed_dim
#         self.time_embed_dim = time_embed_dim
        
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_embed_dim, embed_dim),
#             nn.SiLU(),
#             nn.Linear(embed_dim, embed_dim)
#         )
        
#         self.input_proj = nn.Linear(input_dim, embed_dim)
        
#         self.blocks = nn.ModuleList([
#             DiTBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 hidden_dim=int(embed_dim * mlp_ratio),
#                 dropout=dropout
#             )
#             for _ in range(depth)
#         ])
        
#         self.norm = nn.LayerNorm(embed_dim)
        
#         self.output_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.SiLU(),
#             nn.Linear(embed_dim, input_dim)
#         )
        
#     def forward(self, x, t):
#         # Here x is assumed to have shape [batch, seq_len, input_dim]
#         t_emb = advanced_time_embedding(t, self.time_embed_dim)
#         t_emb = self.time_mlp(t_emb)
        
#         h = self.input_proj(x)
#         h = h + t_emb.unsqueeze(1)
        
#         for block in self.blocks:
#             h = block(h)
        
#         h = self.norm(h)
#         velocity = self.output_proj(h)
#         return velocity

# ##########################################
# # VQ-VAE Components
# ##########################################
# class VectorQuantizer(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
#         super(VectorQuantizer, self).__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.commitment_cost = commitment_cost
        
#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)
#         self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
#     def forward(self, inputs):
#         # inputs: [B, C, H, W] -> convert to BHWC
#         inputs = inputs.permute(0, 2, 3, 1).contiguous()
#         input_shape = inputs.shape  # [B, H, W, C]
        
#         flat_input = inputs.view(-1, self.embedding_dim)
        
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
#                      + torch.sum(self.embedding.weight**2, dim=1)
#                      - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
        
#         quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
#         quantized = inputs + (quantized - inputs).detach()
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
#         return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1),
#                 nn.BatchNorm2d(out_channels)
#             )
            
#     def forward(self, x):
#         residual = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += self.shortcut(residual)
#         x = F.relu(x)
#         return x

# class VQVAE(nn.Module):
#     def __init__(self, h=62, w=47, in_channels=1, hidden_dims=[16, 32, 64], 
#                  embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
#         super(VQVAE, self).__init__()
#         self.h = h
#         self.w = w
#         self.in_channels = in_channels
#         self.embedding_dim = embedding_dim
        
#         encoder_layers = len(hidden_dims)
#         self.h_encoded = h // (2 ** encoder_layers)
#         self.w_encoded = w // (2 ** encoder_layers)
        
#         self.h_padded = self.h_encoded * (2 ** encoder_layers)
#         self.w_padded = self.w_encoded * (2 ** encoder_layers)
        
#         encoder_modules = []
#         in_ch = in_channels
#         for h_dim in hidden_dims:
#             encoder_modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU(),
#                     ResidualBlock(h_dim, h_dim)
#                 )
#             )
#             in_ch = h_dim
            
#         encoder_modules.append(
#             nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1)
#         )
        
#         self.encoder = nn.Sequential(*encoder_modules)
#         self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
#         decoder_modules = []
#         decoder_modules.append(
#             nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=1)
#         )
        
#         for i in range(len(hidden_dims)-1, 0, -1):
#             decoder_modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 
#                                       kernel_size=4, stride=2, padding=1),
#                     nn.BatchNorm2d(hidden_dims[i-1]),
#                     nn.LeakyReLU(),
#                     ResidualBlock(hidden_dims[i-1], hidden_dims[i-1])
#                 )
#             )
            
#         decoder_modules.append(
#             nn.Sequential(
#                 nn.ConvTranspose2d(hidden_dims[0], in_channels, 
#                                   kernel_size=4, stride=2, padding=1),
#                 nn.Sigmoid()
#             )
#         )
        
#         self.decoder = nn.Sequential(*decoder_modules)
        
#     def encode(self, x):
#         x = x.view(-1, self.in_channels, self.h, self.w)
#         encoded = self.encoder(x)
#         return encoded
        
#     def decode(self, z):
#         reconstructed = self.decoder(z)
#         return reconstructed
    
#     def encode_to_indices(self, x):
#         """Encode input to discrete indices with grid shape [batch, h_encoded, w_encoded]"""
#         x = x.view(-1, self.in_channels, self.h, self.w)
#         encoded = self.encoder(x)
#         _, _, _, _, indices = self.vq(encoded)
#         batch_size = x.shape[0]
#         indices = indices.view(batch_size, self.h_encoded, self.w_encoded)
#         return indices
    
#     def decode_from_indices(self, indices, batch_size):
#         """
#         Decode from discrete indices.
#         indices: tensor of shape [batch_size, h_encoded, w_encoded]
#         """
#         # Flatten spatial dimensions: [batch_size, h_encoded*w_encoded]
#         flat_indices = indices.view(batch_size, -1)
#         one_hot = F.one_hot(flat_indices, num_classes=self.vq.num_embeddings).float()
#         quantized = torch.matmul(one_hot, self.vq.embedding.weight)
#         quantized = quantized.view(batch_size, self.h_encoded, self.w_encoded, self.embedding_dim)
#         quantized = quantized.permute(0, 3, 1, 2).contiguous()
#         return self.decode(quantized)
        
#     def forward(self, x):
#         x = x.view(-1, self.in_channels, self.h, self.w)
#         encoded = self.encoder(x)
#         vq_loss, quantized, perplexity, _, _ = self.vq(encoded)
#         reconstructed = self.decoder(quantized)
#         if reconstructed.shape[2:] != x.shape[2:]:
#             reconstructed = reconstructed[:, :, :x.shape[2], :x.shape[3]]
#         return reconstructed, vq_loss, perplexity

# ##########################################
# # Enhanced Flow Matching DiT with VQVAE Integration
# ##########################################
# class EnhancedFlowMatchingDiT(nn.Module):
#     def __init__(
#         self, 
#         input_dim,
#         vae_model=None,
#         use_vae_latent=True,
#         embed_dim=256, 
#         depth=4,
#         num_heads=4,
#         mlp_ratio=4.0,
#         dropout=0.1,
#         time_embed_dim=256
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.embed_dim = embed_dim
#         self.time_embed_dim = time_embed_dim
#         self.use_vae_latent = use_vae_latent
#         self.vae_model = vae_model
        
#         if use_vae_latent and vae_model is not None:
#             self.working_dim = vae_model.vq.num_embeddings  # e.g. 512
#         else:
#             self.working_dim = input_dim
        
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_embed_dim, embed_dim),
#             nn.SiLU(),
#             nn.Linear(embed_dim, embed_dim)
#         )
        
#         self.input_proj = nn.Linear(self.working_dim, embed_dim)
        
#         self.blocks = nn.ModuleList([
#             DiTBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 hidden_dim=int(embed_dim * mlp_ratio),
#                 dropout=dropout
#             )
#             for _ in range(depth)
#         ])
        
#         self.norm = nn.LayerNorm(embed_dim)
        
#         self.output_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.SiLU(),
#             nn.Linear(embed_dim, self.working_dim)
#         )
        
#     def forward(self, x, t):
#         # x can be:
#         #   - [batch, H, W]  (integer indices -> we do one-hot)
#         #   - [batch, H, W, working_dim] (already one-hot, flatten to [batch, H*W, working_dim])
#         #   - [batch, n_tokens, working_dim] (already flattened, do nothing)

#         if x.dim() == 3:
#             # Case: [batch, n_tokens, working_dim] or [batch, H, W]
#             if x.shape[-1] == self.working_dim:
#                 # Already [batch, n_tokens, working_dim]; do nothing
#                 pass
#             else:
#                 # Then it must be integer indices [batch, H, W], so one-hot them:
#                 if x.dtype in [torch.int64, torch.int32]:
#                     x = F.one_hot(x.long(), num_classes=self.working_dim).float()
#                     # Now x is [batch, H, W, working_dim], so flatten:
#                     b, H, W, C = x.shape
#                     x = x.view(b, H*W, C)

#         elif x.dim() == 4:
#             # [batch, H, W, working_dim] -> flatten
#             b, H, W, C = x.shape
#             x = x.view(b, H*W, C)

#         # else if x.dim() == 2, it might be [batch, working_dim], or
#         # x.dim() == 1 is invalid, etc.

#         # Now x is always [batch, n_tokens, working_dim].
#         # Next, do your time-embedding and DiT logic:
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
# # Linear Interpolation for Flow Matching
# ##########################################
# def linear_interpolation(x_0, x_1, t):
#     """Interpolate between x_0 and x_1 at time t (broadcasting over tokens)"""
#     return (1 - t) * x_0 + t * x_1

# ##########################################
# # Training VQ-VAE
# ##########################################
# def train_vqvae(images, h, w, batch_size=64, epochs=20, lr=1e-4, device='cuda'):
#     n_samples = len(images)
#     h_padded = ((h + 7) // 8) * 8
#     w_padded = ((w + 7) // 8) * 8
    
#     vqvae = VQVAE(h=h_padded, w=w_padded, in_channels=1).to(device)
    
#     images_reshaped = images.reshape(-1, 1, h, w)
#     images_tensor = torch.tensor(images_reshaped, dtype=torch.float32) #/ 255.0
    
#     dataset = torch.utils.data.TensorDataset(images_tensor)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=True
#     )
    
#     optimizer = optim.Adam(vqvae.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, verbose=True
#     )
    
#     best_loss = float('inf')
#     losses = []
    
#     for epoch in range(epochs):
#         vqvae.train()
#         epoch_loss = 0
#         reconstruction_loss = 0
#         vq_loss_sum = 0
        
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
#         for i, (data,) in enumerate(pbar):
#             data = data.to(device)
            
#             if h != h_padded or w != w_padded:
#                 padded_data = torch.zeros(data.shape[0], 1, h_padded, w_padded, device=device)
#                 padded_data[:, :, :h, :w] = data
#                 data_input = padded_data
#             else:
#                 data_input = data
                
#             reconstructed, vq_loss, perplexity = vqvae(data_input)
            
#             if h != h_padded or w != w_padded:
#                 reconstructed = reconstructed[:, :, :h, :w]
                
#             recon_loss = F.mse_loss(reconstructed, data)
#             loss = recon_loss + vq_loss
            
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
            
#             epoch_loss += loss.item()
#             reconstruction_loss += recon_loss.item()
#             vq_loss_sum += vq_loss.item()
            
#             pbar.set_postfix({
#                 'loss': loss.item(), 
#                 'recon_loss': recon_loss.item(),
#                 'vq_loss': vq_loss.item(),
#                 'perplexity': perplexity.item()
#             })
        
#         avg_loss = epoch_loss / len(dataloader)
#         avg_recon_loss = reconstruction_loss / len(dataloader)
#         avg_vq_loss = vq_loss_sum / len(dataloader)
#         losses.append(avg_loss)
        
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, "
#               f"Recon Loss: {avg_recon_loss:.6f}, VQ Loss: {avg_vq_loss:.6f}")
        
#         scheduler.step(avg_loss)
        
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(vqvae.state_dict(), "vqvae_best.pt")
    
#     return vqvae, losses

# ##########################################
# # Training Enhanced Flow Model with VQ-VAE Integration
# ##########################################
# def train_enhanced_flow_model(data, vae_model, h, w, steps=5000, batch_size=64, lr=1e-4, 
#                              checkpoint_freq=1000, embed_dim=256, num_heads=4, depth=4):
#     n_samples = len(data)
#     device = next(vae_model.parameters()).device
    
#     os.makedirs("checkpoints", exist_ok=True)
#     indices_cache_path = "checkpoints/vqvae_indices_cache.pt"
#     if os.path.exists(indices_cache_path):
#         print("Loading cached VQ-VAE indices...")
#         all_indices = torch.load(indices_cache_path)
#     else:
#         print("Processing data through VQ-VAE encoder...")
#         indices_list = []
#         vae_model.eval()
#         with torch.no_grad():
#             for i in range(0, n_samples, batch_size):
#                 batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
#                 batch = batch.reshape(-1, 1, h, w) #/ 255.0
                
#                 h_padded = ((h + 7) // 8) * 8
#                 w_padded = ((w + 7) // 8) * 8
                
#                 if h != h_padded or w != w_padded:
#                     padded_batch = torch.zeros(batch.shape[0], 1, h_padded, w_padded, device=device)
#                     padded_batch[:, :, :h, :w] = batch
#                     batch_input = padded_batch
#                 else:
#                     batch_input = batch
                    
#                 encoded = vae_model.encode(batch_input)
#                 _, _, _, _, indices = vae_model.vq(encoded)
#                 indices = indices.view(batch.shape[0], vae_model.h_encoded, vae_model.w_encoded)
#                 indices_list.append(indices.cpu())
        
#         all_indices = torch.cat(indices_list, dim=0)
#         torch.save(all_indices, indices_cache_path)
    
#     input_dim = vae_model.vq.num_embeddings
#     print(f"Training with VQ indices, input dimension: {input_dim}")
    
#     checkpoint_path = "checkpoints/enhanced_flow_model_latest.pt"
#     start_step = 0
#     if os.path.exists(checkpoint_path):
#         print(f"Resuming from checkpoint: {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path)
#         model = EnhancedFlowMatchingDiT(
#             input_dim=data.shape[1],
#             vae_model=vae_model,
#             use_vae_latent=True,
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             depth=depth
#         ).to(device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/10)
#         if 'scheduler_state_dict' in checkpoint:
#             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         start_step = checkpoint['step'] + 1
#         losses = checkpoint.get('losses', [])
#         print(f"Resuming from step {start_step}")
#     else:
#         model = EnhancedFlowMatchingDiT(
#             input_dim=data.shape[1],
#             vae_model=vae_model,
#             use_vae_latent=True,
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             depth=depth
#         ).to(device)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/10)
#         losses = []
    
#     # Ensure we have indices shaped as [n_samples, h_encoded, w_encoded]
#     if all_indices.dim() == 2:
#         all_indices = all_indices.view(n_samples, vae_model.h_encoded, vae_model.w_encoded)
    
#     running_loss = 0.0
#     pbar = tqdm(range(start_step, steps), desc="Training enhanced flow model")
    
#     try:
#         for step in pbar:
#             optimizer.zero_grad()
            
#             batch_indices = torch.randint(0, n_samples, (batch_size,))
#             x_1_indices = all_indices[batch_indices].to(device)  # shape: [batch, h_encoded, w_encoded]
            
#             x_0_indices = torch.randint(
#                 0, vae_model.vq.num_embeddings, 
#                 size=(batch_size, vae_model.h_encoded, vae_model.w_encoded), 
#                 device=device
#             )
            
#             t = torch.rand(batch_size, device=device)
            
#             # One-hot encode and then flatten spatial dims so each sample becomes a token–sequence.
#             x_1_oh = F.one_hot(x_1_indices.long(), num_classes=input_dim).float()  # [batch, H, W, input_dim]
#             x_0_oh = F.one_hot(x_0_indices.long(), num_classes=input_dim).float()  # [batch, H, W, input_dim]
#             x_1_oh = x_1_oh.view(batch_size, -1, input_dim)  # [batch, n_tokens, input_dim]
#             x_0_oh = x_0_oh.view(batch_size, -1, input_dim)
            
#             t_reshaped = t.view(batch_size, 1, 1)
#             x_t_oh = linear_interpolation(x_0_oh, x_1_oh, t_reshaped)  # [batch, n_tokens, input_dim]
            
#             v_pred = model(x_t_oh, t)  # [batch, n_tokens, input_dim]
            
#             v_t = x_1_oh - x_0_oh  # [batch, n_tokens, input_dim]
            
#             loss = F.mse_loss(v_pred, v_t)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
            
#             running_loss += loss.item()
#             losses.append(loss.item())
            
#             if (step + 1) % checkpoint_freq == 0 or step == steps - 1:
#                 checkpoint = {
#                     'step': step,
#                     'model_state_dict': model.state_dict(),
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
#         if 'model' in locals():
#             torch.save({
#                 'step': step if 'step' in locals() else 0,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
#                 'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
#                 'losses': losses
#             }, "checkpoints/enhanced_flow_model_emergency.pt")
#         raise
    
#     return model, losses

# ##########################################
# # Generate Enhanced Samples using Flow Model + VQVAE
# ##########################################
# def generate_enhanced_samples(flow_model, vae_model, n_samples=16, h=125, w=94, steps=50):
#     """Generate face samples using enhanced flow model with VQ-VAE.
#        h and w are the original image dimensions.
#     """
#     import torch.nn.functional as F
    
#     flow_model.eval()
#     vae_model.eval()
#     device = next(vae_model.parameters()).device
    
#     scheduler = FlowMatchingScheduler(num_inference_steps=steps, scheduler_type='cosine')
#     timesteps = scheduler.get_timesteps().to(device)
    
#     num_embeddings = vae_model.vq.num_embeddings
#     h_encoded = vae_model.h_encoded
#     w_encoded = vae_model.w_encoded
#     x_indices = torch.randint(0, num_embeddings, (n_samples, h_encoded, w_encoded), device=device)
    
#     x_oh = F.one_hot(x_indices, num_classes=num_embeddings).float()
#     x_oh = x_oh.view(n_samples, -1, num_embeddings)
    
#     all_states = [x_indices.detach().cpu()]
    
#     with torch.no_grad():
#         for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
#             t_batch = t.expand(n_samples)
            
#             velocity = flow_model(x_oh, t_batch)
#             dt = 1.0 / len(timesteps) if i < len(timesteps) - 1 else 0
#             x_oh = x_oh + velocity * dt
            
#             x_indices = torch.argmax(x_oh, dim=-1)
#             x_indices = x_indices.view(n_samples, h_encoded, w_encoded)
#             x_oh = F.one_hot(x_indices, num_classes=num_embeddings).float()
#             x_oh = x_oh.view(n_samples, -1, num_embeddings)
            
#             all_states.append(x_indices.detach().cpu())
    
#     final_indices = all_states[-1].to(device)
#     final_decoded = vae_model.decode_from_indices(final_indices, n_samples)
    
#     actual_shape = final_decoded.shape
#     if len(actual_shape) == 4:
#         _, channels, actual_h, actual_w = actual_shape
#         if actual_h != h or actual_w != w:
#             if actual_h >= h and actual_w >= w:
#                 h_start = (actual_h - h) // 2
#                 w_start = (actual_w - w) // 2
#                 final_decoded = final_decoded[:, :, h_start:h_start+h, w_start:w_start+w]
#             else:
#                 final_decoded = F.interpolate(final_decoded, size=(h, w), mode='bilinear', align_corners=False)
#         decoded_np = final_decoded.detach().cpu().numpy() * 255
#         if channels == 1:
#             final_images = decoded_np.squeeze(1)
#         else:
#             final_images = decoded_np.transpose(0, 2, 3, 1)
#     else:
#         final_images = (final_decoded.detach().cpu().numpy() * 255)
    
#     all_decoded_images = []
#     with torch.no_grad():
#         for state_indices in all_states:
#             indices = state_indices.to(device)
#             decoded = vae_model.decode_from_indices(indices, n_samples)
#             if len(decoded.shape) == 4:
#                 _, _, decoded_h, decoded_w = decoded.shape
#                 if decoded_h != h or decoded_w != w:
#                     if decoded_h >= h and decoded_w >= w:
#                         h_start = (decoded_h - h) // 2
#                         w_start = (decoded_w - w) // 2
#                         decoded = decoded[:, :, h_start:h_start+h, w_start:w_start+w]
#                     else:
#                         decoded = F.interpolate(decoded, size=(h, w), mode='bilinear', align_corners=False)
#                 images = (decoded.cpu().numpy() * 255).reshape(n_samples, h, w)
#             else:
#                 images = (decoded.cpu().numpy() * 255).reshape(n_samples, h, w)
#             all_decoded_images.append(images)
    
#     return final_images, all_decoded_images

# ##########################################
# # Visualization and Animation Helpers
# ##########################################
# def visualize_samples(images, save_path="generated_faces.png"):
#     n_samples = len(images)
#     rows = int(np.sqrt(n_samples))
#     cols = int(np.ceil(n_samples / rows))
    
#     plt.figure(figsize=(cols*2, rows*2))
    
#     for i, img in enumerate(images):
#         if i >= rows * cols:
#             break
#         plt.subplot(rows, cols, i+1)
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
    
#     h_padded = ((h + 7) // 8) * 8
#     w_padded = ((w + 7) // 8) * 8
    
#     samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
#     samples_tensor = samples_tensor.reshape(-1, 1, h, w) #/ 255.0
    
#     if h != h_padded or w != w_padded:
#         padded_data = torch.zeros(samples_tensor.shape[0], 1, h_padded, w_padded, device=device)
#         padded_data[:, :, :h, :w] = samples_tensor
#         samples_input = padded_data
#     else:
#         samples_input = samples_tensor
    
#     with torch.no_grad():
#         reconstructions, _, _ = vae_model(samples_input)
#         if h != h_padded or w != w_padded:
#             reconstructions = reconstructions[:, :, :h, :w]
    
#     reconstructions = reconstructions.cpu().numpy() * 255
#     reconstructions = reconstructions.reshape(-1, h, w)
#     original_samples = samples.reshape(-1, h, w)
    
#     plt.figure(figsize=(2*n_samples, 4))
    
#     for i in range(n_samples):
#         plt.subplot(2, n_samples, i+1)
#         plt.imshow(original_samples[i], cmap='gray')
#         plt.title("Original")
#         plt.axis('off')
    
#     for i in range(n_samples):
#         plt.subplot(2, n_samples, n_samples+i+1)
#         plt.imshow(reconstructions[i], cmap='gray')
#         plt.title("Reconstructed")
#         plt.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     print(f"Saved VAE reconstructions to {save_path}")
#     plt.close()

# def visualize_training(losses, save_path="training_loss.png"):
#     plt.figure(figsize=(10, 6))
#     window_size = 100
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
#         plt.figure(figsize=(cols*2, rows*2))
        
#         for i in range(n_samples):
#             plt.subplot(rows, cols, i+1)
#             plt.imshow(all_images[t][i], cmap='gray')
#             plt.axis('off')
        
#         plt.suptitle(f"Generation Progress: {t/(n_steps-1)*100:.1f}%", fontsize=16)
#         plt.tight_layout()
        
#         frame_path = f"frames/frame_{t:04d}.png"
#         plt.savefig(frame_path, dpi=150, bbox_inches='tight')
#         frame_paths.append(frame_path)
#         plt.close()
    
#     frames = [Image.open(f) for f in frame_paths]
#     frames[0].save(
#         save_path,
#         save_all=True,
#         append_images=frames[1:],
#         optimize=True,
#         duration=1000//fps,
#         loop=0
#     )
#     print(f"Saved grid animation to {save_path}")

# ##########################################
# # Main Functions
# ##########################################
# def main_with_vqvae():
#     try:
#         print("Starting enhanced face image generation with VQ-VAE + Flow Matching DiT...")
        
#         images_array = X.reshape(n_samples, h, w)
#         os.makedirs("checkpoints", exist_ok=True)
        
#         vqvae_cache_path = "checkpoints/vqvae_model.pt"
#         h_padded = ((h + 7) // 8) * 8
#         w_padded = ((w + 7) // 8) * 8
#         if os.path.exists(vqvae_cache_path):
#             print("Loading cached VQ-VAE model...")
#             vae_model = VQVAE(h=h_padded, w=w_padded, in_channels=1).to(device)
#             vae_model.load_state_dict(torch.load(vqvae_cache_path))
#             vae_losses = []
#         else:
#             print("Training VQ-VAE...")
#             vae_model, vae_losses = train_vqvae(
#                 images=images_array, 
#                 h=h, w=w, 
#                 batch_size=64, 
#                 epochs=15, 
#                 device=device
#             )
#             torch.save(vae_model.state_dict(), vqvae_cache_path)
            
#         recon_path = "vae_reconstructions.png"
#         if not os.path.exists(recon_path):
#             visualize_vae_reconstructions(
#                 vae_model=vae_model,
#                 data=images_array,
#                 h=h, w=w,
#                 save_path=recon_path
#             )
        
#         loss_path = "vae_training_loss.png"
#         if len(vae_losses) > 0 and not os.path.exists(loss_path):
#             visualize_training(vae_losses, save_path=loss_path)
        
#         print("Training enhanced flow model...")
#         flow_model, flow_losses = train_enhanced_flow_model(
#             data=images_array, 
#             vae_model=vae_model, 
#             h=h, w=w, 
#             steps=5000, 
#             batch_size=64
#         )
        
#         visualize_training(flow_losses, save_path="enhanced_flow_training_loss.png")
        
#         print("Generating enhanced samples...")
#         final_images, all_images = generate_enhanced_samples(
#             flow_model=flow_model,
#             vae_model=vae_model,
#             n_samples=50,
#             h=h, w=w,
#             steps=50
#         )
        
#         visualize_samples(final_images, save_path="vqvae_enhanced_faces.png")
#         create_grid_animation(all_images, save_path="vqvae_enhanced_generation.gif")
        
#         print("VQ-VAE + Flow Matching DiT experiment completed successfully!")
        
#     except Exception as e:
#         print(f"Error in main execution: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main_with_vqvae()
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

# Scale the data (for the flow model on raw pixels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

##########################################
# Flow Matching Scheduler
##########################################
class FlowMatchingScheduler:
    """Scheduler for Flow Matching process."""
    def __init__(self, num_inference_steps=50, scheduler_type='cosine', min_t=0.002, max_t=0.998):
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
            return timesteps[:-1]  # Remove the last timestep
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
        return self.timesteps

##########################################
# Attention & Feed-Forward Blocks with Modulation
##########################################
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        x_norm = self.norm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        return x + attn_output

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

# A DiT block with FiLM-style modulation from the time embedding.
class DiTBlockMod(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=64, hidden_dim=None, dropout=0.0, time_embed_dim=256):
        super().__init__()
        self.attn = AttentionBlock(dim, num_heads, dim_head, dropout)
        self.ff = FeedForwardBlock(dim, hidden_dim, dropout)
        # Modulation network: produce scale (gamma) and shift (beta) parameters
        self.modulation = nn.Sequential(
            nn.Linear(time_embed_dim, dim * 2),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, t_emb):
        # t_emb: [batch, time_embed_dim]
        mod_params = self.modulation(t_emb)  # [batch, 2*dim]
        gamma, beta = mod_params.chunk(2, dim=-1)  # each [batch, dim]
        gamma = gamma.unsqueeze(1)  # [batch, 1, dim]
        beta = beta.unsqueeze(1)    # [batch, 1, dim]
        # Apply modulation (FiLM): scale and shift normalized activations.
        x = self.norm(x)
        x = x * (1 + gamma) + beta
        x = self.attn(x)
        x = self.ff(x)
        return x

##########################################
# Advanced Time Embedding
##########################################
def advanced_time_embedding(t, dim=256):
    half_dim = dim // 2
    freqs = torch.exp(-torch.arange(half_dim, device=t.device) * math.log(10000) / half_dim)
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding

##########################################
# Enhanced Flow Matching DiT with VQVAE Integration (with Modulation)
##########################################
class EnhancedFlowMatchingDiT(nn.Module):
    def __init__(self, input_dim, vae_model=None, use_vae_latent=True,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0,
                 dropout=0.1, time_embed_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        self.use_vae_latent = use_vae_latent
        self.vae_model = vae_model
        
        if use_vae_latent and vae_model is not None:
            self.working_dim = vae_model.vq.num_embeddings  # e.g. 512
        else:
            self.working_dim = input_dim
        
        # Pre-compute a time embedding via an MLP for global conditioning if needed
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.input_proj = nn.Linear(self.working_dim, embed_dim)
        
        # Use the modulated DiT blocks instead of vanilla ones.
        self.blocks = nn.ModuleList([
            DiTBlockMod(dim=embed_dim, num_heads=num_heads, hidden_dim=int(embed_dim * mlp_ratio),
                        dropout=dropout, time_embed_dim=time_embed_dim)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, self.working_dim)
        )
        
    def forward(self, x, t):
        # x can be:
        #   - [batch, H, W]  (integer indices -> we do one-hot)
        #   - [batch, H, W, working_dim] (already one-hot, flatten to [batch, H*W, working_dim])
        #   - [batch, n_tokens, working_dim] (already flattened)
        if x.dim() == 3:
            if x.shape[-1] == self.working_dim:
                pass  # already tokenized
            else:
                if x.dtype in [torch.int64, torch.int32]:
                    x = F.one_hot(x.long(), num_classes=self.working_dim).float()
                    b, H, W, C = x.shape
                    x = x.view(b, H*W, C)
        elif x.dim() == 4:
            b, H, W, C = x.shape
            x = x.view(b, H*W, C)
        
        # Compute global time embedding
        t_emb = advanced_time_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # [batch, embed_dim]
        
        h_tokens = self.input_proj(x)
        # Pass the time embedding into each modulated block
        for block in self.blocks:
            h_tokens = block(h_tokens, t_emb)
        h_tokens = self.norm(h_tokens)
        velocity = self.output_proj(h_tokens)
        return velocity

##########################################
# Linear Interpolation for Flow Matching
##########################################
def linear_interpolation(x_0, x_1, t):
    """Interpolate between x_0 and x_1 at time t (broadcasting over tokens)"""
    return (1 - t) * x_0 + t * x_1

##########################################
# VQ-VAE Components (unchanged)
##########################################
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, h=62, w=47, in_channels=1, hidden_dims=[16, 32, 64], 
                 embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        encoder_layers = len(hidden_dims)
        self.h_encoded = h // (2 ** encoder_layers)
        self.w_encoded = w // (2 ** encoder_layers)
        self.h_padded = self.h_encoded * (2 ** encoder_layers)
        self.w_padded = self.w_encoded * (2 ** encoder_layers)
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
        encoder_modules.append(nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1))
        self.encoder = nn.Sequential(*encoder_modules)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        decoder_modules = []
        decoder_modules.append(nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=1))
        for i in range(len(hidden_dims)-1, 0, -1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i-1]),
                    nn.LeakyReLU(),
                    ResidualBlock(hidden_dims[i-1], hidden_dims[i-1])
                )
            )
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0], in_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*decoder_modules)
        
    def encode(self, x):
        x = x.view(-1, self.in_channels, self.h, self.w)
        encoded = self.encoder(x)
        return encoded
        
    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed
    
    def encode_to_indices(self, x):
        x = x.view(-1, self.in_channels, self.h, self.w)
        encoded = self.encoder(x)
        _, _, _, _, indices = self.vq(encoded)
        batch_size = x.shape[0]
        indices = indices.view(batch_size, self.h_encoded, self.w_encoded)
        return indices
    
    def decode_from_indices(self, indices, batch_size):
        flat_indices = indices.view(batch_size, -1)
        one_hot = F.one_hot(flat_indices, num_classes=self.vq.num_embeddings).float()
        quantized = torch.matmul(one_hot, self.vq.embedding.weight)
        quantized = quantized.view(batch_size, self.h_encoded, self.w_encoded, self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decode(quantized)
        
    def forward(self, x):
        x = x.view(-1, self.in_channels, self.h, self.w)
        encoded = self.encoder(x)
        vq_loss, quantized, perplexity, _, _ = self.vq(encoded)
        reconstructed = self.decoder(quantized)
        if reconstructed.shape[2:] != x.shape[2:]:
            reconstructed = reconstructed[:, :, :x.shape[2], :x.shape[3]]
        return reconstructed, vq_loss, perplexity

##########################################
# Training VQ-VAE (unchanged)
##########################################
def train_vqvae(images, h, w, batch_size=64, epochs=20, lr=1e-4, device='cuda'):
    n_samples = len(images)
    h_padded = ((h + 7) // 8) * 8
    w_padded = ((w + 7) // 8) * 8
    vqvae = VQVAE(h=h_padded, w=w_padded, in_channels=1).to(device)
    images_reshaped = images.reshape(-1, 1, h, w)
    images_tensor = torch.tensor(images_reshaped, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(images_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(vqvae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_loss = float('inf')
    losses = []
    for epoch in range(epochs):
        vqvae.train()
        epoch_loss = 0
        reconstruction_loss = 0
        vq_loss_sum = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (data,) in enumerate(pbar):
            data = data.to(device)
            if h != h_padded or w != w_padded:
                padded_data = torch.zeros(data.shape[0], 1, h_padded, w_padded, device=device)
                padded_data[:, :, :h, :w] = data
                data_input = padded_data
            else:
                data_input = data
            reconstructed, vq_loss, perplexity = vqvae(data_input)
            if h != h_padded or w != w_padded:
                reconstructed = reconstructed[:, :, :h, :w]
            recon_loss = F.mse_loss(reconstructed, data)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            reconstruction_loss += recon_loss.item()
            vq_loss_sum += vq_loss.item()
            pbar.set_postfix({
                'loss': loss.item(), 
                'recon_loss': recon_loss.item(),
                'vq_loss': vq_loss.item(),
                'perplexity': perplexity.item()
            })
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = reconstruction_loss / len(dataloader)
        avg_vq_loss = vq_loss_sum / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, VQ Loss: {avg_vq_loss:.6f}")
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vqvae.state_dict(), "vqvae_best.pt")
    return vqvae, losses

##########################################
# Training Enhanced Flow Model with VQ-VAE Integration (Rectified Flow Loss)
##########################################
def train_enhanced_flow_model(data, vae_model, h, w, steps=5000, batch_size=64, lr=1e-4, 
                              checkpoint_freq=1000, embed_dim=256, num_heads=4, depth=4):
    n_samples = len(data)
    device = next(vae_model.parameters()).device
    os.makedirs("checkpoints", exist_ok=True)
    indices_cache_path = "checkpoints/vqvae_indices_cache.pt"
    if os.path.exists(indices_cache_path):
        print("Loading cached VQ-VAE indices...")
        all_indices = torch.load(indices_cache_path)
    else:
        print("Processing data through VQ-VAE encoder...")
        indices_list = []
        vae_model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
                batch = batch.reshape(-1, 1, h, w)
                h_padded = ((h + 7) // 8) * 8
                w_padded = ((w + 7) // 8) * 8
                if h != h_padded or w != w_padded:
                    padded_batch = torch.zeros(batch.shape[0], 1, h_padded, w_padded, device=device)
                    padded_batch[:, :, :h, :w] = batch
                    batch_input = padded_batch
                else:
                    batch_input = batch
                encoded = vae_model.encode(batch_input)
                _, _, _, _, indices = vae_model.vq(encoded)
                indices = indices.view(batch.shape[0], vae_model.h_encoded, vae_model.w_encoded)
                indices_list.append(indices.cpu())
        all_indices = torch.cat(indices_list, dim=0)
        torch.save(all_indices, indices_cache_path)
    
    input_dim = vae_model.vq.num_embeddings
    print(f"Training with VQ indices, input dimension: {input_dim}")
    checkpoint_path = "checkpoints/enhanced_flow_model_latest.pt"
    start_step = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model = EnhancedFlowMatchingDiT(input_dim=data.shape[1],
                                        vae_model=vae_model,
                                        use_vae_latent=True,
                                        embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        depth=depth).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/10)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step'] + 1
        losses = checkpoint.get('losses', [])
        print(f"Resuming from step {start_step}")
    else:
        model = EnhancedFlowMatchingDiT(input_dim=data.shape[1],
                                        vae_model=vae_model,
                                        use_vae_latent=True,
                                        embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        depth=depth).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/10)
        losses = []
    
    if all_indices.dim() == 2:
        all_indices = all_indices.view(n_samples, vae_model.h_encoded, vae_model.w_encoded)
    
    running_loss = 0.0
    pbar = tqdm(range(start_step, steps), desc="Training enhanced flow model")
    try:
        for step in pbar:
            optimizer.zero_grad()
            batch_indices = torch.randint(0, n_samples, (batch_size,))
            x_1_indices = all_indices[batch_indices].to(device)
            x_0_indices = torch.randint(0, vae_model.vq.num_embeddings, 
                                         size=(batch_size, vae_model.h_encoded, vae_model.w_encoded), device=device)
            t = torch.rand(batch_size, device=device)
            x_1_oh = F.one_hot(x_1_indices.long(), num_classes=input_dim).float()  # [B, H, W, input_dim]
            x_0_oh = F.one_hot(x_0_indices.long(), num_classes=input_dim).float()  # [B, H, W, input_dim]
            x_1_oh = x_1_oh.view(batch_size, -1, input_dim)
            x_0_oh = x_0_oh.view(batch_size, -1, input_dim)
            t_reshaped = t.view(batch_size, 1, 1)
            # Compute forward process: zₜ = (1–t)x₀ + t*x₁
            x_t_oh = linear_interpolation(x_0_oh, x_1_oh, t_reshaped)
            # Get predicted velocity/noise from the model
            v_pred = model(x_t_oh, t)
            # Target is the difference (x₁ - x₀)
            v_t = x_1_oh - x_0_oh
            # ----- Rectified Flow Loss Reweighting -----
            # Weight: w(t) = t / clamp(1-t, min=1e-3)
            t_expanded = t.view(batch_size, 1, 1)
            weight = t_expanded / torch.clamp(1 - t_expanded, min=1e-3)
            # Compute per-sample loss
            loss_raw = F.mse_loss(v_pred, v_t, reduction='none')
            loss_raw = loss_raw.mean(dim=[1,2])
            loss = (weight.squeeze() * loss_raw).mean()
            # ----- End Loss Reweighting -----
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            losses.append(loss.item())
            if (step + 1) % checkpoint_freq == 0 or step == steps - 1:
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': losses,
                    'loss': loss.item()
                }
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, f"checkpoints/enhanced_flow_model_step_{step+1}.pt")
                avg_loss = running_loss / checkpoint_freq
                print(f"Step {step+1}, Average Loss: {avg_loss:.6f}")
                running_loss = 0.0
    except Exception as e:
        print(f"Exception during training: {e}")
        if 'model' in locals():
            torch.save({
                'step': step if 'step' in locals() else 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
                'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
                'losses': losses
            }, "checkpoints/enhanced_flow_model_emergency.pt")
        raise
    return model, losses

##########################################
# Generate Enhanced Samples using Flow Model + VQVAE
##########################################
def generate_enhanced_samples(flow_model, vae_model, n_samples=16, h=125, w=94, steps=50):
    """Generate face samples using enhanced flow model with VQ-VAE.
       h and w are the original image dimensions.
    """
    import torch.nn.functional as F
    flow_model.eval()
    vae_model.eval()
    device = next(vae_model.parameters()).device
    scheduler = FlowMatchingScheduler(num_inference_steps=steps, scheduler_type='cosine')
    timesteps = scheduler.get_timesteps().to(device)
    num_embeddings = vae_model.vq.num_embeddings
    h_encoded = vae_model.h_encoded
    w_encoded = vae_model.w_encoded
    x_indices = torch.randint(0, num_embeddings, (n_samples, h_encoded, w_encoded), device=device)
    x_oh = F.one_hot(x_indices, num_classes=num_embeddings).float()
    x_oh = x_oh.view(n_samples, -1, num_embeddings)
    all_states = [x_indices.detach().cpu()]
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = t.expand(n_samples)
            velocity = flow_model(x_oh, t_batch)
            dt = 1.0 / len(timesteps) if i < len(timesteps) - 1 else 0
            x_oh = x_oh + velocity * dt
            x_indices = torch.argmax(x_oh, dim=-1)
            x_indices = x_indices.view(n_samples, h_encoded, w_encoded)
            x_oh = F.one_hot(x_indices, num_classes=num_embeddings).float()
            x_oh = x_oh.view(n_samples, -1, num_embeddings)
            all_states.append(x_indices.detach().cpu())
    final_indices = all_states[-1].to(device)
    final_decoded = vae_model.decode_from_indices(final_indices, n_samples)
    actual_shape = final_decoded.shape
    if len(actual_shape) == 4:
        _, channels, actual_h, actual_w = actual_shape
        if actual_h != h or actual_w != w:
            if actual_h >= h and actual_w >= w:
                h_start = (actual_h - h) // 2
                w_start = (actual_w - w) // 2
                final_decoded = final_decoded[:, :, h_start:h_start+h, w_start:w_start+w]
            else:
                final_decoded = F.interpolate(final_decoded, size=(h, w), mode='bilinear', align_corners=False)
        decoded_np = final_decoded.detach().cpu().numpy() * 255
        if channels == 1:
            final_images = decoded_np.squeeze(1)
        else:
            final_images = decoded_np.transpose(0, 2, 3, 1)
    else:
        final_images = (final_decoded.detach().cpu().numpy() * 255)
    all_decoded_images = []
    with torch.no_grad():
        for state_indices in all_states:
            indices = state_indices.to(device)
            decoded = vae_model.decode_from_indices(indices, n_samples)
            if len(decoded.shape) == 4:
                _, _, decoded_h, decoded_w = decoded.shape
                if decoded_h != h or decoded_w != w:
                    if decoded_h >= h and decoded_w >= w:
                        h_start = (decoded_h - h) // 2
                        w_start = (decoded_w - w) // 2
                        decoded = decoded[:, :, h_start:h_start+h, w_start:w_start+w]
                    else:
                        decoded = F.interpolate(decoded, size=(h, w), mode='bilinear', align_corners=False)
                images = (decoded.cpu().numpy() * 255).reshape(n_samples, h, w)
            else:
                images = (decoded.cpu().numpy() * 255).reshape(n_samples, h, w)
            all_decoded_images.append(images)
    return final_images, all_decoded_images

##########################################
# Visualization and Animation Helpers (unchanged)
##########################################
def visualize_samples(images, save_path="generated_faces.png"):
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

def visualize_vae_reconstructions(vae_model, data, n_samples=10, h=125, w=94, save_path="vae_reconstructions.png"):
    vae_model.eval()
    device = next(vae_model.parameters()).device
    indices = np.random.choice(len(data), n_samples, replace=False)
    samples = data[indices]
    h_padded = ((h + 7) // 8) * 8
    w_padded = ((w + 7) // 8) * 8
    samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
    samples_tensor = samples_tensor.reshape(-1, 1, h, w)
    if h != h_padded or w != w_padded:
        padded_data = torch.zeros(samples_tensor.shape[0], 1, h_padded, w_padded, device=device)
        padded_data[:, :, :h, :w] = samples_tensor
        samples_input = padded_data
    else:
        samples_input = samples_tensor
    with torch.no_grad():
        reconstructions, _, _ = vae_model(samples_input)
        if h != h_padded or w != w_padded:
            reconstructions = reconstructions[:, :, :h, :w]
    reconstructions = reconstructions.cpu().numpy() * 255
    reconstructions = reconstructions.reshape(-1, h, w)
    original_samples = samples.reshape(-1, h, w)
    plt.figure(figsize=(2*n_samples, 4))
    for i in range(n_samples):
        plt.subplot(2, n_samples, i+1)
        plt.imshow(original_samples[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')
    for i in range(n_samples):
        plt.subplot(2, n_samples, n_samples+i+1)
        plt.imshow(reconstructions[i], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved VAE reconstructions to {save_path}")
    plt.close()

def visualize_training(losses, save_path="training_loss.png"):
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

def create_grid_animation(all_images, save_path="face_generation_grid.gif", fps=15):
    n_steps = len(all_images)
    n_samples = min(16, len(all_images[0]))
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))
    os.makedirs("frames", exist_ok=True)
    frame_paths = []
    for t in tqdm(range(n_steps), desc="Creating animation frames"):
        plt.figure(figsize=(cols*2, rows*2))
        for i in range(n_samples):
            plt.subplot(rows, cols, i+1)
            plt.imshow(all_images[t][i], cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Generation Progress: {t/(n_steps-1)*100:.1f}%", fontsize=16)
        plt.tight_layout()
        frame_path = f"frames/frame_{t:04d}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        frame_paths.append(frame_path)
        plt.close()
    frames = [Image.open(f) for f in frame_paths]
    frames[0].save(save_path, save_all=True, append_images=frames[1:], optimize=True, duration=1000//fps, loop=0)
    print(f"Saved grid animation to {save_path}")

##########################################
# Main Function
##########################################
def main_with_vqvae():
    try:
        print("Starting enhanced face image generation with VQ-VAE + Rectified Flow Matching DiT...")
        images_array = X.reshape(n_samples, h, w)
        os.makedirs("checkpoints", exist_ok=True)
        vqvae_cache_path = "checkpoints/vqvae_model.pt"
        h_padded = ((h + 7) // 8) * 8
        w_padded = ((w + 7) // 8) * 8
        if os.path.exists(vqvae_cache_path):
            print("Loading cached VQ-VAE model...")
            vae_model = VQVAE(h=h_padded, w=w_padded, in_channels=1).to(device)
            vae_model.load_state_dict(torch.load(vqvae_cache_path))
            vae_losses = []
        else:
            print("Training VQ-VAE...")
            vae_model, vae_losses = train_vqvae(images=images_array, h=h, w=w, batch_size=64, epochs=15, device=device)
            torch.save(vae_model.state_dict(), vqvae_cache_path)
        recon_path = "vae_reconstructions.png"
        if not os.path.exists(recon_path):
            visualize_vae_reconstructions(vae_model=vae_model, data=images_array, h=h, w=w, save_path=recon_path)
        loss_path = "vae_training_loss.png"
        if len(vae_losses) > 0 and not os.path.exists(loss_path):
            visualize_training(vae_losses, save_path=loss_path)
        print("Training enhanced rectified flow model...")
        flow_model, flow_losses = train_enhanced_flow_model(data=images_array, vae_model=vae_model, h=h, w=w, steps=5000, batch_size=64)
        visualize_training(flow_losses, save_path="enhanced_flow_training_loss.png")
        print("Generating enhanced samples...")
        final_images, all_images = generate_enhanced_samples(flow_model=flow_model, vae_model=vae_model, n_samples=50, h=h, w=w, steps=50)
        visualize_samples(final_images, save_path="vqvae_enhanced_faces.png")
        create_grid_animation(all_images, save_path="vqvae_enhanced_generation.gif")
        print("VQ-VAE + Rectified Flow Matching DiT experiment completed successfully!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_with_vqvae()