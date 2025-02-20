import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)  # GroupNorm for better stability
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(8, out_channels)
        
        if not self.same_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(8, out_channels)
            )
            
    def forward(self, x):
        identity = x
        out = F.silu(self.bn1(self.conv1(x)))  # SiLU activation (GELU alternative)
        out = self.bn2(self.conv2(out))
        
        if not self.same_channels:
            identity = self.shortcut(x)
            
        out += identity
        out = F.silu(out)
        return out

class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style inverted residual block with expansion factor"""
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(InvertedResidualBlock, self).__init__()
        
        hidden_dim = in_channels * expansion_factor
        self.use_res_connection = in_channels == out_channels
        
        layers = []
        # Point-wise expansion
        layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
        layers.append(nn.GroupNorm(8, hidden_dim))
        layers.append(nn.SiLU())
        
        # Depth-wise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, 
                               padding=1, groups=hidden_dim))
        layers.append(nn.GroupNorm(8, hidden_dim))
        layers.append(nn.SiLU())
        
        # Point-wise compression
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
        layers.append(nn.GroupNorm(8, out_channels))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connection:
            return x + self.layers(x)
        else:
            return self.layers(x)

class ContinuousAutoencoder(nn.Module):
    def __init__(self, h=128, w=128, in_channels=3, latent_dim=768,
                 hidden_dims=[64, 128, 256, 512]):
        super(ContinuousAutoencoder, self).__init__()
        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Calculate dimensions after encoding
        encoder_layers = len(hidden_dims)
        self.h_encoded = h // (2 ** encoder_layers)  # 4x downsampling = 16x16
        self.w_encoded = w // (2 ** encoder_layers)
        
        # Encoder with MobileNetV2-style blocks
        encoder_modules = []
        in_ch = in_channels
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, h_dim),
                    nn.SiLU(),
                    InvertedResidualBlock(h_dim, h_dim)
                )
            )
            in_ch = h_dim
            
        # Final layer to get to target latent dimension
        final_latent_ch = latent_dim // (self.h_encoded * self.w_encoded)
        encoder_modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], final_latent_ch, kernel_size=1),
                nn.GroupNorm(8, final_latent_ch)
            )
        )
        
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Latent regularization
        self.latent_regularizer = nn.Tanh()  # Constrain to [-1, 1] range
        
        # Decoder with nearest-neighbor upsampling
        decoder_modules = []
        
        # Initial projection from latent space
        decoder_modules.append(
            nn.Sequential(
                nn.Conv2d(final_latent_ch, hidden_dims[-1], kernel_size=1),
                nn.GroupNorm(8, hidden_dims[-1]),
                nn.SiLU()
            )
        )
        
        # Upsample blocks
        for i in range(len(hidden_dims)-1, 0, -1):
            decoder_modules.append(
                nn.Sequential(
                    # Nearest-neighbor upsampling followed by convolution
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i-1], kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dims[i-1]),
                    nn.SiLU(),
                    ResidualBlock(hidden_dims[i-1], hidden_dims[i-1])
                )
            )
            
        # Final layer
        decoder_modules.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(hidden_dims[0], hidden_dims[0] // 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dims[0] // 2),
                nn.SiLU(),
                nn.Conv2d(hidden_dims[0] // 2, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()  # Output in [0, 1] range
            )
        )
        
        self.decoder = nn.Sequential(*decoder_modules)
        
    def encode(self, x):
        """Encode input to continuous latent representation"""
        x = self._ensure_dimensions(x)
        encoded = self.encoder(x)
        # Apply tanh to constrain latent values to [-1, 1]
        latent = self.latent_regularizer(encoded)
        return latent
    
    def decode(self, z):
        """Decode latent representation"""
        return self.decoder(z)
    
    def _ensure_dimensions(self, x):
        """Ensure input has the correct dimensions"""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale
        if x.shape[2:] != (self.h, self.w):
            x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=False)
        return x
        
    def forward(self, x):
        """Full forward pass: encode, decode"""
        x = self._ensure_dimensions(x)
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        return reconstructed, latent
        
class LaplacianPyramidLoss(nn.Module):
    """Multi-scale loss using Laplacian pyramid for detail preservation"""
    def __init__(self, max_levels=5, weights=None):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.weights = weights if weights is not None else [0.8, 0.4, 0.2, 0.1, 0.05]
        
    def build_pyramid(self, img, max_levels):
        pyramid = []
        current = img
        for _ in range(max_levels - 1):
            down = F.avg_pool2d(current, kernel_size=2, stride=2)
            up = F.interpolate(down, scale_factor=2, mode='bilinear', align_corners=False)
            diff = current - up
            pyramid.append(diff)
            current = down
        pyramid.append(current)
        return pyramid
        
    def forward(self, pred, target):
        pred_pyramid = self.build_pyramid(pred, self.max_levels)
        target_pyramid = self.build_pyramid(target, self.max_levels)
        
        total_loss = 0
        for i in range(self.max_levels):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            total_loss += weight * F.mse_loss(pred_pyramid[i], target_pyramid[i])
            
        return total_loss

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for semantic consistency"""
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        blocks = []
        blocks.append(vgg[:4])    # relu1_2
        blocks.append(vgg[4:9])   # relu2_2
        blocks.append(vgg[9:16])  # relu3_3
        blocks.append(vgg[16:23]) # relu4_3
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, pred, target):
        if pred.shape[1] != 3:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        if self.resize:
            pred = F.interpolate(pred, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        # Normalize to ImageNet stats
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        for i, block in enumerate(self.blocks):
            pred = block(pred)
            with torch.no_grad():
                target = block(target)
            loss += F.mse_loss(pred, target)
            
        return loss

def train_continuous_autoencoder(data, h, w, in_channels, batch_size=64, epochs=50, 
                               lr=3e-4, device='cuda'):
    """Train the continuous autoencoder for rectified flow"""
    n_samples = len(data)
    
    # Create dataset and dataloader
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create continuous autoencoder model
    model = ContinuousAutoencoder(h, w, in_channels=in_channels).to(device)
    
    # Define losses
    recon_loss_fn = nn.MSELoss().to(device)
    laplacian_loss_fn = LaplacianPyramidLoss().to(device)
    perceptual_loss_fn = PerceptualLoss().to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr/10
    )
    
    # Training metrics
    losses = {'total': [], 'recon': [], 'laplacian': [], 'perceptual': [], 'latent_reg': []}
    best_loss = float('inf')
    
    # Train loop
    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in losses.keys()}
        
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon, latent = model(data)
            
            # Calculate losses
            recon_loss = recon_loss_fn(recon, data)
            laplacian_loss = laplacian_loss_fn(recon, data)
            perceptual_loss = perceptual_loss_fn(recon, data)
            
            # L2 regularization on latent values to stay in [-1, 1] range
            # (note: this is in addition to tanh constraint)
            latent_reg_loss = 0.1 * torch.mean(torch.pow(torch.abs(latent) - 0.9, 2) * 
                                             (torch.abs(latent) > 0.9).float())
            
            # Total loss
            total_loss = recon_loss + 0.5 * laplacian_loss + 0.05 * perceptual_loss + latent_reg_loss
            
            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['laplacian'] += laplacian_loss.item()
            epoch_losses['perceptual'] += perceptual_loss.item() 
            epoch_losses['latent_reg'] += latent_reg_loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {total_loss.item():.4f}")
        
        # End of epoch
        for k, v in epoch_losses.items():
            epoch_losses[k] = v / len(dataloader)
            losses[k].append(epoch_losses[k])
            
        # Reduce learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1} summary:")
        for k, v in epoch_losses.items():
            print(f"  {k}: {v:.6f}")
            
        # Save checkpoint if best
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'checkpoints/continuous_ae_best.pt')
            print(f"  Saved new best model with loss: {best_loss:.6f}")
            
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_losses['total'],
    }, 'checkpoints/continuous_ae_final.pt')
    
    return model, losses
