import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Callable
import einops


class FlowMatchingScheduler:
    """
    Scheduler for Flow Matching based diffusion process.
    Implements time-step scheduling strategies for HunyuanVideo.
    """
    def __init__(
        self,
        num_inference_steps=50,
        scheduler_type='shifted',
        shifting_factor=7.0,
        min_t=0.002,
        max_t=0.998
    ):
        self.num_inference_steps = num_inference_steps
        self.scheduler_type = scheduler_type
        self.shifting_factor = shifting_factor
        self.min_t = min_t
        self.max_t = max_t
        
        # Create timestep schedule based on selected strategy
        self.timesteps = self._get_schedule()
        
    def _get_schedule(self):
        """
        Generate timestep schedule based on chosen strategy.
        """
        if self.scheduler_type == 'linear':
            # Linear schedule from max_t to min_t
            return torch.linspace(
                self.max_t, self.min_t, self.num_inference_steps
            )
        
        elif self.scheduler_type == 'quadratic':
            # Quadratic schedule giving more steps to early diffusion
            steps = np.linspace(0, 1, self.num_inference_steps)
            # Convert to quadratic curve
            steps = 1 - np.square(steps)
            # Scale to desired range
            steps = steps * (self.max_t - self.min_t) + self.min_t
            return torch.from_numpy(steps).float()
        
        elif self.scheduler_type == 'cosine':
            # Cosine schedule
            steps = torch.arange(self.num_inference_steps + 1).float() / self.num_inference_steps
            alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            timesteps = torch.clamp(alpha_cumprod, self.min_t, self.max_t)
            return timesteps
        
        elif self.scheduler_type == 'shifted':
            # Shifted schedule using the shifting factor
            # This focuses more steps on early diffusion (crucial for fewer steps)
            steps = np.linspace(0, 1, self.num_inference_steps)
            # Apply shifting function t' = s*t/(1+(s-1)*t)
            s = self.shifting_factor
            steps = s * steps / (1 + (s - 1) * steps)
            # Scale to desired range
            steps = (1 - steps) * (self.max_t - self.min_t) + self.min_t
            return torch.from_numpy(steps).float()
            
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def get_timesteps(self):
        """
        Get all timesteps for inference.
        """
        return self.timesteps
    
    def adjust_shifting_factor(self, num_steps):
        """
        Dynamically adjust shifting factor based on number of steps.
        Lower inference steps require higher shifting factor.
        """
        if num_steps <= 10:
            self.shifting_factor = 17.0
        elif num_steps <= 20:
            self.shifting_factor = 15.0
        elif num_steps <= 30:
            self.shifting_factor = 12.0
        elif num_steps <= 40:
            self.shifting_factor = 9.0
        else:
            self.shifting_factor = 7.0
            
        # Update timesteps with new shifting factor
        if self.scheduler_type == 'shifted':
            self.timesteps = self._get_schedule()
        
        return self.timesteps


class NoiseUtils:
    """
    Utilities for managing noise in diffusion models.
    Provides methods for adding/removing noise and computing diffusion targets.
    """
    @staticmethod
    def q_sample(x_start, x_noise, t):
        """
        Diffuse data to timestep t by interpolating between
        data and noise according to schedule.
        
        Args:
            x_start: Starting clean data
            x_noise: Random noise
            t: Diffusion timesteps [batch_size]
            
        Returns:
            Noisy samples at timestep t
        """
        # Linear interpolation between start and noise
        # t=0 is all noise, t=1 is all signal
        t = t.view(-1, 1, 1, 1, 1)  # For broadcasting
        return (1 - t) * x_noise + t * x_start
    
    @staticmethod
    def compute_flow_velocity(x_start, x_noise):
        """
        Compute the ground-truth velocity field for Flow Matching.
        
        Args:
            x_start: Clean data
            x_noise: Noise data
            
        Returns:
            Velocity field
        """
        # For linear interpolation, velocity is x_target - x_noise
        return x_start - x_noise
    
    @staticmethod
    def get_noise_level_embedding(timesteps, embedding_dim=256):
        """
        Sinusoidal embedding for noise levels/timesteps.
        
        Args:
            timesteps: Timesteps to embed [batch_size]
            embedding_dim: Dimension of embedding
            
        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:  # Zero pad if dim is odd
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
            
        return emb
    
    @staticmethod
    def noise_like(shape, device, repeat=False):
        """
        Create new noise tensor or repeat noise for video frames.
        
        Args:
            shape: Shape of noise tensor
            device: Device to create tensor on
            repeat: If True, generates one noise and repeats across frames
            
        Returns:
            Noise tensor
        """
        if repeat:
            # Create one noise and repeat for all frames
            # Useful for frame-consistent noise
            batch, channel, time, height, width = shape
            noise = torch.randn((batch, channel, 1, height, width), device=device)
            return noise.expand(batch, channel, time, height, width)
        else:
            # Different noise for each frame
            return torch.randn(shape, device=device)


class MemoryBank:
    """
    Memory bank for tracking and propagating features across frames.
    Extends the DiffusionMemory implementation with additional methods.
    """
    def __init__(
        self,
        feature_dim,
        memory_length=16,
        update_rate=0.9,
        use_topk=True,
        topk=5
    ):
        self.feature_dim = feature_dim
        self.memory_length = memory_length
        self.update_rate = update_rate
        self.use_topk = use_topk
        self.topk = min(topk, memory_length)
        
        # Initialize memory as empty
        self.reset()
        
    def reset(self):
        """Reset the memory bank"""
        self.memory = []
        self.current_idx = 0
        self.is_initialized = False
        
    def initialize(self, initial_features):
        """
        Initialize memory bank with features.
        
        Args:
            initial_features: Initial features to store [B, C, H, W] or [B, C, T, H, W]
        """
        if len(initial_features.shape) == 5:
            # If temporal dimension is present, use first frame
            features = initial_features[:, :, 0]
        else:
            features = initial_features
            
        # Create memory bank
        batch_size, channels, height, width = features.shape
        
        # Store mean feature vector for each spatial location
        for _ in range(self.memory_length):
            self.memory.append(features.clone())
            
        self.is_initialized = True
        self.current_idx = 0
        
    def update(self, new_features):
        """
        Update memory bank with new features using moving average.
        
        Args:
            new_features: New features to incorporate [B, C, H, W]
        """
        if not self.is_initialized:
            self.initialize(new_features)
            return
            
        # Update memory at current index
        self.memory[self.current_idx] = (
            self.update_rate * self.memory[self.current_idx] + 
            (1 - self.update_rate) * new_features
        )
        
        # Update index
        self.current_idx = (self.current_idx + 1) % self.memory_length
        
    def get_memory_features(self, query_features=None):
        """
        Retrieve features from memory bank.
        
        Args:
            query_features: Optional query to find closest memories
            
        Returns:
            Memory features [B, C, M, H, W] where M is memory length
        """
        if not self.is_initialized:
            # Return zeros if memory not initialized
            batch_size, channels, height, width = query_features.shape
            return torch.zeros(
                (batch_size, channels, self.memory_length, height, width),
                device=query_features.device
            )
            
        # Stack memory features
        memory_features = torch.stack(self.memory, dim=2)  # [B, C, M, H, W]
        
        if self.use_topk and query_features is not None:
            # Find top-k similar memories if query provided
            batch_size, channels, height, width = query_features.shape
            
            # Reshape for similarity computation
            query_flat = query_features.view(batch_size, channels, -1)  # [B, C, H*W]
            memory_flat = memory_features.view(batch_size, channels, self.memory_length, -1)  # [B, C, M, H*W]
            
            # Compute cosine similarity
            query_norm = torch.norm(query_flat, dim=1, keepdim=True)  # [B, 1, H*W]
            memory_norm = torch.norm(memory_flat, dim=1, keepdim=True)  # [B, 1, M, H*W]
            
            # Avoid division by zero
            query_norm = torch.clamp(query_norm, min=1e-6)
            memory_norm = torch.clamp(memory_norm, min=1e-6)
            
            # Normalize features
            query_flat = query_flat / query_norm
            memory_flat = memory_flat / memory_norm
            
            # Compute similarity
            similarity = torch.einsum('bcp,bcmp->bmp', query_flat, memory_flat)  # [B, M, H*W]
            
            # Average similarity across spatial dimensions
            similarity = similarity.mean(dim=2)  # [B, M]
            
            # Get top-k indices
            _, top_indices = torch.topk(similarity, k=self.topk, dim=1)  # [B, K]
            
            # Gather top-k memories
            top_memories = torch.gather(
                memory_features,
                dim=2,
                index=top_indices.view(batch_size, 1, self.topk, 1, 1).expand(
                    -1, channels, -1, height, width
                )
            )  # [B, C, K, H, W]
            
            return top_memories
            
        return memory_features
    
    def aggregate_with_attention(self, query_features):
        """
        Retrieve memory features and aggregate using attention mechanism.
        
        Args:
            query_features: Query features [B, C, H, W]
            
        Returns:
            Aggregated memory features [B, C, H, W]
        """
        if not self.is_initialized:
            return query_features
            
        # Get memory features
        memory_features = self.get_memory_features(query_features)  # [B, C, M, H, W]
        
        batch_size, channels, mem_len, height, width = memory_features.shape
        
        # Reshape for attention computation
        query = query_features.view(batch_size, channels, -1)  # [B, C, H*W]
        memory = memory_features.view(batch_size, channels, mem_len, -1)  # [B, C, M, H*W]
        
        # Compute attention weights
        attn_weights = torch.einsum('bcp,bcmp->bmp', query, memory)  # [B, M, H*W]
        attn_weights = F.softmax(attn_weights / math.sqrt(channels), dim=1)  # [B, M, H*W]
        
        # Apply attention to memory features
        aggregated = torch.einsum('bmp,bcmp->bcp', attn_weights, memory)  # [B, C, H*W]
        
        # Reshape back to spatial format
        return aggregated.view(batch_size, channels, height, width)


class TextGuidanceDistillation(nn.Module):
    """
    Text-guidance distillation for optimizing classifier-free guidance.
    This approach eliminates the need for two forward passes during inference.
    """
    def __init__(
        self,
        teacher_model,
        guidance_scale_range=(1.0, 8.0)
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = None  # Will be initialized as a copy of teacher
        self.guidance_scale_range = guidance_scale_range
        
    def initialize_student(self):
        """
        Initialize student as a copy of teacher model
        """
        self.student_model = type(self.teacher_model)(
            **{k: v for k, v in self.teacher_model.__dict__.items()
               if k in self.teacher_model.__init__.__code__.co_varnames}
        )
        self.student_model.load_state_dict(self.teacher_model.state_dict())
        
    def teacher_forward_with_cfg(
        self, 
        x, 
        timesteps, 
        text_embeddings,
        uncond_embeddings,
        clip_text_embeddings=None,
        uncond_clip_embeddings=None,
        guidance_scale=7.5,
        **kwargs
    ):
        """
        Teacher model forward pass with classifier-free guidance
        
        Args:
            x: Input latent video
            timesteps: Diffusion timesteps
            text_embeddings: Conditional text embeddings
            uncond_embeddings: Unconditional text embeddings (empty prompt)
            clip_text_embeddings: CLIP text features (conditional)
            uncond_clip_embeddings: CLIP text features (unconditional)
            guidance_scale: Scale for classifier-free guidance
            **kwargs: Additional arguments for model forward
            
        Returns:
            CFG-combined output
        """
        # Get batch size
        batch_size = x.shape[0]
        
        # Double the inputs for conditional and unconditional passes
        x_twice = torch.cat([x, x], dim=0)
        timesteps_twice = torch.cat([timesteps, timesteps], dim=0)
        
        # Text embeddings (conditional and unconditional)
        text_embeddings_twice = torch.cat([text_embeddings, uncond_embeddings], dim=0)
        
        # CLIP features (if provided)
        clip_embeddings_twice = None
        if clip_text_embeddings is not None and uncond_clip_embeddings is not None:
            clip_embeddings_twice = torch.cat(
                [clip_text_embeddings, uncond_clip_embeddings], dim=0
            )
        
        # Forward pass with combined batch
        pred_twice = self.teacher_model(
            x_twice,
            timesteps_twice,
            text_embeddings_twice,
            clip_text_embeddings=clip_embeddings_twice,
            **kwargs
        )
        
        # Split predictions
        pred_cond, pred_uncond = pred_twice.chunk(2)
        
        # Apply classifier-free guidance
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
    def student_forward(
        self,
        x,
        timesteps,
        text_embeddings,
        clip_text_embeddings=None,
        guidance_scale=7.5,
        **kwargs
    ):
        """
        Student model forward pass with guidance scale as input
        
        Args:
            x: Input latent video
            timesteps: Diffusion timesteps
            text_embeddings: Text embeddings
            clip_text_embeddings: CLIP text features
            guidance_scale: Guidance scale value
            **kwargs: Additional arguments for model forward
            
        Returns:
            Output with distilled guidance
        """
        # Inject guidance scale into timestep embedding
        batch_size = x.shape[0]
        
        # Create guidance scale tensor and inject it with timesteps
        guidance_tensor = torch.ones((batch_size, 1), device=x.device) * guidance_scale
        scaled_timesteps = torch.cat([timesteps.unsqueeze(1), guidance_tensor], dim=1)
        
        # Forward pass with modified timesteps
        return self.student_model(
            x,
            scaled_timesteps,
            text_embeddings,
            clip_text_embeddings=clip_text_embeddings,
            **kwargs
        )
    
    def training_step(self, batch, optimizer):
        """
        Execute a single distillation training step
        
        Args:
            batch: Training batch data
            optimizer: Student model optimizer
            
        Returns:
            Loss value
        """
        # Unpack batch
        x = batch['latents']
        timesteps = batch['timesteps']
        text_embeddings = batch['text_embeddings']
        uncond_embeddings = batch['uncond_embeddings']
        clip_text_embeddings = batch.get('clip_embeddings')
        uncond_clip_embeddings = batch.get('uncond_clip_embeddings')
        
        # Sample random guidance scale
        guidance_scale = torch.FloatTensor(1).uniform_(*self.guidance_scale_range)[0]
        
        # Get teacher output
        with torch.no_grad():
            teacher_out = self.teacher_forward_with_cfg(
                x,
                timesteps,
                text_embeddings,
                uncond_embeddings,
                clip_text_embeddings,
                uncond_clip_embeddings,
                guidance_scale=guidance_scale
            )
        
        # Get student output
        student_out = self.student_forward(
            x,
            timesteps,
            text_embeddings,
            clip_text_embeddings,
            guidance_scale=guidance_scale
        )
        
        # Compute distillation loss (MSE)
        loss = F.mse_loss(student_out, teacher_out)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class VideoTilingSampler:
    """
    Implements tiling strategy for handling large videos in VAE.
    Splits video into overlapping tiles to avoid memory issues.
    """
    def __init__(
        self,
        vae,
        tile_size_t=8,
        tile_size_h=64,
        tile_size_w=64,
        tile_overlap_t=2,
        tile_overlap_h=16,
        tile_overlap_w=16
    ):
        self.vae = vae
        self.tile_size_t = tile_size_t
        self.tile_size_h = tile_size_h
        self.tile_size_w = tile_size_w
        self.tile_overlap_t = tile_overlap_t
        self.tile_overlap_h = tile_overlap_h
        self.tile_overlap_w = tile_overlap_w
        
    def get_tile_coordinates(self, time_dim, height, width):
        """
        Generate coordinates for overlapping tiles.
        
        Args:
            time_dim: Temporal dimension of video
            height, width: Spatial dimensions of video
            
        Returns:
            List of (t_start, t_end, h_start, h_end, w_start, w_end) for each tile
        """
        # Calculate effective step sizes
        t_step = self.tile_size_t - self.tile_overlap_t
        h_step = self.tile_size_h - self.tile_overlap_h
        w_step = self.tile_size_w - self.tile_overlap_w
        
        # Calculate number of tiles in each dimension
        num_t = max(1, math.ceil((time_dim - self.tile_overlap_t) / t_step))
        num_h = max(1, math.ceil((height - self.tile_overlap_h) / h_step))
        num_w = max(1, math.ceil((width - self.tile_overlap_w) / w_step))
        
        # Generate tile coordinates
        coords = []
        for t_idx in range(num_t):
            for h_idx in range(num_h):
                for w_idx in range(num_w):
                    # Calculate tile boundaries
                    t_start = min(t_idx * t_step, time_dim - self.tile_size_t)
                    h_start = min(h_idx * h_step, height - self.tile_size_h)
                    w_start = min(w_idx * w_step, width - self.tile_size_w)
                    
                    t_end = t_start + self.tile_size_t
                    h_end = h_start + self.tile_size_h
                    w_end = w_start + self.tile_size_w
                    
                    coords.append((t_start, t_end, h_start, h_end, w_start, w_end))
        
        return coords
    
    def create_blend_mask(self, tile_shape, full_shape):
        """
        Create blending mask for smooth tile transitions.
        
        Args:
            tile_shape: Shape of tile (T, H, W)
            full_shape: Shape of full video (T, H, W)
            
        Returns:
            Blending mask for this tile
        """
        t_size, h_size, w_size = tile_shape
        t_full, h_full, w_full = full_shape
        
        # Create linear ramps for blending at edges
        def linear_ramp(size, overlap):
            if size <= overlap:
                return torch.ones(size)
            
            ramp = torch.ones(size)
            ramp_vals = torch.linspace(0, 1, overlap)
            ramp[:overlap] = ramp_vals
            ramp[-overlap:] = torch.flip(ramp_vals, [0])
            return ramp
        
        # Create 1D masks for each dimension
        t_mask = linear_ramp(t_size, self.tile_overlap_t)
        h_mask = linear_ramp(h_size, self.tile_overlap_h)
        w_mask = linear_ramp(w_size, self.tile_overlap_w)
        
        # Combine 1D masks into 3D mask
        mask_t = t_mask.view(-1, 1, 1)
        mask_h = h_mask.view(1, -1, 1)
        mask_w = w_mask.view(1, 1, -1)
        
        mask_3d = mask_t * mask_h * mask_w
        
        return mask_3d
        
    def encode_tiled(self, video):
        """
        Encode video using tiling approach.
        
        Args:
            video: Input video [B, C, T, H, W]
            
        Returns:
            Encoded latents
        """
        batch_size, channels, time_dim, height, width = video.shape
        
        # Get downsampled latent dimensions
        t_factor = self.vae.time_downscale_factor
        s_factor = self.vae.downscale_factor
        
        latent_time = time_dim // t_factor
        latent_height = height // s_factor
        latent_width = width // s_factor
        
        # Prepare output tensor
        latents = torch.zeros(
            (batch_size, self.vae.latent_channels, latent_time, latent_height, latent_width),
            device=video.device
        )
        
        # Get tile coordinates
        coords = self.get_tile_coordinates(time_dim, height, width)
        
        # Keep track of blending weights
        blend_weights = torch.zeros((latent_time, latent_height, latent_width), device=video.device)
        
        for t_start, t_end, h_start, h_end, w_start, w_end in coords:
            # Extract tile
            tile = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
            
            # Encode tile
            with torch.no_grad():
                tile_latent, _ = self.vae.encode(tile)
            
            # Calculate latent coordinates
            lt_start = t_start // t_factor
            lt_end = t_end // t_factor
            lh_start = h_start // s_factor
            lh_end = h_end // s_factor
            lw_start = w_start // s_factor
            lw_end = w_end // s_factor
            
            # Create blend mask for this tile
            blend_mask = self.create_blend_mask(
                (lt_end - lt_start, lh_end - lh_start, lw_end - lw_start),
                (latent_time, latent_height, latent_width)
            ).to(video.device)
            
            # Update output latents and blend weights
            latents[:, :, lt_start:lt_end, lh_start:lh_end, lw_start:lw_end] += (
                tile_latent * blend_mask.unsqueeze(0).unsqueeze(0)
            )
            blend_weights[lt_start:lt_end, lh_start:lh_end, lw_start:lw_end] += blend_mask
        
        # Normalize by blend weights
        blend_weights = torch.clamp(blend_weights, min=1e-5)
        latents = latents / blend_weights.unsqueeze(0).unsqueeze(0)
        
        return latents
    
    def decode_tiled(self, latents):
        """
        Decode latents using tiling approach.
        
        Args:
            latents: Input latents [B, C, T, H, W]
            
        Returns:
            Decoded video
        """
        batch_size, channels, latent_time, latent_height, latent_width = latents.shape
        
        # Get upsampled video dimensions
        t_factor = self.vae.time_downscale_factor
        s_factor = self.vae.downscale_factor
        
        time_dim = latent_time * t_factor
        height = latent_height * s_factor
        width = latent_width * s_factor
        
        # Prepare output tensor
        video = torch.zeros(
            (batch_size, self.vae.in_channels, time_dim, height, width),
            device=latents.device
        )
        
        # Get tile coordinates in latent space
        coords = self.get_tile_coordinates(latent_time, latent_height, latent_width)
        
        # Keep track of blending weights
        blend_weights = torch.zeros((time_dim, height, width), device=latents.device)
        
        for lt_start, lt_end, lh_start, lh_end, lw_start, lw_end in coords:
            # Extract latent tile
            tile_latent = latents[:, :, lt_start:lt_end, lh_start:lh_end, lw_start:lw_end]
            
            # Decode latent tile
            with torch.no_grad():
                tile = self.vae.decode(tile_latent)
            
            # Calculate pixel coordinates
            t_start = lt_start * t_factor
            t_end = lt_end * t_factor
            h_start = lh_start * s_factor
            h_end = lh_end * s_factor
            w_start = lw_start * s_factor
            w_end = lw_end * s_factor
            
            # Create blend mask for this tile
            blend_mask = self.create_blend_mask(
                (t_end - t_start, h_end - h_start, w_end - w_start),
                (time_dim, height, width)
            ).to(latents.device)
            
            # Update output video and blend weights
            video[:, :, t_start:t_end, h_start:h_end, w_start:w_end] += (
                tile * blend_mask.unsqueeze(0).unsqueeze(0)
            )
            blend_weights[t_start:t_end, h_start:h_end, w_start:w_end] += blend_mask
        
        # Normalize by blend weights
        blend_weights = torch.clamp(blend_weights, min=1e-5)
        video = video / blend_weights.unsqueeze(0).unsqueeze(0)
        
        return video