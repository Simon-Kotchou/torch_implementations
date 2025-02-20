import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any, Union
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Callable, Optional

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding with advanced frequency encoding."""
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.max_period = max_period
        
        # Improved embedding network
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps tensor of shape [batch_size]
            
        Returns:
            Time embeddings of shape [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Calculation in log space for better numerical stability
        freqs = torch.exp(-math.log(self.max_period) * 
                         torch.arange(half_dim, device=device) / half_dim)
        
        # Create sinusoidal embedding: [batch_size, dim]
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embedding = torch.cat([
                embedding,
                torch.zeros_like(t).unsqueeze(-1)
            ], dim=-1)
            
        # Process through MLP for improved representation
        embedding = self.mlp(embedding)
        
        return embedding

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding for improved attention."""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def _build_rotary_pos_emb(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, 'n d -> 1 n 1 d')
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding to queries and keys."""
        device = q.device
        pos_emb = self._build_rotary_pos_emb(seq_len, device)
        
        # Apply to queries and keys
        q_cos, q_sin = q * pos_emb.cos(), q * pos_emb.sin()
        k_cos, k_sin = k * pos_emb.cos(), k * pos_emb.sin()
        
        # Rotate
        q = q_cos + self.rotate_half(q_sin)
        k = k_cos + self.rotate_half(k_sin)
        
        return q, k
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim]
            k: Key tensor of shape [batch, heads, seq_len, head_dim]
            
        Returns:
            Rotary position embedded q and k
        """
        seq_len = q.shape[2]
        return self.apply_rotary_pos_emb(q, k, seq_len)

class LayerNormWithModulation(nn.Module):
    """Layer normalization with modulation parameters."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.alpha_proj = nn.Linear(dim, dim)
        self.beta_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            cond: Conditional input of shape [batch, cond_dim]
            
        Returns:
            Normalized and modulated tensor
        """
        x = self.norm(x)
        
        if cond is not None:
            # Generate modulation parameters from condition
            alpha = self.alpha_proj(cond).unsqueeze(1) + 1.0  # Add 1 for stability
            beta = self.beta_proj(cond).unsqueeze(1)
            
            # Apply modulation
            x = alpha * x + beta
            
        return x

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using RMS."""
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.scale * x * rms

class AttentionWithModulation(nn.Module):
    """Multi-head attention with conditional modulation."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        use_rotary: bool = True,
        use_flash_attn: bool = True,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.use_rotary = use_rotary
        self.use_flash_attn = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        self.cross_attention = cross_attention
        
        # Query, Key, Value projections
        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        
        # Rotary positional embedding
        if use_rotary:
            self.rotary_emb = RotaryPositionalEmbedding(dim_head)
            
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Modulation parameters
        self.alpha_scale = nn.Parameter(torch.ones(1))
        self.beta_shift = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cond_alpha: Optional[torch.Tensor] = None,
        cond_beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            context: Optional context for cross-attention [batch, context_len, dim]
            mask: Optional attention mask [batch, seq_len, seq_len]
            cond_alpha: Conditional alpha modulation [batch, dim]
            cond_beta: Conditional beta modulation [batch, dim]
            
        Returns:
            Attention output
        """
        batch_size, seq_len, _ = x.shape
        kv_input = context if self.cross_attention and context is not None else x
        
        # Apply conditional modulation if provided
        if cond_alpha is not None and cond_beta is not None:
            alpha = cond_alpha.unsqueeze(1)  # [batch, 1, dim]
            beta = cond_beta.unsqueeze(1)    # [batch, 1, dim]
            
            # Modulate input before attention
            x = x * (self.alpha_scale * alpha + 1.0) + self.beta_shift * beta
        
        # Project to queries, keys, values
        q = self.to_q(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary:
            q, k = self.rotary_emb(q, k)
        
        # Perform attention
        if self.use_flash_attn:
            # Use PyTorch's flash attention implementation
            if mask is not None:
                # Expand mask for multi-head attention
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len or context_len]
                
            # Scale query for stability
            q = q / math.sqrt(self.dim_head)
            
            # Apply flash attention
            context_len = k.shape[2]
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=0.0 if not self.training else 0.1,
                is_causal=False
            )
        else:
            # Manual implementation for compatibility
            q = q / math.sqrt(self.dim_head)
            
            # Compute attention scores
            attn = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq_len, context_len]
            
            # Apply mask if provided
            if mask is not None:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len or context_len]
                attn = attn.masked_fill(mask == 0, -1e9)
            
            # Attention weights
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention
            attn_output = torch.matmul(attn, v)  # [batch, heads, seq_len, dim_head]
        
        # Reshape and project back
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        output = self.to_out(attn_output)
        
        return output

class FeedForwardWithModulation(nn.Module):
    """Feed-forward network with conditional modulation."""
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        # Modulation parameters
        self.delta_scale = nn.Parameter(torch.ones(1))
        self.epsilon_shift = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        x: torch.Tensor,
        cond_delta: Optional[torch.Tensor] = None,
        cond_epsilon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            cond_delta: Conditional delta modulation [batch, dim]
            cond_epsilon: Conditional epsilon modulation [batch, dim]
            
        Returns:
            Feed-forward output
        """
        # Apply conditional modulation if provided
        if cond_delta is not None and cond_epsilon is not None:
            delta = cond_delta.unsqueeze(1)     # [batch, 1, dim]
            epsilon = cond_epsilon.unsqueeze(1)  # [batch, 1, dim]
            
            # Modulate input before feed-forward
            x = x * (self.delta_scale * delta + 1.0) + self.epsilon_shift * epsilon
            
        return self.net(x)

class MMDiTBlock(nn.Module):
    """Multi-Modal Diffusion Transformer Block."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        use_rotary: bool = True,
        use_flash_attention: bool = True,
        cross_attention: bool = True,
        use_modulation: bool = True,
        sandwich_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_modulation = use_modulation
        self.cross_attention = cross_attention
        self.sandwich_norm = sandwich_norm
        
        # First normalization (sandwich norm has norm before and after attention)
        self.norm1 = LayerNormWithModulation(dim) if use_modulation else nn.LayerNorm(dim)
        
        # Self-attention
        self.attn = AttentionWithModulation(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=attention_dropout,
            qkv_bias=qkv_bias,
            use_rotary=use_rotary,
            use_flash_attn=use_flash_attention,
            cross_attention=False,
        )
        
        # Optional second normalization for sandwich norm
        self.norm1b = nn.LayerNorm(dim) if sandwich_norm else nn.Identity()
        
        # Cross-attention if enabled
        if cross_attention:
            self.norm_cross = LayerNormWithModulation(dim) if use_modulation else nn.LayerNorm(dim)
            self.cross_attn = AttentionWithModulation(
                dim=dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=attention_dropout,
                qkv_bias=qkv_bias,
                use_rotary=use_rotary,
                use_flash_attn=use_flash_attention,
                cross_attention=True,
            )
            self.norm_cross_b = nn.LayerNorm(dim) if sandwich_norm else nn.Identity()
        
        # Feed-forward normalization
        self.norm2 = LayerNormWithModulation(dim) if use_modulation else nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ff = FeedForwardWithModulation(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )
        
        # Optional final normalization for sandwich norm
        self.norm2b = nn.LayerNorm(dim) if sandwich_norm else nn.Identity()
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            context: Optional context for cross-attention [batch, context_len, dim]
            mask: Attention mask [batch, seq_len, seq_len]
            cond: Conditional parameters dict with alpha, beta, delta, epsilon
            
        Returns:
            Transformed tensor
        """
        # Extract conditional parameters if provided
        cond_alpha = cond.get('alpha') if cond is not None else None
        cond_beta = cond.get('beta') if cond is not None else None
        cond_delta = cond.get('delta') if cond is not None else None
        cond_epsilon = cond.get('epsilon') if cond is not None else None
        
        # Self-attention
        residual = x
        if self.use_modulation:
            x = self.norm1(x, cond=cond_alpha)
        else:
            x = self.norm1(x)
            
        x = self.attn(
            x, 
            mask=mask,
            cond_alpha=cond_alpha,
            cond_beta=cond_beta
        )
        x = self.norm1b(x)
        x = residual + x
        
        # Cross-attention if enabled
        if self.cross_attention and context is not None:
            residual = x
            if self.use_modulation:
                x = self.norm_cross(x, cond=cond_alpha)
            else:
                x = self.norm_cross(x)
                
            x = self.cross_attn(
                x,
                context=context,
                cond_alpha=cond_alpha,
                cond_beta=cond_beta
            )
            x = self.norm_cross_b(x)
            x = residual + x
        
        # Feed-forward
        residual = x
        if self.use_modulation:
            x = self.norm2(x, cond=cond_delta)
        else:
            x = self.norm2(x)
            
        x = self.ff(
            x,
            cond_delta=cond_delta,
            cond_epsilon=cond_epsilon
        )
        x = self.norm2b(x)
        x = residual + x
        
        return x

class AttributeEmbedding(nn.Module):
    """Embedding for CelebA attributes."""
    def __init__(
        self, 
        num_attributes: int, 
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim * 2
        
        # Linear embedding of each attribute
        self.embed = nn.Linear(num_attributes, hidden_dim)
        
        # Nonlinear projection
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attributes: Binary attribute tensor [batch, num_attributes]
            
        Returns:
            Attribute embedding [batch, embedding_dim]
        """
        x = self.embed(attributes)
        return self.proj(x)

class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings."""
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        use_norm: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # Patch projection
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Optional normalization
        self.norm = nn.LayerNorm(embed_dim) if use_norm else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Image tensor [batch, channels, height, width]
            
        Returns:
            Patch embeddings [batch, num_patches, embed_dim]
            Height and width in patches
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image dimensions {H}x{W} must be divisible by patch size {self.patch_size}"
        
        # Project patches
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        
        # Flatten patches to sequence
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        
        return x, h, w

class UnPatchify(nn.Module):
    """Convert patch embeddings back to image."""
    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels * patch_size**2, kernel_size=1),
            nn.Pixel_Shuffle(patch_size),  # Efficient upsampling
        )
        
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings [batch, num_patches, embed_dim]
            h: Height in patches
            w: Width in patches
            
        Returns:
            Image tensor [batch, out_channels, height, width]
        """
        # Reshape to 2D spatial grid
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        # Project to pixels
        x = self.proj(x)
        
        return x

class MMDiTRectifiedFlow(nn.Module):
    """Complete MM-DiT model for rectified flow on CelebA."""
    def __init__(
        self,
        config: Union['FluxModelConfig', Dict],
        img_size: int = 256,
        in_channels: int = 16,
        patch_size: int = 2,
        embed_dim: int = 1280,
        depth: int = 24,
        num_heads: int = 20,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        time_embed_dim: int = 1280,
        num_attributes: int = 40,
        attribute_embed_dim: int = 512,
        use_flash_attention: bool = True,
        use_rotary: bool = True,
        sandwich_norm: bool = True,
    ):
        super().__init__()
        # Store parameters
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.attribute_embed_dim = attribute_embed_dim
        self.num_attributes = num_attributes
        
        # Calculate sequence length based on patching
        self.seq_len = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            use_norm=True,
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Attribute embedding
        self.attribute_embed = AttributeEmbedding(
            num_attributes=num_attributes,
            embedding_dim=attribute_embed_dim,
            hidden_dim=embed_dim,
        )
        
        # Project condition embeddings to modulation parameters
        self.cond_projector = nn.ModuleDict({
            'alpha': nn.Linear(embed_dim + attribute_embed_dim, embed_dim),
            'beta': nn.Linear(embed_dim + attribute_embed_dim, embed_dim),
            'delta': nn.Linear(embed_dim + attribute_embed_dim, embed_dim),
            'epsilon': nn.Linear(embed_dim + attribute_embed_dim, embed_dim),
        })
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                qkv_bias=True,
                use_rotary=use_rotary,
                use_flash_attention=use_flash_attention,
                cross_attention=False,  # No cross-attention in base model
                use_modulation=True,
                sandwich_norm=sandwich_norm,
            )
            for _ in range(depth)
        ])
        
        # Output processing
        self.norm_out = nn.LayerNorm(embed_dim)
        self.unpatchify = UnPatchify(
            embed_dim=embed_dim,
            out_channels=in_channels,
            patch_size=patch_size,
        )
        
    def _prepare_conditions(
        self, 
        timesteps: torch.Tensor, 
        attributes: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare conditional inputs for modulation.
        
        Args:
            timesteps: Time steps tensor [batch]
            attributes: Attributes tensor [batch, num_attributes]
            cond_drop_prob: Probability of dropping condition for classifier-free guidance
            
        Returns:
            Dictionary with modulation parameters
        """
        batch_size = timesteps.shape[0]
        device = timesteps.device
        
        # Time embedding
        t_emb = self.time_embed(timesteps)
        t_emb = self.time_proj(t_emb)
        
        # Attribute embedding with optional dropout for classifier-free guidance
        if attributes is not None:
            # Apply dropout to attributes during training
            if self.training and cond_drop_prob > 0:
                mask = torch.bernoulli(
                    torch.ones(batch_size, device=device) * (1 - cond_drop_prob)
                ).unsqueeze(-1)
                attr_emb = self.attribute_embed(attributes * mask)
            else:
                attr_emb = self.attribute_embed(attributes)
                
            # Concatenate time and attribute embeddings
            cond = torch.cat([t_emb, attr_emb], dim=1)
        else:
            # Use zero attribute embedding if not provided
            zero_attr = torch.zeros(batch_size, self.attribute_embed_dim, device=device)
            cond = torch.cat([t_emb, zero_attr], dim=1)
        
        # Project to modulation parameters
        modulation = {}
        for param_name, projector in self.cond_projector.items():
            modulation[param_name] = projector(cond)
            
        return modulation
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        attributes: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.1,
    ) -> torch.Tensor:
        """
        Forward pass to predict velocity field.
        
        Args:
            x: Input latent tensor [batch, channels, height, width]
            timesteps: Time steps tensor [batch]
            attributes: Attributes tensor [batch, num_attributes]
            cond_drop_prob: Probability of dropping condition for classifier-free guidance
            
        Returns:
            Predicted velocity field [batch, channels, height, width]
        """
        # Prepare modulation parameters
        cond = self._prepare_conditions(timesteps, attributes, cond_drop_prob)
        
        # Convert image to patches
        x, h, w = self.patch_embed(x)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Process with transformer blocks
        for block in self.blocks:
            x = block(x, cond=cond)
            
        # Output processing
        x = self.norm_out(x)
        velocity = self.unpatchify(x, h, w)
        
        return velocity

class CelebaFluxTrainer:
    """Trainer for CelebA-optimized Flux model."""
    def __init__(
        self,
        model: MMDiTRectifiedFlow,
        autoencoder: nn.Module,
        optimizer_config: Dict,
        training_config: Dict,
        data_config: Dict,
        ode_solver_config: Dict,
        device: torch.device,
    ):
        self.model = model
        self.autoencoder = autoencoder
        self.device = device
        self.config = {
            'optimizer': optimizer_config,
            'training': training_config,
            'data': data_config,
            'ode_solver': ode_solver_config,
        }
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # EMA setup if enabled
        self.use_ema = optimizer_config.get('use_ema', True)
        if self.use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = optimizer_config.get('ema_decay', 0.9999)
            self.ema_update_interval = optimizer_config.get('ema_update_interval', 10)
        
        # ODE solver for sampling
        self.ode_solver = DormandPrinceSolver(
            velocity_fn=self._get_velocity_fn(),
            rtol=ode_solver_config.get('rtol', 1e-5),
            atol=ode_solver_config.get('atol', 1e-5),
            safety_factor=ode_solver_config.get('safety_factor', 0.9),
            min_step_size=ode_solver_config.get('min_step_size', 1e-5),
            max_step_size=ode_solver_config.get('max_step_size', 0.1),
            max_steps=ode_solver_config.get('max_steps', 1000),
            device=device,
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        opt_cfg = self.config['optimizer']
        opt_type = opt_cfg.get('optimizer', 'adamw')
        lr = opt_cfg.get('learning_rate', 1e-4)
        weight_decay = opt_cfg.get('weight_decay', 0.01)
        betas = opt_cfg.get('betas', (0.9, 0.999))
        
        if opt_type.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=betas,
            )
        elif opt_type.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        elif opt_type.lower() == 'lion':
            try:
                from lion_pytorch import Lion
                return Lion(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                )
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        opt_cfg = self.config['optimizer']
        train_cfg = self.config['training']
        scheduler_type = opt_cfg.get('lr_scheduler', 'cosine')
        warmup_steps = opt_cfg.get('warmup_steps', 5000)
        min_lr_ratio = opt_cfg.get('min_lr_ratio', 0.1)
        total_steps = train_cfg.get('total_steps', 400000)
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.optimizer.param_groups[0]['lr'] * min_lr_ratio,
            )
        elif scheduler_type == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=min_lr_ratio,
                total_iters=total_steps,
            )
        elif scheduler_type == 'warmup_cosine':
            from transformers import get_cosine_schedule_with_warmup
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif scheduler_type == 'constant':
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=total_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def _create_ema_model(self):
        """Create Exponential Moving Average model copy."""
        ema_model = type(self.model)(**self.model.config)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        return ema_model
    
    def _update_ema_model(self):
        """Update EMA model weights."""
        with torch.no_grad():
            for param_ema, param_model in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                param_ema.data.mul_(self.ema_decay).add_(
                    param_model.data, alpha=1 - self.ema_decay
                )
    
    def _get_velocity_fn(self):
        """Get velocity prediction function for ODE solver."""
        def velocity_fn(x, t):
            # Expand timesteps for batched prediction
            timesteps = t.expand(x.shape[0])
            return self.ema_model(x, timesteps) if self.use_ema else self.model(x, timesteps)
        
        return velocity_fn
    
    def train_step(
        self,
        latents: torch.Tensor,
        attributes: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.
        
        Args:
            latents: Target latent vectors [batch, channels, height, width]
            attributes: Optional attribute tensor [batch, num_attributes]
            cond_drop_prob: Probability of dropping conditioning
            
        Returns:
            Dictionary with loss values
        """
        batch_size = latents.shape[0]
        
        # Sample random noise
        z0 = torch.randn_like(latents)
        
        # Sample random timesteps in (0,1)
        t = torch.rand(batch_size, device=self.device)
        
        # Construct point along the trajectory with linear interpolation
        zt = t.view(-1, 1, 1, 1) * latents + (1 - t.view(-1, 1, 1, 1)) * z0
        
        # True velocity field
        v_target = latents - z0
        
        # Predict velocity field
        v_pred = self.model(
            zt, 
            timesteps=t,
            attributes=attributes, 
            cond_drop_prob=cond_drop_prob
        )
        
        # Calculate loss
        losses = self._calculate_losses(v_pred, v_target, zt, latents)
        loss = self._aggregate_losses(losses)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['optimizer'].get('clip_grad_norm', 1.0)
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update EMA model if enabled
        if self.use_ema and self.step % self.ema_update_interval == 0:
            self._update_ema_model()
            
        # Update step counter
        self.step += 1
        
        # Add grad norm and learning rate to losses dict
        losses['grad_norm'] = grad_norm
        losses['lr'] = self.scheduler.get_last_lr()[0]
        losses['total_loss'] = loss.item()
        
        return losses
    
    def _calculate_losses(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        z_t: torch.Tensor,
        z_1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate all loss components.
        
        Args:
            v_pred: Predicted velocity [batch, channels, height, width]
            v_target: Target velocity [batch, channels, height, width]
            z_t: Current point on trajectory [batch, channels, height, width]
            z_1: Target point (data) [batch, channels, height, width]
            
        Returns:
            Dictionary of loss components
        """
        loss_cfg = self.config['training'].get('loss_weights', {})
        loss_type = self.config['training'].get('loss_type', 'mse')
        
        losses = {}
        
        # Main velocity prediction loss
        if loss_type == 'mse':
            losses['velocity'] = F.mse_loss(v_pred, v_target)
        elif loss_type == 'l1':
            losses['velocity'] = F.l1_loss(v_pred, v_target)
        elif loss_type == 'huber':
            losses['velocity'] = F.smooth_l1_loss(v_pred, v_target)
        else:
            losses['velocity'] = F.mse_loss(v_pred, v_target)
            
        # Velocity regularization loss
        if loss_cfg.get('velocity_reg', 0) > 0:
            losses['velocity_reg'] = torch.mean(torch.norm(
                v_pred.reshape(v_pred.shape[0], -1), 
                dim=1
            ))
            
        # Latent regularization loss
        if loss_cfg.get('latent_reg', 0) > 0:
            losses['latent_reg'] = torch.mean(
                torch.abs(torch.abs(z_t) - 0.9).clamp(min=0) ** 2
            )
            
        return losses
    
    def _aggregate_losses(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine loss components with weights."""
        loss_weights = self.config['training'].get('loss_weights', {})
        total_loss = 0
        
        for loss_name, loss_value in losses.items():
            weight = loss_weights.get(loss_name, 0.0)
            if weight > 0:
                total_loss = total_loss + weight * loss_value
                
        return total_loss
    
    def sample(
        self,
        batch_size: int,
        attributes: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample images using rectified flow.
        
        Args:
            batch_size: Number of samples to generate
            attributes: Optional attribute conditioning [batch, num_attributes]
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated images [batch, channels, height, width]
        """
        # Set models to eval mode
        self.model.eval()
        if self.use_ema:
            self.ema_model.eval()
        
        # Get latent shape from autoencoder
        with torch.no_grad():
            dummy = torch.zeros(
                1, 
                self.autoencoder.in_channels,
                self.autoencoder.img_size,
                self.autoencoder.img_size,
                device=self.device,
            )
            latent = self.autoencoder.encode(dummy)
            latent_shape = latent.shape[1:]
        
        def guided_velocity_fn(x, t):
            """Velocity function with classifier-free guidance."""
            timesteps = t.expand(x.shape[0])
            model = self.ema_model if self.use_ema else self.model
            
            # Get conditional prediction
            if attributes is not None:
                v_cond = model(x, timesteps, attributes=attributes, cond_drop_prob=0.0)
                
                # For guidance, get unconditional prediction
                if guidance_scale != 1.0:
                    v_uncond = model(x, timesteps, attributes=None, cond_drop_prob=1.0) 
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v = v_cond
            else:
                # No conditioning
                v = model(x, timesteps, attributes=None)
                
            return v
        
        # Set up ODE solver with guided velocity function
        solver = DormandPrinceSolver(
            velocity_fn=guided_velocity_fn,
            rtol=self.config['ode_solver'].get('rtol', 1e-5),
            atol=self.config['ode_solver'].get('atol', 1e-5),
            safety_factor=self.config['ode_solver'].get('safety_factor', 0.9),
            min_step_size=self.config['ode_solver'].get('min_step_size', 1e-5),
            max_step_size=self.config['ode_solver'].get('max_step_size', 0.1),
            max_steps=self.config['ode_solver'].get('max_steps', 1000),
            device=self.device,
        )
        
        # Sample initial latent from standard normal
        z0 = torch.randn(batch_size, *latent_shape, device=self.device)
        
        # Set up time span from t=0 to t=1
        t_span = torch.tensor([0.0, 1.0], device=self.device)
        
        # Integrate from t=0 to t=1
        with torch.no_grad():
            result = solver.integrate(
                z0, 
                t_span, 
                record_intermediate=True,
                record_steps=100,
            )
            
            # Decode final latent to image using autoencoder
            z1 = result['y1']
            images = self.autoencoder.decode(z1)
        
        # Return to train mode
        self.model.train()
        
        return images, result['trajectory']