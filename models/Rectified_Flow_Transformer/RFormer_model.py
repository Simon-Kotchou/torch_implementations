import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torch.nn import MultiheadAttention
from transformers import CLIPModel, T5EncoderModel

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Modulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, y):
        return self.scale(y) * x + self.shift(y)

class Block(nn.Module):
    def __init__(self, dim, num_heads, context_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-5)
        self.attn = MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.ls1 = Modulation(context_dim)
        self.ls2 = Modulation(context_dim)

    def forward(self, x, context):
        x = x + self.ls1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0], context)
        x = x + self.ls2(self.mlp(self.norm2(x)), context)
        return x

class T5Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList(
            [T5Block(config) for _ in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids, attention_mask=None):
        input_shape = input_ids.size()
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.embed_positions(torch.arange(input_shape[1]))
        hidden_states += position_embeddings
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class TextEncoder(nn.Module):
    def __init__(self, clip_model, t5_model, projection_dim):
        super().__init__()
        self.clip_model = clip_model
        self.t5_model = t5_model
        self.projection = nn.Linear(
            clip_model.config.projection_dim + t5_model.config.d_model,
            projection_dim
        )

    def forward(self, input_ids, attention_mask=None):
        clip_output = self.clip_model.text_model(input_ids)[1]
        t5_output = self.t5_model(input_ids, attention_mask=attention_mask)

        combined_output = torch.cat([clip_output, t5_output[:, 0]], dim=-1)
        projected_output = self.projection(combined_output)

        return projected_output

class MMDiT(nn.Module):
    def __init__(self, x_dim, c_dim, depth, num_heads):
        super().__init__()
        self.depth = depth

        self.x_pos_embedding = nn.Embedding(1024, c_dim)
        self.c_pos_embedding = nn.Embedding(256, c_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(c_dim, c_dim * 4),
            nn.SiLU(),
            nn.Linear(c_dim * 4, c_dim)
        )

        self.x_blocks = nn.ModuleList([
            Block(c_dim, num_heads=num_heads, context_dim=c_dim) for _ in range(depth)
        ])
        self.c_blocks = nn.ModuleList([
            Block(c_dim, num_heads=num_heads, context_dim=c_dim) for _ in range(depth)
        ])

        self.norm_x = RMSNorm(c_dim, eps=1e-5)
        self.norm_c = RMSNorm(c_dim, eps=1e-5)
        self.out = nn.Sequential(
            nn.Linear(c_dim, x_dim),
        )

    def forward(self, x, c, t):
        B, HW, _ = x.shape

        x += self.x_pos_embedding(torch.arange(x.shape[1], device=x.device))
        c += self.c_pos_embedding(torch.arange(c.shape[1], device=c.device))

        t = self.time_embed(timestep_embedding(t, c.shape[-1]))

        for i in range(self.depth):
            x_context = torch.cat([x, c], dim=1)
            c_context = torch.cat([c, x], dim=1)

            x = self.x_blocks[i](x, t) + x
            c = self.c_blocks[i](c, t) + c

        x = self.norm_x(x)
        c = self.norm_c(c)

        x = self.out(x)
        return x

class AutoencoderKL(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dims, hidden_dims, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dims, out_channels, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, hidden_dims, 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dims, in_channels, 4, 2, 1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

class RectifiedFlow(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        hidden_dims,
        depth,
        num_heads,
    ):
        super().__init__()
        self.autoencoder = AutoencoderKL(in_channels, out_channels, hidden_dims)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.t5_model = T5EncoderModel.from_pretrained("google/t5-xxl-ssm-nq")
        self.text_encoder = TextEncoder(self.clip_model, self.t5_model, out_channels)
        self.mmdim = MMDiT(out_channels, out_channels, depth, num_heads)

    def forward(self, x, input_ids, attention_mask, t):
        z = self.autoencoder.encode(x)
        c = self.text_encoder(input_ids, attention_mask)

        z = rearrange(z, 'b c h w -> b (h w) c')
        c = rearrange(c, 'b l c -> b l c')

        z = self.mmdim(z, c, t)

        z = rearrange(z, 'b (h w) c -> b c h w', h=x.shape[-2] // 8)
        x = self.autoencoder.decode(z)

        return x