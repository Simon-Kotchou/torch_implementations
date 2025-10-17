import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import time
import os

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "deepdream_results"
os.makedirs(output_dir, exist_ok=True)

def save_image(img, filename):
    """Save image to disk."""
    if isinstance(img, torch.Tensor):
        if img.is_cuda:
            img = img.cpu()
        if len(img.shape) == 4:
            img = img.squeeze(0)
        img = img.permute(1, 2, 0).numpy()
        if img.max() <= 1:
            img = img * 255
        img = img.astype(np.uint8)
    
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    else:
        pil_img = img
    
    filepath = os.path.join(output_dir, filename)
    pil_img.save(filepath)
    print(f"Saved: {filepath}")

def print_layer_guide():
    """Print layer mapping between models."""
    print("\n" + "="*70)
    print("LAYER NAMING GUIDE - ProGamerGov vs InceptionV3")
    print("="*70)
    print("\nProGamerGov uses BVLC GoogleNet (InceptionV1) with these layer names:")
    print("  inception_3a, inception_3b       (early features - textures)")
    print("  inception_4a, inception_4b       (mid features - objects)")
    print("  inception_4c, inception_4d       (deeper features)")
    print("  inception_4e, inception_5a       (semantic features)")
    print("  inception_5b                     (late features)")
    print("\nWIthin each module are sub-layers like:")
    print("  inception_4d_1x1")
    print("  inception_4d_3x3_reduce          ← The layer you want!")
    print("  inception_4d_3x3")
    print("  inception_4d_5x5_reduce")
    print("  inception_4d_5x5")
    print("  inception_4d_pool_proj")
    
    print("\n" + "-"*70)
    print("PyTorch InceptionV3 uses DIFFERENT names:")
    print("  Mixed_5b, Mixed_5c, Mixed_5d     (early/mid features)")
    print("  Mixed_6a, Mixed_6b, Mixed_6c     (mid features)")
    print("  Mixed_7a, Mixed_7b, Mixed_7c     (late/semantic features)")
    
    print("\n" + "-"*70)
    print("MAPPING (approximate):")
    print("  inception_3a/3b       ≈ Mixed_5b")
    print("  inception_4a/4b/4c    ≈ Mixed_6a/6b")
    print("  inception_4d/4e       ≈ Mixed_6c")
    print("  inception_5a/5b       ≈ Mixed_7b/7c")
    print("="*70 + "\n")

class ChannelAnalyzer:
    """Analyze what individual channels respond to."""
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
    
    def get_channel_stats(self, img):
        """Analyze channel activations."""
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach().clone()
            return hook
        
        try:
            layer = dict(self.model.named_modules())[self.layer_name]
        except KeyError:
            print(f"❌ Layer '{self.layer_name}' not found in model")
            return None, None
        
        hook = layer.register_forward_hook(make_hook(self.layer_name))
        hooks.append(hook)
        
        self.model.aux_logits = False
        with torch.no_grad():
            _ = self.model(img)
        
        for hook in hooks:
            hook.remove()
        
        activation = activations.get(self.layer_name, None)
        if activation is None:
            return None, None
        
        batch_size, num_channels, h, w = activation.shape
        
        stats = {}
        for ch in range(num_channels):
            ch_activation = activation[:, ch, :, :].flatten()
            stats[ch] = {
                'mean': ch_activation.mean().item(),
                'std': ch_activation.std().item(),
                'max': ch_activation.max().item(),
            }
        
        return stats, num_channels

class InceptionV3DeepDream:
    """ProGamerGov-style: Manual channel selection."""
    
    def __init__(self, model, layer_name, channels=None):
        """
        Args:
            channels: Can be:
                - None: Use all channels
                - int: Single channel (119)
                - list: Multiple [119, 1, 29]
                - str: Comma-separated "119,1,29"
        """
        self.model = model
        self.layer_name = layer_name
        self.channels = self._parse_channels(channels)
        self.analyzer = ChannelAnalyzer(model, layer_name)
        
    def _parse_channels(self, channels):
        """Parse channel specification."""
        if channels is None:
            return None
        
        if isinstance(channels, int):
            return [channels]
        
        if isinstance(channels, list):
            return channels
        
        if isinstance(channels, str):
            if '&' in channels:
                return [int(p.strip()) for p in channels.split('&')]
            else:
                return [int(c.strip()) for c in channels.split(',')]
        
        return None
    
    def get_layer_activation(self, img):
        """Get activation from target layer."""
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        try:
            layer = dict(self.model.named_modules())[self.layer_name]
        except KeyError:
            return None
        
        hook = layer.register_forward_hook(make_hook(self.layer_name))
        hooks.append(hook)
        
        self.model.aux_logits = False
        _ = self.model(img)
        
        for hook in hooks:
            hook.remove()
        
        return activations.get(self.layer_name, None)
    
    def calc_loss(self, img, channels=None):
        """Calculate loss from specific channels."""
        activation = self.get_layer_activation(img)
        if activation is None:
            return torch.tensor(0.0, device=device)
        
        target_channels = channels if channels is not None else self.channels
        
        if target_channels is not None:
            activation = activation[:, target_channels, :, :]
        
        loss = activation.norm()
        return loss
    
    def dream_step(self, img, step_size=0.01, jitter=16, channels=None):
        """Single optimization step."""
        if img.grad is not None:
            img.grad.data.zero_()
        
        if jitter > 0:
            shift_x = torch.randint(-jitter, jitter + 1, (1,)).item()
            shift_y = torch.randint(-jitter, jitter + 1, (1,)).item()
            img_shifted = torch.roll(img, shifts=(shift_y, shift_x), dims=(2, 3))
        else:
            img_shifted = img
        
        loss = self.calc_loss(img_shifted, channels)
        loss.backward()
        
        gradients = img.grad.data
        rms = torch.sqrt(torch.mean(gradients ** 2)) + 1e-8
        gradients = gradients / rms
        
        img.data = img.data + step_size * gradients
        img.data = torch.clamp(img.data, 0, 1)
        
        return loss.item()
    
    def dream(self, img, steps=100, step_size=0.01, jitter=16, channels=None):
        """Run DeepDream with manual channel selection."""
        img = img.clone().detach()
        img.requires_grad = True
        
        target_channels = channels if channels is not None else self.channels
        if target_channels:
            print(f"  Targeting channels: {target_channels}")
        else:
            print(f"  Using ALL channels in {self.layer_name}")
        
        for step in range(steps):
            loss = self.dream_step(img, step_size=step_size, jitter=jitter,
                                   channels=target_channels)
            
            if step % 10 == 0:
                print(f"  Step {step}/{steps}, Loss: {loss:.4f}")
            
            if step > 0 and step % 50 == 0:
                intermediate = self._deprocess(img.detach())
                save_image(intermediate, f"step_{step:04d}.png")
        
        return img.detach()
    
    def _deprocess(self, tensor):
        """Convert tensor back to PIL image."""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        array = tensor.permute(1, 2, 0).numpy() * 255
        return Image.fromarray(array.astype(np.uint8))

def explore_channels(model, layer_name, img_tensor, num_top=10):
    """Discover which channels are most active."""
    analyzer = ChannelAnalyzer(model, layer_name)
    stats, total_channels = analyzer.get_channel_stats(img_tensor)
    
    if stats is None:
        print(f"❌ Could not analyze {layer_name}")
        return
    
    sorted_channels = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"📊 Channel Analysis: {layer_name}")
    print(f"{'='*60}")
    print(f"Total channels: {total_channels}")
    
    print(f"\n🔥 Top {num_top} STRONGEST channels (high activation):")
    for i, (ch, stat) in enumerate(sorted_channels[:num_top]):
        bar = "█" * int(stat['mean'] * 20)
        print(f"  [{i+1:2d}] Channel {ch:3d}: {bar} mean={stat['mean']:.4f}")
    
    print(f"\n❄️  Top {num_top} WEAKEST channels (barely respond):")
    for i, (ch, stat) in enumerate(sorted_channels[-num_top:]):
        bar = "█" * max(1, int(stat['mean'] * 20))
        print(f"  [{i+1:2d}] Channel {ch:3d}: {bar} mean={stat['mean']:.4f}")
    
    top_3_strong = [ch for ch, _ in sorted_channels[:3]]
    top_3_weak = [ch for ch, _ in sorted_channels[-3:]]
    
    print(f"\n💡 Try these channel combinations:")
    print(f"  channels = \"{','.join(map(str, top_3_strong))}\"     # Top 3 strongest")
    print(f"  channels = \"{top_3_strong[0]}\"                      # Single strongest")
    print(f"  channels = \"{','.join(map(str, top_3_weak))}\"       # Top 3 weakest (experimental)")
    print(f"  channels = None                      # All channels")
    print(f"{'='*60}\n")

def preprocess_image(image_path, target_size=None):
    """Preprocess image for InceptionV3."""
    img = Image.open(image_path).convert('RGB')
    
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    img_array = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensor = (tensor - mean) / std
    
    return tensor, img.size

def run_octaves(model, img, layer_name, channels, steps_per_octave=100,
                step_size=0.01, num_octaves=4, octave_scale=1.3, jitter=16):
    """Run DeepDream with octaves."""
    
    dreamer = InceptionV3DeepDream(model, layer_name, channels=channels)
    base_shape = img.shape[-2:]
    img = img.clone()
    
    for octave in range(num_octaves):
        octave_index = octave - num_octaves + 1
        octave_size = [int(s * (octave_scale ** octave_index)) for s in base_shape]
        
        if min(octave_size) < 64:
            continue
        
        img = F.interpolate(img, size=octave_size, mode='bilinear', align_corners=False)
        print(f"\n🎨 Octave {octave + 1}/{num_octaves}, Size: {octave_size}")
        
        img = dreamer.dream(img, steps=steps_per_octave, step_size=step_size,
                           jitter=jitter, channels=channels)
        
        result = dreamer._deprocess(img)
        save_image(result, f"octave_{octave:02d}.png")
    
    return img

def main(image_path='./image.png', layer_name='Mixed_6c', channels=None,
         explore=True, num_octaves=3, steps_per_octave=150, step_size=0.01, jitter=16):
    """
    Main DeepDream with manual channel selection for InceptionV3.
    
    Args:
        layer_name: InceptionV3 layer (NOT GoogleNet layer names!)
                   Mixed_5b, Mixed_6a, Mixed_6b, Mixed_6c, Mixed_7a, Mixed_7b, Mixed_7c
        channels: Specific channels to target
    """
    
    print("\n" + "="*70)
    print("🌀 DeepDream with Manual Channel Selection (InceptionV3)")
    print("="*70)
    print(f"Layer: {layer_name}")
    print(f"Channels: {channels if channels else 'ALL'}")
    
    # Show layer guide once
    print_layer_guide()
    
    # Load model
    print("📥 Loading InceptionV3...")
    model = models.inception_v3(pretrained=True)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Load image
    print(f"📷 Loading image: {image_path}")
    max_dim = 1024
    try:
        orig_img = Image.open(image_path)
        orig_size = orig_img.size
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    if max(orig_size) > max_dim:
        scale = max_dim / max(orig_size)
        target_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        print(f"   Scaling to {target_size}")
    else:
        target_size = None
    
    img_tensor, proc_size = preprocess_image(image_path, target_size)
    save_image(orig_img, "original.png")
    
    # Explore channels
    if explore:
        explore_channels(model, layer_name, img_tensor, num_top=12)
    
    # Run DeepDream
    print("\n🚀 Starting DeepDream...")
    start = time.time()
    
    dreamed = run_octaves(
        model, img_tensor, layer_name, channels,
        steps_per_octave=steps_per_octave,
        step_size=step_size,
        num_octaves=num_octaves,
        octave_scale=1.3,
        jitter=jitter
    )
    
    elapsed = time.time() - start
    
    # Save result
    dreamer = InceptionV3DeepDream(model, layer_name)
    result = dreamer._deprocess(dreamed)
    
    if target_size:
        result = result.resize(orig_size, Image.LANCZOS)
    
    save_image(result, "final_result.png")
    
    # Comparison
    comparison = Image.new('RGB', (orig_size[0] * 2, orig_size[1]))
    comparison.paste(orig_img.resize(orig_size), (0, 0))
    comparison.paste(result, (orig_size[0], 0))
    save_image(comparison, "comparison.png")
    
    print(f"\n✅ Completed in {elapsed:.1f}s")
    print(f"📁 Results saved to {output_dir}/\n")

if __name__ == "__main__":
    # STEP 1: Explore channels first
    main(
        image_path='./Albert_Wider_Priestergrab_in_Widnau_02.png',
        layer_name='Mixed_6c',        # Use InceptionV3 layer names!
        channels=[302],                 # None = analyze all
        explore=True,
        num_octaves=3,
        steps_per_octave=100
    )
    
    # STEP 2: After exploring, try specific channels:
    # main(image_path='./your_image.png', layer_name='Mixed_6c', channels="45,12,88")