import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from urllib.request import urlopen
import time
import os

# For reproducibility
torch.manual_seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
output_dir = "deepdream_dinov3_results"
os.makedirs(output_dir, exist_ok=True)

def save_image(img, filename):
    """Save a PIL image or PyTorch tensor to file."""
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
    print(f"Image saved to {filepath}")
    return filepath

def load_dinov3_model():
    """Load DINOv3 ConvNeXt Large model from HuggingFace."""
    model = AutoModel.from_pretrained(
        "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def get_image_processor():
    """Get the image processor for DINOv3."""
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        trust_remote_code=True
    )
    return processor

def calc_loss_targeted_layers(model, img, target_layers):
    """Calculate loss from specific layers within Stage 2.
    
    Uses modern 2025 techniques:
    - Channel diversity loss
    - Spatial coherence
    - Multi-scale features
    
    Args:
        model: DINOv3 model
        img: Input tensor
        target_layers: List of layer indices within stage 2 to hook (e.g., [10, 15, 20])
    """
    activations = {}
    hooks = []
    
    # Register hooks on specific layers in Stage 2
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    for layer_idx in target_layers:
        # Hook into the pointwise_conv2 for most semantic features
        layer = model.stages[2].layers[layer_idx].pointwise_conv2
        hook = layer.register_forward_hook(make_hook(f'stage2_layer{layer_idx}'))
        hooks.append(hook)
    
    # Forward pass
    _ = model(img)
    
    # Calculate loss with 2025-style objectives
    total_loss = 0
    
    for name, activation in activations.items():
        # Extract layer index from name
        layer_idx = int(name.split('layer')[1])
        
        # Weight deeper layers more (they have more abstract features)
        depth_weight = (layer_idx - min(target_layers)) / (max(target_layers) - min(target_layers) + 1)
        depth_weight = 0.5 + depth_weight * 0.5  # Scale to [0.5, 1.0]
        
        # 1. Maximize mean activation (excite neurons)
        mean_loss = activation.mean()
        
        # 2. Maximize channel diversity (encourage different patterns)
        # Calculate variance across spatial dimensions for each channel
        B, C, H, W = activation.shape
        channel_means = activation.mean(dim=[2, 3])  # [B, C]
        diversity_loss = channel_means.std()
        
        # 3. Spatial coherence (encourage connected patterns, not noise)
        # L2 gradient penalty - penalize high-frequency noise
        dx = activation[:, :, :, 1:] - activation[:, :, :, :-1]
        dy = activation[:, :, 1:, :] - activation[:, :, :-1, :]
        smoothness = -(dx.pow(2).mean() + dy.pow(2).mean())
        
        # Combine losses
        layer_loss = (
            1.0 * mean_loss + 
            0.3 * diversity_loss + 
            0.1 * smoothness
        ) * depth_weight
        
        total_loss += layer_loss
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_loss

class DeepDreamDINOv3:
    def __init__(self, model, target_layers):
        """Initialize DeepDream with DINOv3 and target layers."""
        self.model = model
        self.target_layers = target_layers
        
    def __call__(self, img, steps=100, step_size=0.01):
        """Run the DeepDream algorithm."""
        img = img.clone()
        img.requires_grad = True
        
        for step in range(steps):
            if img.grad is not None:
                img.grad.data.zero_()
            
            # Calculate loss from target layers
            loss = calc_loss_targeted_layers(self.model, img, self.target_layers)
            
            # Backward pass
            loss.backward()
            
            # Normalize gradients (2025 style - use RMS normalization)
            gradients = img.grad.data
            rms = torch.sqrt(torch.mean(gradients ** 2)) + 1e-8
            gradients = gradients / rms
            
            # Update image with gradient ascent
            img.data = img.data + step_size * gradients
            
            # Clip to maintain valid image values
            img.data = torch.clamp(img.data, 0, 1)
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                
                if step > 0 and step % 40 == 0:
                    intermediate = deprocess_image(img.detach())
                    save_image(intermediate, f"step_{step}_intermediate.png")
                
        return img.detach()

def random_roll(img, max_roll):
    """Randomly shift the image to avoid tiled boundaries."""
    _, _, h, w = img.shape
    dy = torch.randint(-max_roll, max_roll + 1, (1,)).item()
    dx = torch.randint(-max_roll, max_roll + 1, (1,)).item()
    return torch.roll(img, shifts=(dy, dx), dims=(2, 3)), (dy, dx)

class TiledDeepDreamDINOv3:
    def __init__(self, model, target_layers):
        """Initialize TiledDeepDream with DINOv3 and target layers."""
        self.model = model
        self.target_layers = target_layers
        
    def __call__(self, img, tile_size=512):
        """Calculate gradients using tiled approach for large images."""
        b, c, h, w = img.shape
        
        # Random roll to avoid tile boundaries
        shifted_img, (dy, dx) = random_roll(img, tile_size // 8)
        
        # Initialize gradients
        gradients = torch.zeros_like(shifted_img)
        
        # Process each tile
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                if y_end - y < 64 or x_end - x < 64:
                    continue
                
                # Extract tile
                tile = shifted_img[:, :, y:y_end, x:x_end].clone()
                tile.requires_grad = True
                
                # Calculate loss for this tile
                loss = calc_loss_targeted_layers(self.model, tile, self.target_layers)
                
                # Backward pass
                loss.backward()
                
                # Store gradients
                gradients[:, :, y:y_end, x:x_end] = tile.grad.data
                
                if tile.grad is not None:
                    tile.grad.data.zero_()
        
        # Normalize gradients using RMS
        rms = torch.sqrt(torch.mean(gradients ** 2)) + 1e-8
        gradients = gradients / rms
        
        # Un-shift gradients
        gradients = torch.roll(gradients, shifts=(-dy, -dx), dims=(2, 3))
        
        return gradients

def preprocess_image(image, processor, target_size=None):
    """Preprocess image for DINOv3."""
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # Use the processor
    inputs = processor(images=image, return_tensors="pt")
    tensor = inputs['pixel_values'].to(device)
    
    return tensor, image.size

def deprocess_image(tensor, target_size=None):
    """Convert tensor back to PIL image."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # DINOv3 uses ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    array = tensor.permute(1, 2, 0).numpy() * 255
    img = Image.fromarray(array.astype(np.uint8))
    
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
        
    return img

def run_deepdream_with_octaves(model, img, target_layers,
                              steps_per_octave=100, step_size=0.01,
                              octaves=range(-2, 0, 3), octave_scale=1.3,
                              use_tiling=True, tile_size=512,
                              original_size=None):
    """Run DeepDream with octaves."""
    input_shape = img.shape[-2:]
    
    if original_size is not None:
        base_shape = (original_size[1], original_size[0])
    else:
        base_shape = input_shape
        
    img = img.clone()
    
    if use_tiling:
        dreamer = TiledDeepDreamDINOv3(model, target_layers)
    else:
        dreamer = DeepDreamDINOv3(model, target_layers)
    
    for octave in octaves:
        new_size = [int(s * (octave_scale ** octave)) for s in base_shape]
        
        if min(new_size) < 64:
            continue
            
        img = torch.nn.functional.interpolate(
            img, size=new_size, mode='bilinear', align_corners=False
        )
        
        print(f"\nProcessing octave {octave} with image size {img.shape[-2:]} pixels")
        
        if use_tiling:
            for step in range(steps_per_octave):
                gradients = dreamer(img, tile_size=min(tile_size, min(new_size)))
                img.data = img.data + step_size * gradients
                img.data = torch.clamp(img.data, 0, 1)
                
                if step % 20 == 0:
                    print(f"Octave {octave}, Step {step}")
        else:
            img = dreamer(img, steps=steps_per_octave, step_size=step_size)
        
        octave_img = deprocess_image(img)
        save_image(octave_img, f"octave_{octave}_result.png")
    
    img = torch.nn.functional.interpolate(
        img, size=base_shape, mode='bilinear', align_corners=False
    )
    return img

def generate_deepdream_dinov3(
    image_path='./gettyimages.jpg',
    target_layers=[10, 15, 20],  # Middle-late Stage 2 layers
    use_tiling=True,
    tile_size=512
):
    """Main function for DINOv3 DeepDream.
    
    Args:
        image_path: Path to input image
        target_layers: Layer indices within Stage 2 to target (0-26)
                      Recommended: [10, 15, 20] for rich abstract features
        use_tiling: Whether to use tiling (for larger images)
        tile_size: Tile size if using tiling
    """
    print(f"DINOv3 ConvNeXt Large DeepDream initialized:")
    print(f"- Stage 2 target layers: {target_layers}")
    print(f"  (out of 27 total layers in Stage 2)")
    print(f"- Use tiling: {use_tiling}")
    print(f"- Tile size: {tile_size}")
    
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"\nLoaded image from '{image_path}'")
    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'")
        return
        
    original_size = img.size
    print(f"Image size: {original_size[0]}x{original_size[1]} pixels")
    save_image(img, "original_image.png")
    
    # Load DINOv3 model and processor
    print("\nLoading DINOv3 ConvNeXt Large model...")
    model = load_dinov3_model()
    processor = get_image_processor()
    
    # Print architecture info
    print("\nDINOv3 ConvNeXt Large architecture:")
    print(f"Stage 0: 3 layers, 192 channels")
    print(f"Stage 1: 3 layers, 384 channels")
    print(f"Stage 2: 27 layers, 768 channels ← TARGETING LAYERS {target_layers}")
    print(f"Stage 3: 3 layers, 1536 channels")
    
    # Handle image size
    max_dimension = 1024
    if max(original_size) > max_dimension:
        scale = max_dimension / max(original_size)
        target_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        print(f"\nScaling to {target_size[0]}x{target_size[1]} to avoid memory issues")
    else:
        target_size = None
        print(f"\nPreserving original resolution")
    
    # Preprocess
    img_tensor, proc_size = preprocess_image(img, processor, target_size)
    
    start_time = time.time()
    
    # Run DeepDream
    print("\nStarting DeepDream processing...")
    dreamed_img = run_deepdream_with_octaves(
        model, img_tensor, target_layers,
        steps_per_octave=200,  # More steps for richer details
        step_size=0.01,       # Slightly higher for stronger effect
        octaves=(-2, 0, 3),           # Single octave for clarity
        octave_scale=1.3,
        use_tiling=use_tiling,
        tile_size=tile_size,
        original_size=original_size if target_size is None else None
    )
    
    end_time = time.time()
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
    
    # Save results
    result_img = deprocess_image(
        dreamed_img, 
        target_size=original_size if target_size else None
    )
    save_image(result_img, "deepdream_dinov3_result.png")
    
    # Create comparison
    comparison = Image.new('RGB', (original_size[0]*2, original_size[1]))
    original_resized = img.resize(original_size)
    comparison.paste(original_resized, (0, 0))
    comparison.paste(result_img, (original_size[0], 0))
    save_image(comparison, "comparison_dinov3.png")
    
    print("\nDeepDream processing complete!")
    print(f"All outputs saved to '{output_dir}/' directory")

if __name__ == "__main__":
    # Target middle-late Stage 2 layers for maximum abstract feature beauty
    generate_deepdream_dinov3(
        image_path='./Albert_Wider_Priestergrab_in_Widnau_02.png',
        target_layers=[5, 10, 15],  # Sweet spot for abstract patterns
        use_tiling=True,
        tile_size=256
    )