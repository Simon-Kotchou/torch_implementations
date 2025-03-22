import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
import timm
import time
import os

# For reproducibility
torch.manual_seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
output_dir = "deepdream_results"
os.makedirs(output_dir, exist_ok=True)

def download_image(url):
    """Download an image from a URL and convert to PIL Image."""
    return Image.open(urlopen(url))

def save_image(img, filename):
    """Save a PIL image or PyTorch tensor to file.
    This avoids Qt/display issues on some systems."""
    if isinstance(img, torch.Tensor):
        # Convert PyTorch tensor to numpy array
        if img.is_cuda:
            img = img.cpu()
        if len(img.shape) == 4:  # Batch dimension
            img = img.squeeze(0)
        img = img.permute(1, 2, 0).numpy()
        # Convert from [0,1] to [0,255]
        if img.max() <= 1:
            img = img * 255
        img = img.astype(np.uint8)
    
    # Convert numpy array to PIL Image for saving
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    else:  # Already a PIL Image
        pil_img = img
    
    # Create full path with output directory
    filepath = os.path.join(output_dir, filename)
    pil_img.save(filepath)
    print(f"Image saved to {filepath}")
    return filepath

def load_model():
    """Load ConvNeXt model with features_only to get intermediate activations.
    Configured specifically for ConvNeXt architecture."""
    model = timm.create_model(
        'convnext_xxlarge.clip_laion2b_soup_ft_in1k',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),  # Get all stages: 384, 768, 1536, 3072 channels
    )
    model = model.to(device)
    model.eval()
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_model_transforms():
    """Get the transforms required by the model."""
    model = timm.create_model(
        'convnext_xxlarge.clip_laion2b_soup_ft_in1k',
        pretrained=True
    )
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms

def calc_loss(activations, layer_indices):
    """Calculate loss optimized for ConvNeXt architecture.
    
    Args:
        activations: List of activation tensors from different model layers
        layer_indices: Indices of layers to use for loss calculation
    
    Returns:
        Scalar loss value
    """
    loss = 0
    for i, activation in enumerate(activations):
        if i in layer_indices:
            # For ConvNeXt, based on the architecture:
            # - Layer 0 (384 channels): Low-level features
            # - Layer 1 (768 channels): Mid-level features
            # - Layer 2 (1536 channels): Higher-level patterns (with 30 blocks)
            # - Layer 3 (3072 channels): Highest-level features
            
            # Maximize both mean and variance to create diverse patterns
            # Weight the loss components based on layer depth
            if i == 0:  # Stage 0 - low level features (edges, textures)
                layer_loss = activation.mean() + 0.05 * activation.std()
            elif i == 1:  # Stage 1 - mid level features
                layer_loss = activation.mean() + 0.1 * activation.std()
            elif i == 2:  # Stage 2 - higher level patterns (the massive 30-block stage)
                layer_loss = activation.mean() + 0.2 * activation.std()
            else:  # Stage 3 - highest level features
                layer_loss = activation.mean() + 0.15 * activation.std()
                
            loss += layer_loss
            
    return loss

class DeepDream:
    def __init__(self, model, layer_indices):
        """Initialize DeepDream with model and chosen layers."""
        self.model = model
        self.layer_indices = layer_indices
        
    def __call__(self, img, steps=100, step_size=0.01):
        """Run the DeepDream algorithm on an image."""
        img = img.clone()
        img.requires_grad = True
        
        for step in range(steps):
            # Clear previous gradients
            if img.grad is not None:
                img.grad.data.zero_()
            
            # Forward pass
            activations = self.model(img)
            
            # Calculate loss
            loss = calc_loss(activations, self.layer_indices)
            
            # Backward pass (compute gradients)
            loss.backward()
            
            # Normalize gradients (same as TF tutorial)
            gradients = img.grad.data
            gradients /= (torch.std(gradients) + 1e-8)
            
            # Update image using gradient ascent
            img.data = img.data + step_size * gradients
            
            # Clip to maintain valid image values
            img.data = torch.clamp(img.data, 0, 1)
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                
                # Save intermediate result every 20 steps
                if step > 0 and step % 40 == 0:
                    intermediate = deprocess_image(img.detach())
                    save_image(intermediate, f"step_{step}_intermediate.png")
                
        return img.detach()

def run_deepdream_simple(model, img, layer_indices, steps=100, step_size=0.01):
    """Run basic DeepDream without octaves or tiling."""
    deepdream = DeepDream(model, layer_indices)
    result = deepdream(img, steps=steps, step_size=step_size)
    
    # Save intermediate result
    intermediate_img = deprocess_image(result)
    save_image(intermediate_img, f"deepdream_simple.png")
    
    return result

def random_roll(img, max_roll):
    """Randomly shift the image to avoid tiled boundaries."""
    _, _, h, w = img.shape
    
    # Random shift
    dy = torch.randint(-max_roll, max_roll + 1, (1,)).item()
    dx = torch.randint(-max_roll, max_roll + 1, (1,)).item()
    
    # Roll image
    return torch.roll(img, shifts=(dy, dx), dims=(2, 3)), (dy, dx)

class TiledDeepDream:
    def __init__(self, model, layer_indices):
        """Initialize TiledDeepDream with model and chosen layers."""
        self.model = model
        self.layer_indices = layer_indices
        
    def __call__(self, img, tile_size=512):
        """Calculate gradients using tiled approach to handle large images."""
        b, c, h, w = img.shape
        
        # Random roll to avoid tile boundaries
        shifted_img, (dy, dx) = random_roll(img, tile_size // 8)
        
        # Initialize gradients to zero
        gradients = torch.zeros_like(shifted_img)
        
        # For each tile
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Ensure we don't go out of bounds
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Skip if tile is too small
                if y_end - y < 32 or x_end - x < 32:
                    continue
                
                # Extract tile
                tile = shifted_img[:, :, y:y_end, x:x_end].clone()
                
                # Calculate gradients for this tile
                tile.requires_grad = True
                
                # Forward pass
                activations = self.model(tile)
                
                # Calculate loss
                loss = calc_loss(activations, self.layer_indices)
                
                # Backward pass
                loss.backward()
                
                # Store gradients for this tile
                gradients[:, :, y:y_end, x:x_end] = tile.grad.data
                
                # Reset gradients for next tile
                if tile.grad is not None:
                    tile.grad.data.zero_()
        
        # Normalize gradients
        gradients = gradients / (torch.std(gradients) + 1e-8)
        
        # Un-shift gradients
        gradients = torch.roll(gradients, shifts=(-dy, -dx), dims=(2, 3))
        
        return gradients

def preprocess_image(image, transforms_fn, target_size=None):
    """Preprocess an image for the model with optional resizing.
    
    Args:
        image: PIL Image
        transforms_fn: Model-specific transforms
        target_size: Optional (height, width) to resize image
                     If None, keeps original size
    """
    # Get original size before any transforms
    original_size = image.size  # (width, height)
    
    # If target_size is specified, resize first
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # Apply model transforms
    tensor = transforms_fn(image).unsqueeze(0)
    return tensor.to(device), original_size

def deprocess_image(tensor, target_size=None):
    """Convert tensor back to PIL image with optional resizing.
    
    Args:
        tensor: PyTorch tensor with values in [0,1]
        target_size: Optional (width, height) for the output image
                     If None, uses the tensor's size
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL image
    array = tensor.permute(1, 2, 0).numpy() * 255
    img = Image.fromarray(array.astype(np.uint8))
    
    # Resize to target size if specified
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
        
    return img

def run_deepdream_with_octaves(model, img, layer_indices,
                              steps_per_octave=100, step_size=0.01,
                              octaves=range(-2,3), octave_scale=1.3,
                              use_tiling=True, tile_size=512,
                              original_size=None):
    """Run DeepDream with octaves for better results."""
    # Store the input shape for final resizing
    input_shape = img.shape[-2:]  # (H, W)
    
    # Determine base shape (either original size or input shape)
    if original_size is not None:
        # Convert (width, height) to (height, width)
        base_shape = (original_size[1], original_size[0])
    else:
        base_shape = input_shape
        
    img = img.clone()
    
    if use_tiling:
        # Create tiled deepdream object
        tiled_deepdream = TiledDeepDream(model, layer_indices)
    else:
        # Create regular deepdream object
        deepdream = DeepDream(model, layer_indices)
    
    # Save intermediate results for comparison
    octave_results = []
    
    for octave in octaves:
        # Scale the image based on the octave
        new_size = [int(s * (octave_scale ** octave)) for s in base_shape]
        
        # Don't go too small
        if min(new_size) < 32:
            continue
            
        img = torch.nn.functional.interpolate(img, size=new_size, mode='bilinear', align_corners=False)
        
        print(f"Processing octave {octave} with image size {img.shape[-2:]} pixels")
        
        if use_tiling:
            # For each step in this octave
            for step in range(steps_per_octave):
                # Calculate gradients using tiled approach
                gradients = tiled_deepdream(img, tile_size=min(tile_size, min(new_size)))
                
                # Update image (gradient ascent)
                img.data = img.data + step_size * gradients
                
                # Clip to maintain valid image values
                img.data = torch.clamp(img.data, 0, 1)
                
                if step % 10 == 0:
                    print(f"Octave {octave}, Step {step}")
        else:
            # Use regular DeepDream for this octave
            img = deepdream(img, steps=steps_per_octave, step_size=step_size)
        
        # Save intermediate result for this octave
        octave_img = deprocess_image(img)
        save_image(octave_img, f"octave_{octave}_result.png")
        octave_results.append((octave, octave_img))
    
    # Resize back to original size
    img = torch.nn.functional.interpolate(img, size=base_shape, mode='bilinear', align_corners=False)
    return img

def generate_deepdream(image_path='./gettyimages.jpg', layer_choices=[1, 2], use_tiling=False, tile_size=64):
    """Main function to run DeepDream with configurable parameters.
    
    Args:
        image_path: Path to input image
        layer_choices: Indices of feature maps to use for loss calculation
        use_tiling: Whether to use tiling approach (for larger images)
        tile_size: Size of tiles if using tiling
    """
    print(f"ConvNeXt DeepDream initialized with settings:")
    print(f"- Layer indices: {layer_choices}")
    print(f"- Use tiling: {use_tiling}")
    print(f"- Tile size: {tile_size}")
    
    # Load local image
    try:
        img = Image.open(image_path)
        print(f"Loaded image from '{image_path}'")
    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'")
        return
        
    original_size = img.size  # (width, height)
    print(f"Image size: {original_size[0]}x{original_size[1]} pixels")
    save_image(img, "original_image.png")
    
    # Load model and transforms
    model = load_model()
    transforms_fn = get_model_transforms()
    
    # Print model feature map info
    dummy_img = transforms_fn(img.resize((224, 224))).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_maps = model(dummy_img)
        print("\nConvNeXt feature maps architecture:")
        for i, fm in enumerate(feature_maps):
            print(f"Layer {i}: {fm.shape} - {fm.shape[1]} channels")
    
    # Determine if we should maintain original resolution
    # Only go up to certain size to avoid CUDA memory issues
    max_dimension = 1024
    preserve_resolution = max(original_size) <= max_dimension
    
    if preserve_resolution:
        print(f"Preserving original resolution: {original_size[0]}x{original_size[1]}")
        # Keep original resolution
        target_size = None
    else:
        # Scale down to avoid memory issues
        scale = max_dimension / max(original_size)
        target_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        print(f"Scaling to {target_size[0]}x{target_size[1]} to avoid memory issues")
    
    # Preprocess image
    img_tensor, original_size = preprocess_image(img, transforms_fn, target_size)
    
    start_time = time.time()
    
    # Run DeepDream with specified settings
    dreamed_img = run_deepdream_with_octaves(
        model, img_tensor, layer_choices, 
        steps_per_octave=500, step_size=0.01,
        octaves=[0], octave_scale=1.3,
        use_tiling=use_tiling, tile_size=tile_size,
        original_size=original_size
    )
    
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    # Convert back to PIL image and save
    result_img = deprocess_image(dreamed_img, target_size=original_size)
    print("Final DeepDream result:")
    save_image(result_img, "deepdream_result.png")
    
    # Save a side-by-side comparison
    comparison = Image.new('RGB', (original_size[0]*2, original_size[1]), color='white')
    original_resized = img.resize((original_size[0], original_size[1]))
    comparison.paste(original_resized, (0, 0))
    comparison.paste(result_img, (original_size[0], 0))
    save_image(comparison, "comparison.png")
    
    print("\nDeepDream processing complete!")
    print(f"All outputs saved to '{output_dir}/' directory")

if __name__ == "__main__":
    # Use layers 1 and 2 (768 and 1536 channels) for best results based on ConvNeXt architecture
    generate_deepdream(
        image_path='./gettyimages.jpg',
        layer_choices=[2],
        use_tiling=True, 
        tile_size=512
    )  