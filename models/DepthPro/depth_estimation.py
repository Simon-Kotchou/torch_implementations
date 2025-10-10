import os
import requests
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from typing import Union, Optional, Dict, Tuple, List
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast


class EnhancedDepthEstimator:
    """
    An enhanced wrapper for DepthPro that provides high-resolution depth estimation
    with advanced visualization capabilities.
    
    Attributes:
        model: The DepthPro transformer model
        processor: Image processor for preparing inputs
        device: Device for inference (CPU/GPU)
        native_size: Native model input size (typically 1536x1536)
    """
    
    def __init__(
        self, 
        model_name: str = "apple/DepthPro-hf", 
        use_fov_model: bool = True,
        device: Optional[torch.device] = None,
        native_size: int = 1536
    ):
        """
        Initialize the EnhancedDepthEstimator with a DepthPro model.
        
        Args:
            model_name: Name or path of the pre-trained model
            use_fov_model: Whether to use the field-of-view estimation head
            device: Device for inference (default: GPU if available, else CPU)
            native_size: Native model resolution (DepthPro uses 1536x1536)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.native_size = native_size
        
        print(f"Loading DepthPro model on {self.device}...")
        self.processor = DepthProImageProcessorFast.from_pretrained(model_name)
        self.model = DepthProForDepthEstimation.from_pretrained(
            model_name, 
            use_fov_model=use_fov_model
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded successfully. Native resolution: {self.native_size}x{self.native_size}")
        
    def load_image(self, image_path_or_url: str) -> Image.Image:
        """
        Load an image from a file path or URL.
        
        Args:
            image_path_or_url: Path to an image file or a URL
            
        Returns:
            A PIL Image object
        """
        if image_path_or_url.startswith(('http://', 'https://')):
            return Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            return Image.open(image_path_or_url)
    
    def estimate_depth(
        self, 
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        preserve_aspect_ratio: bool = True,
        force_native_resolution: bool = True
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Estimate depth from a single image at high resolution.
        
        Args:
            image: Input image (file path, PIL Image, numpy array, or torch tensor)
            preserve_aspect_ratio: Whether to preserve aspect ratio during preprocessing
            force_native_resolution: Whether to force using the model's native resolution
            
        Returns:
            Dictionary containing depth estimation results and metadata
        """
        # Load the image if a path is provided
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Store original image size for reference
        if isinstance(image, Image.Image):
            original_size = image.size  # (width, height)
        elif isinstance(image, np.ndarray):
            original_size = (image.shape[1], image.shape[0])  # (width, height)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:  # (C, H, W)
                original_size = (image.shape[2], image.shape[1])
            else:  # (B, C, H, W)
                original_size = (image.shape[3], image.shape[2])
        
        # Process the image
        if preserve_aspect_ratio and force_native_resolution and isinstance(image, Image.Image):
            # Calculate target size while preserving aspect ratio
            w, h = image.size
            if w > h:
                new_w = self.native_size
                new_h = int(h * (self.native_size / w))
            else:
                new_h = self.native_size
                new_w = int(w * (self.native_size / h))
                
            # Resize the image
            image_resized = image.resize((new_w, new_h), Image.LANCZOS)
            
            # Create a background image (black) of native size
            background = Image.new('RGB', (self.native_size, self.native_size), (0, 0, 0))
            
            # Paste the resized image onto the background, centered
            offset = ((self.native_size - new_w) // 2, (self.native_size - new_h) // 2)
            background.paste(image_resized, offset)
            
            # Convert to tensor for the model
            inputs = self.processor(images=background, return_tensors="pt").to(self.device)
            
            # Store resize info for post-processing
            resize_info = {
                'original_size': original_size,
                'resized_size': (new_w, new_h),
                'offset': offset
            }
        else:
            # Standard processing without aspect ratio preservation
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            resize_info = None
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process the output based on original size
        if isinstance(image, Image.Image):
            target_size = (image.height, image.width)  # (height, width)
        elif isinstance(image, np.ndarray):
            target_size = image.shape[:2]  # (height, width)
        else:  # torch.Tensor
            if image.ndim == 3:  # (C, H, W)
                target_size = (image.shape[1], image.shape[2])
            else:  # (B, C, H, W)
                target_size = (image.shape[2], image.shape[3])
        
        results = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[target_size]
        )[0]
        
        # Extract raw depth map and metadata
        depth_map = results["predicted_depth"]  # shape (H, W)
        field_of_view = results.get("field_of_view")  # Horizontal FOV in degrees
        focal_length = results.get("focal_length")  # Focal length in pixels
        
        # Calculate additional depth statistics
        depth_np = depth_map.detach().cpu().numpy()
        depth_stats = {
            "min_depth": float(depth_np.min()),
            "max_depth": float(depth_np.max()),
            "mean_depth": float(depth_np.mean()),
            "median_depth": float(np.median(depth_np)),
            "std_depth": float(np.std(depth_np)),
            "percentile_5": float(np.percentile(depth_np, 5)),
            "percentile_95": float(np.percentile(depth_np, 95))
        }
        
        # Return comprehensive results
        return {
            "depth_map": depth_map,  # Original tensor
            "field_of_view": field_of_view,
            "focal_length": focal_length,
            "original_size": original_size,
            "resize_info": resize_info,
            "statistics": depth_stats,
            "raw_depth_np": depth_np  # Raw numpy array for further processing
        }
        
    def visualize(
        self, 
        depth_result: Dict[str, Union[torch.Tensor, float]],
        image: Optional[Union[str, Image.Image, np.ndarray]] = None,
        mode: str = "color",
        colormap: str = 'inferno',
        invert: bool = False,
        alpha: float = 1.0,
        show_histogram: bool = True,
        show_stats: bool = True,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        dpi: int = 150,
        figsize: Tuple[int, int] = (16, 10)
    ) -> np.ndarray:
        """
        Create enhanced visualizations for the depth estimation results.
        
        Args:
            depth_result: Result from estimate_depth
            image: Original image for overlay visualization (optional)
            mode: Visualization mode ('color', 'overlay', 'side', '3d', 'all')
            colormap: Matplotlib colormap for depth visualization
            invert: Whether to invert the depth map (far=bright, near=dark)
            alpha: Opacity for overlay mode
            show_histogram: Whether to show depth histogram
            show_stats: Whether to show depth statistics
            save_path: Path to save visualization (if None, don't save)
            show_plot: Whether to display the visualization
            dpi: DPI for saved plots
            figsize: Figure size for plots
            
        Returns:
            The visualization as a numpy array
        """
        # Extract depth map and metadata
        depth_map = depth_result["depth_map"]
        depth_np = depth_map.detach().cpu().numpy()
        
        # Normalize for visualization
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        if invert:
            depth_norm = 1.0 - depth_norm
        
        # Load original image if provided as path
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Convert original image to numpy if provided
        if image is not None:
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
                
            # Ensure image and depth map have the same dimensions
            if image_np.shape[:2] != depth_np.shape:
                if isinstance(image, Image.Image):
                    image = image.resize((depth_np.shape[1], depth_np.shape[0]), Image.LANCZOS)
                    image_np = np.array(image)
                else:
                    image_np = np.array(Image.fromarray(image_np).resize(
                        (depth_np.shape[1], depth_np.shape[0]), Image.LANCZOS))
        
        # Apply colormap to depth
        cmap = plt.get_cmap(colormap)
        depth_colored = cmap(depth_norm)
        depth_colored_np = (depth_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Create visualization based on mode
        if mode == 'overlay' or mode == 'all':
            if image is None:
                raise ValueError("Original image must be provided for overlay visualization")
                
            # Create overlay visualization
            overlay = image_np.copy()
            overlay = (overlay.astype(float) * (1 - alpha) + 
                      depth_colored_np.astype(float) * alpha).astype(np.uint8)
        
        # Setup figure layout based on mode
        if mode == 'side' or mode == 'all':
            if image is None:
                raise ValueError("Original image must be provided for side-by-side visualization")
            
            # Create figure with multiple subplots
            n_plots = 2  # Original + Depth
            if show_histogram:
                n_plots += 1
            
            fig = plt.figure(figsize=figsize)
            grid = plt.GridSpec(2, 6)
            
            # Original image
            ax1 = fig.add_subplot(grid[0, :3])
            ax1.imshow(image_np)
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Depth map
            ax2 = fig.add_subplot(grid[0, 3:])
            im = ax2.imshow(depth_norm, cmap=colormap)
            title = "Depth Map"
            if depth_result.get("field_of_view") is not None:
                title += f" (FOV: {depth_result['field_of_view']:.1f}°, f: {depth_result['focal_length']:.1f}px)"
            ax2.set_title(title)
            ax2.axis('off')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax2)
            cbar.set_label('Depth (normalized)', rotation=270, labelpad=15)
            
            # Histogram
            if show_histogram:
                ax3 = fig.add_subplot(grid[1, :3])
                ax3.hist(depth_np.flatten(), bins=100, color='steelblue', alpha=0.7)
                ax3.set_title('Depth Distribution')
                ax3.set_xlabel('Depth (meters)')
                ax3.set_ylabel('Pixel Count')
                ax3.grid(alpha=0.3)
            
            # Statistics
            if show_stats:
                ax4 = fig.add_subplot(grid[1, 3:])
                ax4.axis('off')
                stats_text = "Depth Statistics:\n\n"
                stats_text += f"Min depth: {depth_result['statistics']['min_depth']:.2f} m\n"
                stats_text += f"Max depth: {depth_result['statistics']['max_depth']:.2f} m\n"
                stats_text += f"Mean depth: {depth_result['statistics']['mean_depth']:.2f} m\n"
                stats_text += f"Median depth: {depth_result['statistics']['median_depth']:.2f} m\n"
                stats_text += f"Std deviation: {depth_result['statistics']['std_depth']:.2f} m\n"
                stats_text += f"5th percentile: {depth_result['statistics']['percentile_5']:.2f} m\n"
                stats_text += f"95th percentile: {depth_result['statistics']['percentile_95']:.2f} m\n"
                if depth_result.get("field_of_view") is not None:
                    stats_text += f"\nField of View: {depth_result['field_of_view']:.2f}°\n"
                    stats_text += f"Focal Length: {depth_result['focal_length']:.2f} px"
                    
                ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                         fontsize=10, verticalalignment='top')
            
            plt.tight_layout()
        
        elif mode == '3d' or mode == 'all':
            # Create 3D point cloud visualization
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Subsample points for faster rendering
            step = max(1, int(min(depth_np.shape) / 100))
            y, x = np.mgrid[0:depth_np.shape[0]:step, 0:depth_np.shape[1]:step]
            z = depth_np[::step, ::step]
            
            # Normalize for visualization
            z = (z - z.min()) / (z.max() - z.min() + 1e-8)
            if invert:
                z = 1.0 - z
            
            # Get colors for points if original image is available
            if image is not None:
                img_small = np.array(Image.fromarray(image_np).resize(
                    (x.shape[1], x.shape[0]), Image.LANCZOS))
                colors = img_small.reshape(-1, 3) / 255.0
            else:
                colors = cmap(z.flatten())[:, :3]
            
            # Plot 3D surface
            surf = ax.plot_surface(x, y, z, facecolors=colors.reshape(*x.shape, 3),
                                   rstride=1, cstride=1, shade=False)
            
            ax.set_title('3D Depth Visualization')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Depth')
            ax.view_init(elev=30, azim=45)
            
        elif mode == 'color':
            # Simple colormap visualization
            plt.figure(figsize=figsize)
            plt.imshow(depth_norm, cmap=colormap)
            title = "Depth Map"
            if depth_result.get("field_of_view") is not None:
                title += f" (FOV: {depth_result['field_of_view']:.1f}°, f: {depth_result['focal_length']:.1f}px)"
            plt.title(title)
            plt.axis('off')
            cbar = plt.colorbar()
            cbar.set_label('Depth (normalized)', rotation=270, labelpad=15)
            
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")
        
        # Save if requested
        if save_path:
            dirname = os.path.dirname(save_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Visualization saved to {save_path}")
            
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        # Return colored depth map for further use
        return depth_colored_np
    
    def process_and_visualize(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        mode: str = "side",
        colormap: str = 'turbo',
        invert: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        **kwargs
    ) -> Dict:
        """
        Convenience method to estimate depth and visualize in one step.
        
        Args:
            image: Input image
            mode: Visualization mode
            colormap: Colormap to use
            invert: Whether to invert the depth map
            save_path: Path to save visualization
            show_plot: Whether to display the plot
            **kwargs: Additional arguments for estimate_depth and visualize
            
        Returns:
            Dictionary containing depth estimation results and visualization
        """
        # Separate kwargs for estimate_depth and visualize
        depth_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['preserve_aspect_ratio', 'force_native_resolution']}
        
        vis_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['alpha', 'show_histogram', 'show_stats', 'dpi', 'figsize']}
        
        # First estimate depth
        depth_result = self.estimate_depth(image, **depth_kwargs)
        
        # Then visualize
        vis_result = self.visualize(
            depth_result,
            image=image,
            mode=mode,
            colormap=colormap,
            invert=invert,
            save_path=save_path,
            show_plot=show_plot,
            **vis_kwargs
        )
        
        # Return both results
        return {
            "depth_result": depth_result,
            "visualization": vis_result
        }
    
    def batch_process(
        self,
        image_list: List[Union[str, Image.Image, np.ndarray, torch.Tensor]],
        output_dir: str = "output",
        mode: str = "side",
        colormap: str = 'turbo',
        **kwargs
    ) -> List[Dict]:
        """
        Process a batch of images and save visualizations.
        
        Args:
            image_list: List of input images
            output_dir: Directory to save outputs
            mode: Visualization mode
            colormap: Colormap to use
            **kwargs: Additional arguments for process_and_visualize
            
        Returns:
            List of result dictionaries
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Separate kwargs for estimate_depth and visualize
        depth_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['preserve_aspect_ratio', 'force_native_resolution']}
        
        vis_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['alpha', 'show_histogram', 'show_stats', 'dpi', 'figsize']}
            
        results = []
        
        for i, image in enumerate(image_list):
            # Get base name if image is a file path
            if isinstance(image, str):
                basename = os.path.splitext(os.path.basename(image))[0]
            else:
                basename = f"image_{i}"
                
            # Process image
            save_path = os.path.join(output_dir, f"{basename}_depth.png")
            result = self.process_and_visualize(
                image,
                mode=mode,
                colormap=colormap,
                save_path=save_path,
                **depth_kwargs,
                **vis_kwargs
            )
            
            # Store result
            result["image_index"] = i
            result["basename"] = basename
            result["save_path"] = save_path
            results.append(result)
            
        return results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced DepthPro Depth Estimation")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="depth_output.png", help="Path to save output visualization")
    parser.add_argument("--mode", type=str, default="side", choices=["color", "overlay", "side", "3d", "all"], 
                       help="Visualization mode")
    parser.add_argument("--colormap", type=str, default="turbo", help="Colormap for visualization")
    parser.add_argument("--invert", action="store_true", help="Invert depth map (far=bright, near=dark)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Opacity for overlay mode")
    parser.add_argument("--no-display", action="store_true", help="Don't display the result")
    parser.add_argument("--native", action="store_true", help="Use native model resolution (1536x1536)")
    parser.add_argument("--no-stats", action="store_true", help="Don't show depth statistics")
    parser.add_argument("--no-histogram", action="store_true", help="Don't show depth histogram")
    args = parser.parse_args()
    
    # Initialize the estimator
    depth_estimator = EnhancedDepthEstimator()
    
    # Process and visualize
    result = depth_estimator.process_and_visualize(
        args.image,
        mode=args.mode,
        colormap=args.colormap,
        invert=args.invert,
        alpha=args.alpha,
        save_path=args.output,
        show_plot=not args.no_display,
        show_stats=not args.no_stats,
        show_histogram=not args.no_histogram,
        force_native_resolution=args.native
    )
    
    print(f"Depth analysis complete. Visualization saved to {args.output}")
    print("\nDepth Statistics:")
    for k, v in result["depth_result"]["statistics"].items():
        print(f"  {k}: {v:.2f}")
    if result["depth_result"]["field_of_view"] is not None:
        print(f"\nField of View: {result['depth_result']['field_of_view']:.2f}°")
        print(f"Focal Length: {result['depth_result']['focal_length']:.2f} px")