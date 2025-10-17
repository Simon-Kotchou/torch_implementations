import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast


class VideoDepthEstimator:
    """
    Streams depth estimation from a video file using DepthPro.
    """
    
    def __init__(
        self, 
        model_name="apple/DepthPro-hf", 
        use_fov_model=True, 
        process_size=384,  # Smaller process size to avoid memory issues
        display_size=640
    ):
        """
        Initialize the video depth estimator.
        
        Args:
            model_name: Name of the DepthPro model
            use_fov_model: Whether to use FOV estimation
            process_size: Size to resize frames for processing (square)
            display_size: Maximum dimension for display
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.process_size = process_size
        self.display_size = display_size
        
        print(f"Loading DepthPro model on {self.device}...")
        self.processor = DepthProImageProcessorFast.from_pretrained(model_name)
        self.model = DepthProForDepthEstimation.from_pretrained(
            model_name, 
            use_fov_model=use_fov_model
        ).to(self.device)
        self.model.eval()
        print("Model loaded successfully.")
    
    def process_frame(self, frame):
        """
        Process a single video frame to generate a depth map.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Colored depth map and metadata
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to square for processing (DepthPro expects square input)
        h, w = rgb_frame.shape[:2]
        # Create a square image (black padding)
        size = self.process_size
        square_img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Resize preserving aspect ratio
        if w > h:
            new_w = size
            new_h = int(h * (size / w))
            resized = cv2.resize(rgb_frame, (new_w, new_h))
            # Center in square
            offset_y = (size - new_h) // 2
            square_img[offset_y:offset_y+new_h, 0:new_w] = resized
        else:
            new_h = size
            new_w = int(w * (size / h))
            resized = cv2.resize(rgb_frame, (new_w, new_h))
            # Center in square
            offset_x = (size - new_w) // 2
            square_img[0:new_h, offset_x:offset_x+new_w] = resized
        
        # Convert to PIL Image
        pil_image = Image.fromarray(square_img)
        
        # Process image for the model
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Get depth prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process to get depth map at the original frame size
        results = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(frame.shape[0], frame.shape[1])]
        )[0]
        
        # Get depth map
        depth_map = results["predicted_depth"].detach().cpu().numpy()
        
        # Normalize and colorize
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        
        # Get metadata if available
        metadata = {}
        if "field_of_view" in results:
            metadata["field_of_view"] = float(results["field_of_view"])
        if "focal_length" in results:
            metadata["focal_length"] = float(results["focal_length"])
        
        return depth_colored, metadata

    def stream_video(self, video_path, output_path=None, frame_skip=2):
        """
        Stream depth estimation from a video file.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video (optional)
            frame_skip: Process every Nth frame for speed (default: 2)
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate resize dimensions for display
        if max(width, height) > self.display_size:
            scale = self.display_size / max(width, height)
            display_width = int(width * scale)
            display_height = int(height * scale)
        else:
            display_width, display_height = width, height
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps/frame_skip, (display_width*2, display_height))
        
        print(f"Processing video at {self.process_size}x{self.process_size} (processing)")
        print(f"Displaying at {display_width}x{display_height}")
        print(f"Processing every {frame_skip} frames")
        print("Press 'q' to quit, 's' to save a frame")
        
        frame_count = 0
        processed_count = 0
        last_depth = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for speed
            if frame_count % frame_skip != 0 and last_depth is not None:
                # Resize frame for display
                display_frame = cv2.resize(frame, (display_width, display_height))
                
                # Create side-by-side display
                combined = np.hstack((display_frame, last_depth))
                
                # Display the result
                cv2.imshow('Original | Depth', combined)
                
                # Write frame if output is enabled
                if output_path:
                    out.write(combined)
                
                # Handle keyboard input with a smaller wait time for skipped frames
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                continue
            
            # Process frame to get depth
            depth_colored, metadata = self.process_frame(frame)
            
            # Resize for display
            display_frame = cv2.resize(frame, (display_width, display_height))
            display_depth = cv2.resize(depth_colored, (display_width, display_height))
            
            # Add metadata text
            if metadata:
                text = []
                if "field_of_view" in metadata:
                    text.append(f"FOV: {metadata['field_of_view']:.1f}°")
                if "focal_length" in metadata:
                    text.append(f"f: {metadata['focal_length']:.1f}px")
                
                if text:
                    text_str = ", ".join(text)
                    cv2.putText(display_depth, text_str, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save for skipped frames
            last_depth = display_depth
            
            # Create side-by-side display
            combined = np.hstack((display_frame, display_depth))
            
            # Display the result
            cv2.imshow('Original | Depth', combined)
            
            # Write frame if output is enabled
            if output_path:
                out.write(combined)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"depth_frame_{processed_count:04d}.png"
                cv2.imwrite(save_path, combined)
                print(f"Saved frame to {save_path}")
            
            processed_count += 1
            
            # Print progress every 10 frames
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames")
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Total frames: {frame_count}, Processed frames: {processed_count}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Depth Estimation")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (optional)")
    parser.add_argument("--process-size", type=int, default=384, 
                        help="Size for processing (square, smaller values use less memory)")
    parser.add_argument("--display-size", type=int, default=640, help="Maximum dimension for display")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame for speed")
    args = parser.parse_args()
    
    # Create the estimator and process video
    estimator = VideoDepthEstimator(
        process_size=args.process_size,
        display_size=args.display_size
    )
    estimator.stream_video(args.video, args.output, frame_skip=args.frame_skip)