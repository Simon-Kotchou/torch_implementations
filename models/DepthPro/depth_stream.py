import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
import time # Added for simple FPS calculation

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
            display_size: Maximum dimension for display side (original or depth)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.process_size = process_size
        self.display_size = display_size # This is max width/height for EACH side (original, depth)

        print(f"Loading DepthPro model on {self.device}...")
        self.processor = DepthProImageProcessorFast.from_pretrained(model_name)
        self.model = DepthProForDepthEstimation.from_pretrained(
            model_name,
            use_fov_model=use_fov_model
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded successfully. Processing at {self.process_size}x{self.process_size}")

    def _resize_for_display(self, frame, target_width, target_height):
        """ Resizes a frame for display using OpenCV """
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    def process_frame(self, frame, use_fov_model=True):
        """
        Process a single video frame to generate a depth map.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Colored depth map (BGR numpy array) and metadata dictionary
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_h, original_w = rgb_frame.shape[:2]

        # --- Prepare frame for model (resize with padding) ---
        size = self.process_size
        # Calculate new dimensions maintaining aspect ratio
        if original_w >= original_h:
            new_w = size
            new_h = int(original_h * (size / original_w))
        else:
            new_h = size
            new_w = int(original_w * (size / original_h))

        # Resize
        resized_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR) # Use INTER_LINEAR for speed

        # Create square canvas and paste resized frame
        top_pad = (size - new_h) // 2
        bottom_pad = size - new_h - top_pad
        left_pad = (size - new_w) // 2
        right_pad = size - new_w - left_pad

        # Use cv2.copyMakeBorder for padding - often efficient
        padded_frame = cv2.copyMakeBorder(resized_frame, top_pad, bottom_pad, left_pad, right_pad,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0]) # Pad with black

        # Convert padded frame to PIL Image
        pil_image = Image.fromarray(padded_frame)

        # --- Process image for the model ---
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # --- Get depth prediction ---
        with torch.no_grad():
            outputs = self.model(**inputs)

        # --- Post-process to get depth map at the *original* frame size ---
        # This step resizes the prediction back to the original video dimensions
        results = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(original_h, original_w)] # Target original H, W
        )[0]

        # Get depth map tensor and convert to numpy
        depth_map_tensor = results["predicted_depth"] # Shape (H, W)
        depth_map_np = depth_map_tensor.detach().cpu().numpy()

        # --- Normalize and colorize ---
        min_depth, max_depth = depth_map_np.min(), depth_map_np.max()
        if max_depth - min_depth > 1e-8:
             depth_norm = (depth_map_np - min_depth) / (max_depth - min_depth)
        else:
             depth_norm = np.zeros_like(depth_map_np) # Avoid division by zero if depth is flat

        depth_colored_bgr = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

        # --- Get metadata ---
        metadata = {}
        if use_fov_model and "field_of_view" in results: # Only access if requested and available
             metadata["field_of_view"] = float(results["field_of_view"])
        if use_fov_model and "focal_length" in results:
             metadata["focal_length"] = float(results["focal_length"])

        return depth_colored_bgr, metadata

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
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total frames if possible

        print(f"Source Video: {source_width}x{source_height} @ {fps:.2f} FPS, ~{total_frames} frames")

        # --- Calculate resize dimensions for display ---
        scale = self.display_size / max(source_width, source_height)
        display_width = int(source_width * scale)
        display_height = int(source_height * scale)

        # Output video width will be double the single display width
        output_width = display_width * 2
        output_height = display_height

        # --- Initialize video writer if output path is provided ---
        out = None
        if output_path:
            # Use mp4v codec for .mp4 files
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Adjust FPS for frame skipping
            output_fps = fps / frame_skip if frame_skip > 0 else fps
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
            if not out.isOpened():
                print(f"Error: Could not open video writer for {output_path}")
                output_path = None # Disable saving if writer fails
            else:
                 print(f"Saving output to {output_path} at {output_width}x{output_height}, {output_fps:.2f} FPS")


        print(f"Displaying at {display_width}x{display_height} per side (WxH)")
        print(f"Processing every {frame_skip} frames (frame_skip=1 means process all)")
        print("Press 'q' to quit, 's' to save a snapshot")

        frame_count = 0
        processed_count = 0
        last_display_depth = None # Store the *last successfully processed* depth map resized for display
        start_time = time.time()
        display_interval = 2 # seconds to display FPS

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame_count += 1

            # Resize original frame ONCE for display
            display_frame = self._resize_for_display(frame, display_width, display_height)

            current_depth_for_display = None # Placeholder for depth map of this frame

            # --- Process frame or use last depth ---
            if frame_skip <= 0 or frame_count % frame_skip == 0:
                process_start_time = time.time()
                # Process this frame
                try:
                    depth_colored, metadata = self.process_frame(frame)
                    # Resize the *processed* depth map for display
                    current_depth_for_display = self._resize_for_display(depth_colored, display_width, display_height)
                    last_display_depth = current_depth_for_display # Update cache
                    processed_count += 1
                    process_end_time = time.time()
                    print(f"Frame {frame_count}: Processed in {process_end_time - process_start_time:.3f}s")

                    # Add metadata text to the *current* depth display
                    if metadata:
                        text = []
                        if "field_of_view" in metadata: text.append(f"FOV:{metadata['field_of_view']:.1f}")
                        if "focal_length" in metadata: text.append(f"f:{metadata['focal_length']:.1f}")
                        if text:
                            text_str = ", ".join(text)
                            cv2.putText(current_depth_for_display, text_str, (10, 20), # Adjusted position
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Smaller font

                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    # If processing fails, reuse last depth if available
                    if last_display_depth is not None:
                        current_depth_for_display = last_display_depth
                    else:
                        # Create a black frame if no depth available yet
                        current_depth_for_display = np.zeros_like(display_frame)

            else:
                # Skip processing, use the last processed depth map for display
                if last_display_depth is not None:
                    current_depth_for_display = last_display_depth
                else:
                    # If no depth processed yet, show black
                    current_depth_for_display = np.zeros_like(display_frame)


            # --- Combine and display ---
            if current_depth_for_display is not None:
                combined = np.hstack((display_frame, current_depth_for_display))

                # Calculate and display FPS periodically
                elapsed_time = time.time() - start_time
                if elapsed_time > display_interval:
                    actual_fps = processed_count / elapsed_time
                    print(f"Avg Processing FPS: {actual_fps:.2f}")
                    start_time = time.time()
                    processed_count = 0 # Reset counter for next interval

                cv2.imshow('Original | Depth', combined)

                # Write frame if output is enabled
                if out is not None:
                    out.write(combined)

            # --- Handle keyboard input ---
            key = cv2.waitKey(1) & 0xFF # Wait 1ms - crucial for display update
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                if current_depth_for_display is not None:
                    save_path = f"snapshot_frame_{frame_count:05d}.png"
                    cv2.imwrite(save_path, combined)
                    print(f"Saved snapshot to {save_path}")


        # --- Release resources ---
        cap.release()
        if out is not None:
            print(f"Releasing video writer for {output_path}")
            out.release()
        cv2.destroyAllWindows()
        print("Processing finished.")


# Example usage from command line
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video Depth Estimation Streamer")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (e.g., output.mp4)")
    parser.add_argument("--process-size", type=int, default=384,
                        help="Square size for model processing (smaller=faster, e.g., 256)")
    parser.add_argument("--display-size", type=int, default=640,
                        help="Maximum width/height for the displayed original/depth frame")
    parser.add_argument("--frame-skip", type=int, default=2,
                        help="Process every Nth frame (1=process all, 0=process all)")
    parser.add_argument("--no-fov", action="store_true", help="Disable FOV/focal length estimation head (saves minor computation)")
    args = parser.parse_args()

    # Create the estimator and process video
    estimator = VideoDepthEstimator(
        process_size=args.process_size,
        display_size=args.display_size,
        use_fov_model=(not args.no_fov) # Use FOV unless disabled
    )
    estimator.stream_video(
        args.video,
        output_path=args.output,
        frame_skip=args.frame_skip
    )