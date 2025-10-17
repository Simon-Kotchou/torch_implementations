import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
import time
import multiprocessing as mp # Use multiprocessing

# --- Worker Process Function ---
# This function runs in a separate process for each GPU.
def depth_worker(gpu_id, input_queue, output_queue, model_name, use_fov_model, process_size):
    try:
        # 1. Set CUDA device for this specific process
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[Worker {gpu_id}]: Initialized on {device}")

        # 2. Load model and processor *within this process*
        processor = DepthProImageProcessorFast.from_pretrained(model_name)
        model = DepthProForDepthEstimation.from_pretrained(
            model_name,
            use_fov_model=use_fov_model
        ).to(device)
        model.eval()
        print(f"[Worker {gpu_id}]: Model loaded.")

        while True:
            # 3. Get work from the input queue
            work_item = input_queue.get() # Blocking call

            # 4. Check for termination signal
            if work_item is None:
                print(f"[Worker {gpu_id}]: Termination signal received.")
                break

            frame_index, frame_data = work_item
            # print(f"[Worker {gpu_id}]: Processing frame {frame_index}") # Debugging

            try:
                 # --- Frame Processing Logic (Adapted from single GPU version) ---
                rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                original_h, original_w = rgb_frame.shape[:2]

                # Resize/Pad frame for model
                size = process_size
                if original_w >= original_h:
                    new_w = size; new_h = int(original_h * (size / original_w))
                else:
                    new_h = size; new_w = int(original_w * (size / original_h))
                resized_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                top_pad = (size - new_h) // 2; bottom_pad = size - new_h - top_pad
                left_pad = (size - new_w) // 2; right_pad = size - new_w - left_pad
                padded_frame = cv2.copyMakeBorder(resized_frame, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                pil_image = Image.fromarray(padded_frame)

                # Prepare input tensor
                inputs = processor(images=pil_image, return_tensors="pt").to(device)

                # Inference
                with torch.no_grad():
                    outputs = model(**inputs)

                # Post-process
                results = processor.post_process_depth_estimation(
                    outputs, target_sizes=[(original_h, original_w)]
                )[0]

                depth_map_tensor = results["predicted_depth"]
                depth_map_np = depth_map_tensor.detach().cpu().numpy() # Move result to CPU

                # Normalize and colorize
                min_depth, max_depth = depth_map_np.min(), depth_map_np.max()
                if max_depth - min_depth > 1e-8:
                    depth_norm = (depth_map_np - min_depth) / (max_depth - min_depth)
                else:
                    depth_norm = np.zeros_like(depth_map_np)
                depth_colored_bgr = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

                # Metadata
                metadata = {}
                if use_fov_model and "field_of_view" in results: metadata["field_of_view"] = float(results["field_of_view"])
                if use_fov_model and "focal_length" in results: metadata["focal_length"] = float(results["focal_length"])
                # --- End Frame Processing ---

                # 5. Put result (on CPU) into output queue
                # Ensure data sent through queue is CPU data (numpy arrays are fine)
                output_queue.put((frame_index, depth_colored_bgr, metadata))
                # print(f"[Worker {gpu_id}]: Finished frame {frame_index}") # Debugging

            except Exception as e:
                print(f"[Worker {gpu_id}]: Error processing frame {frame_index}: {e}")
                # Optionally put an error marker in the output queue
                output_queue.put((frame_index, None, {"error": str(e)}))


    except Exception as e:
        print(f"[Worker {gpu_id}]: Worker failed catastrophically: {e}")
    finally:
        print(f"[Worker {gpu_id}]: Exiting.")


# --- Main Class for Orchestration ---
class MultiGPUVideoDepthEstimator:
    def __init__(
        self,
        num_gpus,
        model_name="apple/DepthPro-hf",
        use_fov_model=True,
        process_size=384,
        display_size=640
    ):
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.use_fov_model = use_fov_model
        self.process_size = process_size
        self.display_size = display_size

        if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
            raise RuntimeError(f"Requires {num_gpus} CUDA GPUs, but found {torch.cuda.device_count()}.")

        # Use 'spawn' start method for CUDA compatibility in multiprocessing
        # Needs to be set *before* creating queues or processes
        if mp.get_start_method(allow_none=True) != 'spawn':
             mp.set_start_method('spawn', force=True)
             print("Set multiprocessing start method to 'spawn'")

        # Create communication queues (limit size to prevent memory explosion)
        self.input_queue = mp.Queue(maxsize=num_gpus * 2)
        self.output_queue = mp.Queue(maxsize=num_gpus * 2)
        self.workers = []

    def _resize_for_display(self, frame, target_width, target_height):
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    def start_workers(self):
        print("Starting worker processes...")
        for gpu_id in range(self.num_gpus):
            process = mp.Process(
                target=depth_worker,
                args=(
                    gpu_id,
                    self.input_queue,
                    self.output_queue,
                    self.model_name,
                    self.use_fov_model,
                    self.process_size
                ),
                daemon=True # Make workers daemons so they exit if main process crashes
            )
            self.workers.append(process)
            process.start()
            print(f"Worker for GPU {gpu_id} started (PID: {process.pid}).")

    def stop_workers(self):
        print("Stopping worker processes...")
        # Send termination signal (None) to each worker
        for _ in range(self.num_gpus):
            self.input_queue.put(None)

        # Wait for workers to finish
        for i, process in enumerate(self.workers):
            try:
                process.join(timeout=10) # Wait for 10 seconds
                if process.is_alive():
                    print(f"Warning: Worker {i} did not terminate gracefully, forcing termination.")
                    process.terminate() # Force terminate if needed
                else:
                     print(f"Worker {i} joined successfully.")
            except Exception as e:
                 print(f"Error joining worker {i}: {e}")
        self.workers = []
        print("All workers stopped.")


    def stream_video(self, video_path, output_path=None, frame_skip=1):
        # Start workers before processing
        self.start_workers()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            self.stop_workers()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Source Video: {source_width}x{source_height} @ {fps:.2f} FPS, ~{total_frames} frames")

        scale = self.display_size / max(source_width, source_height)
        display_width = int(source_width * scale)
        display_height = int(source_height * scale)
        output_width = display_width * 2
        output_height = display_height

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_fps = fps # Save at original FPS, display might skip
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
            if not out.isOpened(): print(f"Error opening video writer for {output_path}"); output_path = None
            else: print(f"Saving output to {output_path}")

        print(f"Displaying at {display_width}x{display_height} per side")
        print(f"Using {self.num_gpus} GPUs. Frame skip = {frame_skip}")
        print("Press 'q' to quit")

        frame_read_count = 0
        frame_sent_count = 0
        next_frame_to_display = 0
        results_buffer = {} # Buffer for out-of-order results {frame_idx: (depth, meta)}
        last_displayed_frame = None # Store the last successfully displayed frame
        last_displayed_depth = None

        try:
            while True:
                # --- Read Frames and Fill Input Queue ---
                can_read_more = True
                # Try to keep the input queue reasonably full
                while not self.input_queue.full():
                    ret, frame = cap.read()
                    if not ret:
                        can_read_more = False # End of video
                        break

                    frame_read_count += 1

                    # Apply frame skipping *before* putting in queue
                    if frame_skip <= 0 or frame_read_count % frame_skip == 0:
                        # Put a *copy* of the frame data into the queue
                        self.input_queue.put((frame_read_count -1, frame.copy())) # Use 0-based index
                        frame_sent_count += 1
                        # print(f"Sent frame {frame_read_count - 1} to input queue") # Debug
                    # We still need the original frame for display later if not skipped
                    # Store the *original* frame associated with the index being displayed
                    if frame_read_count -1 == next_frame_to_display:
                         last_displayed_frame = frame.copy()


                # --- Process Output Queue and Display Chronologically ---
                while not self.output_queue.empty() or next_frame_to_display in results_buffer:
                     # Try to get next result directly from queue if available
                     if next_frame_to_display not in results_buffer:
                          try:
                              # Use non-blocking get to avoid waiting if the next frame isn't ready
                              idx, depth, meta = self.output_queue.get_nowait()
                              results_buffer[idx] = (depth, meta)
                              # print(f"Received result for frame {idx}") # Debug
                          except mp.queues.Empty:
                              # If the next frame isn't in the buffer and not in the queue yet, break inner loop
                              # and wait for more results or read more frames.
                              if not can_read_more and self.output_queue.empty() and self.input_queue.empty():
                                   break # Exit display loop if EOF and queues empty
                              else:
                                   pass # Continue outer loop to read more or wait
                                   break # Break inner loop to avoid busy wait if next frame not ready


                     # Check if the *next chronological frame* is now in the buffer
                     if next_frame_to_display in results_buffer:
                          depth_result, meta_result = results_buffer.pop(next_frame_to_display)

                          # Retrieve the corresponding original frame (needs careful handling)
                          # For simplicity: Reread or store original frames if needed
                          # Here, we assume we might need to handle skipped frames visually
                          # Need a way to associate original frame with index if skipped
                          # Let's simplify: we'll display the *latest read frame* for simplicity
                          # A more robust solution would store original frames in a buffer too.

                          # Display logic
                          if depth_result is not None and last_displayed_frame is not None:
                              display_frame_orig = self._resize_for_display(last_displayed_frame, display_width, display_height)
                              display_depth = self._resize_for_display(depth_result, display_width, display_height)
                              last_displayed_depth = display_depth # Update last good depth

                              # Add metadata
                              if meta_result and 'error' not in meta_result:
                                   text = []
                                   if "field_of_view" in meta_result: text.append(f"FOV:{meta_result['field_of_view']:.1f}")
                                   if "focal_length" in meta_result: text.append(f"f:{meta_result['focal_length']:.1f}")
                                   if text:
                                        cv2.putText(display_depth, ", ".join(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                              elif meta_result and 'error' in meta_result:
                                   cv2.putText(display_depth, "ERROR", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


                              combined = np.hstack((display_frame_orig, display_depth))
                              cv2.imshow("Original | Depth (Multi-GPU)", combined)
                              if out: out.write(combined)

                          elif last_displayed_frame is not None and last_displayed_depth is not None:
                               # If current depth failed, show last good depth
                                display_frame_orig = self._resize_for_display(last_displayed_frame, display_width, display_height)
                                combined = np.hstack((display_frame_orig, last_displayed_depth))
                                cv2.imshow("Original | Depth (Multi-GPU)", combined)
                                if out: out.write(combined)


                          # Move to the next frame index
                          next_frame_to_display += 1 # Increment to look for next frame

                     else:
                          # If the next frame is not in buffer, break inner loop and wait
                           break


                # --- Handle Keyboard Input & Check Termination ---
                key = cv2.waitKey(1) & 0xFF # Crucial for display updates
                if key == ord('q'):
                    print("Quit key pressed.")
                    break

                # Check if video ended AND all sent frames have been processed and displayed
                if not can_read_more and self.input_queue.empty() and self.output_queue.empty() and not results_buffer:
                     # Make sure we displayed the very last frame index processed
                     if next_frame_to_display >= frame_sent_count:
                         print("End of video and processing queue empty.")
                         break
                     else:
                         # Still waiting for last results
                         time.sleep(0.01) # Small sleep to yield


        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            # --- Cleanup ---
            print("Cleaning up...")
            cap.release()
            if out:
                print("Releasing video writer...")
                out.release()
            cv2.destroyAllWindows()
            # Ensure workers are stopped properly
            self.stop_workers()
            print("Cleanup complete.")


# --- Main Execution Block ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-GPU Video Depth Estimation")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (e.g., output_multi.mp4)")
    parser.add_argument("--process-size", type=int, default=384, help="Square size for model processing (smaller=faster)")
    parser.add_argument("--display-size", type=int, default=640, help="Max width/height for displayed frames")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (1=process all)")
    parser.add_argument("--no-fov", action="store_true", help="Disable FOV/focal length estimation")
    args = parser.parse_args()

    # Set start method globally if needed (can be finicky)
    # try:
    #      if mp.get_start_method(allow_none=True) is None:
    #           mp.set_start_method('spawn')
    #           print("Set start method to 'spawn' globally.")
    # except RuntimeError:
    #      print("Start method already set.")


    estimator = MultiGPUVideoDepthEstimator(
        num_gpus=args.gpus,
        process_size=args.process_size,
        display_size=args.display_size,
        use_fov_model=(not args.no_fov)
    )

    estimator.stream_video(
        args.video,
        output_path=args.output,
        frame_skip=args.frame_skip
    )

    print("Main process finished.")