# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image
# import argparse
# import sys
# import os
# import numpy as np
# import cv2
# import time
# import gc

# class SmolVLMInference:
#     def __init__(self, model_path="HuggingFaceTB/SmolVLM2-500M-Instruct"):
#         """Initialize the SmolVLM model and processor."""
#         try:
#             self.processor = AutoProcessor.from_pretrained(model_path)
            
#             # First initialize on CPU, then move to GPU
#             self.model = AutoModelForImageTextToText.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.bfloat16,
#             )
            
#             # Explicitly move to GPU after initialization
#             if torch.cuda.is_available():
#                 self.model = self.model.to("cuda")
#                 self.device = torch.device("cuda")
#             else:
#                 self.device = torch.device("cpu")
                
#             print(f"Model loaded on {self.device}")
#         except Exception as e:
#             print(f"Error initializing model: {str(e)}")
#             sys.exit(1)

#     def extract_frames_by_interval(self, video_path, interval_seconds=10, max_frames=4):
#         """Extract frames at specified time intervals."""
#         try:
#             # Check if file exists
#             if not os.path.isfile(video_path):
#                 print(f"Video file not found: {video_path}")
#                 return []
                
#             # Open video file
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"Error: Could not open video file: {video_path}")
#                 return []
                
#             # Get video properties
#             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = cap.get(cv2.CAP_PROP_FPS)
            
#             duration = frame_count / fps if fps > 0 else 0
            
#             print(f"Video successfully loaded: {video_path}")
#             print(f"Total frames: {frame_count}")
#             print(f"Video dimensions: {width}x{height}")
#             print(f"FPS: {fps}")
#             print(f"Duration: {duration:.2f} seconds")
            
#             # Calculate frame positions at intervals
#             interval_frames = int(fps * interval_seconds)
            
#             # Create batches of frame indices
#             all_frame_indices = []
#             current_frame = 0
            
#             while current_frame < frame_count:
#                 batch_indices = []
#                 for _ in range(max_frames):
#                     if current_frame < frame_count:
#                         batch_indices.append(current_frame)
#                         current_frame += interval_frames
#                     else:
#                         break
                
#                 if batch_indices:
#                     all_frame_indices.append(batch_indices)
            
#             # Extract all batches
#             batches = []
#             for batch_idx, batch_indices in enumerate(all_frame_indices):
#                 batch_frames = []
#                 for idx in batch_indices:
#                     # Set frame position
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#                     ret, frame = cap.read()
                    
#                     if ret:
#                         # Convert BGR to RGB for PIL and resize to reduce memory
#                         # Resize to lower resolution to save memory
#                         target_width = 640  # Reduced resolution
#                         target_height = int(height * (target_width / width))
                        
#                         frame = cv2.resize(frame, (target_width, target_height))
#                         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                         pil_frame = Image.fromarray(rgb_frame)
                        
#                         # Calculate timestamp
#                         timestamp = idx / fps if fps > 0 else 0
#                         time_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
                        
#                         batch_frames.append((pil_frame, time_str, idx))
#                         print(f"Extracted frame {idx} at {time_str} with shape {rgb_frame.shape}")
#                     else:
#                         print(f"Failed to extract frame {idx}")
                
#                 if batch_frames:
#                     batches.append(batch_frames)
            
#             # Release video capture
#             cap.release()
            
#             print(f"Extracted {sum(len(batch) for batch in batches)} frames in {len(batches)} batches")
#             return batches
            
#         except Exception as e:
#             print(f"Error extracting video frames: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return []

#     def generate_response(self, messages, max_new_tokens=128):
#         """Generate response from the model."""
#         try:
#             inputs = self.processor.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 tokenize=True,
#                 return_dict=True,
#                 return_tensors="pt"
#             ).to(self.device, dtype=torch.bfloat16)

#             generated_ids = self.model.generate(
#                 **inputs,
#                 do_sample=False,
#                 max_new_tokens=max_new_tokens
#             )
            
#             generated_text = self.processor.batch_decode(
#                 generated_ids,
#                 skip_special_tokens=True
#             )[0]
            
#             return generated_text
#         except Exception as e:
#             print(f"Error generating response: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return None

#     def run_batch_inference(self, batch_frames, text_query):
#         """Run inference on a batch of frames."""
#         messages = []
#         content = []

#         # Add the text query
#         if text_query:
#             content.append({"type": "text", "text": text_query})

#         # Add frames with timestamps
#         for frame, time_str, _ in batch_frames:
#             content.append({"type": "text", "text": f"Time: {time_str}"})
#             content.append({"type": "image", "image": frame})

#         messages.append({
#             "role": "user",
#             "content": content
#         })
        
#         print(f"Processing batch with {len(batch_frames)} frames...")
#         response = self.generate_response(messages)
        
#         # Clean up to help with memory
#         del messages
#         del content
#         gc.collect()
#         torch.cuda.empty_cache()
        
#         return response

#     def process_video_in_batches(self, video_path, text_query, interval_seconds=10, frames_per_batch=2):
#         """Process a video in batches to avoid memory issues."""
#         # Extract frames in batches
#         batches = self.extract_frames_by_interval(video_path, interval_seconds, frames_per_batch)
        
#         if not batches:
#             print("No frames extracted from video.")
#             return "Could not process video."
        
#         # Process each batch
#         all_responses = []
#         for i, batch in enumerate(batches):
#             print(f"\nProcessing batch {i+1}/{len(batches)}")
            
#             # Get timestamp range for this batch
#             start_time = batch[0][1]  # First frame timestamp
#             end_time = batch[-1][1]   # Last frame timestamp
            
#             # Customize query for this segment
#             segment_query = f"{text_query} [Time segment: {start_time} to {end_time}]"
            
#             # Process batch
#             response = self.run_batch_inference(batch, segment_query)
            
#             if response:
#                 # Format response with timestamp information
#                 formatted_response = f"--- Time segment: {start_time} to {end_time} ---\n{response}\n"
#                 all_responses.append(formatted_response)
#                 print(f"Completed batch {i+1}/{len(batches)}")
#             else:
#                 all_responses.append(f"--- Time segment: {start_time} to {end_time} ---\nError processing this segment\n")
#                 print(f"Error in batch {i+1}/{len(batches)}")
            
#             # Brief pause to allow memory cleanup
#             time.sleep(1)
        
#         # Combine all responses
#         full_response = "\n".join(all_responses)
#         return full_response

# def main():
#     parser = argparse.ArgumentParser(description="SmolVLM2 Batch Video Inference Script")
#     parser.add_argument("--type", choices=["image", "video"], required=True,
#                       help="Type of input (image or video)")
#     parser.add_argument("--input", nargs="+", required=True,
#                       help="Path to input file(s)")
#     parser.add_argument("--query", type=str, required=True,
#                       help="Text query for the model")
#     parser.add_argument("--model_path", type=str,
#                       default="HuggingFaceTB/SmolVLM2-500M-Instruct",
#                       help="Path to model checkpoint")
#     parser.add_argument("--interval", type=int, default=10,
#                       help="Interval between frames in seconds (for video)")
#     parser.add_argument("--batch_size", type=int, default=2,
#                       help="Number of frames to process in each batch (for video)")

#     args = parser.parse_args()

#     # Initialize inference
#     inference = SmolVLMInference(args.model_path)
    
#     # Run inference based on type
#     if args.type == "video":
#         result = inference.process_video_in_batches(
#             args.input[0], 
#             args.query,
#             args.interval,
#             args.batch_size
#         )
#     else:
#         print("Image processing not implemented in batch mode.")
#         return
    
#     if result:
#         print("\nCombined Model Response:")
#         print(result)
        
#         # Save to file
#         output_file = "video_analysis_results.txt"
#         with open(output_file, "w") as f:
#             f.write(result)
#         print(f"\nResults saved to {output_file}")

# if __name__ == "__main__":
#     main()
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse
import sys
import os
import numpy as np
import cv2
import time
import gc

class SmolVLMInference:
    def __init__(self, model_path="HuggingFaceTB/SmolVLM2-500M-Instruct", use_smaller_model=False, use_quantization=False):
        """Initialize the SmolVLM model and processor."""
        try:
            # Use the smaller 256M model if requested
            if use_smaller_model:
                model_path = "HuggingFaceTB/SmolVLM-Instruct-250M"
                print(f"Using smaller 250M model: {model_path}")
            else:
                print(f"Using model: {model_path}")
                
            # Initialize processor
            self.processor = AutoProcessor.from_pretrained(model_path)
            
            # Load the model with appropriate optimizations
            if torch.cuda.is_available():
                print("CUDA is available. Loading model on GPU...")
                
                # Only try quantization if explicitly requested
                if use_quantization:
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        print("Using 8-bit quantization")
                        
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            model_path,
                            quantization_config=quantization_config
                        )
                    except Exception as e:
                        print(f"Quantization failed, falling back to standard loading: {e}")
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16
                        ).to("cuda")
                else:
                    # Standard GPU loading without quantization
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16
                    ).to("cuda")
                    
                self.device = torch.device("cuda")
            else:
                # CPU loading
                print("CUDA not available. Loading model on CPU...")
                self.model = AutoModelForImageTextToText.from_pretrained(model_path)
                self.device = torch.device("cpu")
                
            print(f"Model successfully loaded on {self.device}")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def extract_frames_by_interval(self, video_path, interval_seconds=10, max_frames=2):
        """Extract frames at specified time intervals."""
        try:
            # Check if file exists
            if not os.path.isfile(video_path):
                print(f"Video file not found: {video_path}")
                return []
                
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return []
                
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"Video successfully loaded: {video_path}")
            print(f"Total frames: {frame_count}")
            print(f"Video dimensions: {width}x{height}")
            print(f"FPS: {fps}")
            print(f"Duration: {duration:.2f} seconds")
            
            # Calculate frame positions at intervals
            interval_frames = int(fps * interval_seconds)
            
            # Create batches of frame indices
            all_frame_indices = []
            current_frame = 0
            
            while current_frame < frame_count:
                batch_indices = []
                for _ in range(max_frames):
                    if current_frame < frame_count:
                        batch_indices.append(current_frame)
                        current_frame += interval_frames
                    else:
                        break
                
                if batch_indices:
                    all_frame_indices.append(batch_indices)
            
            # Extract all batches
            batches = []
            for batch_idx, batch_indices in enumerate(all_frame_indices):
                batch_frames = []
                for idx in batch_indices:
                    # Set frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Convert BGR to RGB for PIL and resize to reduce memory usage
                        target_width = 640
                        target_height = 360
                        
                        frame = cv2.resize(frame, (target_width, target_height))
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(rgb_frame)
                        
                        # Calculate timestamp
                        timestamp = idx / fps if fps > 0 else 0
                        time_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
                        
                        batch_frames.append((pil_frame, time_str, idx))
                        print(f"Extracted frame {idx} at {time_str} with shape {rgb_frame.shape}")
                    else:
                        print(f"Failed to extract frame {idx}")
                
                if batch_frames:
                    batches.append(batch_frames)
            
            # Release video capture
            cap.release()
            
            print(f"Extracted {sum(len(batch) for batch in batches)} frames in {len(batches)} batches")
            return batches
            
        except Exception as e:
            print(f"Error extracting video frames: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def generate_response(self, messages, max_new_tokens=128):
        """Generate response from the model."""
        try:
            # Apply chat template to format conversation
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Convert to bfloat16 if on CUDA
            if self.device.type == "cuda":
                inputs = {k: v.to(dtype=torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens
            )
            
            # Decode the generated text
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            return generated_text
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run_batch_inference(self, batch_frames, text_query):
        """Run inference on a batch of frames."""
        try:
            messages = []
            content = []

            # Add frames with timestamps
            for i, (frame, time_str, _) in enumerate(batch_frames):
                # For each frame, create a new message with the query and timestamp
                frame_content = [
                    {"type": "text", "text": f"{text_query} [Time: {time_str}]"},
                    {"type": "image", "image": frame}
                ]
                content.extend(frame_content)

            messages.append({
                "role": "user",
                "content": content
            })
            
            print(f"Processing batch with {len(batch_frames)} frames...")
            response = self.generate_response(messages)
            
            # Clean up to help with memory
            del messages
            del content
            gc.collect()
            torch.cuda.empty_cache()
            
            return response
        except Exception as e:
            print(f"Error in batch inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error processing batch: {str(e)}"

    def process_video_in_batches(self, video_path, text_query, interval_seconds=10, frames_per_batch=2):
        """Process a video in batches to avoid memory issues."""
        # Extract frames in batches
        batches = self.extract_frames_by_interval(video_path, interval_seconds, frames_per_batch)
        
        if not batches:
            print("No frames extracted from video.")
            return "Could not process video."
        
        # Process each batch
        all_responses = []
        for i, batch in enumerate(batches):
            print(f"\nProcessing batch {i+1}/{len(batches)}")
            
            # Get timestamp range for this batch
            start_time = batch[0][1]  # First frame timestamp
            end_time = batch[-1][1]   # Last frame timestamp
            
            # Process batch
            response = self.run_batch_inference(batch, text_query)
            
            if response:
                # Format response with timestamp information
                formatted_response = f"--- Time segment: {start_time} to {end_time} ---\n{response}\n"
                all_responses.append(formatted_response)
                print(f"Completed batch {i+1}/{len(batches)}")
            else:
                all_responses.append(f"--- Time segment: {start_time} to {end_time} ---\nError processing this segment\n")
                print(f"Error in batch {i+1}/{len(batches)}")
            
            # Brief pause to allow memory cleanup
            time.sleep(1)
        
        # Combine all responses
        full_response = "\n".join(all_responses)
        return full_response

def main():
    parser = argparse.ArgumentParser(description="SmolVLM Batch Video Inference Script")
    parser.add_argument("--type", choices=["image", "video"], required=True,
                      help="Type of input (image or video)")
    parser.add_argument("--input", nargs="+", required=True,
                      help="Path to input file(s)")
    parser.add_argument("--query", type=str, required=True,
                      help="Text query for the model")
    parser.add_argument("--model_path", type=str,
                      default="HuggingFaceTB/SmolVLM2-500M-Instruct",
                      help="Path to model checkpoint")
    parser.add_argument("--interval", type=int, default=10,
                      help="Interval between frames in seconds (for video)")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Number of frames to process in each batch (for video)")
    parser.add_argument("--use_small_model", action="store_true",
                      help="Use the smaller 250M model for better memory efficiency")
    parser.add_argument("--use_quantization", action="store_true",
                      help="Use 8-bit quantization (requires bitsandbytes library)")
    parser.add_argument("--output_file", type=str, default="video_analysis_results.txt",
                      help="Path to save output results")

    args = parser.parse_args()

    # Initialize inference
    inference = SmolVLMInference(
        args.model_path, 
        args.use_small_model,
        args.use_quantization
    )
    
    # Run inference based on type
    if args.type == "video":
        result = inference.process_video_in_batches(
            args.input[0], 
            args.query,
            args.interval,
            args.batch_size
        )
    else:
        print("Image processing not implemented in batch mode.")
        return
    
    if result:
        print("\nCombined Model Response:")
        print(result)
        
        # Save to file
        output_file = args.output_file
        with open(output_file, "w") as f:
            f.write(result)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()