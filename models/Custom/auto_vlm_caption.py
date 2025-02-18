import json
import re
import random
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class StructuredCaptionGenerator:
    """
    Generates structured JSON captions for videos using the approach described in HunyuanVideo.
    Uses a Vision-Language Model to create multi-dimensional descriptions.
    """
    def __init__(
        self,
        vlm_model_path: str,
        device: str = "cuda",
        max_length: int = 2048,
    ):
        self.device = device
        self.vlm_model = self._load_vlm_model(vlm_model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(vlm_model_path)
        self.max_length = max_length
        
        # Caption components to generate
        self.caption_components = [
            "short_description",
            "dense_description",
            "background",
            "style", 
            "shot_type",
            "lighting",
            "atmosphere",
            "camera_movement"
        ]
        
        # Default prompt templates
        self.prompt_templates = {
            "base": (
                "Generate a detailed structured caption for this video. "
                "Analyze the content and provide information in JSON format "
                "with the following fields: short_description, dense_description, "
                "background, style, shot_type, lighting, atmosphere, and camera_movement."
            ),
            "short_description": "Provide a brief one-sentence summary of the main content.",
            "dense_description": "Describe the scene in detail including objects, actions, and visual elements.",
            "background": "Describe the environment or setting.",
            "style": "Identify the visual style (documentary, cinematic, realistic, animation, etc.).",
            "shot_type": "Identify the camera shot type (aerial, close-up, medium, long shot, etc.).",
            "lighting": "Describe the lighting conditions.",
            "atmosphere": "Describe the mood or atmosphere of the scene.",
            "camera_movement": "Identify camera movements (static, pan, tilt, zoom, handheld, etc.)."
        }
        
        # Camera movement classifier model if available
        self.camera_movement_classifier = None
        
    def _load_vlm_model(self, model_path, device):
        """Load the Vision-Language Model"""
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        model.eval()
        return model
    
    def _load_camera_movement_classifier(self, model_path):
        """Load dedicated camera movement classifier model"""
        # This would be implemented based on the specific model architecture
        # The paper mentions a classifier supporting 14 movement types
        pass
    
    def generate_caption_from_video_frames(
        self, 
        video_frames: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Generate structured caption from video frames
        
        Args:
            video_frames: Video frames tensor [T, C, H, W]
            metadata: Optional metadata about the video
            
        Returns:
            Structured caption in JSON format
        """
        # Process video with VLM to get base descriptions
        prompt = self.prompt_templates["base"]
        
        # In a real implementation, we would process the video frames through the VLM
        # Here we simulate the output of a VLM
        raw_caption = self._process_with_vlm(video_frames, prompt)
        
        # Parse structured components
        structured_caption = self._parse_to_json(raw_caption)
        
        # Add camera movement prediction if available
        if self.camera_movement_classifier is not None:
            camera_movement = self._predict_camera_movement(video_frames)
            structured_caption["camera_movement"] = camera_movement
            
        # Incorporate metadata if provided
        if metadata:
            # Add metadata tags, source information, etc.
            structured_caption["metadata"] = metadata
            
        return structured_caption
    
    def _process_with_vlm(self, video_frames, prompt):
        """Process video frames with Vision-Language Model"""
        # This would call the actual VLM with the frames and prompt
        # Simulated output for this implementation
        return {
            "short_description": "A person walking through a forest.",
            "dense_description": "A young woman with a backpack walks along a forest trail. The sunlight filters through the trees, creating dappled patterns on the ground. She pauses occasionally to look around at the scenery.",
            "background": "Dense forest with tall pine trees and undergrowth. A well-maintained hiking trail winds through the forest floor covered with fallen leaves.",
            "style": "documentary, realistic, natural",
            "shot_type": "medium shot, tracking shot",
            "lighting": "natural daylight, dappled sunlight through tree canopy",
            "atmosphere": "peaceful, serene, contemplative",
        }
    
    def _predict_camera_movement(self, video_frames):
        """Predict camera movement types"""
        # This would use the dedicated classifier mentioned in the paper
        # Simulated output for this implementation
        movement_types = [
            "static", "pan left", "pan right", "tilt up", "tilt down",
            "zoom in", "zoom out", "handheld", "tracking", "dolly",
            "crane up", "crane down", "arc left", "arc right"
        ]
        return random.choice(movement_types)
    
    def _parse_to_json(self, caption_dict):
        """Ensure caption is in proper JSON format"""
        # In a real implementation, this would parse text to JSON
        # Here we already have a dictionary
        return caption_dict
    
    def create_training_caption(self, structured_caption: Dict) -> str:
        """
        Create training caption from structured JSON
        
        Args:
            structured_caption: Caption in structured JSON format
            
        Returns:
            Training caption string with optional component dropout
        """
        components = []
        
        # Apply random dropout to create variation
        available_components = list(self.caption_components)
        
        # 70% chance to use all components, 30% chance to drop some
        if random.random() < 0.3:
            # Drop 1-3 random components
            num_to_drop = random.randint(1, min(3, len(available_components) - 2))
            # Always keep short_description and dense_description
            droppable = [c for c in available_components 
                         if c not in ["short_description", "dense_description"]]
            to_drop = random.sample(droppable, num_to_drop)
            available_components = [c for c in available_components if c not in to_drop]
        
        # Add components with 90% probability of including descriptions
        for component in available_components:
            if component in structured_caption:
                if component in ["short_description", "dense_description"]:
                    # Higher probability to include descriptions
                    if random.random() < 0.9:
                        components.append(structured_caption[component])
                else:
                    components.append(structured_caption[component])
        
        # Join components with period separation
        caption = ". ".join(c for c in components if c)
        if not caption.endswith("."):
            caption += "."
            
        return caption


class PromptRewriter:
    """
    Implements the prompt rewriting mechanism from HunyuanVideo.
    Uses LLM to adapt user prompts to model-preferred format.
    """
    def __init__(
        self, 
        llm_model_path: str,
        device: str = "cuda",
        use_lora: bool = True,
    ):
        self.device = device
        self.use_lora = use_lora
        
        # Load LLM model
        self.model, self.tokenizer = self._load_model(llm_model_path)
        
        # Example prompts for in-context learning
        self.example_pairs = [
            {
                "user": "make a video of a cat playing",
                "rewritten": "A fluffy cat playing with a ball of yarn on a carpeted floor. The cat enthusiastically paws at the yarn, occasionally pouncing and rolling around. The scene is captured in a medium shot with soft natural lighting creating a warm, playful atmosphere."
            },
            {
                "user": "show me space",
                "rewritten": "A stunning view of distant galaxies and nebulae in outer space. The cosmic scene features swirling clouds of colorful gas, twinkling stars, and celestial bodies against the vast blackness of space. The camera slowly pans across the breathtaking cosmic landscape, captured in a wide-angle shot with dramatic lighting that emphasizes the mysterious and awe-inspiring atmosphere."
            }
        ]
        
        # System prompt for rewriting
        self.system_prompt = (
            "You are an expert prompt engineer for video generation. "
            "Your task is to rewrite brief user prompts into detailed, "
            "well-structured prompts that will produce high-quality videos. "
            "The rewritten prompts should:\n"
            "1. Maintain the original intent and subject matter\n"
            "2. Add relevant details about setting, lighting, camera movement, and style\n"
            "3. Use clear, descriptive language\n"
            "4. Follow a consistent structure\n"
            "5. Be 2-4 sentences long\n\n"
            "Rewrite the user's prompt to create the best possible video generation prompt."
        )
        
    def _load_model(self, model_path):
        """Load LLM model for prompt rewriting"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.use_lora:
            # Load base model with LoRA weights for prompt rewriting
            from peft import PeftModel, PeftConfig
            
            # This is a simplified version - actual implementation would
            # load the specific LoRA configuration for prompt rewriting
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            model = PeftModel.from_pretrained(base_model, model_path + "-lora")
        else:
            # Load standard model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            
        model.eval()
        return model, tokenizer
    
    def rewrite_prompt(self, user_prompt: str) -> str:
        """
        Rewrite user prompt to model-preferred format
        
        Args:
            user_prompt: Original user prompt
            
        Returns:
            Rewritten prompt
        """
        # Construct full prompt with system instructions and examples
        full_prompt = self._construct_prompt(user_prompt)
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # Generate completion
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
        # Decode completion
        completion = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract rewritten prompt
        rewritten = self._extract_rewritten_prompt(completion, full_prompt)
        
        # Apply self-revision if needed
        if random.random() < 0.3:  # 30% chance to perform self-revision
            rewritten = self._self_revise(user_prompt, rewritten)
            
        return rewritten
    
    def _construct_prompt(self, user_prompt: str) -> str:
        """Construct full prompt with system instructions and examples"""
        prompt_parts = [self.system_prompt, "\n\nExamples:"]
        
        # Add example pairs
        for example in self.example_pairs:
            prompt_parts.append(f"\nUser: {example['user']}")
            prompt_parts.append(f"Rewritten: {example['rewritten']}")
            
        # Add user prompt
        prompt_parts.append(f"\nUser: {user_prompt}")
        prompt_parts.append("Rewritten:")
        
        return "\n".join(prompt_parts)
    
    def _extract_rewritten_prompt(self, completion: str, full_prompt: str) -> str:
        """Extract rewritten prompt from model completion"""
        # Find where our input prompt ends and the completion begins
        completion = completion[len(full_prompt):].strip()
        
        # Use regex to find the rewritten prompt - everything until end or next heading
        match = re.search(r'^(.*?)(?:$|User:)', completion, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return completion.strip()
    
    def _self_revise(self, original_prompt: str, rewritten_prompt: str) -> str:
        """
        Apply self-revision technique to improve rewritten prompt
        
        Args:
            original_prompt: Original user prompt
            rewritten_prompt: Initial rewritten prompt
            
        Returns:
            Improved prompt after self-revision
        """
        revision_prompt = (
            f"Review these two prompts and create an improved version:\n\n"
            f"Original: {original_prompt}\n\n"
            f"Rewritten: {rewritten_prompt}\n\n"
            f"Ensure the improved version:\n"
            f"1. Preserves the original intent\n"
            f"2. Adds appropriate details for video generation\n"
            f"3. Removes any redundancy or unclear elements\n"
            f"4. Uses descriptive, concise language\n\n"
            f"Improved version:"
        )
        
        # Tokenize
        inputs = self.tokenizer(revision_prompt, return_tensors="pt").to(self.device)
        
        # Generate completion
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
        # Decode and extract revised prompt
        completion = self.tokenizer.decode(output[0], skip_special_tokens=True)
        revised = completion[len(revision_prompt):].strip()
        
        return revised


class CameraMovementClassifier:
    """
    Classifier for predicting camera movement types in videos.
    Supports 14 movement types as mentioned in the HunyuanVideo paper.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        self.device = device
        self.model = self._load_model(model_path)
        
        # Camera movement types from the paper
        self.movement_types = [
            "zoom in", "zoom out", 
            "pan up", "pan down", "pan left", "pan right",
            "tilt up", "tilt down", "tilt left", "tilt right",
            "around left", "around right", 
            "static shot", "handheld shot"
        ]
    
    def _load_model(self, model_path):
        """
        Load camera movement classifier model.
        This would be implemented based on the specific architecture.
        """
        # Placeholder - actual implementation would load a pretrained model
        # Likely a 3D CNN or Video Transformer
        pass
    
    def predict(self, video_frames: torch.Tensor) -> Dict[str, float]:
        """
        Predict camera movement probabilities from video frames
        
        Args:
            video_frames: Video tensor [T, C, H, W]
            
        Returns:
            Dictionary of movement types and confidence scores
        """
        # Process frames
        with torch.no_grad():
            # Actual implementation would run inference
            # Here we simulate with random predictions
            confidences = {
                movement: random.random()
                for movement in self.movement_types
            }
            
        # Normalize to probabilities
        total = sum(confidences.values())
        normalized = {k: v/total for k, v in confidences.items()}
        
        return normalized
    
    def get_primary_movement(
        self, 
        video_frames: torch.Tensor,
        confidence_threshold: float = 0.3
    ) -> Optional[str]:
        """
        Get primary camera movement if confidence exceeds threshold
        
        Args:
            video_frames: Video tensor
            confidence_threshold: Minimum confidence to report movement
            
        Returns:
            Primary movement type or None if low confidence
        """
        predictions = self.predict(video_frames)
        primary = max(predictions.items(), key=lambda x: x[1])
        
        if primary[1] >= confidence_threshold:
            return primary[0]
        return None
    
    def get_movement_sequence(
        self,
        video_frames: torch.Tensor,
        window_size: int = 16,
        stride: int = 8
    ) -> List[str]:
        """
        Get sequence of camera movements throughout video
        
        Args:
            video_frames: Video tensor [T, C, H, W]
            window_size: Analysis window size
            stride: Window stride
            
        Returns:
            List of movement types for each window
        """
        num_frames = video_frames.shape[0]
        movements = []
        
        for start_idx in range(0, num_frames - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = video_frames[start_idx:end_idx]
            
            movement = self.get_primary_movement(window)
            if movement:
                movements.append(movement)
                
        return movements