"""
Unified DINOv3 feature extraction for semantic matching.

This module provides a consistent feature extractor used by both:
- Database building phase (faiss_ds.py)
- Query/inference phase (core.py)

Using the same feature space ensures proper semantic matching.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DINOv3Extractor:
    """
    Unified DINOv3 feature extraction for semantic matching.

    Features:
    - Single and batch inference
    - Masked region extraction
    - L2 normalization for cosine similarity
    - Memory efficient

    Usage:
        # Single image
        extractor = DINOv3Extractor()
        embedding = extractor.extract_single(image_crop)

        # Batch processing (10-50x faster)
        embeddings = extractor.extract_batch(image_crops)

        # Masked region
        embedding = extractor.extract_masked_region(image, mask)
    """

    def __init__(
        self,
        model_name: str = 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32
    ):
        """
        Initialize DINOv3 extractor.

        Args:
            model_name: HuggingFace model identifier
            device: Device for inference ('cuda' or 'cpu')
            batch_size: Batch size for batch inference
        """
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        logger.info(f"Loading DINOv3: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = self.model(dummy_input)
            self.embedding_dim = dummy_output.pooler_output.shape[-1]

        logger.info(f"✓ DINOv3 loaded. Embedding dim: {self.embedding_dim}")

    def extract_single(self, image_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract DINOv3 embedding for a single image crop.

        Args:
            image_crop: RGB image as numpy array (H, W, 3), uint8

        Returns:
            L2-normalized embedding vector of shape (embedding_dim,)
            Returns None if extraction fails
        """
        try:
            # Convert to PIL
            if image_crop.size == 0:
                return None

            pil_image = Image.fromarray(image_crop.astype(np.uint8))

            # Process with DINOv3
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.pooler_output[0].cpu().numpy()

            # L2 normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            return None

    def extract_batch(self, image_crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract DINOv3 embeddings for multiple crops in batches.

        This is 10-50x faster than sequential extraction due to:
        - Batched GPU inference
        - Reduced overhead

        Args:
            image_crops: List of RGB images as numpy arrays

        Returns:
            Array of shape (num_crops, embedding_dim)
            L2-normalized embeddings
        """
        if len(image_crops) == 0:
            return np.array([]).reshape(0, self.embedding_dim)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(image_crops), self.batch_size):
            batch_crops = image_crops[i:i + self.batch_size]

            # Convert to PIL
            batch_pil = []
            for crop in batch_crops:
                if crop.size > 0:
                    batch_pil.append(Image.fromarray(crop.astype(np.uint8)))

            if len(batch_pil) == 0:
                continue

            # Process batch with DINOv3
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()

            # L2 normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.where(norms > 0, embeddings / norms, embeddings)

            all_embeddings.append(embeddings.astype(np.float32))

        return np.vstack(all_embeddings) if all_embeddings else np.array([]).reshape(0, self.embedding_dim)

    def extract_masked_region(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        padding: int = 10,
        use_mask_fill: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract DINOv3 embedding for a masked region.

        This is compatible with the SAM2-based workflow:
        1. Get bounding box from mask
        2. Extract crop with padding
        3. Optionally fill background with mean color
        4. Extract embedding

        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W), bool or uint8
            padding: Padding around bbox in pixels
            use_mask_fill: If True, fill background with mean color

        Returns:
            L2-normalized embedding or None if mask is empty
        """
        # Get bounding box from mask
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return None

        y1, x1 = coords[0].min(), coords[1].min()
        y2, x2 = coords[0].max(), coords[1].max()

        # Add padding
        H, W = image.shape[:2]
        y1 = max(0, y1 - padding)
        x1 = max(0, x1 - padding)
        y2 = min(H, y2 + padding)
        x2 = min(W, x2 + padding)

        # Extract crop
        crop = image[y1:y2+1, x1:x2+1]
        mask_crop = mask[y1:y2+1, x1:x2+1]

        # Apply mask (fill background with mean color)
        if use_mask_fill:
            masked_crop = crop.copy()
            if mask_crop.any():
                mean_color = crop[mask_crop].mean(axis=0)
                masked_crop[~mask_crop] = mean_color
            crop = masked_crop

        return self.extract_single(crop)

    def cleanup(self):
        """Cleanup GPU resources."""
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("DINOv3 resources cleaned up")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass


def extract_segment_crop_from_bbox(
    image: np.ndarray,
    bbox: Union[tuple, list, np.ndarray],
    mask: Optional[np.ndarray] = None,
    padding: int = 10
) -> np.ndarray:
    """
    Helper function to extract a cropped region around a segment.

    Compatible with both bbox formats:
    - SAM2 format: [x, y, w, h]
    - Standard format: [y1, x1, y2, x2]

    Args:
        image: RGB image (H, W, 3)
        bbox: Bounding box in [x, y, w, h] or [y1, x1, y2, x2] format
        mask: Optional binary mask to apply
        padding: Padding in pixels

    Returns:
        Cropped region as RGB image
    """
    H, W = image.shape[:2]

    # Detect bbox format and normalize to [x1, y1, x2, y2]
    if len(bbox) == 4:
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            # Format is [x, y, w, h] (SAM2 format)
            x, y, w, h = [int(v) for v in bbox]
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            # Format is [y1, x1, y2, x2] (standard format)
            y1, x1, y2, x2 = [int(v) for v in bbox]
    else:
        raise ValueError(f"Invalid bbox format: {bbox}")

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W, x2 + padding)
    y2 = min(H, y2 + padding)

    # Crop image
    crop = image[y1:y2, x1:x2]

    # Apply mask if provided
    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        if mask_crop.any():
            masked_crop = crop.copy()
            mean_color = crop[mask_crop].mean(axis=0)
            masked_crop[~mask_crop] = mean_color
            crop = masked_crop

    return crop


if __name__ == "__main__":
    # Quick test
    print("Testing DINOv3Extractor...")

    # Create dummy image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_mask = np.zeros((512, 512), dtype=bool)
    test_mask[100:200, 100:200] = True

    # Initialize extractor
    extractor = DINOv3Extractor(device='cpu')

    # Test single extraction
    print("\n1. Testing single extraction...")
    embedding = extractor.extract_single(test_image[100:200, 100:200])
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {np.linalg.norm(embedding):.3f} (should be ~1.0)")

    # Test masked extraction
    print("\n2. Testing masked region extraction...")
    embedding = extractor.extract_masked_region(test_image, test_mask)
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {np.linalg.norm(embedding):.3f}")

    # Test batch extraction
    print("\n3. Testing batch extraction...")
    crops = [test_image[i:i+100, i:i+100] for i in range(0, 200, 50)]
    embeddings = extractor.extract_batch(crops)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.3f}")

    print("\n✓ All tests passed!")
