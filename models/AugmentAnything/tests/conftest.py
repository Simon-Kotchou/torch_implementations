"""
Pytest configuration and shared fixtures for AugmentAnything tests.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import cv2


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Generate a simple test image (RGB numpy array)."""
    # Create a 256x256 RGB image with colored regions
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128, :128] = [255, 0, 0]  # Red square
    img[:128, 128:] = [0, 255, 0]  # Green square
    img[128:, :128] = [0, 0, 255]  # Blue square
    img[128:, 128:] = [255, 255, 0]  # Yellow square
    return img


@pytest.fixture
def sample_images_dataset(temp_dir, sample_image):
    """Create a small dataset of test images."""
    image_dir = temp_dir / "test_images"
    image_dir.mkdir()

    # Create 5 test images with different patterns
    for i in range(5):
        img = sample_image.copy()
        # Add some variation
        img = np.roll(img, shift=i * 20, axis=0)
        img = np.roll(img, shift=i * 20, axis=1)

        # Save as both numpy array and file
        img_path = image_dir / f"test_{i}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return image_dir


@pytest.fixture
def sample_mask():
    """Generate a simple binary mask."""
    mask = np.zeros((256, 256), dtype=bool)
    mask[64:192, 64:192] = True  # Square mask in center
    return mask


@pytest.fixture
def sample_bbox():
    """Return a sample bounding box [x, y, w, h]."""
    return [64, 64, 128, 128]


@pytest.fixture
def sample_rle_mask():
    """Generate a sample RLE-encoded mask (COCO format)."""
    # This is a simplified RLE for testing
    # In real usage, this comes from pycocotools
    return {
        'size': [256, 256],
        'counts': b'mock_rle_data'  # Mock data for testing
    }


@pytest.fixture
def mock_features():
    """Generate mock SAM2 features."""
    # Typical SAM2 feature shape: [C, H, W]
    return np.random.randn(256, 64, 64).astype(np.float32)


@pytest.fixture
def mock_dino_embedding():
    """Generate mock DINOv3 embedding."""
    # DINOv3 ConvNeXt-Large outputs 1536-dim embeddings
    embedding = np.random.randn(1536).astype(np.float32)
    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def sample_segment_data():
    """Generate sample segment metadata."""
    return {
        'image_idx': 0,
        'image_shape': np.array([256, 256], dtype=np.int16),
        'bbox': np.array([64, 64, 128, 128], dtype=np.float32),
        'area': np.int32(16384),
        'aspect_ratio': np.float32(1.0),
        'predicted_iou': np.float32(0.85),
        'stability_score': np.float32(0.90),
    }


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_no_models():
    """Skip test if model downloads would be required."""
    # This can be controlled via environment variable for CI/CD
    import os
    if os.environ.get("SKIP_MODEL_TESTS", "false").lower() == "true":
        pytest.skip("Model tests disabled")
