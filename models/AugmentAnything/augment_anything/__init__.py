"""
Augment Anything - Blazingly fast semantic augmentation using SAM2 + DINOv3 + FAISS.

Build Phase: SAM2 segmentation → DINOv3 embeddings → FAISS index
Inference Phase: Query masks → DINOv3 lookup → Sample candidates → Blend regions
"""

__version__ = "0.1.0"
__author__ = "Simon Kotchou"
__email__ = "simonkotchou@mines.edu"

# Core inference API
from .core import (
    AugmentAnything,
    AugmentAnythingDataset,
    AugmentConfig,
    quick_augment,
)

# Database builder
from .faiss_ds import (
    SegmentDatabaseBuilder,
    load_segment_database,
    rle_to_mask,
)

# Public API
__all__ = [
    "AugmentAnything",
    "AugmentAnythingDataset",
    "AugmentConfig",
    "quick_augment",
    "SegmentDatabaseBuilder",
    "load_segment_database",
    "rle_to_mask",
    "__version__",
]
