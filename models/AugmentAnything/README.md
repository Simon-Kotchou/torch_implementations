# Augment Anything

**Intelligent data augmentation using SAM2 (Segment Anything Model 2) for semantic-aware image synthesis.**

Augment Anything automatically segments images, finds semantically compatible segments across your dataset, and creates realistic augmented training data through intelligent segment swapping and blending.

## Features

- **Semantic Similarity Matching**: Uses SAM2 features and FAISS for fast, semantic-aware segment matching
- **Scalable Architecture**: Handles 1000+ images efficiently with FAISS indexing
- **Multiple Workflows**:
  - Basic augmentation for small datasets (~50 images)
  - FAISS-accelerated augmentation for large datasets (1000+ images)
  - Database builder for training pipelines with DINOv3 embeddings
- **Intelligent Blending**: Gaussian blur-based smooth compositing
- **Quality Filtering**: IoU and stability score thresholds ensure high-quality segments
- **Caching Support**: Save/load preprocessed data for instant reuse

## Installation

### From Source

```bash
git clone https://github.com/simon-kotchou/augment-anything.git
cd augment-anything
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for GPU acceleration)

Core dependencies:
- `torch` - Deep learning framework
- `transformers` - Hugging Face model hub
- `sam2` - Segment Anything Model 2
- `faiss-gpu` or `faiss-cpu` - Fast similarity search
- `opencv-python` - Image processing
- `numpy`, `matplotlib`, `tqdm`, `pillow`

## Quick Start

### 1. Basic Augmentation (Small Datasets)

For datasets with ~50 images, use the brute-force approach:

```bash
python augment_sam.py \
  --image_dir /path/to/images \
  --n_samples 3 \
  --output_dir ./results \
  --max_images 50
```

### 2. FAISS-Accelerated Augmentation (Large Datasets)

For 1000+ images, use FAISS indexing:

```bash
python faiss_augment.py \
  --image_dir /path/to/images \
  --n_samples 5 \
  --output_dir ./augment_results \
  --max_images 1000 \
  --max_masks_per_image 20 \
  --cache_path ./cache/dataset.cache
```

**Caching**: The first run extracts features and builds the index. Subsequent runs with the same `--cache_path` load instantly.

### 3. Build Segment Database (Training Pipelines)

For production training pipelines, build a searchable segment database with DINOv3 embeddings:

```bash
python faiss_ds.py \
  --image_dir /path/to/images \
  --output_dir ./segment_db \
  --batch_size 32 \
  --min_segment_area 500 \
  --max_images 5000
```

**Outputs**:
- `segments.faiss` - FAISS index for fast search
- `segments_metadata.pkl.gz` - Compressed segment metadata (RLE masks, quality scores)
- `image_paths.pkl.gz` - Image path lookup table
- `database_info.pkl` - Database statistics

### 4. Use as a Library

```python
from augment_anything import AugmentAnythingDemo

# Initialize with caching
demo = AugmentAnythingDemo(
    image_dir='./images',
    device='cuda',
    max_images=1000,
    cache_path='./cache/dataset.cache'
)

# Augment a specific image
original, augmented, info = demo.augment_image(img_idx=0, num_augmentations=2)

# Generate multiple demos
demo.demo_multiple_samples(n_samples=5, output_dir='./results')
```

**Load Pre-built Database**:

```python
from faiss_ds import load_segment_database

# Load database
faiss_index, segments, image_paths, info = load_segment_database('./segment_db')

# Search for similar segments
similarities, indices = faiss_index.search(query_embedding, k=10)
```

## How It Works

### Architecture Overview

1. **Segmentation** (SAM2)
   - Automatic mask generation with quality filtering
   - Extracts high-resolution feature maps

2. **Descriptor Computation**
   - Masked average pooling over SAM2 features
   - L2 normalization for cosine similarity

3. **Similarity Search**
   - FAISS IndexFlatIP for fast cosine similarity
   - Compatibility filtering (area ratio, aspect ratio)

4. **Blending**
   - Gaussian blur on blend masks (21x21, sigma=7)
   - Smooth alpha compositing

### Compatibility Filters

Segments are matched based on:
- **Semantic similarity**: Cosine similarity > 0.2 (default)
- **Area ratio**: 0.3 < ratio < 3.0
- **Aspect ratio**: Difference < 2.0
- **Source exclusion**: No segments from the same image

## Command-Line Options

### faiss_augment.py

```
--image_dir         Directory containing images (required)
--n_samples         Number of demo augmentations (default: 3)
--output_dir        Output directory (default: ./augment_results)
--device            Device: cuda or cpu (default: cuda)
--max_images        Maximum images to process (default: 1000)
--max_masks_per_image  Max masks per image (default: 20)
--cache_path        Cache file path for fast reloading
--seed              Random seed (default: 42)
--no_save           Show results interactively instead of saving
```

### faiss_ds.py

```
--image_dir         Directory containing images (required)
--output_dir        Output directory for database (required)
--batch_size        DINOv3 batch size (default: 32)
--min_segment_area  Minimum segment area in pixels (default: 500)
--max_images        Maximum images to process (default: 5000)
--device            Device: cuda or cpu (default: cuda)
```

## Models Used

- **SAM2**:
  - `facebook/sam2-hiera-tiny` (faiss_augment.py) - Faster, good quality
  - `facebook/sam2-hiera-large` (augment_sam.py, faiss_ds.py) - Highest quality

- **DINOv3**:
  - `facebook/dinov3-convnext-large-pretrain-lvd1689m` (faiss_ds.py only)

## Performance Notes

- **SAM2 inference**: ~2-5 seconds per image (GPU)
- **DINOv3 batching**: 10-50x speedup over sequential inference
- **FAISS search**: <1ms for 100K segments
- **Caching**: Subsequent runs load in <5 seconds

## Output Examples

Each augmentation visualization shows:
- Original image
- Augmented image with blended segments
- Difference map highlighting changes
- All detected segments
- Source and target segment pairs with similarity scores

## File Formats

### Cache Files (faiss_augment.py)
- `.faiss` - FAISS index (binary)
- `.pkl` - Pickled metadata (images_data, segment_metadata)

### Database Files (faiss_ds.py)
- `segments.faiss` - FAISS index
- `segments_metadata.pkl.gz` - Gzipped pickle with COCO RLE masks
- `image_paths.pkl.gz` - Gzipped pickle
- `database_info.pkl` - Plain pickle

## Configuration

Key parameters in the code:

```python
# SAM2 mask generation
points_per_side = 16              # Grid density for automatic masks
pred_iou_thresh = 0.7             # Minimum predicted IoU
stability_score_thresh = 0.85     # Minimum stability score

# Compatibility filtering
similarity_threshold = 0.2        # Minimum cosine similarity
area_ratio_range = (0.3, 3.0)     # Acceptable area ratio
aspect_diff_max = 2.0             # Max aspect ratio difference

# Blending
blur_kernel = (21, 21)            # Gaussian blur kernel size
blur_sigma = 7                    # Gaussian blur sigma
```

## Use Cases

- **Data Augmentation**: Generate diverse training data for computer vision tasks
- **Dataset Expansion**: Increase dataset size with semantically valid variations
- **Domain Adaptation**: Mix segments across different domains
- **Visual Analysis**: Study segment compatibility and feature similarity
- **Research**: Explore SAM2 feature representations

## Limitations

- Requires GPU for practical use on large datasets
- Quality depends on SAM2 segmentation accuracy
- Best results with datasets containing similar object categories
- Blending may be visible on highly textured backgrounds

## Testing

A comprehensive test suite is available in the `tests/` directory.

### Running Tests

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=augment_anything --cov-report=html
```

Skip slow integration tests:
```bash
pytest -m "not slow"
```

Skip model download tests (useful for CI/CD):
```bash
SKIP_MODEL_TESTS=true pytest
```

### Test Structure

- `test_core.py` - Unit tests for core augmentation functionality
- `test_faiss_ds.py` - Unit tests for database builder
- `test_integration.py` - End-to-end integration tests
- `conftest.py` - Shared fixtures and test utilities

Tests use mocking to avoid model downloads and run quickly in CI/CD environments.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [SAM2](https://github.com/facebookresearch/segment-anything-2) by Meta AI
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research

## Contact

For questions or feedback, please open an issue on GitHub.
