# DepthPro: Transformer-based Monocular Depth Estimation

## Overview

DepthPro is a state-of-the-art monocular depth estimation model introduced by Apple in 2024. It is a foundation model for zero-shot metric depth, capable of predicting absolute depth from a single RGB image without camera parameters or further training.

### Key Features

- **High-Resolution Sharp Depth Maps**: Produces 2.25 MP (1536×1536) depth maps with unparalleled sharpness and high-frequency detail
- **Metric Depth with No Intrinsics**: Outputs metric-scaled depth (in meters) with absolute scale, no camera intrinsics required
- **Efficient Multi-Scale Transformer Architecture**: Uses an efficient multi-scale Vision Transformer (ViT) architecture for dense depth prediction
- **Real + Synthetic Training**: Comprehensive training protocol mixing real-world datasets and synthetic data
- **State-of-the-Art Results**: Significantly outperforms prior monocular depth models in both accuracy and inference speed

## Repository Structure

This repository contains both documentation and implementation code for DepthPro:

### Documentation

- [Model Architecture](model_architecture.md): Detailed explanation of DepthPro's multi-scale transformer architecture
- [Depth Estimation Theory](depth_estimation_theory.md): Mathematical formulation of monocular depth estimation
- [Hugging Face Integration](huggingface_integration.md): Guide to using the Hugging Face checkpoint

### Implementation

- [`depth_estimation.py`](depth_estimation.py): Core module for depth estimation using DepthPro
- [`video_stream.py`](video_stream.py): Flask application for streaming depth estimation on video

## Getting Started

### Installation

```bash
pip install transformers torch opencv-python flask
```

### Basic Usage

```python
from depth_estimation import DepthEstimator

# Initialize the estimator
estimator = DepthEstimator()

# Estimate depth from an image
depth_map = estimator.estimate_from_image("path/to/image.jpg")

# Visualize the depth map
estimator.visualize(depth_map)
```

### Running the Video Streaming Demo

```bash
python video_stream.py --input path/to/video.mp4
```

Then open your browser at http://localhost:5000 to view the live depth estimation.

## References

- [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/html/2410.02073v1)
- [Hugging Face Model: apple/DepthPro-hf](https://huggingface.co/apple/DepthPro-hf)
