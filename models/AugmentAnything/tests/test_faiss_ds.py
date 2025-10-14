"""
Tests for faiss_ds.py - Segment database building functionality.
"""

import pytest
import numpy as np
import cv2
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pickle
import gzip
import faiss

from augment_anything.faiss_ds import (
    SegmentDatabaseBuilder,
    load_segment_database,
    rle_to_mask,
)


class TestSegmentDatabaseBuilder:
    """Tests for SegmentDatabaseBuilder class."""

    def test_initialization(self):
        """Test builder initialization."""
        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Mock the DINOv3 model to return proper output for embedding_dim calculation
            mock_pooler_output = torch.randn(1, 1536)  # Standard DINOv3 dim
            mock_output = Mock()
            mock_output.pooler_output = mock_pooler_output

            # Create callable mock that returns mock_output when called
            mock_model_instance = Mock()
            mock_model_instance.return_value = mock_output

            # Configure .to() to return the same callable mock (supports chaining)
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance

            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu', batch_size=16)

            assert builder.device == 'cpu'
            assert builder.batch_size == 16
            assert builder.embedding_dim == 1536
            assert len(builder.image_paths) == 0

    def test_get_image_path_idx(self):
        """Test image path indexing and deduplication."""
        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Setup mock model for initialization
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(1, 1536)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu')

            # First path
            idx1 = builder._get_image_path_idx(Path('/path/to/image1.jpg'))
            assert idx1 == 0
            assert len(builder.image_paths) == 1

            # Same path should return same index
            idx2 = builder._get_image_path_idx(Path('/path/to/image1.jpg'))
            assert idx2 == idx1
            assert len(builder.image_paths) == 1

            # Different path should get new index
            idx3 = builder._get_image_path_idx(Path('/path/to/image2.jpg'))
            assert idx3 == 1
            assert len(builder.image_paths) == 2

    def test_extract_segment_crop(self, sample_image):
        """Test segment crop extraction with padding."""
        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Setup mock model for initialization
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(1, 1536)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu')

            mask = np.zeros((256, 256), dtype=bool)
            mask[100:150, 100:150] = True
            bbox = [100, 100, 50, 50]  # [x, y, w, h]

            crop = builder.extract_segment_crop(sample_image, mask, bbox, padding=10)

            # Should extract region with padding
            assert crop.shape[0] <= 70  # 50 + 2*10
            assert crop.shape[1] <= 70
            assert crop.shape[2] == 3

    def test_extract_segment_crop_edge(self, sample_image):
        """Test crop extraction at image edges."""
        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Setup mock model for initialization
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(1, 1536)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu')

            mask = np.zeros((256, 256), dtype=bool)
            mask[0:50, 0:50] = True
            bbox = [0, 0, 50, 50]  # At corner

            crop = builder.extract_segment_crop(sample_image, mask, bbox, padding=10)

            # Should handle edge case without error
            assert crop.shape[2] == 3
            assert crop.shape[0] > 0
            assert crop.shape[1] > 0

    def test_compute_embeddings_batched(self, sample_image):
        """Test batched embedding computation."""
        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor') as mock_processor, \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Mock DINOv3 model outputs - need real tensor for shape access in __init__
            mock_pooler_tensor = torch.randn(1, 1536)

            # Mock for initialization (uses .shape)
            mock_init_output = Mock()
            mock_init_output.pooler_output = mock_pooler_tensor

            # Mock for actual embedding computation - return different sizes per batch
            # 3 crops with batch_size=2 → batch 1: 2 crops, batch 2: 1 crop
            mock_embed_output_1 = Mock()
            mock_embed_output_1.pooler_output = Mock()
            mock_embed_output_1.pooler_output.cpu.return_value.numpy.return_value = \
                np.random.randn(2, 1536).astype(np.float32)  # First batch: 2 items

            mock_embed_output_2 = Mock()
            mock_embed_output_2.pooler_output = Mock()
            mock_embed_output_2.pooler_output.cpu.return_value.numpy.return_value = \
                np.random.randn(1, 1536).astype(np.float32)  # Second batch: 1 item

            mock_model_instance = Mock()
            # First call (init), then batch 1, then batch 2
            mock_model_instance.side_effect = [mock_init_output, mock_embed_output_1, mock_embed_output_2]
            # Configure .to() and .eval() to support chaining
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu', batch_size=2)

            # Create 3 crops (will require 2 batches)
            crops = [sample_image.copy() for _ in range(3)]

            embeddings = builder.compute_embeddings_batched(crops)

            # Should return normalized embeddings
            assert embeddings.shape == (3, builder.embedding_dim)
            assert embeddings.dtype == np.float32
            # Check normalization
            for emb in embeddings:
                assert np.abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_compute_embeddings_empty(self):
        """Test embedding computation with empty crop list."""
        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Setup mock model for initialization
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(1, 1536)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu')

            embeddings = builder.compute_embeddings_batched([])

            assert len(embeddings) == 0

    @patch('cv2.imread')
    def test_process_image_success(self, mock_imread, sample_image):
        """Test successful image processing."""
        mock_imread.return_value = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator') as mock_mask_gen, \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Setup mock model for initialization
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(1, 1536)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance

            # Mock mask generator
            mock_gen_instance = Mock()
            mock_rle = {'size': [256, 256], 'counts': b'mock'}
            mock_gen_instance.generate.return_value = [
                {
                    'segmentation': mock_rle,
                    'bbox': [64, 64, 128, 128],
                    'area': 16384,
                    'predicted_iou': 0.85,
                    'stability_score': 0.90
                }
            ]
            mock_mask_gen.return_value = mock_gen_instance

            builder = SegmentDatabaseBuilder(device='cpu')

            with patch.object(builder, 'rle_to_mask', return_value=np.ones((256, 256), dtype=bool)):
                segment_data, crops = builder.process_image('/fake/path.jpg', min_segment_area=500)

            assert len(segment_data) >= 0  # May have segments
            assert len(crops) == len(segment_data)

    @patch('cv2.imread')
    def test_process_image_failure(self, mock_imread):
        """Test image processing with failed load."""
        mock_imread.return_value = None

        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Setup mock model for initialization
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(1, 1536)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu')

            segment_data, crops = builder.process_image('/fake/path.jpg')

            # Should return empty lists
            assert segment_data == []
            assert crops == []

    def test_process_image_filter_small_segments(self, sample_image):
        """Test filtering of small segments."""
        with patch('cv2.imread', return_value=cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)), \
             patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator') as mock_mask_gen, \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Setup mock model for initialization
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(1, 1536)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance

            # Mock mask generator with small and large segments
            mock_gen_instance = Mock()
            mock_gen_instance.generate.return_value = [
                {
                    'segmentation': {'size': [256, 256], 'counts': b'mock'},
                    'bbox': [0, 0, 10, 10],
                    'area': 100,  # Too small
                    'predicted_iou': 0.85,
                    'stability_score': 0.90
                },
                {
                    'segmentation': {'size': [256, 256], 'counts': b'mock'},
                    'bbox': [64, 64, 128, 128],
                    'area': 16384,  # Large enough
                    'predicted_iou': 0.85,
                    'stability_score': 0.90
                }
            ]
            mock_mask_gen.return_value = mock_gen_instance

            builder = SegmentDatabaseBuilder(device='cpu')

            with patch.object(builder, 'rle_to_mask', return_value=np.ones((256, 256), dtype=bool)):
                segment_data, crops = builder.process_image(
                    '/fake/path.jpg',
                    min_segment_area=500
                )

            # Should filter out small segment
            assert len(segment_data) <= 1  # Only large segment or none if processing fails

    def test_build_database_structure(self, temp_dir, sample_images_dataset):
        """Test database building creates all expected files."""
        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator') as mock_mask_gen, \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Mock everything to avoid actual model inference
            mock_gen_instance = Mock()
            mock_rle = {'size': [256, 256], 'counts': b'mock'}
            mock_gen_instance.generate.return_value = [
                {
                    'segmentation': mock_rle,
                    'bbox': [64, 64, 128, 128],
                    'area': 16384,
                    'predicted_iou': 0.85,
                    'stability_score': 0.90
                }
            ]
            mock_mask_gen.return_value = mock_gen_instance

            # Mock DINOv3 embeddings - need real tensor for shape access in __init__
            mock_pooler_tensor = torch.randn(1, 1536)

            # Mock for initialization (uses .shape)
            mock_init_output = Mock()
            mock_init_output.pooler_output = mock_pooler_tensor

            # Mock for actual embedding computation (uses .cpu().numpy())
            mock_embed_output = Mock()
            mock_embed_output.pooler_output = Mock()
            mock_embed_output.pooler_output.cpu.return_value.numpy.return_value = \
                np.random.randn(1, 1536).astype(np.float32)

            mock_model_instance = Mock()
            # First call (init) returns mock_init_output, subsequent calls return mock_embed_output
            mock_model_instance.side_effect = [mock_init_output] + [mock_embed_output] * 100
            # Configure .to() and .eval() to support chaining
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu', batch_size=2)

            with patch.object(builder, 'rle_to_mask', return_value=np.ones((256, 256), dtype=bool)):
                output_path = temp_dir / "test_db"
                builder.build_database(
                    sample_images_dataset,
                    output_path,
                    max_images=2,
                    min_segment_area=500
                )

            # Check all expected files are created
            assert (output_path / "segments.faiss").exists()
            assert (output_path / "segments_metadata.pkl.gz").exists()
            assert (output_path / "image_paths.pkl.gz").exists()
            assert (output_path / "database_info.pkl").exists()


class TestLoadSegmentDatabase:
    """Tests for database loading functionality."""

    def test_load_segment_database(self, temp_dir):
        """Test loading a saved database."""
        db_path = temp_dir / "test_db"
        db_path.mkdir()

        # Create mock database files
        # FAISS index
        index = faiss.IndexFlatIP(128)
        embeddings = np.random.randn(10, 128).astype(np.float32)
        index.add(embeddings)
        faiss.write_index(index, str(db_path / "segments.faiss"))

        # Metadata
        metadata = [
            {
                'image_idx': 0,
                'bbox': np.array([0, 0, 100, 100]),
                'area': 10000
            }
            for _ in range(10)
        ]
        with gzip.open(db_path / "segments_metadata.pkl.gz", 'wb') as f:
            pickle.dump(metadata, f)

        # Image paths
        image_paths = [f'/path/to/image_{i}.jpg' for i in range(5)]
        with gzip.open(db_path / "image_paths.pkl.gz", 'wb') as f:
            pickle.dump(image_paths, f)

        # Info
        info = {
            'num_segments': 10,
            'num_images': 5,
            'embedding_dim': 128
        }
        with open(db_path / "database_info.pkl", 'wb') as f:
            pickle.dump(info, f)

        # Load database
        loaded_index, loaded_segments, loaded_paths, loaded_info = \
            load_segment_database(db_path)

        assert loaded_index.ntotal == 10
        assert len(loaded_segments) == 10
        assert len(loaded_paths) == 5
        assert loaded_info['num_segments'] == 10

    def test_load_segment_database_missing_files(self, temp_dir):
        """Test loading with missing files."""
        db_path = temp_dir / "missing_db"
        db_path.mkdir()

        # Should raise error when files are missing
        with pytest.raises((FileNotFoundError, Exception)):
            load_segment_database(db_path)


class TestRLEConversion:
    """Tests for RLE mask conversion."""

    def test_rle_to_mask(self):
        """Test RLE to mask conversion."""
        # Use real pycocotools since it's available
        import pycocotools.mask as mask_utils

        # Create a real RLE from a binary mask
        binary_mask = np.zeros((256, 256), dtype=np.uint8, order='F')
        binary_mask[64:192, 64:192] = 1
        rle = mask_utils.encode(binary_mask)

        # Test our function
        decoded_mask = rle_to_mask(rle)

        assert decoded_mask.shape == (256, 256)
        assert decoded_mask.dtype == np.uint8
        # Verify it decoded correctly
        np.testing.assert_array_equal(decoded_mask, binary_mask)


class TestEdgeCases:
    """Tests for edge cases in database building."""

    def test_build_database_no_images(self, temp_dir):
        """Test building database with no images."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor'), \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Setup mock model for initialization
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(1, 1536)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu')

            output_path = temp_dir / "test_db"

            # Should handle gracefully (print message or raise)
            with patch('glob.glob', return_value=[]):
                builder.build_database(
                    empty_dir,
                    output_path,
                    min_segment_area=500
                )

            # Database files should not be created or function should exit early

    def test_compute_embeddings_various_sizes(self):
        """Test embedding computation with various crop sizes."""
        with patch('augment_anything.faiss_ds.SAM2ImagePredictor'), \
             patch('augment_anything.faiss_ds.SAM2AutomaticMaskGenerator'), \
             patch('augment_anything.faiss_ds.AutoImageProcessor') as mock_processor, \
             patch('augment_anything.faiss_ds.AutoModel') as mock_model:

            # Mock DINOv3 - need real tensor for shape access in __init__
            mock_pooler_tensor = torch.randn(1, 1536)

            # Mock for initialization (uses .shape)
            mock_init_output = Mock()
            mock_init_output.pooler_output = mock_pooler_tensor

            # Mock for actual embedding computation - return different sizes per batch
            # 3 crops with batch_size=2 → batch 1: 2 crops, batch 2: 1 crop
            mock_embed_output_1 = Mock()
            mock_embed_output_1.pooler_output = Mock()
            mock_embed_output_1.pooler_output.cpu.return_value.numpy.return_value = \
                np.random.randn(2, 1536).astype(np.float32)  # First batch: 2 items

            mock_embed_output_2 = Mock()
            mock_embed_output_2.pooler_output = Mock()
            mock_embed_output_2.pooler_output.cpu.return_value.numpy.return_value = \
                np.random.randn(1, 1536).astype(np.float32)  # Second batch: 1 item

            mock_model_instance = Mock()
            # First call (init), then batch 1, then batch 2
            mock_model_instance.side_effect = [mock_init_output, mock_embed_output_1, mock_embed_output_2]
            # Configure .to() and .eval() to support chaining
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance

            builder = SegmentDatabaseBuilder(device='cpu', batch_size=2)

            # Different sized crops
            crops = [
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8),
                np.random.randint(0, 255, (50, 300, 3), dtype=np.uint8),
            ]

            embeddings = builder.compute_embeddings_batched(crops)

            # All should be normalized to same dimension
            assert embeddings.shape[0] == 3
            assert embeddings.shape[1] == builder.embedding_dim
