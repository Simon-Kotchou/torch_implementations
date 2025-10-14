"""
Tests for core.py - AugmentAnything main functionality.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from augment_anything import (
    AugmentAnything,
    AugmentAnythingDataset,
    AugmentConfig,
    quick_augment,
)


class TestAugmentConfig:
    """Tests for AugmentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AugmentConfig()
        assert config.sam_model == "facebook/sam2-hiera-tiny"
        assert config.points_per_side == 16
        assert config.pred_iou_thresh == 0.7
        assert config.similarity_threshold == 0.2
        assert config.auto_cache is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = AugmentConfig(
            sam_model="facebook/sam2-hiera-large",
            similarity_threshold=0.5,
            max_masks_per_image=30
        )
        assert config.sam_model == "facebook/sam2-hiera-large"
        assert config.similarity_threshold == 0.5
        assert config.max_masks_per_image == 30

    def test_device_selection(self):
        """Test device selection logic."""
        config = AugmentConfig()
        # Device should be cuda if available, else cpu
        assert config.device in ["cuda", "cpu"]


class TestAugmentAnything:
    """Tests for AugmentAnything class."""

    @patch('augment_anything.core.SAM2ImagePredictor')
    @patch('augment_anything.core.SAM2AutomaticMaskGenerator')
    def test_initialization_no_cache(self, mock_mask_gen, mock_predictor,
                                     sample_images_dataset, temp_dir):
        """Test initialization without existing cache."""
        config = AugmentConfig(cache_dir=str(temp_dir), auto_cache=False)

        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            augmenter = AugmentAnything(sample_images_dataset, config=config)
            assert augmenter.image_dir == sample_images_dataset
            assert augmenter.config == config

    def test_find_images(self, sample_images_dataset):
        """Test image file discovery."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(sample_images_dataset, config=config)
            images = augmenter._find_images()

            # Should find all test images
            assert len(images) == 5
            assert all(img.suffix.lower() in ['.jpg', '.jpeg', '.png'] for img in images)

    def test_compute_descriptor(self, mock_features, sample_mask):
        """Test feature descriptor computation."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            descriptor = augmenter._compute_descriptor(mock_features, sample_mask)

            # Should return normalized descriptor
            assert descriptor is not None
            assert descriptor.shape == (mock_features.shape[0],)
            # Check L2 normalization
            assert np.abs(np.linalg.norm(descriptor) - 1.0) < 1e-5

    def test_compute_descriptor_small_mask(self, mock_features):
        """Test descriptor computation with very small mask."""
        small_mask = np.zeros((256, 256), dtype=bool)
        small_mask[0:2, 0:2] = True  # Tiny mask

        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            descriptor = augmenter._compute_descriptor(mock_features, small_mask)

            # Should fallback to global pooling for small masks
            assert descriptor is not None

    def test_mask_to_bbox(self, sample_mask):
        """Test bounding box extraction from mask."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            bbox = augmenter._mask_to_bbox(sample_mask)

            # Should return [y1, x1, y2, x2]
            assert len(bbox) == 4
            assert bbox[0] == 64  # y1
            assert bbox[1] == 64  # x1
            assert bbox[2] == 191  # y2
            assert bbox[3] == 191  # x2

    def test_mask_to_bbox_empty(self):
        """Test bounding box for empty mask."""
        empty_mask = np.zeros((256, 256), dtype=bool)

        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            bbox = augmenter._mask_to_bbox(empty_mask)
            assert bbox == [0, 0, 0, 0]

    def test_find_best_match_filters(self, mock_dino_embedding):
        """Test segment matching with area/aspect ratio filters."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(
                auto_cache=False,
                area_ratio_range=(0.5, 2.0),
                aspect_ratio_diff_max=1.0
            )
            augmenter = AugmentAnything(Path('.'), config=config)

            # Mock FAISS index and segments
            augmenter.index = MagicMock()
            augmenter.index.search.return_value = (
                np.array([[0.8, 0.7]]),  # similarities
                np.array([[0, 1]])  # indices
            )

            augmenter.segments = [
                {
                    'img_idx': 0,
                    'mask': np.ones((256, 256), dtype=bool),
                    'area': 10000,  # Similar area
                    'bbox': [64, 64, 100, 100]  # w, h = 100, 100
                },
                {
                    'img_idx': 1,
                    'mask': np.ones((256, 256), dtype=bool),
                    'area': 50000,  # Too large (5x)
                    'bbox': [0, 0, 200, 100]
                }
            ]

            source_mask = np.zeros((256, 256), dtype=bool)
            source_mask[64:164, 64:164] = True

            match = augmenter._find_best_match(source_mask, mock_dino_embedding, 10000)

            # Should match first segment (similar area), not second (too large)
            assert match is not None
            assert match['img_idx'] == 0

    def test_blend_segment_same_size(self, sample_image):
        """Test segment blending with same-sized images."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False, blend_kernel_size=0)
            augmenter = AugmentAnything(Path('.'), config=config)

            target_img = sample_image.copy()
            source_img = np.full_like(sample_image, 128)  # Gray image

            target_mask = np.zeros((256, 256), dtype=bool)
            target_mask[64:192, 64:192] = True
            source_mask = target_mask.copy()

            result = augmenter._blend_segment(target_img, source_img, target_mask, source_mask)

            # Result should have source pixels in masked region
            assert result.shape == target_img.shape
            assert result.dtype == np.uint8
            # Center should be blended
            assert np.any(result[128, 128] != target_img[128, 128])

    def test_augment_with_seed(self, sample_image):
        """Test augmentation with fixed seed for reproducibility."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'), \
             patch.object(AugmentAnything, '_extract_features_and_masks') as mock_extract, \
             patch.object(AugmentAnything, '_find_best_match') as mock_match:

            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)
            augmenter.image_data = [{'image': sample_image, 'path': Path('test.jpg')}]

            # Mock feature extraction
            mock_mask = {'segmentation': np.ones((256, 256), dtype=bool), 'area': 65536}
            mock_extract.return_value = (np.random.randn(256, 64, 64), [mock_mask])
            mock_match.return_value = {'img_idx': 0, 'mask': mock_mask['segmentation'], 'similarity': 0.9}

            # Augment with same seed should give same result
            result1 = augmenter.augment(sample_image, num_swaps=1, seed=42)
            result2 = augmenter.augment(sample_image, num_swaps=1, seed=42)

            np.testing.assert_array_equal(result1, result2)

    def test_augment_return_info(self, sample_image):
        """Test augmentation with return_info flag."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'), \
             patch.object(AugmentAnything, '_extract_features_and_masks') as mock_extract:

            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            # Mock no masks found
            mock_extract.return_value = (None, [])

            augmented, info = augmenter.augment(sample_image, return_info=True)

            # Should return original image and empty info
            np.testing.assert_array_equal(augmented, sample_image)
            assert info == []


class TestAugmentAnythingDataset:
    """Tests for PyTorch Dataset wrapper."""

    @patch('augment_anything.core.AugmentAnything')
    def test_dataset_initialization(self, mock_augmenter, sample_images_dataset):
        """Test dataset initialization."""
        mock_augmenter.return_value.image_data = [
            {'path': Path(f'test_{i}.jpg')} for i in range(5)
        ]

        dataset = AugmentAnythingDataset(
            sample_images_dataset,
            augment_prob=0.5,
            num_swaps=2
        )

        assert len(dataset) == 5
        assert dataset.augment_prob == 0.5
        assert dataset.num_swaps == 2

    @patch('augment_anything.core.AugmentAnything')
    def test_dataset_getitem(self, mock_augmenter, sample_images_dataset, sample_image):
        """Test dataset item retrieval."""
        mock_instance = Mock()
        mock_instance.image_data = [{'path': sample_images_dataset / 'test_0.jpg'}]
        mock_instance.augment.return_value = sample_image
        mock_augmenter.return_value = mock_instance

        dataset = AugmentAnythingDataset(sample_images_dataset, augment_prob=1.0)

        with patch('cv2.imread', return_value=cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)):
            item = dataset[0]
            # Should return augmented image
            assert isinstance(item, np.ndarray)

    @patch('augment_anything.core.AugmentAnything')
    def test_dataset_with_transform(self, mock_augmenter, sample_images_dataset, sample_image):
        """Test dataset with transform function."""
        mock_instance = Mock()
        mock_instance.image_data = [{'path': sample_images_dataset / 'test_0.jpg'}]
        mock_instance.augment.return_value = sample_image
        mock_augmenter.return_value = mock_instance

        def transform(img):
            return img / 255.0

        dataset = AugmentAnythingDataset(
            sample_images_dataset,
            augment_prob=0.0,  # No augmentation
            transform=transform
        )

        with patch('cv2.imread', return_value=cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)):
            item = dataset[0]
            # Transform should be applied
            assert item.dtype == np.float64
            assert item.max() <= 1.0


class TestQuickAugment:
    """Tests for quick_augment helper function."""

    @patch('augment_anything.core.AugmentAnything')
    def test_quick_augment(self, mock_augmenter, sample_image):
        """Test quick augmentation helper."""
        mock_instance = Mock()
        mock_instance.augment.return_value = sample_image
        mock_augmenter.return_value = mock_instance

        result = quick_augment('/fake/path', sample_image, num_swaps=3)

        mock_augmenter.assert_called_once_with('/fake/path')
        mock_instance.augment.assert_called_once_with(sample_image, num_swaps=3)
        np.testing.assert_array_equal(result, sample_image)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_augment_with_invalid_image_path(self):
        """Test augmentation with non-existent image path."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            with patch('cv2.imread', return_value=None):
                result = augmenter.augment('/fake/path/to/image.jpg')
                # Should handle gracefully (returns None or original)

    def test_blend_segment_different_sizes(self, sample_image):
        """Test blending with different sized source and target images."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            target_img = sample_image  # 256x256
            source_img = np.full((512, 512, 3), 128, dtype=np.uint8)  # Different size

            target_mask = np.zeros((256, 256), dtype=bool)
            target_mask[64:192, 64:192] = True
            source_mask = np.ones((512, 512), dtype=bool)

            result = augmenter._blend_segment(target_img, source_img, target_mask, source_mask)

            # Should resize and blend successfully
            assert result.shape == target_img.shape

    def test_compute_descriptor_zero_norm(self):
        """Test descriptor computation with zero-norm features."""
        with patch.object(AugmentAnything, '_init_models'), \
             patch.object(AugmentAnything, '_build_database'):
            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            zero_features = np.zeros((256, 64, 64), dtype=np.float32)
            mask = np.ones((256, 256), dtype=bool)

            descriptor = augmenter._compute_descriptor(zero_features, mask)

            # Should handle zero norm gracefully
            assert descriptor is not None
