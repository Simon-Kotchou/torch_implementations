"""
Integration tests for AugmentAnything - End-to-end workflows.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, Mock
import torch
from torch.utils.data import DataLoader

from augment_anything import (
    AugmentAnything,
    AugmentAnythingDataset,
    AugmentConfig,
)


class TestEndToEndWorkflow:
    """Test complete augmentation workflows."""

    @pytest.mark.slow
    @patch('augment_anything.core.SAM2ImagePredictor')
    @patch('augment_anything.core.SAM2AutomaticMaskGenerator')
    def test_full_augmentation_pipeline(self, mock_mask_gen, mock_predictor,
                                       sample_images_dataset, temp_dir):
        """Test full pipeline from dataset to augmented output."""
        # Mock SAM2 components
        mock_predictor_instance = Mock()
        mock_predictor_instance._features = {
            'high_res_feats': [[np.random.randn(256, 64, 64)]]
        }
        mock_predictor.from_pretrained.return_value = mock_predictor_instance

        mock_mask_gen_instance = Mock()
        mock_mask_gen_instance.generate.return_value = [
            {
                'segmentation': np.ones((256, 256), dtype=bool),
                'bbox': [64, 64, 128, 128],
                'area': 16384
            }
        ]
        mock_mask_gen.return_value = mock_mask_gen_instance

        # Create augmenter
        config = AugmentConfig(
            cache_dir=str(temp_dir),
            auto_cache=False,
            max_masks_per_image=5
        )

        augmenter = AugmentAnything(sample_images_dataset, config=config)

        # Load a test image
        test_images = list(sample_images_dataset.glob('*.jpg'))
        if test_images:
            test_img = cv2.imread(str(test_images[0]))
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

            # Augment
            augmented = augmenter.augment(test_img_rgb, num_swaps=2)

            # Verify output
            assert augmented.shape == test_img_rgb.shape
            assert augmented.dtype == np.uint8

    @pytest.mark.slow
    @patch('augment_anything.core.AugmentAnything')
    def test_pytorch_dataloader_integration(self, mock_augmenter,
                                           sample_images_dataset, sample_image):
        """Test integration with PyTorch DataLoader."""
        # Mock augmenter
        mock_instance = Mock()
        mock_instance.image_data = [
            {'path': sample_images_dataset / f'test_{i}.jpg'} for i in range(5)
        ]
        mock_instance.augment.return_value = sample_image
        mock_augmenter.return_value = mock_instance

        # Create dataset
        dataset = AugmentAnythingDataset(
            sample_images_dataset,
            augment_prob=0.5,
            num_swaps=2
        )

        # Create dataloader
        with patch('cv2.imread', return_value=cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)):
            loader = DataLoader(dataset, batch_size=2, shuffle=False)

            # Iterate through batches
            batch_count = 0
            for batch in loader:
                assert isinstance(batch, (torch.Tensor, np.ndarray))
                batch_count += 1

            assert batch_count > 0

    @patch('augment_anything.core.SAM2ImagePredictor')
    @patch('augment_anything.core.SAM2AutomaticMaskGenerator')
    def test_cache_save_and_load(self, mock_mask_gen, mock_predictor,
                                 sample_images_dataset, temp_dir):
        """Test caching mechanism."""
        # Setup mocks
        mock_predictor_instance = Mock()
        mock_predictor_instance._features = {
            'high_res_feats': [[np.random.randn(256, 64, 64)]]
        }
        mock_predictor.from_pretrained.return_value = mock_predictor_instance

        mock_mask_gen_instance = Mock()
        mock_mask_gen_instance.generate.return_value = [
            {
                'segmentation': np.ones((256, 256), dtype=bool),
                'bbox': [64, 64, 128, 128],
                'area': 16384
            }
        ]
        mock_mask_gen.return_value = mock_mask_gen_instance

        config = AugmentConfig(
            cache_dir=str(temp_dir),
            auto_cache=True
        )

        # First run - build and save cache
        augmenter1 = AugmentAnything(
            sample_images_dataset,
            config=config,
            cache_name="test_cache"
        )

        cache_path = temp_dir / "test_cache.augcache.faiss"
        assert cache_path.exists()

        # Second run - load from cache
        augmenter2 = AugmentAnything(
            sample_images_dataset,
            config=config,
            cache_name="test_cache"
        )

        assert len(augmenter2.image_data) > 0


class TestRobustness:
    """Test robustness to various inputs and edge cases."""

    @patch('augment_anything.core.SAM2ImagePredictor')
    @patch('augment_anything.core.SAM2AutomaticMaskGenerator')
    def test_augment_same_image_multiple_times(self, mock_mask_gen, mock_predictor,
                                               sample_image, temp_dir):
        """Test augmenting the same image multiple times."""
        mock_predictor_instance = Mock()
        mock_predictor_instance._features = {
            'high_res_feats': [[np.random.randn(256, 64, 64)]]
        }
        mock_predictor.from_pretrained.return_value = mock_predictor_instance

        mock_mask_gen_instance = Mock()
        mock_mask_gen_instance.generate.return_value = [
            {
                'segmentation': np.ones((256, 256), dtype=bool),
                'bbox': [64, 64, 128, 128],
                'area': 16384
            }
        ]
        mock_mask_gen.return_value = mock_mask_gen_instance

        # Create minimal augmenter
        config = AugmentConfig(auto_cache=False)
        augmenter = AugmentAnything(Path('.'), config=config)
        augmenter.image_data = [{'image': sample_image, 'path': Path('test.jpg')}]
        augmenter.segments = [
            {'img_idx': 0, 'mask': np.ones((256, 256), dtype=bool),
             'bbox': [0, 0, 256, 256], 'area': 65536}
        ]

        # Augment multiple times
        results = []
        for i in range(3):
            result = augmenter.augment(sample_image, num_swaps=1, seed=i)
            results.append(result)

        # All should succeed
        assert all(r.shape == sample_image.shape for r in results)

    def test_various_image_sizes(self, temp_dir):
        """Test handling of different image sizes."""
        with patch('augment_anything.core.SAM2ImagePredictor'), \
             patch('augment_anything.core.SAM2AutomaticMaskGenerator'):

            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            # Test different image sizes
            sizes = [(256, 256), (512, 512), (1024, 768), (300, 400)]

            for h, w in sizes:
                img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                mask = np.ones((h, w), dtype=bool)

                # Should handle all sizes
                descriptor = augmenter._compute_descriptor(
                    np.random.randn(256, 64, 64),
                    mask
                )
                assert descriptor is not None

    def test_zero_swaps(self, sample_image):
        """Test augmentation with num_swaps=0."""
        with patch('augment_anything.core.SAM2ImagePredictor'), \
             patch('augment_anything.core.SAM2AutomaticMaskGenerator'):

            config = AugmentConfig(auto_cache=False)
            augmenter = AugmentAnything(Path('.'), config=config)

            with patch.object(augmenter, '_extract_features_and_masks') as mock_extract:
                mock_extract.return_value = (np.random.randn(256, 64, 64), [])

                # Should return original image
                result = augmenter.augment(sample_image, num_swaps=0)
                np.testing.assert_array_equal(result, sample_image)


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.slow
    @patch('augment_anything.core.AugmentAnything')
    def test_dataset_iteration_performance(self, mock_augmenter,
                                          sample_images_dataset, sample_image):
        """Test that dataset iteration is reasonably fast."""
        import time

        mock_instance = Mock()
        mock_instance.image_data = [
            {'path': sample_images_dataset / f'test_{i}.jpg'} for i in range(5)
        ]
        mock_instance.augment.return_value = sample_image
        mock_augmenter.return_value = mock_instance

        dataset = AugmentAnythingDataset(
            sample_images_dataset,
            augment_prob=0.5
        )

        with patch('cv2.imread', return_value=cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)):
            start = time.time()

            for i in range(len(dataset)):
                _ = dataset[i]

            elapsed = time.time() - start

            # Should complete in reasonable time (very generous for CI)
            assert elapsed < 10.0  # seconds


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_corrupt_cache_file(self, temp_dir):
        """Test handling of corrupted cache file."""
        cache_path = temp_dir / "corrupt.augcache.pkl.gz"
        cache_path.write_bytes(b"corrupted data")

        with patch('augment_anything.core.SAM2ImagePredictor'), \
             patch('augment_anything.core.SAM2AutomaticMaskGenerator'):

            config = AugmentConfig(cache_dir=str(temp_dir), auto_cache=True)

            # Should either rebuild or raise informative error
            try:
                augmenter = AugmentAnything(
                    Path('.'),
                    config=config,
                    cache_name="corrupt"
                )
            except Exception as e:
                # Should fail gracefully with reasonable error
                assert isinstance(e, (ValueError, OSError, pickle.UnpicklingError))

    @patch('augment_anything.core.SAM2ImagePredictor')
    @patch('augment_anything.core.SAM2AutomaticMaskGenerator')
    def test_missing_image_in_cache(self, mock_mask_gen, mock_predictor, temp_dir):
        """Test loading cache when source images have been deleted."""
        # This tests the brittleness issue mentioned in code review
        config = AugmentConfig(cache_dir=str(temp_dir), auto_cache=True)

        with patch.object(AugmentAnything, '_init_models'):
            augmenter = AugmentAnything(Path('.'), config=config)

            # Manually create cache data with non-existent paths
            augmenter.image_data = [
                {'path': '/nonexistent/image.jpg', 'masks': [], 'features': np.random.randn(256, 64, 64)}
            ]

            augmenter._save_cache()

            # Try to load - should handle missing images
            with patch('cv2.imread', return_value=None):
                augmenter._load_cache()
                # image_data should handle None images or skip them
