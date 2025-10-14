"""
Test feature space compatibility between build and query phases.

This test ensures the critical bug fix - using DINOv3 for both phases.
"""

import pytest
import numpy as np
import os


def test_dino_extractor_import():
    """Test that DINOv3Extractor can be imported."""
    from augment_anything import DINOv3Extractor
    assert DINOv3Extractor is not None


@pytest.mark.skipif(os.getenv('SKIP_MODEL_TESTS') == 'true',
                   reason="Skipping model tests in CI")
def test_dino_extractor_basic():
    """Test basic DINOv3Extractor functionality."""
    from augment_anything import DINOv3Extractor

    # Initialize on CPU for testing
    extractor = DINOv3Extractor(device='cpu')

    # Check embedding dimension
    assert extractor.embedding_dim > 0
    print(f"DINOv3 embedding dim: {extractor.embedding_dim}")

    # Test single extraction
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    embedding = extractor.extract_single(test_image)

    assert embedding is not None
    assert embedding.shape == (extractor.embedding_dim,)
    assert abs(np.linalg.norm(embedding) - 1.0) < 0.01  # Should be normalized

    extractor.cleanup()


@pytest.mark.skipif(os.getenv('SKIP_MODEL_TESTS') == 'true',
                   reason="Skipping model tests in CI")
def test_feature_dimension_match():
    """
    CRITICAL TEST: Ensure core.py and faiss_ds.py use the same feature dimension.

    This verifies the fix for the feature space mismatch bug.
    """
    from augment_anything import DINOv3Extractor
    from augment_anything.faiss_ds import SegmentDatabaseBuilder

    # Core.py uses DINOv3Extractor
    core_extractor = DINOv3Extractor(device='cpu')
    core_dim = core_extractor.embedding_dim

    # faiss_ds.py uses SegmentDatabaseBuilder with DINOv3
    builder = SegmentDatabaseBuilder(device='cpu', batch_size=4)
    builder_dim = builder.embedding_dim

    # CRITICAL: These must match!
    assert core_dim == builder_dim, (
        f"Feature dimension mismatch! "
        f"core.py: {core_dim}, faiss_ds.py: {builder_dim}"
    )

    print(f"✓ Feature dimensions match: {core_dim}")

    core_extractor.cleanup()


@pytest.mark.skipif(os.getenv('SKIP_MODEL_TESTS') == 'true',
                   reason="Skipping model tests in CI")
def test_masked_region_extraction():
    """Test masked region extraction produces correct dimensions."""
    from augment_anything import DINOv3Extractor

    extractor = DINOv3Extractor(device='cpu')

    # Create test image and mask
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_mask = np.zeros((512, 512), dtype=bool)
    test_mask[100:200, 100:200] = True

    # Extract embedding
    embedding = extractor.extract_masked_region(test_image, test_mask)

    assert embedding is not None
    assert embedding.shape == (extractor.embedding_dim,)
    assert abs(np.linalg.norm(embedding) - 1.0) < 0.01

    extractor.cleanup()


@pytest.mark.skipif(os.getenv('SKIP_MODEL_TESTS') == 'true',
                   reason="Skipping model tests in CI")
def test_batch_extraction():
    """Test batch extraction produces correct dimensions."""
    from augment_anything import DINOv3Extractor

    extractor = DINOv3Extractor(device='cpu', batch_size=4)

    # Create test crops
    crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
             for _ in range(8)]

    # Extract embeddings
    embeddings = extractor.extract_batch(crops)

    assert embeddings.shape == (8, extractor.embedding_dim)

    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01)

    extractor.cleanup()


def test_extract_segment_crop_helper():
    """Test the helper function for extracting segment crops."""
    from augment_anything import extract_segment_crop_from_bbox

    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Test with SAM2 bbox format [x, y, w, h]
    bbox_sam2 = [100, 100, 50, 50]  # x, y, w, h
    crop = extract_segment_crop_from_bbox(test_image, bbox_sam2, padding=10)

    assert crop.shape[0] > 50  # Height should be bigger due to padding
    assert crop.shape[1] > 50  # Width should be bigger due to padding
    assert crop.shape[2] == 3  # RGB

    # Test with standard bbox format [y1, x1, y2, x2]
    bbox_std = [100, 100, 150, 150]  # y1, x1, y2, x2
    crop = extract_segment_crop_from_bbox(test_image, bbox_std, padding=10)

    assert crop.shape[0] > 50
    assert crop.shape[1] > 50
    assert crop.shape[2] == 3


if __name__ == "__main__":
    print("Running feature compatibility tests...")

    print("\n1. Testing DINOv3Extractor import...")
    test_dino_extractor_import()
    print("   ✓ Import successful")

    if os.getenv('SKIP_MODEL_TESTS') != 'true':
        print("\n2. Testing DINOv3Extractor basic functionality...")
        test_dino_extractor_basic()
        print("   ✓ Basic functionality works")

        print("\n3. Testing feature dimension match...")
        test_feature_dimension_match()
        print("   ✓ Feature dimensions match!")

        print("\n4. Testing masked region extraction...")
        test_masked_region_extraction()
        print("   ✓ Masked region extraction works")

        print("\n5. Testing batch extraction...")
        test_batch_extraction()
        print("   ✓ Batch extraction works")

    print("\n6. Testing bbox extraction helper...")
    test_extract_segment_crop_helper()
    print("   ✓ Bbox extraction helper works")

    print("\n" + "="*70)
    print("✓ All feature compatibility tests passed!")
    print("="*70)
