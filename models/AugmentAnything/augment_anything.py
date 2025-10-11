"""
AugmentAnything - Semantic-Aware Data Augmentation

A clean, first-principles implementation that makes augmentation super easy:

    # 3 lines to augment your dataset!
    augmenter = AugmentAnything('/path/to/images')
    augmented_image = augmenter.augment(image)

Or use with PyTorch:

    dataset = AugmentAnythingDataset('/path/to/images', transform=my_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Tuple, Union, Callable
import pickle
import gzip
from tqdm import tqdm
import faiss
from dataclasses import dataclass
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


@dataclass
class AugmentConfig:
    """Configuration for AugmentAnything"""

    # Model settings
    sam_model: str = "facebook/sam2-hiera-tiny"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Segmentation settings
    points_per_side: int = 16
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.85
    min_mask_area: int = 500
    max_masks_per_image: int = 20

    # Search settings
    similarity_threshold: float = 0.2
    area_ratio_range: Tuple[float, float] = (0.3, 3.0)
    aspect_ratio_diff_max: float = 2.0

    # Blending settings
    blend_kernel_size: int = 21
    blend_sigma: float = 7.0

    # Performance settings
    cache_dir: Optional[str] = "./augment_cache"
    auto_cache: bool = True


class AugmentAnything:
    """
    Semantic-aware data augmentation using SAM2 + FAISS.

    First Principles:
    1. Segment images into semantic parts
    2. Build searchable feature database
    3. Find compatible segments by similarity + constraints
    4. Blend segments to create augmented images

    Usage:
        # Simple
        augmenter = AugmentAnything('/path/to/images')
        augmented = augmenter.augment(image)

        # With config
        config = AugmentConfig(sam_model="facebook/sam2-hiera-large")
        augmenter = AugmentAnything('/path/to/images', config=config)

        # Augment with control
        augmented = augmenter.augment(image, num_swaps=3, seed=42)
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        config: Optional[AugmentConfig] = None,
        cache_name: Optional[str] = None
    ):
        self.config = config or AugmentConfig()
        self.image_dir = Path(image_dir)

        # Set cache path
        if self.config.cache_dir and self.config.auto_cache:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(exist_ok=True, parents=True)
            cache_name = cache_name or self.image_dir.name
            self.cache_path = cache_dir / f"{cache_name}.augcache"
        else:
            self.cache_path = None

        # Try loading from cache
        if self.cache_path and self.cache_path.exists():
            print(f"Loading from cache: {self.cache_path}")
            self._load_cache()
        else:
            # Build from scratch
            print("Initializing AugmentAnything...")
            self._init_models()
            self._build_database()

            if self.cache_path:
                self._save_cache()

    def _init_models(self):
        """Initialize SAM2 model"""
        print(f"Loading SAM2: {self.config.sam_model}")
        self.predictor = SAM2ImagePredictor.from_pretrained(self.config.sam_model)
        self.predictor.model.to(self.config.device)

        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.predictor.model,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

    def _build_database(self):
        """Build segment database from image directory"""
        # Load images
        image_files = self._find_images()
        print(f"Found {len(image_files)} images")

        self.image_data = []
        all_descriptors = []
        self.segments = []

        # Process each image
        for img_path in tqdm(image_files, desc="Building database"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract features and masks
            features, masks = self._extract_features_and_masks(img_rgb)

            if features is None or len(masks) == 0:
                continue

            # Store image data
            img_idx = len(self.image_data)
            self.image_data.append({
                'path': img_path,
                'image': img_rgb,
                'features': features,
                'masks': masks[:self.config.max_masks_per_image]
            })

            # Compute descriptors for each mask
            for mask_idx, mask_dict in enumerate(masks[:self.config.max_masks_per_image]):
                mask = mask_dict['segmentation']
                descriptor = self._compute_descriptor(features, mask)

                if descriptor is not None:
                    all_descriptors.append(descriptor)
                    self.segments.append({
                        'img_idx': img_idx,
                        'mask_idx': mask_idx,
                        'mask': mask,
                        'bbox': mask_dict['bbox'],
                        'area': mask_dict['area']
                    })

        # Build FAISS index
        if len(all_descriptors) == 0:
            raise ValueError("No valid segments found!")

        descriptors_array = np.array(all_descriptors).astype('float32')
        self.feature_dim = descriptors_array.shape[1]

        print(f"Building FAISS index with {len(all_descriptors)} segments...")
        self.index = faiss.IndexFlatIP(self.feature_dim)
        self.index.add(descriptors_array)

        print(f"✓ Database ready: {len(self.image_data)} images, {len(self.segments)} segments")

    def _find_images(self) -> List[Path]:
        """Find all images in directory"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(self.image_dir.glob(ext))
            image_files.extend(self.image_dir.glob(ext.upper()))
        return sorted(image_files)

    def _extract_features_and_masks(self, image: np.ndarray):
        """Extract SAM2 features and generate masks"""
        try:
            # Set image
            self.predictor.set_image(image)

            # Get features
            features = self.predictor._features
            if 'high_res_feats' in features and len(features['high_res_feats']) > 0:
                features = features['high_res_feats'][0][0]
            else:
                features = self.predictor.get_image_embedding()[0]

            # Generate masks
            masks = self.mask_generator.generate(image)
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)

            return features, masks

        except Exception as e:
            print(f"Error processing image: {e}")
            return None, []

    def _compute_descriptor(self, features, mask):
        """Compute normalized feature descriptor for a segment"""
        if features is None:
            return None

        # Convert to numpy if needed
        if torch.is_tensor(features):
            features = features.cpu().numpy()

        C, fH, fW = features.shape
        H, W = mask.shape

        # Resize mask to feature resolution
        mask_resized = cv2.resize(
            mask.astype(np.float32),
            (fW, fH),
            interpolation=cv2.INTER_LINEAR
        ) > 0.5

        # Masked pooling
        if mask_resized.sum() > 10:
            descriptor = features[:, mask_resized].mean(axis=1)
        else:
            descriptor = features.mean(axis=(1, 2))

        # L2 normalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm

        return descriptor

    def augment(
        self,
        image: Union[np.ndarray, str, Path],
        num_swaps: int = 2,
        seed: Optional[int] = None,
        return_info: bool = False
    ):
        """
        Augment an image by swapping segments with similar ones from database.

        Args:
            image: Input image (numpy array or path)
            num_swaps: Number of segments to swap
            seed: Random seed for reproducibility
            return_info: If True, return (augmented, swap_info)

        Returns:
            augmented image (and optionally swap info)
        """
        if seed is not None:
            np.random.seed(seed)

        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract features and masks for this image
        features, masks = self._extract_features_and_masks(image)

        if features is None or len(masks) == 0:
            return (image, []) if return_info else image

        augmented = image.copy()
        swap_info = []

        # Perform swaps
        for i in range(min(num_swaps, len(masks))):
            source_mask = masks[i]['segmentation']
            source_desc = self._compute_descriptor(features, source_mask)

            if source_desc is None:
                continue

            # Find compatible segment
            match = self._find_best_match(source_mask, source_desc, masks[i]['area'])

            if match is None:
                continue

            # Blend the matched segment
            target_img = self.image_data[match['img_idx']]['image']
            target_mask = match['mask']

            augmented = self._blend_segment(augmented, target_img, source_mask, target_mask)

            swap_info.append({
                'source_mask': source_mask,
                'target_img_path': str(self.image_data[match['img_idx']]['path']),
                'similarity': match['similarity']
            })

        return (augmented, swap_info) if return_info else augmented

    def _find_best_match(self, source_mask, source_descriptor, source_area):
        """Find best matching segment using FAISS"""
        # Search
        query = source_descriptor.reshape(1, -1).astype('float32')
        similarities, indices = self.index.search(query, k=50)

        # Get source properties
        source_bbox = self._mask_to_bbox(source_mask)
        source_h = source_bbox[2] - source_bbox[0]
        source_w = source_bbox[3] - source_bbox[1]
        source_aspect = source_w / max(source_h, 1)

        # Filter by constraints
        for sim, idx in zip(similarities[0], indices[0]):
            if sim < self.config.similarity_threshold:
                continue

            segment = self.segments[idx]

            # Area ratio check
            area_ratio = segment['area'] / max(source_area, 1)
            if not (self.config.area_ratio_range[0] < area_ratio < self.config.area_ratio_range[1]):
                continue

            # Aspect ratio check
            bbox = segment['bbox']
            target_aspect = bbox[2] / max(bbox[3], 1)
            aspect_diff = abs(target_aspect - source_aspect)
            if aspect_diff > self.config.aspect_ratio_diff_max:
                continue

            # Valid match found
            return {
                'img_idx': segment['img_idx'],
                'mask': segment['mask'],
                'similarity': float(sim)
            }

        return None

    def _mask_to_bbox(self, mask):
        """Get bounding box [y1, x1, y2, x2]"""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return [0, 0, 0, 0]
        return [coords[0].min(), coords[1].min(), coords[0].max(), coords[1].max()]

    def _blend_segment(self, target_img, source_img, target_mask, source_mask):
        """Blend source segment into target"""
        H, W = target_img.shape[:2]

        # Resize source if needed
        if source_mask.shape != (H, W):
            source_mask = cv2.resize(
                source_mask.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        if source_img.shape[:2] != (H, W):
            source_img = cv2.resize(source_img, (W, H))

        # Create smooth blend mask
        blend_mask = target_mask.astype(np.float32)
        blend_mask = cv2.GaussianBlur(
            blend_mask,
            (self.config.blend_kernel_size, self.config.blend_kernel_size),
            self.config.blend_sigma
        )
        blend_mask = np.clip(blend_mask, 0, 1)[:, :, np.newaxis]

        # Blend
        result = (blend_mask * source_img + (1 - blend_mask) * target_img).astype(np.uint8)
        return result

    def _save_cache(self):
        """Save database to cache"""
        print(f"Saving cache to {self.cache_path}...")

        # Save FAISS index
        faiss.write_index(self.index, str(self.cache_path.with_suffix('.faiss')))

        # Save everything else
        cache_data = {
            'image_data': [
                {
                    'path': str(d['path']),
                    'masks': d['masks'],
                    'features': d['features'].cpu().numpy() if torch.is_tensor(d['features']) else d['features']
                }
                for d in self.image_data
            ],
            'segments': self.segments,
            'feature_dim': self.feature_dim,
            'config': self.config
        }

        with gzip.open(self.cache_path.with_suffix('.pkl.gz'), 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("✓ Cache saved")

    def _load_cache(self):
        """Load database from cache"""
        # Load FAISS index
        self.index = faiss.read_index(str(self.cache_path.with_suffix('.faiss')))

        # Load data
        with gzip.open(self.cache_path.with_suffix('.pkl.gz'), 'rb') as f:
            cache_data = pickle.load(f)

        # Reconstruct image_data (reload images)
        self.image_data = []
        for d in cache_data['image_data']:
            img = cv2.imread(d['path'])
            if img is not None:
                self.image_data.append({
                    'path': Path(d['path']),
                    'image': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    'masks': d['masks'],
                    'features': d['features']
                })

        self.segments = cache_data['segments']
        self.feature_dim = cache_data['feature_dim']
        self.config = cache_data.get('config', self.config)

        # Reinitialize SAM2 for augmentation
        self._init_models()

        print(f"✓ Loaded from cache: {len(self.image_data)} images, {len(self.segments)} segments")


class AugmentAnythingDataset(Dataset):
    """
    PyTorch Dataset with built-in augmentation.

    Usage:
        dataset = AugmentAnythingDataset(
            '/path/to/images',
            augment_prob=0.5,
            transform=transforms.Compose([...])
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        augment_prob: float = 0.5,
        num_swaps: int = 2,
        transform: Optional[Callable] = None,
        config: Optional[AugmentConfig] = None,
        cache_name: Optional[str] = None
    ):
        self.augmenter = AugmentAnything(image_dir, config=config, cache_name=cache_name)
        self.augment_prob = augment_prob
        self.num_swaps = num_swaps
        self.transform = transform

        # Get list of images
        self.images = [d['path'] for d in self.augmenter.image_data]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation with probability
        if np.random.rand() < self.augment_prob:
            image = self.augmenter.augment(image, num_swaps=self.num_swaps)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image

    def get_sample(self, idx: int, augmented: bool = True):
        """Get a sample for visualization (returns numpy array)"""
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if augmented:
            image = self.augmenter.augment(image, num_swaps=self.num_swaps)

        return image


def quick_augment(image_dir: str, image: np.ndarray, num_swaps: int = 2) -> np.ndarray:
    """
    Quick augmentation without configuration.

    Args:
        image_dir: Directory containing reference images
        image: Image to augment
        num_swaps: Number of segment swaps

    Returns:
        Augmented image
    """
    augmenter = AugmentAnything(image_dir)
    return augmenter.augment(image, num_swaps=num_swaps)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="AugmentAnything - Easy semantic augmentation")
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--num_swaps', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='./augment_outputs')
    parser.add_argument('--cache_name', type=str, default=None)

    args = parser.parse_args()

    # Create augmenter
    augmenter = AugmentAnything(args.image_dir, cache_name=args.cache_name)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Augment random samples
    indices = np.random.choice(len(augmenter.image_data), min(args.num_samples, len(augmenter.image_data)), replace=False)

    for idx in indices:
        original = augmenter.image_data[idx]['image']
        augmented, info = augmenter.augment(original, num_swaps=args.num_swaps, return_info=True)

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(augmented)
        axes[1].set_title(f'Augmented ({len(info)} swaps)')
        axes[1].axis('off')

        save_path = output_dir / f"augment_{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")

    print(f"\n✓ Done! Saved {len(indices)} augmented images to {output_dir}")
