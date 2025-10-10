import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import random
import glob
import faiss
import pickle
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class AugmentAnythingDemo:
    """
    Intelligent data augmentation using SAM2 features + FAISS.
    Scales to 1000+ images efficiently.
    """
    
    def __init__(self, image_dir, device='cuda', max_images=1000, 
                 max_masks_per_image=20, cache_path=None):
        self.device = device
        self.max_images = max_images
        self.max_masks_per_image = max_masks_per_image
        self.cache_path = cache_path
        
        # Load SAM2
        print("Loading SAM2 model...")
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
        self.predictor.model.to(device)
        
        # Initialize automatic mask generator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.predictor.model,
            points_per_side=16,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )
        
        # Check if we have cached data
        if cache_path and Path(cache_path).exists():
            print(f"Loading from cache: {cache_path}")
            self.load_cache(cache_path)
        else:
            # Load images
            self.load_images(image_dir)
            
            # Extract features for all images
            self.extract_features()
            
            # Build FAISS index
            self.build_faiss_index()
            
            # Save cache if path provided
            if cache_path:
                self.save_cache(cache_path)
    
    def load_images(self, image_dir):
        """Load all images from directory."""
        image_dir = Path(image_dir)
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(str(image_dir / ext)))
            image_files.extend(glob.glob(str(image_dir / ext.upper())))
        
        image_files = sorted(image_files)[:self.max_images]
        
        print(f"Found {len(image_files)} images")
        
        self.images_data = []
        for img_path in tqdm(image_files, desc="Loading images"):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images_data.append({
                    'path': Path(img_path),
                    'image': img_rgb,
                    'masks': None,
                    'features': None,
                    'segments': []  # Will store segment metadata
                })
        
        print(f"Loaded {len(self.images_data)} images successfully")
    
    def extract_features(self):
        """Extract SAM2 features and masks for all images."""
        print("\nExtracting SAM2 features and generating masks...")
        
        for idx, img_data in enumerate(tqdm(self.images_data, desc="Feature extraction")):
            try:
                # Set image in predictor
                self.predictor.set_image(img_data['image'])
                
                # Get features
                features = self.predictor._features
                
                # Store features (use high res features if available)
                if 'high_res_feats' in features and len(features['high_res_feats']) > 0:
                    img_data['features'] = features['high_res_feats'][0][0]  # First level, batch index 0
                else:
                    # Fallback to image embeddings
                    img_data['features'] = self.predictor.get_image_embedding()[0]
                
                # Generate masks
                masks = self.mask_generator.generate(img_data['image'])
                
                # Sort masks by area and keep top N
                masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                img_data['masks'] = masks[:self.max_masks_per_image]
                
            except Exception as e:
                print(f"\nError processing image {idx}: {e}")
                img_data['features'] = None
                img_data['masks'] = []
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        print("\nBuilding FAISS index...")
        
        # First pass: compute all descriptors and get feature dimension
        all_descriptors = []
        segment_metadata = []
        
        for img_idx, img_data in enumerate(tqdm(self.images_data, desc="Computing descriptors")):
            if img_data['features'] is None or len(img_data['masks']) == 0:
                continue
            
            for mask_idx, mask_dict in enumerate(img_data['masks']):
                mask = mask_dict['segmentation']
                
                # Compute descriptor
                descriptor = self.compute_segment_descriptor(img_data['features'], mask)
                
                if descriptor is not None:
                    all_descriptors.append(descriptor)
                    
                    # Store metadata
                    segment_metadata.append({
                        'image_idx': img_idx,
                        'mask_idx': mask_idx,
                        'mask': mask,
                        'bbox': mask_dict['bbox'],
                        'area': mask_dict['area'],
                    })
        
        if len(all_descriptors) == 0:
            print("No valid descriptors found!")
            return
        
        # Convert to numpy array
        descriptors_array = np.array(all_descriptors).astype('float32')
        feature_dim = descriptors_array.shape[1]
        
        print(f"Total segments indexed: {len(all_descriptors)}")
        print(f"Feature dimension: {feature_dim}")
        
        # Build FAISS index
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(feature_dim)
        
        # Add vectors to index
        self.faiss_index.add(descriptors_array)
        
        # Store metadata
        self.segment_metadata = segment_metadata
        
        print(f"FAISS index built with {self.faiss_index.ntotal} segments")
    
    def compute_segment_descriptor(self, features, mask):
        """
        Compute feature descriptor for a segment.
        """
        if features is None:
            return None
        
        # Convert to numpy if it's a torch tensor
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
            masked_features = features[:, mask_resized]
            descriptor = masked_features.mean(axis=1)
        else:
            descriptor = features.mean(axis=(1, 2))
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        return descriptor
    
    def find_compatible_segment_faiss(self, source_idx, source_mask, source_descriptor, 
                                      top_k=20, similarity_threshold=0.2):
        """
        Find compatible segments using FAISS for fast search.
        """
        if source_descriptor is None:
            return []
        
        # Query FAISS index
        query = source_descriptor.reshape(1, -1).astype('float32')
        similarities, indices = self.faiss_index.search(query, top_k * 3)  # Get more candidates
        
        # Get source segment properties
        source_area = source_mask.sum()
        source_bbox = self.mask_to_bbox(source_mask)
        source_h = source_bbox[2] - source_bbox[0]
        source_w = source_bbox[3] - source_bbox[1]
        source_aspect = source_w / max(source_h, 1)
        
        candidates = []
        
        # Filter candidates
        for sim, idx in zip(similarities[0], indices[0]):
            if sim < similarity_threshold:
                continue
            
            metadata = self.segment_metadata[idx]
            
            # Skip segments from same image
            if metadata['image_idx'] == source_idx:
                continue
            
            # Check shape compatibility
            target_area = metadata['area']
            target_bbox = metadata['bbox']  # [x, y, w, h]
            
            area_ratio = target_area / max(source_area, 1)
            target_aspect = target_bbox[2] / max(target_bbox[3], 1)
            aspect_diff = abs(target_aspect - source_aspect)
            
            # Filters
            if not (0.3 < area_ratio < 3.0 and aspect_diff < 2.0):
                continue
            
            candidates.append({
                'target_idx': metadata['image_idx'],
                'mask': metadata['mask'],
                'bbox': metadata['bbox'],
                'similarity': float(sim),
                'area_ratio': float(area_ratio),
            })
            
            # Early stopping if we have enough good candidates
            if len(candidates) >= top_k:
                break
        
        return candidates
    
    def mask_to_bbox(self, mask):
        """Get bounding box from mask [y1, x1, y2, x2]."""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return [0, 0, 0, 0]
        return [coords[0].min(), coords[1].min(), coords[0].max(), coords[1].max()]
    
    def blend_segment(self, target_image, source_image, source_mask, target_mask):
        """
        Blend source segment into target at masked location.
        """
        H, W = target_image.shape[:2]
        
        # Resize source mask if needed
        if source_mask.shape != (H, W):
            source_mask = cv2.resize(
                source_mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        # Resize source image if needed
        if source_image.shape[:2] != (H, W):
            source_image = cv2.resize(source_image, (W, H))
        
        # Create smooth blend mask
        blend_mask = target_mask.astype(np.float32)
        blend_mask = cv2.GaussianBlur(blend_mask, (21, 21), 7)
        blend_mask = np.clip(blend_mask, 0, 1)[:, :, np.newaxis]
        
        # Blend
        result = (blend_mask * source_image + (1 - blend_mask) * target_image).astype(np.uint8)
        
        return result
    
    def augment_image(self, img_idx, num_augmentations=2):
        """
        Apply intelligent augmentation to an image using FAISS search.
        """
        source_data = self.images_data[img_idx]
        original = source_data['image'].copy()
        
        if source_data['masks'] is None or len(source_data['masks']) == 0:
            print(f"No masks available for image {img_idx}")
            return original, original, []
        
        augmented = original.copy()
        aug_info = []
        
        print(f"\nProcessing: {source_data['path'].name}")
        print(f"Found {len(source_data['masks'])} segments")
        
        # Apply augmentations
        for aug_idx in range(min(num_augmentations, len(source_data['masks']))):
            source_mask_dict = source_data['masks'][aug_idx]
            source_mask = source_mask_dict['segmentation']
            
            print(f"\n  Augmentation {aug_idx + 1}:")
            print(f"    Source segment area: {source_mask_dict['area']}")
            
            # Compute descriptor
            source_descriptor = self.compute_segment_descriptor(
                source_data['features'],
                source_mask
            )
            
            if source_descriptor is None:
                continue
            
            # Find compatible segments using FAISS
            candidates = self.find_compatible_segment_faiss(
                img_idx,
                source_mask,
                source_descriptor,
                top_k=10
            )
            
            if len(candidates) == 0:
                print(f"    No compatible segments found")
                continue
            
            # Use best match
            best = candidates[0]
            target_data = self.images_data[best['target_idx']]
            
            print(f"    Best match: {target_data['path'].name}")
            print(f"    Similarity: {best['similarity']:.3f} (FAISS search)")
            
            # Blend
            augmented = self.blend_segment(
                augmented,
                target_data['image'],
                best['mask'],
                source_mask
            )
            
            aug_info.append({
                'source_mask': source_mask,
                'target_idx': best['target_idx'],
                'target_mask': best['mask'],
                'similarity': best['similarity']
            })
        
        return original, augmented, aug_info
    
    def visualize_augmentation(self, img_idx, save_path=None):
        """
        Create visualization showing the augmentation process.
        """
        original, augmented, aug_info = self.augment_image(img_idx, num_augmentations=2)
        
        source_data = self.images_data[img_idx]
        
        # Create figure
        n_augs = len(aug_info)
        if n_augs == 0:
            print("No augmentations applied, skipping visualization")
            return None
        
        fig = plt.figure(figsize=(20, 6 + 4 * n_augs))
        gs = fig.add_gridspec(2 + n_augs, 4, hspace=0.3, wspace=0.3)
        
        # Row 0: Original and Augmented
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.imshow(original)
        ax1.set_title(f'Original: {source_data["path"].name}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.imshow(augmented)
        ax2.set_title('Augmented (FAISS-Accelerated)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Row 1: Difference and all masks
        ax3 = fig.add_subplot(gs[1, :2])
        diff = np.abs(augmented.astype(float) - original.astype(float)).mean(axis=2)
        im = ax3.imshow(diff, cmap='hot')
        ax3.set_title('Difference Map', fontsize=11)
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)
        
        ax4 = fig.add_subplot(gs[1, 2:])
        mask_vis = self.visualize_masks(original, source_data['masks'][:10])
        ax4.imshow(mask_vis)
        ax4.set_title(f'Detected Segments (top 10 of {len(source_data["masks"])})', fontsize=11)
        ax4.axis('off')
        
        # Rows 2+: Show each augmentation
        for idx, info in enumerate(aug_info):
            row = 2 + idx
            
            # Source segment
            ax_src = fig.add_subplot(gs[row, 0])
            src_overlay = original.copy()
            src_overlay[info['source_mask']] = [255, 0, 0]
            ax_src.imshow(src_overlay)
            ax_src.set_title(f'Source Segment {idx+1}', fontsize=10)
            ax_src.axis('off')
            
            # Source mask only
            ax_src_mask = fig.add_subplot(gs[row, 1])
            ax_src_mask.imshow(info['source_mask'], cmap='gray')
            ax_src_mask.set_title('Source Mask', fontsize=10)
            ax_src_mask.axis('off')
            
            # Target image with segment
            target_data = self.images_data[info['target_idx']]
            ax_tgt = fig.add_subplot(gs[row, 2])
            tgt_overlay = target_data['image'].copy()
            tgt_overlay[info['target_mask']] = [0, 255, 0]
            ax_tgt.imshow(tgt_overlay)
            ax_tgt.set_title(f'Match from: {target_data["path"].name}\nSimilarity: {info["similarity"]:.3f}', 
                           fontsize=10)
            ax_tgt.axis('off')
            
            # Target mask only
            ax_tgt_mask = fig.add_subplot(gs[row, 3])
            ax_tgt_mask.imshow(info['target_mask'], cmap='gray')
            ax_tgt_mask.set_title('Target Mask', fontsize=10)
            ax_tgt_mask.axis('off')
        
        plt.suptitle(f'Augment Anything (FAISS): Scaled to {len(self.images_data)} images, {self.faiss_index.ntotal} segments', 
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved to {save_path}")
        
        return fig
    
    def visualize_masks(self, image, masks):
        """Colorful visualization of masks."""
        overlay = image.copy()
        np.random.seed(42)
        
        for idx, mask_dict in enumerate(masks):
            color = np.random.randint(50, 255, 3)
            mask = mask_dict['segmentation']
            overlay[mask] = overlay[mask] * 0.6 + color * 0.4
        
        return overlay.astype(np.uint8)
    
    def demo_multiple_samples(self, n_samples=3, output_dir=None, seed=None):
        """Generate demos for multiple random images."""
        if seed is not None:
            random.seed(seed)
        
        # Filter images that have masks
        valid_indices = [i for i, d in enumerate(self.images_data) 
                        if d['masks'] is not None and len(d['masks']) > 0]
        
        if len(valid_indices) == 0:
            print("No images with valid masks found!")
            return
        
        n_samples = min(n_samples, len(valid_indices))
        selected = random.sample(valid_indices, n_samples)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*70}")
        print(f"Generating {n_samples} Augment Anything demos...")
        print(f"Searching across {self.faiss_index.ntotal} indexed segments")
        print(f"{'='*70}")
        
        for idx, img_idx in enumerate(selected):
            print(f"\n{'='*70}")
            print(f"Demo {idx+1}/{n_samples}")
            print(f"{'='*70}")
            
            save_path = None
            if output_dir:
                img_name = self.images_data[img_idx]['path'].stem
                save_path = output_path / f"augment_demo_{idx+1}_{img_name}.png"
            
            try:
                fig = self.visualize_augmentation(img_idx, save_path)
                
                if fig is None:
                    continue
                
                if not output_dir:
                    plt.show()
                else:
                    plt.close(fig)
                    
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print("Demo complete!")
        if output_dir:
            print(f"Results saved to: {output_dir}")
    
    def save_cache(self, cache_path):
        """Save processed data to cache for fast reloading."""
        print(f"\nSaving cache to {cache_path}...")
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(cache_path.with_suffix('.faiss')))
        
        # Save metadata
        cache_data = {
            'images_data': [
                {
                    'path': str(d['path']),
                    'masks': d['masks'],
                    'features': d['features'].cpu().numpy() if torch.is_tensor(d['features']) else d['features']
                }
                for d in self.images_data
            ],
            'segment_metadata': self.segment_metadata,
        }
        
        with open(cache_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cache saved successfully!")
    
    def load_cache(self, cache_path):
        """Load from cache."""
        cache_path = Path(cache_path)
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(cache_path.with_suffix('.faiss')))
        
        # Load metadata
        with open(cache_path.with_suffix('.pkl'), 'rb') as f:
            cache_data = pickle.load(f)
        
        # Reconstruct images_data (but don't load actual images yet)
        self.images_data = []
        for d in cache_data['images_data']:
            # Load image
            img = cv2.imread(d['path'])
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images_data.append({
                    'path': Path(d['path']),
                    'image': img_rgb,
                    'masks': d['masks'],
                    'features': d['features'],
                })
        
        self.segment_metadata = cache_data['segment_metadata']
        
        print(f"Loaded {len(self.images_data)} images and {self.faiss_index.ntotal} segments from cache")


def main():
    parser = argparse.ArgumentParser(
        description='Augment Anything - FAISS-accelerated for 1000+ images'
    )
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--n_samples', type=int, default=3,
                       help='Number of augmentation demos to generate')
    parser.add_argument('--output_dir', type=str, default='./augment_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--max_images', type=int, default=1000,
                       help='Maximum number of images to process')
    parser.add_argument('--max_masks_per_image', type=int, default=20,
                       help='Maximum masks to keep per image')
    parser.add_argument('--cache_path', type=str, default=None,
                       help='Path to cache file (for fast reloading)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_save', action='store_true',
                       help='Show results interactively instead of saving')
    
    args = parser.parse_args()
    
    # Create demo
    demo = AugmentAnythingDemo(
        args.image_dir,
        device=args.device,
        max_images=args.max_images,
        max_masks_per_image=args.max_masks_per_image,
        cache_path=args.cache_path
    )
    
    # Run
    output_dir = None if args.no_save else args.output_dir
    demo.demo_multiple_samples(
        n_samples=args.n_samples,
        output_dir=output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()