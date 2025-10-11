"""
Build searchable segment database using:
- SAM2 for segmentation
- DINOv3 ConvNeXt-Large for embeddings
- FAISS for similarity search

OPTIMIZED VERSION with:
- COCO RLE compression (better than packbits)
- Quality scores (predicted_iou, stability_score) stored
- Batched DINOv3 inference (10-50x faster)
- Efficient memory management
"""

import numpy as np
import cv2
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle
import gzip
import glob
import faiss
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import List, Dict


class SegmentDatabaseBuilder:
    """
    Build segment database with SAM2 masks and DINOv3 embeddings.
    """
    
    def __init__(self, device='cuda', embedding_model='facebook/dinov3-convnext-large-pretrain-lvd1689m',
                 batch_size=32):
        self.device = device
        self.batch_size = batch_size
        
        print("Loading SAM2 for segmentation...")
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        self.predictor.model.to(device)
        
        # Use COCO RLE output for compressed masks
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.predictor.model,
            points_per_side=16,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            output_mode='coco_rle',  # ← Already compressed!
        )
        
        print(f"Loading DINOv3 model: {embedding_model}...")
        self.dino_processor = AutoImageProcessor.from_pretrained(embedding_model)
        self.dino_model = AutoModel.from_pretrained(embedding_model).to(device)
        self.dino_model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = self.dino_model(dummy_input)
            self.embedding_dim = dummy_output.pooler_output.shape[-1]
        
        print(f"✓ Models loaded. Embedding dim: {self.embedding_dim}, Batch size: {batch_size}")
        
        self.image_paths = []
        self.image_path_to_idx = {}
    
    def _get_image_path_idx(self, image_path):
        """Get or create index for image path (deduplication)."""
        path_str = str(image_path)
        if path_str not in self.image_path_to_idx:
            idx = len(self.image_paths)
            self.image_paths.append(path_str)
            self.image_path_to_idx[path_str] = idx
        return self.image_path_to_idx[path_str]
    
    def rle_to_mask(self, rle):
        """Convert COCO RLE to binary mask for cropping."""
        from pycocotools import mask as mask_utils
        if isinstance(rle, dict):
            return mask_utils.decode(rle).astype(bool)
        return rle
    
    def extract_segment_crop(self, image, mask, bbox, padding=10):
        """
        Extract a cropped region around the segment with padding.
        bbox: [x, y, w, h]
        """
        H, W = image.shape[:2]
        x, y, w, h = [int(v) for v in bbox]
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)
        
        # Crop image and mask
        crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        
        # Create masked version
        masked_crop = crop.copy()
        if mask_crop.any():
            mean_color = crop[mask_crop].mean(axis=0)
            masked_crop[~mask_crop] = mean_color
        
        return masked_crop
    
    def compute_embeddings_batched(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Compute DINOv3 embeddings for multiple crops in batches.
        Much faster than one-by-one processing!
        """
        if len(crops) == 0:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i:i + self.batch_size]
            
            # Convert to PIL
            batch_pil = [Image.fromarray(crop) for crop in batch_crops]
            
            # Process batch with DINOv3
            inputs = self.dino_processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.where(norms > 0, embeddings / norms, embeddings)
            
            all_embeddings.append(embeddings.astype(np.float32))
        
        return np.vstack(all_embeddings)
    
    def process_image(self, image_path, min_segment_area=500):
        """
        Process single image: segment with SAM2, prepare crops for batched embedding.
        Returns: (segment_data_list, crops_list)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return [], []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]
        
        # Generate masks with SAM2 (COCO RLE format)
        try:
            masks = self.mask_generator.generate(image_rgb)
        except Exception as e:
            return [], []
        
        # Get image path index
        image_idx = self._get_image_path_idx(image_path)
        
        # Prepare segment data and crops
        segment_data_list = []
        crops_list = []
        
        for mask_dict in masks:
            area = mask_dict['area']
            
            # Filter small segments
            if area < min_segment_area:
                continue
            
            # Extract data
            bbox = mask_dict['bbox']  # [x, y, w, h]
            rle_mask = mask_dict['segmentation']
            
            # Convert RLE to binary mask for crop extraction
            try:
                binary_mask = self.rle_to_mask(rle_mask)
                crop = self.extract_segment_crop(image_rgb, binary_mask, bbox)
                crops_list.append(crop)
            except Exception as e:
                continue
            
            # Store segment metadata (without embedding yet)
            segment_data = {
                'image_idx': image_idx,
                'image_shape': np.array([H, W], dtype=np.int16),
                'rle_mask': rle_mask,  # ← Already compressed!
                'bbox': np.array(bbox, dtype=np.float32),
                'area': np.int32(area),
                'aspect_ratio': np.float32(bbox[2] / max(bbox[3], 1)),
                'predicted_iou': np.float32(mask_dict['predicted_iou']),  # ← Quality score
                'stability_score': np.float32(mask_dict['stability_score']),  # ← Quality score
            }
            
            segment_data_list.append(segment_data)
        
        return segment_data_list, crops_list
    
    def build_database(self, image_dir, output_path, 
                      max_images=None, min_segment_area=500, 
                      extensions=None):
        """
        Build complete segment database with FAISS index.
        """
        if extensions is None:
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        image_dir = Path(image_dir)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(str(image_dir / ext)))
            image_files.extend(glob.glob(str(image_dir / ext.upper())))
        
        image_files = sorted(image_files)
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n{'='*70}")
        print(f"Building OPTIMIZED Segment Database with DINOv3 Embeddings")
        print(f"{'='*70}")
        print(f"Images: {len(image_files)}")
        print(f"Min segment area: {min_segment_area} pixels")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"DINOv3 batch size: {self.batch_size}")
        print(f"Output: {output_path}")
        print(f"Optimizations: COCO RLE, quality scores, batched embeddings")
        print(f"{'='*70}\n")
        
        # Phase 1: Segment all images and collect crops
        print("Phase 1: Segmenting images with SAM2...")
        all_segment_data = []
        all_crops = []
        
        for img_path in tqdm(image_files, desc="Segmenting"):
            segment_data_list, crops_list = self.process_image(img_path, min_segment_area)
            all_segment_data.extend(segment_data_list)
            all_crops.extend(crops_list)
        
        if len(all_segment_data) == 0:
            print("❌ No segments found! Try lowering min_segment_area")
            return
        
        print(f"✓ Extracted {len(all_segment_data)} segments")
        print(f"  Average: {len(all_segment_data)/max(len(image_files),1):.1f} segments/image")
        
        # Phase 2: Compute embeddings in batches
        print(f"\nPhase 2: Computing DINOv3 embeddings (batched)...")
        embeddings = self.compute_embeddings_batched(all_crops)
        
        # Add embeddings to segment data
        for seg_data, embedding in zip(all_segment_data, embeddings):
            seg_data['embedding'] = embedding
        
        print(f"✓ Computed {len(embeddings)} embeddings")
        
        # Build FAISS index
        print("\nPhase 3: Building FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings)
        
        print(f"✓ FAISS index built: {self.faiss_index.ntotal} segments indexed")
        
        # Prepare metadata (remove embeddings to save space)
        segment_metadata = []
        for seg in all_segment_data:
            meta = {k: v for k, v in seg.items() if k != 'embedding'}
            segment_metadata.append(meta)
        
        # Save everything
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        faiss_path = output_path / "segments.faiss"
        faiss.write_index(self.faiss_index, str(faiss_path))
        print(f"✓ Saved FAISS index: {faiss_path}")
        
        # Save metadata with compression
        metadata_path = output_path / "segments_metadata.pkl.gz"
        with gzip.open(metadata_path, 'wb') as f:
            pickle.dump(segment_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved metadata (compressed): {metadata_path}")
        
        # Save image paths lookup
        paths_path = output_path / "image_paths.pkl.gz"
        with gzip.open(paths_path, 'wb') as f:
            pickle.dump(self.image_paths, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved image paths: {paths_path}")
        
        # Save info
        info = {
            'num_segments': len(all_segment_data),
            'num_images': len(image_files),
            'embedding_dim': self.embedding_dim,
            'min_segment_area': min_segment_area,
            'image_dir': str(image_dir),
            'embedding_model': 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
            'optimizations': ['coco_rle_masks', 'quality_scores', 'batched_embeddings']
        }
        info_path = output_path / "database_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved database info: {info_path}")
        
        # Statistics
        areas = [seg['area'] for seg in all_segment_data]
        aspects = [seg['aspect_ratio'] for seg in all_segment_data]
        ious = [seg['predicted_iou'] for seg in all_segment_data]
        stabilities = [seg['stability_score'] for seg in all_segment_data]
        
        print(f"\n{'='*70}")
        print("Database Statistics")
        print(f"{'='*70}")
        print(f"Total segments: {len(all_segment_data)}")
        print(f"Unique images: {len(self.image_paths)}")
        print(f"\nArea (pixels):")
        print(f"  Mean: {np.mean(areas):.0f}")
        print(f"  Median: {np.median(areas):.0f}")
        print(f"  Range: [{np.min(areas):.0f}, {np.max(areas):.0f}]")
        print(f"\nAspect ratio:")
        print(f"  Mean: {np.mean(aspects):.2f}")
        print(f"  Range: [{np.min(aspects):.2f}, {np.max(aspects):.2f}]")
        print(f"\nQuality scores:")
        print(f"  Predicted IoU - Mean: {np.mean(ious):.3f}, Range: [{np.min(ious):.3f}, {np.max(ious):.3f}]")
        print(f"  Stability - Mean: {np.mean(stabilities):.3f}, Range: [{np.min(stabilities):.3f}, {np.max(stabilities):.3f}]")
        
        total_size = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file())
        print(f"\nTotal database size: {total_size/(1024**2):.1f} MB")
        print(f"{'='*70}")
        print(f"\n✓ Optimized database ready for training!")


def load_segment_database(database_path):
    """
    Helper function to load the optimized database.
    Returns: (faiss_index, segments, image_paths, info)
    """
    database_path = Path(database_path)
    
    # Load FAISS index
    faiss_index = faiss.read_index(str(database_path / "segments.faiss"))
    
    # Load compressed metadata
    with gzip.open(database_path / "segments_metadata.pkl.gz", 'rb') as f:
        segments = pickle.load(f)
    
    # Load image paths
    with gzip.open(database_path / "image_paths.pkl.gz", 'rb') as f:
        image_paths = pickle.load(f)
    
    # Load info
    with open(database_path / "database_info.pkl", 'rb') as f:
        info = pickle.load(f)
    
    return faiss_index, segments, image_paths, info


def rle_to_mask(rle):
    """Helper to decode COCO RLE mask."""
    from pycocotools import mask as mask_utils
    return mask_utils.decode(rle).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description='Build OPTIMIZED segment database with SAM2 + DINOv3 + FAISS',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Where to save the database')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum images to process (None = all)')
    parser.add_argument('--min_segment_area', type=int, default=500,
                       help='Minimum segment area in pixels')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for DINOv3 embedding computation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['*.jpg', '*.jpeg', '*.png'],
                       help='Image extensions to process')
    
    args = parser.parse_args()
    
    # Build database
    builder = SegmentDatabaseBuilder(
        device=args.device,
        batch_size=args.batch_size
    )
    
    builder.build_database(
        image_dir=args.image_dir,
        output_path=args.output_dir,
        max_images=args.max_images,
        min_segment_area=args.min_segment_area,
        extensions=args.extensions
    )


if __name__ == "__main__":
    main()