"""
Build searchable segment database using:
- SAM2 for segmentation
- DINOv3 ConvNeXt-Large for embeddings
- FAISS for similarity search

OPTIMIZED VERSION with:
- Packbits compression for masks (~8x smaller)
- Image path deduplication (store as indices)
- Compressed pickle with highest protocol
- Efficient data types (int16/int32)
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


class SegmentDatabaseBuilder:
    """
    Build segment database with SAM2 masks and DINOv3 embeddings.
    """
    
    def __init__(self, device='cuda', embedding_model='facebook/dinov3-convnext-large-pretrain-lvd1689m'):
        self.device = device
        
        print("Loading SAM2 for segmentation...")
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        self.predictor.model.to(device)
        
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.predictor.model,
            points_per_side=16,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
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
        
        print(f"✓ Models loaded. Embedding dim: {self.embedding_dim}")
        
        self.segments = []
        self.faiss_index = None
        self.image_paths = []  # Deduplicated list of image paths
        self.image_path_to_idx = {}  # Map path to index
    
    def _compress_mask(self, mask):
        """
        Compress binary mask using packbits (8x compression).
        Returns: (packed_bytes, original_shape)
        """
        # Ensure boolean
        mask_bool = mask.astype(bool)
        # Pack into bytes
        packed = np.packbits(mask_bool.ravel())
        return packed, mask_bool.shape
    
    def _decompress_mask(self, packed, shape):
        """
        Decompress packbits mask back to original shape.
        """
        unpacked = np.unpackbits(packed)
        # Trim to original size (packbits pads to multiple of 8)
        total_pixels = shape[0] * shape[1]
        unpacked = unpacked[:total_pixels]
        return unpacked.reshape(shape).astype(np.uint8)
    
    def _get_image_path_idx(self, image_path):
        """
        Get or create index for image path (deduplication).
        """
        path_str = str(image_path)
        if path_str not in self.image_path_to_idx:
            idx = len(self.image_paths)
            self.image_paths.append(path_str)
            self.image_path_to_idx[path_str] = idx
        return self.image_path_to_idx[path_str]
    
    def extract_segment_crop(self, image, mask, bbox, padding=10):
        """
        Extract a cropped region around the segment with padding.
        bbox: [x, y, w, h]
        """
        H, W = image.shape[:2]
        x, y, w, h = bbox
        
        # Convert to integers (SAM2 returns floats)
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)
        
        # Crop image
        crop = image[y1:y2, x1:x2]
        
        # Apply mask to crop
        mask_crop = mask[y1:y2, x1:x2]
        
        # Create masked version (background to mean color)
        masked_crop = crop.copy()
        if not mask_crop.any():
            return crop
        
        # Set background to mean color of the segment
        mean_color = crop[mask_crop].mean(axis=0)
        masked_crop[~mask_crop] = mean_color
        
        return masked_crop
    
    def compute_segment_embedding(self, image, mask, bbox):
        """
        Compute DINOv3 embedding for a segment.
        """
        # Extract segment crop
        segment_crop = self.extract_segment_crop(image, mask, bbox)
        
        # Convert to PIL
        segment_pil = Image.fromarray(segment_crop)
        
        # Process with DINOv3
        inputs = self.dino_processor(images=segment_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            embedding = outputs.pooler_output.squeeze().cpu().numpy()
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def process_image(self, image_path, min_segment_area=500):
        """
        Process single image: segment with SAM2, embed with DINOv3.
        Returns: (list of segments, failed count)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return [], 0
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]
        
        # Generate masks with SAM2
        try:
            masks = self.mask_generator.generate(image_rgb)
        except Exception as e:
            return [], 0
        
        # Get image path index
        image_idx = self._get_image_path_idx(image_path)
        
        # Process each segment
        image_segments = []
        failed_count = 0
        
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            area = mask_dict['area']
            bbox = mask_dict['bbox']  # [x, y, w, h]
            
            # Filter small segments
            if area < min_segment_area:
                continue
            
            # Compute DINOv3 embedding
            try:
                embedding = self.compute_segment_embedding(image_rgb, mask, bbox)
            except Exception as e:
                failed_count += 1
                continue
            
            # Compress mask with packbits
            packed_mask, mask_shape = self._compress_mask(mask)
            
            # Store segment info with optimized data types
            segment = {
                'image_idx': image_idx,  # Index instead of string
                'image_shape': np.array([H, W], dtype=np.int16),  # Smaller int type
                'packed_mask': packed_mask,  # Compressed mask
                'mask_shape': np.array(mask_shape, dtype=np.int16),  # For decompression
                'bbox': np.array(bbox, dtype=np.int32),  # Int instead of float
                'area': np.int32(area),
                'aspect_ratio': np.float32(bbox[2] / max(bbox[3], 1)),
                'embedding': embedding
            }
            
            image_segments.append(segment)
        
        return image_segments, failed_count
    
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
        print(f"Output: {output_path}")
        print(f"Optimizations: packbits masks, deduplicated paths, compressed storage")
        print(f"{'='*70}\n")
        
        # Process all images
        all_segments = []
        total_failed = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            segments, failed = self.process_image(img_path, min_segment_area)
            all_segments.extend(segments)
            total_failed += failed
        
        print(f"\n✓ Extracted {len(all_segments)} segments from {len(image_files)} images")
        print(f"  Average: {len(all_segments)/max(len(image_files),1):.1f} segments/image")
        if total_failed > 0:
            print(f"  Skipped {total_failed} segments (failed embedding extraction)")
        
        if len(all_segments) == 0:
            print("❌ No segments found! Try lowering min_segment_area")
            return
        
        # Build FAISS index
        print("\nBuilding FAISS index for similarity search...")
        embeddings = np.stack([seg['embedding'] for seg in all_segments])
        
        # IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings)
        
        print(f"✓ FAISS index built: {self.faiss_index.ntotal} segments indexed")
        
        # Prepare metadata (remove embeddings to save space)
        segment_metadata = []
        for seg in all_segments:
            meta = {k: v for k, v in seg.items() if k != 'embedding'}
            segment_metadata.append(meta)
        
        # Save everything
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        faiss_path = output_path / "segments.faiss"
        faiss.write_index(self.faiss_index, str(faiss_path))
        print(f"✓ Saved FAISS index: {faiss_path}")
        
        # Save metadata with compression (gzip + pickle protocol 5)
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
            'num_segments': len(all_segments),
            'num_images': len(image_files),
            'embedding_dim': self.embedding_dim,
            'min_segment_area': min_segment_area,
            'image_dir': str(image_dir),
            'embedding_model': 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
            'optimizations': ['packbits_masks', 'deduplicated_paths', 'compressed_pickle']
        }
        info_path = output_path / "database_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved database info: {info_path}")
        
        # Statistics
        areas = [seg['area'] for seg in all_segments]
        aspects = [seg['aspect_ratio'] for seg in all_segments]
        
        # Calculate space savings
        uncompressed_mask_size = sum(s['mask_shape'][0] * s['mask_shape'][1] for s in segment_metadata)
        compressed_mask_size = sum(len(s['packed_mask']) for s in segment_metadata)
        compression_ratio = uncompressed_mask_size / max(compressed_mask_size, 1)
        
        print(f"\n{'='*70}")
        print("Database Statistics")
        print(f"{'='*70}")
        print(f"Total segments: {len(all_segments)}")
        print(f"Unique images: {len(self.image_paths)}")
        print(f"\nArea (pixels):")
        print(f"  Mean: {np.mean(areas):.0f}")
        print(f"  Median: {np.median(areas):.0f}")
        print(f"  Range: [{np.min(areas):.0f}, {np.max(areas):.0f}]")
        print(f"\nAspect ratio:")
        print(f"  Mean: {np.mean(aspects):.2f}")
        print(f"  Median: {np.median(aspects):.2f}")
        print(f"  Range: [{np.min(aspects):.2f}, {np.max(aspects):.2f}]")
        
        print(f"\nMask compression:")
        print(f"  Uncompressed: {uncompressed_mask_size/(1024**2):.1f} MB")
        print(f"  Compressed: {compressed_mask_size/(1024**2):.1f} MB")
        print(f"  Ratio: {compression_ratio:.1f}x smaller")
        
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


def decompress_mask(segment):
    """
    Helper to decompress a mask from a segment.
    """
    packed = segment['packed_mask']
    shape = tuple(segment['mask_shape'])
    unpacked = np.unpackbits(packed)
    total_pixels = shape[0] * shape[1]
    unpacked = unpacked[:total_pixels]
    return unpacked.reshape(shape).astype(np.uint8)


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
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['*.jpg', '*.jpeg', '*.png'],
                       help='Image extensions to process')
    
    args = parser.parse_args()
    
    # Build database
    builder = SegmentDatabaseBuilder(device=args.device)
    
    builder.build_database(
        image_dir=args.image_dir,
        output_path=args.output_dir,
        max_images=args.max_images,
        min_segment_area=args.min_segment_area,
        extensions=args.extensions
    )


if __name__ == "__main__":
    main()