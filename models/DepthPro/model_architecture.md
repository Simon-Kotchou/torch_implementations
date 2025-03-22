# DepthPro Model Architecture

DepthPro's architecture is designed to efficiently handle high-resolution input using a combination of global and patch-based vision transformers. The model operates on a fixed 1536×1536 resolution input (cropping or resizing the image as needed).

## Multi-Scale Processing Pipeline

The input image is processed at multiple scales to capture both global context and local details:

### Image Encoder (Global Context)

The full image is downsampled to a base resolution (384×384) and passed through a ViT-based image encoder. This provides global context, anchoring the depth prediction in the overall scene layout. DepthPro uses a ViT-L DinoV2 model as the backbone for this encoder by default.

### Patch Encoder (Local Details)

In parallel, the model creates higher-resolution views:
- The image at 1536² (original) and 768² are each tiled into overlapping 384×384 patches
- 5×5 patches for 1536², and 3×3 for 768², with overlap to avoid seams
- These patches, plus the 384² image (which itself is a 1×1 "patch"), yield 35 total patches covering the image
- Each patch is processed by a shared ViT patch encoder (also using a DinoV2/ViT backbone)
- This extracts features from local regions at multiple scales
- Sharing the encoder weights across scales encourages scale-invariant features and keeps the model efficient
- All patch features are then merged back into their spatial positions to form multiscale feature maps

### Multi-Scale Fusion Decoder

The DepthPro decoder is a DPT-like (Dense Prediction Transformer) fusion module. It:
- Takes the combined patch feature maps along with the global image features
- Progressively upsamples and fuses them to output a full-resolution depth map
- Uses skip-connections at multiple feature scales merged via attention and convolution
- Produces an inverse depth map at 1536×1536 resolution as the output prediction

### Focal Length (FOV) Head

In addition to depth, DepthPro can estimate the camera's horizontal field-of-view (or equivalently focal length):
- A small convolutional head uses features from the depth network
- A dedicated ViT encoder for FOV predicts the camera focal length (in pixels)
- This enables the model to output metric scale depth even when camera intrinsics are unknown
- The focal length head can be disabled to save computation if absolute scaling is not needed

## Efficiency Considerations

This design balances detail and speed:
- Processing patches independently at multiple scales is trivially parallelizable
- Avoids the quadratic cost of a single huge ViT on a 1536² image
- By keeping transformer input size modest (384² per patch) and reusing a pretrained backbone, DepthPro achieves high resolution with manageable memory and runtime
- The fixed input size ensures consistent runtime regardless of image size and avoids out-of-memory issues
- The authors report DepthPro is orders of magnitude faster than prior high-res models while delivering better accuracy and sharper boundaries

## Architecture Diagram

```
Input Image (1536×1536)
│
├─────────────────┬─────────────────┐
│                 │                 │
▼                 ▼                 ▼
Downsample        Downsample        Global View
to 768×768        to 384×384        (384×384)
│                 │                 │
▼                 ▼                 ▼
Divide into       ViT Image         ViT Patch
3×3 patches       Encoder           Encoder
│                 │                 │
▼                 ▼                 ▼
Divide 1536×1536  Global            Patch Features
into 5×5 patches  Features          (1×1 patch)
│                 │                 │
▼                 │                 │
Apply shared ViT  │                 │
Patch Encoder     │                 │
to all patches    │                 │
│                 │                 │
▼                 │                 │
Patch Features    │                 │
(35 total)        │                 │
│                 │                 │
└─────────────────┼─────────────────┘
                  │
                  ▼
          Multi-Scale Fusion Decoder
                  │
                  ├─────────────────┐
                  │                 │
                  ▼                 ▼
          Inverse Depth Map   Focal Length
          (1536×1536)         Estimation Head
                              │
                              ▼
                        Field of View
                        (in degrees)
```
