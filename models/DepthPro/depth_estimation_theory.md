# Monocular Depth Estimation Formulation

DepthPro predicts depth in a way that yields metric scale depth from a single image. This document explains the mathematical formulation behind this process.

## Inverse Depth Representation

The network's raw output is a dense inverse depth map $\tilde{D_c}(x)$, referred to as *canonical inverse depth*. This is essentially depth in arbitrary units (or disparity-like) normalized to a canonical camera setting.

### From Inverse Depth to Metric Depth

To convert this to actual metric depth $D(x)$ (in meters), DepthPro scales the inverse depth by the camera's field of view. In practice, if $f$ is the predicted (or known) focal length and $W$ is image width, the depth can be recovered by a formula of the form:

$$D(x) \propto \frac{f}{\tilde{D_c}(x) \cdot W}$$

Up to constant factors (the exact formulation is based on the horizontal FOV and camera model). Intuitively, a larger focal length (narrower FOV) implies a larger depth for the same inverse depth value. By predicting $f$, the model rescales the depth map to the correct absolute scale.

This approach allows DepthPro to output metric depths without any known intrinsics, making it truly zero-shot metric depth estimation.

## Training Objectives

DepthPro is trained with a mix of loss functions on the predicted canonical inverse depth. By training on inverse depth (rather than direct depth), the loss emphasizes nearer objects, which have higher inverse-depth values, thereby focusing the model on getting fine details and close-range structure accurate.

### Loss Functions

The loss setup includes:

#### Scale-Invariant Reconstruction Losses

- e.g., scale-and-shift invariant mean squared error (SILog)
- Measures depth error up to an arbitrary scale/shift
- Helps when mixing data from different sources
- Ensures the model learns relative depth relationships even if absolute scale differs

#### Metric Losses

- On datasets where ground truth metric depth is available
- Direct losses like MAE (Mean Absolute Error) and MSE on depth are used
- These ensure accuracy in absolute terms

#### Gradient and Edge Losses

- Loss terms like MAGE (Mean Absolute Gradient Error) or MSGE
- Penalize differences in depth gradients
- Promote sharp depth discontinuities at object boundaries
- Helps the model learn to reproduce crisp edges in the depth map corresponding to object borders
- Important for downstream tasks like segmentation or view synthesis

### Curriculum Training

The paper employs a two-stage training curriculum:

1. **Stage 1**: The model is trained on a broad set of datasets (including synthetic data) with simpler losses to learn general depth structure
   
2. **Stage 2**: Training continues on a focused set of data with additional loss terms (e.g., adding MSE, edge-aware losses) to fine-tune metric accuracy and boundary sharpness

This curriculum, along with a mix of real indoor/outdoor and synthetic scenes, allows DepthPro to generalize well and produce both qualitatively sharp and quantitatively accurate depth predictions.

## Zero-Shot Depth Performance

Through this design, DepthPro achieves zero-shot depth performance:
- It can be applied to new images or videos and produce metric-scaled depth maps with no additional calibration
- The combination of inverse-depth based training and focal length prediction enables the model to output true-scale depth from a single image
- This solves a task that is inherently ambiguous without such learned priors
