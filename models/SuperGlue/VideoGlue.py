import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter

# Configuration
SUPERGLUE_MODEL = "magic-leap-community/superglue_outdoor"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.6
FRAME_SCALE = 0.5
FLOW_BINS = 16
SMOOTHING_SIGMA = 1.0

class ComplexFlowVisualizer:
    def __init__(self):
        print("Loading SuperGlue model...")
        self.processor = AutoImageProcessor.from_pretrained(SUPERGLUE_MODEL)
        self.model = AutoModel.from_pretrained(SUPERGLUE_MODEL)
        self.model = self.model.to(DEVICE).eval()
        self.color_wheel = self._make_color_wheel()
        
    def _make_color_wheel(self):
        """Create color wheel for flow visualization"""
        angles = np.linspace(0, 2*np.pi, FLOW_BINS, endpoint=False)
        colors = []
        for angle in angles:
            hue = (angle / (2*np.pi)) * 180
            hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            colors.append(rgb[0,0])
        return np.array(colors)

    @torch.inference_mode()
    def process_frames(self, frame1, frame2):
        """Get keypoint matches between frames"""
        inputs = self.processor([frame1, frame2], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        
        image_sizes = [[(frame1.height, frame1.width) for _ in range(2)]]
        matches = self.processor.post_process_keypoint_matching(
            outputs, image_sizes, threshold=CONF_THRESHOLD
        )[0]
        
        return {
            "keypoints0": matches["keypoints0"].cpu().numpy(),
            "keypoints1": matches["keypoints1"].cpu().numpy(),
            "scores": matches["matching_scores"].cpu().numpy()
        }

    def compute_complex_flow(self, matches, shape):
        """Compute flow field using complex vector field analysis"""
        h, w = shape
        
        # Filter high-confidence matches
        mask = matches["scores"] > CONF_THRESHOLD
        if not mask.any():
            return None, None
        
        points0 = matches["keypoints0"][mask]
        points1 = matches["keypoints1"][mask]
        
        # Convert to complex coordinates
        z0 = points0[:, 0] + 1j * points0[:, 1]
        z1 = points1[:, 0] + 1j * points1[:, 1]
        
        # Compute complex flow vectors
        flow_vectors = z1 - z0
        
        # Create regular grid
        y, x = np.mgrid[0:h, 0:w]
        grid_z = x + 1j * y
        
        # Compute weights based on distance and confidence
        weights = matches["scores"][mask]
        
        # Initialize flow field
        complex_flow = np.zeros_like(grid_z, dtype=np.complex128)
        weight_sum = np.zeros_like(grid_z, dtype=float)
        
        # Compute center of the image
        center_z = (w/2) + 1j * (h/2)
        
        for p0, flow, weight in zip(z0, flow_vectors, weights):
            # Compute distance-based influence
            r = np.abs(grid_z - p0)
            # Use gaussian influence
            influence = np.exp(-r**2 / (100**2)) * weight
            
            # Adjust flow based on position relative to center
            rel_pos = (p0 - center_z) / max(w, h)
            flow_adjustment = flow * (1 - 0.5 * np.abs(rel_pos))
            
            # Proper broadcasting for complex flow
            complex_flow += influence * flow_adjustment
            weight_sum += influence
        
        # Normalize flow
        mask = weight_sum > 0
        complex_flow[mask] /= weight_sum[mask]
        
        # Decompose into real and imaginary parts
        vx = np.real(complex_flow)
        vy = np.imag(complex_flow)
        
        # Apply gaussian smoothing
        vx = gaussian_filter(vx, SMOOTHING_SIGMA)
        vy = gaussian_filter(vy, SMOOTHING_SIGMA)
        
        return vx, vy

    def visualize_flow(self, frame, vx, vy):
        """Visualize flow field with enhanced motion patterns"""
        if vx is None or vy is None:
            return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # Compute flow properties
        magnitude = np.sqrt(vx**2 + vy**2)
        direction = np.arctan2(vy, vx) % (2*np.pi)
        
        # Compute curl and divergence for enhanced visualization
        vy_dx = np.gradient(vy, axis=1)
        vx_dy = np.gradient(vx, axis=0)
        vx_dx = np.gradient(vx, axis=1)
        vy_dy = np.gradient(vy, axis=0)
        
        curl = vy_dx - vx_dy
        div = vx_dx + vy_dy
        
        # Normalize magnitude with enhanced contrast
        max_mag = np.percentile(magnitude, 95)
        magnitude = np.clip(magnitude / max_mag, 0, 1)
        
        # Get dimensions
        h, w = vx.shape
        total_pixels = h * w
        
        # Calculate bin indices for the whole array
        bin_indices = ((direction / (2*np.pi) * FLOW_BINS) % FLOW_BINS).astype(int)
        
        # Create flow visualization
        vis_flow = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors based on direction and magnitude
        for i in range(FLOW_BINS):
            mask = (bin_indices == i)
            base_color = self.color_wheel[i]
            
            # Calculate flow characteristics for this direction
            div_strength = abs(div[mask].mean()) if mask.any() else 0
            curl_strength = abs(curl[mask].mean()) if mask.any() else 0
            
            # Create enhanced color
            enhanced_color = np.array([
                min(255, base_color[0] + curl_strength * 50),  # Blue for rotation
                base_color[1],                                 # Green unchanged
                min(255, base_color[2] + div_strength * 50)   # Red for divergence
            ], dtype=np.uint8)
            
            # Apply color where mask is True
            vis_flow[mask] = enhanced_color * magnitude[mask, np.newaxis]
        
        # Blend with original frame
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        blend = cv2.addWeighted(frame_bgr, 0.7, vis_flow, 0.3, 0)
        
        # Add streamlines
        step = 20
        for y in range(0, h, step):
            for x in range(0, w, step):
                if magnitude[y, x] > 0.1:
                    # Get local flow properties
                    flow_mag = magnitude[y, x]
                    local_curl = abs(curl[y, x])
                    local_div = abs(div[y, x])
                    
                    # Adjust line properties based on flow characteristics
                    length = int(20 * flow_mag)
                    thickness = 1
                    if local_curl > np.percentile(abs(curl), 90):
                        thickness = 2  # Emphasize rotational motion
                    if local_div > np.percentile(abs(div), 90):
                        thickness = 2  # Emphasize expansion/contraction
                    
                    # Draw flow line
                    end_x = int(x + vx[y, x] * length)
                    end_y = int(y + vy[y, x] * length)
                    if 0 <= end_x < w and 0 <= end_y < h:
                        # Get color from the current bin index
                        color = tuple(map(int, self.color_wheel[bin_indices[y, x]]))
                        cv2.line(blend, (x, y), (end_x, end_y), color, thickness)
        
        return blend

    def process_video(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        prev_frame = None
        frame_count = 0
        last_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert and resize
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                scaled_frame = pil_frame.resize(
                    (int(pil_frame.width * FRAME_SCALE),
                     int(pil_frame.height * FRAME_SCALE))
                )
                
                if prev_frame is not None:
                    # Process frames
                    matches = self.process_frames(prev_frame, scaled_frame)
                    vx, vy = self.compute_complex_flow(
                        matches,
                        (scaled_frame.height, scaled_frame.width)
                    )
                    vis_frame = self.visualize_flow(scaled_frame, vx, vy)
                    
                    # Calculate and display FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        current_time = cv2.getTickCount() / cv2.getTickFrequency()
                        fps = 30 / (current_time - last_time)
                        last_time = current_time
                        cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Flow Field', vis_frame)
                
                prev_frame = scaled_frame
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ComplexFlowVisualizer()
    # Use webcam (0) or specify video file path
    #tracker.process_video(0)  # For webcam
    tracker.process_video()