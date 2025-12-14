"""
Segmentation algorithms with object tracking.
Optimized for CPU with memory management.
"""

import os
import gc
import cv2
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict

from ..config import (
    SAM2_AVAILABLE, SAM3_AVAILABLE, CUDA_AVAILABLE,
    get_checkpoint
)


def demo_segment(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    bboxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Color-based segmentation using HSV matching."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Process points
    for px, py in points:
        px, py = max(0, min(w-1, int(px))), max(0, min(h-1, int(py)))
        c = hsv[py, px]
        lower = np.array([max(0, int(c[0])-20), 40, 40])
        upper = np.array([min(180, int(c[0])+20), 255, 255])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    
    # Process bboxes
    for x1, y1, x2, y2 in bboxes:
        x1, y1 = max(0, min(w-1, int(x1))), max(0, min(h-1, int(y1)))
        x2, y2 = max(0, min(w-1, int(x2))), max(0, min(h-1, int(y2)))
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        
        roi = hsv[y1:y2, x1:x2]
        if roi.size > 0:
            avg_h = int(np.mean(roi[:, :, 0]))
            lower = np.array([max(0, avg_h-25), 30, 30])
            upper = np.array([min(180, avg_h+25), 255, 255])
            region = cv2.inRange(hsv, lower, upper)
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            bbox_mask[y1:y2, x1:x2] = region[y1:y2, x1:x2]
            mask = cv2.bitwise_or(mask, bbox_mask)
    
    if mask.sum() == 0:
        return frame
    
    # Apply overlay
    overlay = np.zeros_like(frame)
    overlay[:] = color
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (frame * (1 - mask_3ch * 0.4) + overlay * mask_3ch * 0.4).astype(np.uint8)
    
    # Draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 0), 2)
    
    return result


def apply_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 200, 255),
    alpha: float = 0.5
) -> np.ndarray:
    """Apply colored mask overlay to frame."""
    result = frame.copy()
    
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    
    overlay = np.zeros_like(result)
    overlay[mask > 0] = color
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    
    # Draw contour
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 0), 2)
    
    return result


class SAM2Tracker:
    """
    SAM2 Video Object Tracker.
    Optimized for CPU with frame sampling to reduce memory usage.
    """
    
    def __init__(self):
        self.predictor = None
        self.model_size = "tiny"
        self.device = "cpu"
        self.frame_step = 1  # Process every Nth frame
    
    def load(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        status_cb: Callable = None,
        download_cb: Callable = None
    ):
        """Load SAM2 model."""
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 not installed")
        
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        
        # Validate device
        if device == "cuda" and not CUDA_AVAILABLE:
            device = "cpu"
            if status_cb:
                status_cb("CUDA unavailable, using CPU")
        
        self.device = device
        self.model_size = model_size
        
        if status_cb:
            status_cb(f"Loading SAM2 {model_size}...")
        
        config, ckpt = get_checkpoint(model_size, download_cb)
        if not ckpt:
            raise RuntimeError("Failed to get checkpoint")
        
        self.predictor = build_sam2_video_predictor(config, ckpt, device=device)
        
        if status_cb:
            status_cb(f"SAM2 {model_size} ready on {device.upper()}!")
    
    def track(
        self,
        video_path: str,
        points: List[Tuple[int, int]],
        bboxes: List[Tuple[int, int, int, int]],
        max_frames: int = 100,
        status_cb: Callable = None,
        progress_cb: Callable = None,
        frame_cb: Callable = None
    ) -> Dict[int, np.ndarray]:
        """
        Track objects through video.
        
        Args:
            video_path: Path to video file
            points: List of (x, y) points
            bboxes: List of (x1, y1, x2, y2) boxes
            max_frames: Max frames to process (memory limit)
            status_cb: Status callback
            progress_cb: Progress callback (0-100)
        
        Returns:
            Dict mapping frame_idx -> mask
        """
        import torch
        import tempfile
        import shutil
        
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        
        temp_dir = tempfile.mkdtemp(prefix="sam2_")
        
        try:
            if status_cb:
                status_cb("Extracting frames...")
            
            # Extract frames (limit for memory)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate step to limit frames
            step = max(1, total_frames // max_frames)
            self.frame_step = step
            
            frame_idx = 0
            saved_count = 0
            frame_map = {}  # Maps saved index to original index
            saved_frames = {}  # Store frames for preview
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % step == 0:
                    fname = f"{saved_count:05d}.jpg"
                    cv2.imwrite(os.path.join(temp_dir, fname), frame)
                    frame_map[saved_count] = frame_idx
                    # Store downscaled frame for preview (save memory)
                    h, w = frame.shape[:2]
                    preview = cv2.resize(frame, (w // 2, h // 2)) if w > 640 else frame.copy()
                    saved_frames[saved_count] = preview
                    saved_count += 1
                
                frame_idx += 1
            
            cap.release()
            
            if status_cb:
                status_cb(f"Processing {saved_count} frames...")
            
            # Run tracking
            with torch.inference_mode():
                state = self.predictor.init_state(video_path=temp_dir)
                
                # Add prompts
                obj_id = 1
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=np.array([x1, y1, x2, y2], dtype=np.float32)
                    )
                    obj_id += 1
                
                if points:
                    self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=np.array(points, dtype=np.float32),
                        labels=np.ones(len(points), dtype=np.int32)
                    )
                
                # Propagate
                masks = {}
                for idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                    # Combine masks
                    combined = np.zeros(mask_logits.shape[-2:], dtype=np.uint8)
                    for m in mask_logits:
                        m_np = (m.cpu().numpy() > 0.5).astype(np.uint8)
                        if m_np.ndim == 3:
                            m_np = m_np[0]
                        combined = np.maximum(combined, m_np)
                    
                    # Map back to original frame index
                    orig_idx = frame_map.get(idx, idx * step)
                    masks[orig_idx] = combined
                    
                    if progress_cb:
                        progress_cb(int((idx + 1) / saved_count * 100))
                    
                    # Send preview every 5 frames
                    if frame_cb and idx % 5 == 0 and idx in saved_frames:
                        preview = saved_frames[idx]
                        # Resize mask to preview size
                        h, w = preview.shape[:2]
                        mask_resized = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
                        result = apply_mask_overlay(preview, mask_resized)
                        frame_cb(preview, result)
                    
                    # Clean up memory
                    del mask_logits
                    if idx % 10 == 0:
                        gc.collect()
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                
                # Clear saved frames to free memory
                saved_frames.clear()
            
            if status_cb:
                status_cb(f"Tracked {len(masks)} frames!")
            
            return masks
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()
    
    def interpolate_masks(
        self,
        masks: Dict[int, np.ndarray],
        total_frames: int
    ) -> Dict[int, np.ndarray]:
        """Interpolate masks for skipped frames."""
        if not masks:
            return {}
        
        result = dict(masks)
        sorted_keys = sorted(masks.keys())
        
        for i in range(len(sorted_keys) - 1):
            start_idx = sorted_keys[i]
            end_idx = sorted_keys[i + 1]
            
            if end_idx - start_idx <= 1:
                continue
            
            # Use start mask for intermediate frames
            for j in range(start_idx + 1, end_idx):
                result[j] = masks[start_idx].copy()
        
        return result


class SAM2FrameSegmenter:
    """Per-frame segmentation (no tracking)."""
    
    def __init__(self):
        self.predictor = None
        self.device = "cpu"
    
    def load(self, model_size: str = "tiny", device: str = "cpu", status_cb=None, download_cb=None):
        """Load SAM2 image predictor."""
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 not installed")
        
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        if device == "cuda" and not CUDA_AVAILABLE:
            device = "cpu"
        
        self.device = device
        
        config, ckpt = get_checkpoint(model_size, download_cb)
        if not ckpt:
            raise RuntimeError("Failed to get checkpoint")
        
        if status_cb:
            status_cb(f"Loading SAM2 {model_size}...")
        
        model = build_sam2(config, ckpt, device=device)
        self.predictor = SAM2ImagePredictor(model)
        
        if status_cb:
            status_cb("Ready!")
    
    def segment(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        bboxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Segment a single frame."""
        import torch
        
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        
        with torch.inference_mode():
            self.predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            masks = None
            if bboxes:
                x1, y1, x2, y2 = bboxes[0]
                masks, _, _ = self.predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False
                )
            elif points:
                masks, _, _ = self.predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.ones(len(points)),
                    multimask_output=False
                )
            
            if masks is not None and len(masks) > 0:
                return masks[0].astype(np.uint8)
        
        return np.zeros(frame.shape[:2], dtype=np.uint8)
