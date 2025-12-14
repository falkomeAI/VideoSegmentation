"""
Background segmentation thread with object tracking.
Optimized for memory efficiency on CPU.
"""

import gc
import cv2
from typing import List, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from ..config import SAM2_AVAILABLE, check_sam2_available
from ..core import demo_segment, apply_mask_overlay, SAM2Tracker


class SegmentationThread(QThread):
    """Background thread for video segmentation with tracking."""
    
    progress = pyqtSignal(int)
    frame_ready = pyqtSignal(object, object)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    download_progress = pyqtSignal(int, int)
    
    # Memory limit: max frames to process at once
    MAX_FRAMES = 80
    
    def __init__(
        self,
        video_path: str,
        output_path: str,
        points: List[Tuple[int, int]],
        bboxes: List[Tuple[int, int, int, int]],
        model_type: str = "demo",
        model_size: str = "tiny",
        device: str = "cpu"
    ):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.points = points
        self.bboxes = bboxes
        self.model_type = model_type
        self.model_size = model_size
        self.device = device
        self.running = True
    
    def run(self):
        try:
            if self.model_type == "sam2":
                self._run_sam2()
            else:
                self._run_demo()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            gc.collect()
    
    def _download_cb(self, downloaded: int, total: int):
        self.download_progress.emit(downloaded, total)
        if total > 0:
            pct = downloaded / total * 100
            mb = downloaded / 1024 / 1024
            self.status.emit(f"Downloading: {pct:.0f}% ({mb:.0f}MB)")
    
    def _run_sam2(self):
        """Run SAM2 with object tracking."""
        device_str = "GPU" if self.device == "cuda" else "CPU"
        self.status.emit(f"Starting SAM2 ({self.model_size}) on {device_str}...")
        
        # Check SAM2
        if not check_sam2_available()[0]:
            self.error.emit(
                "SAM2 not installed.\n\n"
                "Install: git clone https://github.com/facebookresearch/sam2.git\n"
                "         cd sam2 && pip install -e ."
            )
            return
        
        # Load tracker
        tracker = SAM2Tracker()
        try:
            tracker.load(
                model_size=self.model_size,
                device=self.device,
                status_cb=lambda msg: self.status.emit(msg),
                download_cb=self._download_cb
            )
        except Exception as e:
            self.error.emit(f"Failed to load model: {e}")
            return
        
        if not self.running:
            return
        
        # Get video info
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Adjust max frames for CPU
        max_frames = self.MAX_FRAMES if self.device == "cpu" else total_frames
        
        self.status.emit(f"Tracking through {total_frames} frames...")
        
        # Run tracking with preview
        def emit_frame(orig, seg):
            self.frame_ready.emit(orig, seg)
        
        try:
            masks = tracker.track(
                video_path=self.video_path,
                points=self.points,
                bboxes=self.bboxes,
                max_frames=max_frames,
                status_cb=lambda msg: self.status.emit(msg),
                progress_cb=lambda p: self.progress.emit(int(p * 0.5)),
                frame_cb=emit_frame
            )
        except Exception as e:
            self.error.emit(f"Tracking failed: {e}")
            return
        
        if not self.running:
            return
        
        # Interpolate masks for skipped frames
        if tracker.frame_step > 1:
            self.status.emit("Interpolating frames...")
            masks = tracker.interpolate_masks(masks, total_frames)
        
        # Write output
        self.status.emit("Writing output video...")
        
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        
        frame_idx = 0
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in masks:
                result = apply_mask_overlay(frame, masks[frame_idx])
            else:
                result = frame.copy()
            
            out.write(result)
            
            progress = 50 + int((frame_idx + 1) / total_frames * 50)
            self.progress.emit(progress)
            
            # Show every 3rd frame for smoother preview
            if frame_idx % 3 == 0:
                self.frame_ready.emit(frame.copy(), result.copy())
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Cleanup
        del tracker, masks
        gc.collect()
        
        if self.running:
            self.status.emit("✓ Complete!")
            self.finished.emit(self.output_path)
    
    def _run_demo(self):
        """Run demo color-based segmentation."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit("Cannot open video")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        
        self.status.emit("Processing with Demo mode...")
        
        count = 0
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = demo_segment(frame, self.points, self.bboxes)
            out.write(result)
            
            count += 1
            self.progress.emit(int(count / total * 100))
            
            if count % 5 == 0:
                self.frame_ready.emit(frame, result)
        
        cap.release()
        out.release()
        
        if self.running:
            self.finished.emit(self.output_path)
    
    def stop(self):
        self.running = False
