"""
Custom UI widgets for video segmentation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


class VideoLabel(QLabel):
    """
    Video display widget with point and rectangle selection support.
    
    Signals:
        point_added(x, y): Emitted when a point is added
        bbox_added(x1, y1, x2, y2): Emitted when a bounding box is added
    """
    
    point_added = pyqtSignal(int, int)
    bbox_added = pyqtSignal(int, int, int, int)
    
    def __init__(self, title: str = ""):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #0D1117; border-radius: 8px;")
        self.setText(f"📹 {title}\n\nLoad a video")
        
        # Scaling info
        self._scale = 1.0
        self._offset = (0, 0)
        self._size = (1, 1)
        
        # Selection mode: "point" or "bbox"
        self.selection_mode = "point"
        
        # Rectangle drawing state
        self._drawing = False
        self._start_pos: Optional[Tuple[int, int]] = None
        self._current_pos: Optional[Tuple[int, int]] = None
        self._current_frame: Optional[np.ndarray] = None
        
        self.setMouseTracking(True)
    
    def mousePressEvent(self, event):
        """Handle mouse press for point/bbox selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = int(event.position().x()), int(event.position().y())
            
            if self.selection_mode == "point":
                fx, fy = self.to_frame_coords(x, y)
                self.point_added.emit(fx, fy)
            else:  # bbox mode
                self._drawing = True
                self._start_pos = (x, y)
                self._current_pos = (x, y)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for bbox drawing."""
        if self._drawing and self.selection_mode == "bbox":
            self._current_pos = (int(event.position().x()), int(event.position().y()))
            self._draw_temp_bbox()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to finalize bbox."""
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            
            if self._start_pos and self._current_pos:
                x1, y1 = self.to_frame_coords(*self._start_pos)
                x2, y2 = self.to_frame_coords(*self._current_pos)
                
                # Only emit if bbox has some size
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    self.bbox_added.emit(x1, y1, x2, y2)
            
            self._start_pos = None
            self._current_pos = None
    
    def _draw_temp_bbox(self):
        """Draw temporary rectangle while dragging."""
        if self._current_frame is None:
            return
        
        rgb = cv2.cvtColor(self._current_frame.copy(), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        lw, lh = self.width(), self.height()
        scale = min(lw / w, lh / h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(rgb, (nw, nh))
        
        # Draw the temp rectangle
        if self._start_pos and self._current_pos:
            sx1 = int(self._start_pos[0] - self._offset[0])
            sy1 = int(self._start_pos[1] - self._offset[1])
            sx2 = int(self._current_pos[0] - self._offset[0])
            sy2 = int(self._current_pos[1] - self._offset[1])
            cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
        
        img = QImage(resized.data, nw, nh, 3 * nw, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img))
    
    def show_frame(
        self,
        frame: np.ndarray,
        points: Optional[List[Tuple[int, int]]] = None,
        bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ):
        """
        Display a frame with optional points and bboxes overlay.
        
        Args:
            frame: BGR frame to display
            points: List of (x, y) points to draw
            bboxes: List of (x1, y1, x2, y2) boxes to draw
        """
        self._current_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        lw, lh = self.width(), self.height()
        scale = min(lw / w, lh / h)
        nw, nh = int(w * scale), int(h * scale)
        
        self._scale = scale
        self._offset = ((lw - nw) // 2, (lh - nh) // 2)
        self._size = (w, h)
        
        resized = cv2.resize(rgb, (nw, nh))
        
        # Draw points
        if points:
            for px, py in points:
                sx, sy = int(px * scale), int(py * scale)
                cv2.circle(resized, (sx, sy), 6, (255, 0, 0), -1)
                cv2.circle(resized, (sx, sy), 8, (255, 255, 255), 2)
        
        # Draw bboxes
        if bboxes:
            for x1, y1, x2, y2 in bboxes:
                sx1, sy1 = int(x1 * scale), int(y1 * scale)
                sx2, sy2 = int(x2 * scale), int(y2 * scale)
                cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
                cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), (255, 255, 0), 1)
        
        img = QImage(resized.data, nw, nh, 3 * nw, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img))
    
    def to_frame_coords(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert widget coordinates to frame coordinates.
        
        Args:
            x: Widget x coordinate
            y: Widget y coordinate
        
        Returns:
            (frame_x, frame_y) tuple
        """
        fx = int((x - self._offset[0]) / self._scale)
        fy = int((y - self._offset[1]) / self._scale)
        return (
            max(0, min(self._size[0] - 1, fx)),
            max(0, min(self._size[1] - 1, fy))
        )
