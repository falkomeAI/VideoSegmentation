"""
Main application window for SAM2/SAM3 Video Segmentation.
"""

import os
from datetime import datetime
from typing import List, Tuple

import cv2

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QFrame,
    QProgressBar, QMessageBox, QGroupBox, QSplitter,
    QComboBox, QRadioButton, QApplication, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer

from ..config import (
    SAM2_AVAILABLE, SAM2_MODELS, CUDA_AVAILABLE, OUTPUT_DIR,
    check_sam2_available, install_sam2
)
from ..threads import SegmentationThread
from .widgets import VideoLabel


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Video Segmentation")
        self.setMinimumSize(1200, 700)
        self.setGeometry(50, 50, 1350, 800)
        
        # State
        self.cap = None
        self.video_path = ""
        self.points: List[Tuple[int, int]] = []
        self.bboxes: List[Tuple[int, int, int, int]] = []
        self.playing = False
        self.seg_thread = None
        self.output_path = None
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)
        
        self._build_ui()
    
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        self._build_header(layout)
        self._build_split_view(layout)
        self._build_controls(layout)
        self._build_status(layout)
    
    def _build_header(self, layout):
        header = QFrame()
        header.setStyleSheet("background: #161B22; border-radius: 8px; padding: 8px;")
        h = QHBoxLayout(header)
        
        title = QLabel("🎬 SAM2 Video Segmentation")
        title.setProperty("class", "title")
        h.addWidget(title)
        h.addStretch()
        
        # Device selector
        h.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("🖥️ CPU")
        self.device_combo.addItem("🚀 GPU" + (" ✓" if CUDA_AVAILABLE else " (N/A)"))
        self.device_combo.setCurrentIndex(1 if CUDA_AVAILABLE else 0)
        h.addWidget(self.device_combo)
        
        # Model selector
        h.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self._update_model_list()
        self.model_combo.currentIndexChanged.connect(self._on_model_change)
        h.addWidget(self.model_combo)
        
        # Size selector
        h.addWidget(QLabel("Size:"))
        self.size_combo = QComboBox()
        for key, info in SAM2_MODELS.items():
            self.size_combo.addItem(f"{info['name']} (~{info['size_mb']}MB)")
        self.size_combo.setCurrentIndex(0)
        h.addWidget(self.size_combo)
        
        # Load button
        self.load_btn = QPushButton("📂 Load Video")
        self.load_btn.setProperty("class", "primary")
        self.load_btn.clicked.connect(self.load_video)
        h.addWidget(self.load_btn)
        
        layout.addWidget(header)
    
    def _build_split_view(self, layout):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Original
        left = QFrame()
        left.setProperty("class", "panel")
        l_layout = QVBoxLayout(left)
        
        l_header = QHBoxLayout()
        l_header.addWidget(QLabel("📹 Original Video"))
        l_header.addStretch()
        
        self.point_radio = QRadioButton("📍 Point")
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(self._on_mode_change)
        l_header.addWidget(self.point_radio)
        
        self.bbox_radio = QRadioButton("⬜ Rectangle")
        l_header.addWidget(self.bbox_radio)
        
        self.clear_btn = QPushButton("🗑 Clear")
        self.clear_btn.clicked.connect(self._clear_selections)
        l_header.addWidget(self.clear_btn)
        
        self.selection_lbl = QLabel("0 points, 0 boxes")
        self.selection_lbl.setProperty("class", "info")
        l_header.addWidget(self.selection_lbl)
        l_layout.addLayout(l_header)
        
        self.input_video = VideoLabel("Original")
        self.input_video.point_added.connect(self._add_point)
        self.input_video.bbox_added.connect(self._add_bbox)
        l_layout.addWidget(self.input_video, 1)
        
        self.input_info = QLabel("Load a video to begin")
        self.input_info.setProperty("class", "info")
        self.input_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l_layout.addWidget(self.input_info)
        
        splitter.addWidget(left)
        
        # Right panel - Output
        right = QFrame()
        right.setProperty("class", "panel")
        r_layout = QVBoxLayout(right)
        
        r_header = QHBoxLayout()
        r_header.addWidget(QLabel("🎯 Tracked Output"))
        r_header.addStretch()
        self.model_lbl = QLabel("Demo Mode")
        self.model_lbl.setStyleSheet("color: #7EE787;")
        r_header.addWidget(self.model_lbl)
        r_layout.addLayout(r_header)
        
        self.output_video = VideoLabel("Output")
        r_layout.addWidget(self.output_video, 1)
        
        self.output_info = QLabel("Select objects, then Start")
        self.output_info.setProperty("class", "info")
        self.output_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        r_layout.addWidget(self.output_info)
        
        splitter.addWidget(right)
        splitter.setSizes([550, 550])
        layout.addWidget(splitter, 1)
    
    def _build_controls(self, layout):
        controls = QFrame()
        controls.setStyleSheet("background: #161B22; border-radius: 8px; padding: 8px;")
        c = QHBoxLayout(controls)
        
        # Playback
        play_grp = QGroupBox("Playback")
        play_layout = QHBoxLayout(play_grp)
        
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.clicked.connect(self._toggle_play)
        play_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setFixedSize(40, 40)
        self.stop_btn.clicked.connect(self._stop)
        play_layout.addWidget(self.stop_btn)
        
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.sliderMoved.connect(self._seek)
        play_layout.addWidget(self.timeline, 1)
        
        self.time_lbl = QLabel("00:00 / 00:00")
        self.time_lbl.setProperty("class", "info")
        play_layout.addWidget(self.time_lbl)
        
        c.addWidget(play_grp, 1)
        
        # Segmentation
        seg_grp = QGroupBox("Tracking")
        seg_layout = QHBoxLayout(seg_grp)
        
        self.seg_btn = QPushButton("🎯 Start Tracking")
        self.seg_btn.setProperty("class", "primary")
        self.seg_btn.clicked.connect(self._start_seg)
        seg_layout.addWidget(self.seg_btn)
        
        self.cancel_btn = QPushButton("✖ Cancel")
        self.cancel_btn.setProperty("class", "danger")
        self.cancel_btn.clicked.connect(self._cancel_seg)
        self.cancel_btn.setEnabled(False)
        seg_layout.addWidget(self.cancel_btn)
        
        c.addWidget(seg_grp)
        
        # Output
        out_grp = QGroupBox("Output")
        out_layout = QHBoxLayout(out_grp)
        
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        out_layout.addWidget(self.save_btn)
        
        self.screenshot_btn = QPushButton("📷")
        self.screenshot_btn.clicked.connect(self._screenshot)
        out_layout.addWidget(self.screenshot_btn)
        
        c.addWidget(out_grp)
        layout.addWidget(controls)
    
    def _build_status(self, layout):
        status = QFrame()
        status.setStyleSheet("background: #161B22; border-radius: 6px; padding: 6px;")
        s = QHBoxLayout(status)
        
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setProperty("class", "info")
        s.addWidget(self.status_lbl)
        s.addStretch()
        
        self.progress = QProgressBar()
        self.progress.setFixedWidth(250)
        self.progress.setVisible(False)
        s.addWidget(self.progress)
        
        layout.addWidget(status)
    
    def _update_model_list(self):
        self.model_combo.clear()
        sam2_ok, _ = check_sam2_available()
        
        self.model_combo.addItem("Demo (Color-based)")
        self.model_combo.addItem("SAM2 " + ("✓" if sam2_ok else "(install)"))
        self.model_combo.addItem("SAM3 (coming soon)")
    
    def _on_model_change(self, idx):
        text = self.model_combo.currentText()
        
        if "install" in text:
            reply = QMessageBox.question(
                self, "Install SAM2?",
                "SAM2 not installed. Install now?\n"
                "(Downloads ~200MB + model checkpoint)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._install_sam2()
            else:
                self.model_combo.setCurrentIndex(0)
        elif "coming soon" in text:
            QMessageBox.information(self, "SAM3", "SAM3 is not yet released.\nPlease use SAM2.")
            self.model_combo.setCurrentIndex(0)
        else:
            self.model_lbl.setText(text.split()[0])
    
    def _install_sam2(self):
        progress = QProgressDialog("Installing SAM2...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        success = install_sam2(lambda msg: progress.setLabelText(msg))
        progress.close()
        
        if success:
            QMessageBox.information(self, "Success", "SAM2 installed!")
            self._update_model_list()
            self.model_combo.setCurrentIndex(1)
        else:
            QMessageBox.critical(self, "Error", "Installation failed.\nSee console for details.")
            self.model_combo.setCurrentIndex(0)
    
    def _on_mode_change(self):
        if self.point_radio.isChecked():
            self.input_video.selection_mode = "point"
            self.status_lbl.setText("Click to add points")
        else:
            self.input_video.selection_mode = "bbox"
            self.status_lbl.setText("Drag to draw rectangle")
    
    def _get_device(self) -> str:
        if self.device_combo.currentIndex() == 1 and CUDA_AVAILABLE:
            return "cuda"
        return "cpu"
    
    def _get_model_size(self) -> str:
        sizes = list(SAM2_MODELS.keys())
        return sizes[self.size_combo.currentIndex()]
    
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*)"
        )
        if path:
            self._load_video_file(path)
    
    def _load_video_file(self, path: str):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video")
            return
        
        self.video_path = path
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps
        
        self.timeline.setMaximum(max(1, total - 1))
        self.input_info.setText(f"{w}x{h} • {fps:.0f}fps • {duration:.1f}s")
        self.status_lbl.setText(f"Loaded: {os.path.basename(path)}")
        
        self._clear_selections()
        self._show_frame(0)
    
    def _show_frame(self, n=None):
        if not self.cap:
            return
        
        if n is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        
        ret, frame = self.cap.read()
        if ret:
            self.input_video.show_frame(frame, self.points, self.bboxes)
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.timeline.setValue(pos)
            
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ct, tt = pos / fps, total / fps
            self.time_lbl.setText(f"{int(ct//60):02d}:{int(ct%60):02d} / {int(tt//60):02d}:{int(tt%60):02d}")
    
    def _add_point(self, x: int, y: int):
        if self.cap:
            self.points.append((x, y))
            self._update_selection_label()
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self._show_frame(max(0, pos))
    
    def _add_bbox(self, x1: int, y1: int, x2: int, y2: int):
        if self.cap:
            self.bboxes.append((x1, y1, x2, y2))
            self._update_selection_label()
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self._show_frame(max(0, pos))
    
    def _clear_selections(self):
        self.points = []
        self.bboxes = []
        self._update_selection_label()
        if self.cap:
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self._show_frame(max(0, pos))
    
    def _update_selection_label(self):
        self.selection_lbl.setText(f"{len(self.points)} points, {len(self.bboxes)} boxes")
    
    def _toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("⏸")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.timer.start(int(1000 / fps))
        else:
            self.play_btn.setText("▶")
            self.timer.stop()
    
    def _next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.input_video.show_frame(frame, self.points, self.bboxes)
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.timeline.setValue(pos)
            
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ct, tt = pos / fps, total / fps
            self.time_lbl.setText(f"{int(ct//60):02d}:{int(ct%60):02d} / {int(tt//60):02d}:{int(tt%60):02d}")
        else:
            self._stop()
    
    def _stop(self):
        self.playing = False
        self.play_btn.setText("▶")
        self.timer.stop()
        if self.cap:
            self._show_frame(0)
    
    def _seek(self, v):
        self._show_frame(v)
    
    def _start_seg(self):
        if not self.cap:
            QMessageBox.warning(self, "Warning", "Load a video first!")
            return
        if not self.points and not self.bboxes:
            QMessageBox.warning(self, "Warning", "Select objects first!")
            return
        
        # Check device
        device = self._get_device()
        if self.device_combo.currentIndex() == 1 and not CUDA_AVAILABLE:
            QMessageBox.warning(self, "GPU Not Available", "Using CPU instead.")
            device = "cpu"
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_text = self.model_combo.currentText().lower()
        model_type = "sam2" if "sam2" in model_text else "demo"
        model_size = self._get_model_size()
        
        self.output_path = str(OUTPUT_DIR / f"tracked_{model_type}_{ts}.mp4")
        
        self.seg_thread = SegmentationThread(
            self.video_path, self.output_path,
            self.points.copy(), self.bboxes.copy(),
            model_type, model_size, device
        )
        self.seg_thread.progress.connect(self._on_progress)
        self.seg_thread.frame_ready.connect(self._on_frame)
        self.seg_thread.finished.connect(self._on_done)
        self.seg_thread.error.connect(self._on_error)
        self.seg_thread.status.connect(self._on_status)
        self.seg_thread.download_progress.connect(self._on_download)
        self.seg_thread.start()
        
        self.seg_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
    
    def _cancel_seg(self):
        if self.seg_thread:
            self.seg_thread.stop()
            self.seg_thread.wait()
        self._reset_ui()
        self.status_lbl.setText("Cancelled")
    
    def _on_progress(self, v):
        self.progress.setValue(v)
    
    def _on_frame(self, orig, seg):
        """Update both video displays with current frame."""
        if orig is not None:
            self.input_video.show_frame(orig, self.points, self.bboxes)
        if seg is not None:
            self.output_video.show_frame(seg)
    
    def _on_status(self, msg):
        self.status_lbl.setText(msg)
    
    def _on_download(self, downloaded: int, total: int):
        if total > 0:
            pct = int(downloaded / total * 100)
            self.progress.setValue(pct)
    
    def _on_done(self, path):
        self._reset_ui()
        self.save_btn.setEnabled(True)
        self.output_info.setText(f"Saved: {os.path.basename(path)}")
        self.status_lbl.setText("✓ Complete!")
        QMessageBox.information(self, "Done", f"Saved to:\n{path}")
    
    def _on_error(self, e):
        self._reset_ui()
        self.status_lbl.setText(f"Error: {e[:50]}")
        QMessageBox.critical(self, "Error", e)
    
    def _reset_ui(self):
        self.seg_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setVisible(False)
    
    def _save(self):
        if not self.output_path or not os.path.exists(self.output_path):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save", "output.mp4", "Video (*.mp4)")
        if path:
            import shutil
            shutil.copy(self.output_path, path)
            QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
    
    def _screenshot(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(OUTPUT_DIR / f"screenshot_{ts}.png")
        self.grab().save(path)
        QMessageBox.information(self, "Screenshot", f"Saved:\n{path}")
    
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.seg_thread:
            self.seg_thread.stop()
            self.seg_thread.wait()
        event.accept()
