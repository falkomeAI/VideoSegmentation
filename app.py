#!/usr/bin/env python3
"""
SAM2 Video Segmentation Application

Segment and track objects in videos using SAM2.
Supports CPU and GPU modes with automatic model download.

Usage:
    python app.py
"""

import os
import sys

# Setup paths
os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont

from src.config import CV2_AVAILABLE, print_status, OUTPUT_DIR, CHECKPOINTS_DIR
from src.config.styles import STYLE
from src.ui import MainWindow


def main():
    """Application entry point."""
    
    if not CV2_AVAILABLE:
        print("\n❌ OpenCV not installed!")
        print("Run: pip install opencv-python\n")
        return 1
    
    # Create directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    
    # Print status
    print_status()
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    
    # Create window
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    
    # Auto-load test video
    test_video = "test_video.mp4"
    if os.path.exists(test_video):
        print(f"✓ Auto-loading {test_video}")
        QTimer.singleShot(500, lambda: window._load_video_file(test_video))
    
    print("✓ Ready")
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
