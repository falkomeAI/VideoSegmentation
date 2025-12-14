"""Core segmentation module."""

from .segmentation import (
    demo_segment,
    apply_mask_overlay,
    SAM2Tracker,
    SAM2FrameSegmenter
)

__all__ = [
    'demo_segment',
    'apply_mask_overlay',
    'SAM2Tracker',
    'SAM2FrameSegmenter'
]
