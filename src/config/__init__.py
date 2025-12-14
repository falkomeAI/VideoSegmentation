"""Configuration module."""

from .settings import (
    CV2_AVAILABLE,
    TORCH_AVAILABLE,
    CUDA_AVAILABLE,
    SAM2_AVAILABLE,
    SAM3_AVAILABLE,
    DEVICE,
    CHECKPOINTS_DIR,
    OUTPUT_DIR,
    SAM2_MODELS,
    DEFAULT_MODEL_SIZE,
    get_device,
    get_checkpoint,
    check_sam2_available,
    check_sam3_available,
    install_sam2,
    download_with_progress,
    print_status
)
from .styles import STYLE

__all__ = [
    'CV2_AVAILABLE', 'TORCH_AVAILABLE', 'CUDA_AVAILABLE',
    'SAM2_AVAILABLE', 'SAM3_AVAILABLE', 'DEVICE',
    'CHECKPOINTS_DIR', 'OUTPUT_DIR', 'SAM2_MODELS',
    'DEFAULT_MODEL_SIZE', 'get_device', 'get_checkpoint',
    'check_sam2_available', 'check_sam3_available',
    'install_sam2', 'download_with_progress', 'print_status',
    'STYLE'
]
