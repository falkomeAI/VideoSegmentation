"""
Configuration and model management for SAM2/SAM3 Video Segmentation.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

# ============================================================
# Directories
# ============================================================
BASE_DIR = Path(__file__).parent.parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"

# ============================================================
# SAM2 Model Configurations
# ============================================================
SAM2_MODELS = {
    "tiny": {
        "name": "Tiny",
        "config": "sam2_hiera_t.yaml",
        "checkpoint": "sam2_hiera_tiny.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "size_mb": 40
    },
    "small": {
        "name": "Small",
        "config": "sam2_hiera_s.yaml",
        "checkpoint": "sam2_hiera_small.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "size_mb": 185
    },
    "base_plus": {
        "name": "Base+",
        "config": "sam2_hiera_b+.yaml",
        "checkpoint": "sam2_hiera_base_plus.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "size_mb": 325
    },
    "large": {
        "name": "Large",
        "config": "sam2_hiera_l.yaml",
        "checkpoint": "sam2_hiera_large.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "size_mb": 900
    }
}

DEFAULT_MODEL_SIZE = "tiny"

# ============================================================
# Dependency Checks
# ============================================================
CV2_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    pass

TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
DEVICE = "cpu"

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
except ImportError:
    pass


def get_device(use_gpu: bool = True) -> str:
    """Get device based on preference."""
    if use_gpu and CUDA_AVAILABLE:
        return "cuda"
    return "cpu"


def check_sam2_available() -> tuple:
    """Check if SAM2 is installed."""
    try:
        from sam2.build_sam import build_sam2
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_sam3_available() -> tuple:
    """Check if SAM3 is available (placeholder)."""
    return False, "SAM3 not yet released"


SAM2_AVAILABLE, SAM2_ERROR = check_sam2_available()
SAM3_AVAILABLE, SAM3_ERROR = check_sam3_available()


# ============================================================
# Download & Installation
# ============================================================
def download_with_progress(url: str, dest: Path, callback=None) -> bool:
    """Download file with progress."""
    try:
        print(f"Downloading: {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with urllib.request.urlopen(url, timeout=30) as response:
            total = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(8192 * 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if callback:
                        callback(downloaded, total)
                    elif total > 0:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:.1f}%", end="", flush=True)
        
        print("\n✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def get_checkpoint(model_size: str = "tiny", callback=None) -> tuple:
    """Get or download SAM2 checkpoint."""
    if model_size not in SAM2_MODELS:
        model_size = DEFAULT_MODEL_SIZE
    
    info = SAM2_MODELS[model_size]
    ckpt_path = CHECKPOINTS_DIR / info["checkpoint"]
    
    if ckpt_path.exists():
        return info["config"], str(ckpt_path)
    
    # Download
    print(f"Downloading SAM2 {info['name']} (~{info['size_mb']}MB)...")
    if download_with_progress(info["url"], ckpt_path, callback):
        return info["config"], str(ckpt_path)
    
    return None, None


def install_sam2(status_callback=None) -> bool:
    """Install SAM2 from GitHub."""
    global SAM2_AVAILABLE, SAM2_ERROR
    
    try:
        import tempfile
        install_dir = Path(tempfile.gettempdir()) / "sam2_install"
        sam2_dir = install_dir / "sam2"
        
        if status_callback:
            status_callback("Cloning SAM2 repository...")
        
        if not sam2_dir.exists():
            install_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", 
                 "https://github.com/facebookresearch/sam2.git", str(sam2_dir)],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                return False
        
        if status_callback:
            status_callback("Installing SAM2...")
        
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)],
            capture_output=True, text=True, timeout=600
        )
        
        if result.returncode != 0:
            return False
        
        SAM2_AVAILABLE, SAM2_ERROR = check_sam2_available()
        return SAM2_AVAILABLE
        
    except Exception as e:
        if status_callback:
            status_callback(f"Error: {e}")
        return False


def print_status():
    """Print configuration status."""
    print("=" * 50)
    print("SAM2/SAM3 Video Segmentation")
    print("=" * 50)
    print(f"OpenCV:   {'✓' if CV2_AVAILABLE else '✗'}")
    print(f"PyTorch:  {'✓' if TORCH_AVAILABLE else '✗'}")
    print(f"CUDA/GPU: {'✓ ' + DEVICE if CUDA_AVAILABLE else '✗ CPU only'}")
    print(f"SAM2:     {'✓' if SAM2_AVAILABLE else '○ (auto-install)'}")
    print(f"SAM3:     ○ Coming soon")
    print("-" * 50)
    print("Models:")
    for key, info in SAM2_MODELS.items():
        ckpt = CHECKPOINTS_DIR / info["checkpoint"]
        status = "✓" if ckpt.exists() else f"~{info['size_mb']}MB"
        print(f"  {info['name']:8} {status}")
    print("=" * 50)
