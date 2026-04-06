#!/usr/bin/env python3
"""
download.py — Download Qwen3.5-27B model from Hugging Face

Downloads to /home/models/ with resume capability.
Requires: huggingface_hub, requests
Run: python3 download.py

After successful download, this file should be deleted (cleanup).
"""

import os
import sys
import subprocess
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3.5-27B"
MODEL_PATH = Path("/home/models")
VENV_PYTHON = "/home/venv-qwen35/bin/python3"

def ensure_venv():
    """Verify venv exists and is usable."""
    if not os.path.exists(VENV_PYTHON):
        print(f"❌ Error: venv not found at {VENV_PYTHON}")
        print("Run setup.py first to create environment.")
        sys.exit(1)

def ensure_model_dir():
    """Create model directory if needed."""
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    if not os.access(MODEL_PATH, os.W_OK):
        print(f"❌ Error: No write access to {MODEL_PATH}")
        sys.exit(1)

def download_model():
    """Download model using huggingface_hub."""
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  Downloading {MODEL_NAME}                  ║
╠════════════════════════════════════════════════════════════╣
║  Destination: {str(MODEL_PATH):<45} ║
║  Size: ~16 GB (bfloat16)                                   ║
║  Duration: 5-15 min depending on network                   ║
║                                                            ║
║  NOTE: Download can be resumed if interrupted.            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    script = f"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download

print("Starting download...")
try:
    local_dir = snapshot_download(
        repo_id="{MODEL_NAME}",
        local_dir="{str(MODEL_PATH / 'Qwen3.5-27B')}",
        repo_type="model",
        cache_dir=None,  # Don't use HF cache, download directly to target
        resume_download=True,
    )
    print(f"✅ Download complete: {{local_dir}}")
except Exception as e:
    print(f"❌ Download failed: {{e}}")
    raise
"""
    
    try:
        subprocess.run(
            [VENV_PYTHON, "-c", script],
            check=True
        )
        print("\n✅ Model download successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Model download failed with exit code {e.returncode}")
        return False

def verify_model():
    """Verify model files exist."""
    model_dir = MODEL_PATH / "Qwen3.5-27B"
    if not model_dir.exists():
        print(f"⚠ Model directory not found: {model_dir}")
        return False
    
    required_files = ["config.json", "model.safetensors.index.json"]
    for file in required_files:
        if not (model_dir / file).exists():
            print(f"⚠ Missing: {file}")
            return False
    
    print(f"✅ Model verified at {model_dir}")
    return True

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║  Qwen3.5-27B Model Download — Client H100                 ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    ensure_venv()
    ensure_model_dir()
    
    if verify_model():
        print("\n✓ Model already exists. Skipping download.")
        sys.exit(0)
    
    if not download_model():
        sys.exit(1)
    
    if not verify_model():
        print("❌ Model verification failed after download")
        sys.exit(1)
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  ✅ Model Ready!                                           ║
╠════════════════════════════════════════════════════════════╣
║  Path: {str(MODEL_PATH / 'Qwen3.5-27B'):<42} ║
║                                                            ║
║  Next: bash start_qwen35_h100.sh                           ║
║        python3 test_inference.py                           ║
║                                                            ║
║  Cleanup: rm download.py  (after first successful run)     ║
╚════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
