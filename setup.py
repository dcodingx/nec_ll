#!/usr/bin/env python3
"""
setup.py — Initialize Python environment on client H100

Creates venv, installs vLLM and dependencies, verifies CUDA.
Run: python3 setup.py
"""

import os
import sys
import subprocess
import shutil

VENV_PATH = "/home/venv-qwen35"
PYTHON = sys.executable
VENV_BIN = os.path.join(VENV_PATH, "bin")
VENV_PYTHON = os.path.join(VENV_BIN, "python")
VENV_PIP = os.path.join(VENV_BIN, "pip")

def run_cmd(cmd, description):
    """Run shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} — SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} — FAILED (exit code {e.returncode})")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║  Client H100 — Environment Setup for Qwen3.5-27B vLLM     ║
║  venv: /home/venv-qwen35                                   ║
║  Model path: /home/models/Qwen3.5-27B                      ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Remove old venv if exists
    if os.path.exists(VENV_PATH):
        print(f"\n⚠ Existing venv found at {VENV_PATH}")
        response = input("Delete and recreate? (y/n): ").strip().lower()
        if response == "y":
            print(f"Removing {VENV_PATH}...")
            shutil.rmtree(VENV_PATH)
        else:
            print("Using existing venv. Skipping creation.")
    
    # Step 2: Create venv
    if not os.path.exists(VENV_PATH):
        if not run_cmd(f"python3 -m venv {VENV_PATH}", "Create Python venv"):
            sys.exit(1)
    
    # Step 3: Upgrade pip
    if not run_cmd(f"{VENV_PIP} install --upgrade pip", "Upgrade pip"):
        sys.exit(1)
    
    # Step 4: Install vLLM (nightly for best compatibility)
    print("\nℹ Installing vLLM with CUDA support...")
    if not run_cmd(
        f"{VENV_PIP} install vllm",
        "Install vLLM (GPU-enabled)"
    ):
        sys.exit(1)
    
    # Step 5: Install dependencies from requirements.txt
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        if not run_cmd(
            f"{VENV_PIP} install -r {req_file}",
            "Install dependencies from requirements.txt"
        ):
            sys.exit(1)
    else:
        # Install key dependencies
        deps = [
            "fastapi>=0.110.0",
            "uvicorn[standard]>=0.29.0",
            "huggingface_hub>=0.34.0",
            "requests>=2.31.0",
            "psutil>=5.9.0",
        ]
        if not run_cmd(
            f"{VENV_PIP} install {' '.join(deps)}",
            "Install core dependencies"
        ):
            sys.exit(1)
    
    # Step 6: Verify CUDA
    print("\n" + "="*60)
    print("▶ Verify CUDA/GPU access")
    print("="*60)
    result = subprocess.run(
        f"{VENV_PYTHON} -c \"import torch; print(f'✅ CUDA available: {{torch.cuda.is_available()}}'); print(f'Device: {{torch.cuda.get_device_name(0)}}'); print(f'GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB')\"",
        shell=True
    )
    
    if result.returncode != 0:
        print("⚠ Could not verify CUDA. Continuing anyway...")
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  ✅ Environment Setup Complete!                            ║
╠════════════════════════════════════════════════════════════╣
║  venv: {VENV_PATH}                              ║
║  python: {VENV_PYTHON}              ║
║  pip: {VENV_PIP}                          ║
║                                                            ║
║  Next steps:                                               ║
║  1. Run: python3 download.py  (download model)            ║
║  2. Run: bash start_qwen35_h100.sh  (start vLLM)          ║
║  3. Run: python3 test_inference.py  (health check)        ║
║  4. Run: bash binary.sh  (create service + binary)        ║
╚════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
