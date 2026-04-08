#!/usr/bin/env python3
"""
download_model.py — Download Qwen3.5-27B from HuggingFace

Downloads to LLM_DOWNLOAD_PATH (default: /home/models/Qwen3.5-27B)
setup.sh renames it to LLM_MODEL_PATH (default: /home/models/Q3_LLM_V2)

No HF token required — Qwen3.5-27B is publicly available.

Usage (called by setup.sh automatically):
    python3 download_model.py
"""

import os
import sys
from pathlib import Path

HF_MODEL_ID   = os.environ.get("HF_MODEL_ID", "Qwen/Qwen3.5-27B")
MODEL_PATH     = os.environ.get("LLM_MODEL_PATH", "/home/models/Qwen3.5-27B")
HF_TOKEN       = os.environ.get("HF_TOKEN", "") or None

print(f"[download] Model  : {HF_MODEL_ID}")
print(f"[download] Target : {MODEL_PATH}")
print(f"[download] Token  : {'set' if HF_TOKEN else 'not required (public model)'}")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[download] Installing huggingface_hub …")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "huggingface_hub>=0.34.0,<1.0", "-q"])
    from huggingface_hub import snapshot_download

Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=HF_MODEL_ID,
    local_dir=MODEL_PATH,
    token=HF_TOKEN,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.ot"],
)

print(f"\n[download] ✅ Model downloaded to: {MODEL_PATH}")
