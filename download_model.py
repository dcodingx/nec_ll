#!/usr/bin/env python3
"""
One-time download of shisa-ai/LLM_Q3_V1 from HuggingFace.

Run this ONCE on the client server before starting vLLM.
The script is idempotent: if the model is already present it exits immediately.

Usage:
    python download_model.py

Environment overrides (optional):
    HF_MODEL_ID   — HuggingFace repo id  (default: shisa-ai/LLM_Q3_V1)
    LLM_MODEL_PATH — local destination   (default: /opt/voicebot/models/LLM_Q3_V1)
    HF_TOKEN       — HuggingFace token   (required for private/gated repos)
"""

import os
import sys

HF_MODEL_ID   = os.getenv("HF_MODEL_ID",    "shisa-ai/LLM_Q3_V1")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "/opt/voicebot/models/LLM_Q3_V1")
HF_TOKEN       = os.getenv("HF_TOKEN",       None)

SENTINEL_FILE = os.path.join(LLM_MODEL_PATH, "config.json")


def main():
    print("=" * 60)
    print("  LLM_Q3_V1 — Model Download")
    print("=" * 60)
    print(f"  Source : {HF_MODEL_ID}")
    print(f"  Target : {LLM_MODEL_PATH}")
    print("=" * 60)

    # ── Already downloaded? ──────────────────────────────────
    if os.path.isfile(SENTINEL_FILE):
        print(f"\n[OK] Model already present at {LLM_MODEL_PATH}")
        print("     Skipping download.")
        return

    # ── Import huggingface_hub ───────────────────────────────
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n[ERROR] huggingface_hub is not installed.")
        print("        Run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(LLM_MODEL_PATH, exist_ok=True)

    print(f"\nDownloading {HF_MODEL_ID} …")
    print("This may take 10–30 minutes depending on bandwidth.\n")

    kwargs = dict(
        repo_id=HF_MODEL_ID,
        local_dir=LLM_MODEL_PATH,
        local_dir_use_symlinks=False,
    )
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    try:
        snapshot_download(**kwargs)
    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        print("Hints:")
        print("  - If this is a gated model, set HF_TOKEN=<your_token>")
        print("  - Check your internet connection")
        print("  - Retry: the download resumes from where it stopped")
        sys.exit(1)

    if not os.path.isfile(SENTINEL_FILE):
        print("\n[ERROR] Download finished but config.json not found.")
        print(f"        Check {LLM_MODEL_PATH} manually.")
        sys.exit(1)

    print("\n[OK] Model downloaded successfully.")
    print(f"     Path: {LLM_MODEL_PATH}")


if __name__ == "__main__":
    main()
