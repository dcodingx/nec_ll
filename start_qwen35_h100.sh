#!/usr/bin/env bash
# start_qwen35_h100.sh
# Start Qwen3.5-27B vLLM on client H100 (port 8004)
#
# Single GPU (no tensor parallel)
# OpenAI-compatible API on port 8004
# Model: /home/models/Qwen3.5-27B

set -euo pipefail

MODEL="${QWEN35_MODEL:-/home/models/Qwen3.5-27B}"
PORT="${QWEN35_PORT:-8004}"
GPU_MEM="${QWEN35_GPU_MEM:-0.72}"
MAX_LEN="${QWEN35_MAX_LEN:-32768}"
VENV="${VENV_QWEN35:-/home/venv-qwen35}"

# Single GPU — no tensor parallel needed for H100
export CUDA_VISIBLE_DEVICES="${QWEN35_CUDA_DEVICES:-0}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Starting Qwen3.5-27B vLLM — Single H100                  ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Model        : ${MODEL}"
echo "║  Port         : ${PORT}"
echo "║  GPU          : CUDA:${CUDA_VISIBLE_DEVICES}"
echo "║  GPU Memory   : ${GPU_MEM}"
echo "║  Max Tokens   : ${MAX_LEN}"
echo "║  venv         : ${VENV}"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check venv exists
if [ ! -d "${VENV}" ]; then
    echo "❌ Error: venv not found at ${VENV}"
    echo "Run setup.py first"
    exit 1
fi

# Check model exists
if [ ! -d "${MODEL}" ]; then
    echo "❌ Error: Model not found at ${MODEL}"
    echo "Run download.py first"
    exit 1
fi

# Activate venv
source "${VENV}/bin/activate"

echo "✅ venv activated"
echo ""

# Start vLLM
echo "Starting vLLM server..."
vllm serve "${MODEL}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len "${MAX_LEN}" \
    --dtype bfloat16 \
    --served-model-name "Qwen3.5-27B" \
    --disable-log-requests \
    2>&1

# Note: No --tensor-parallel-size for single GPU
# If deployment uses multiple GPUs, add: --tensor-parallel-size N
