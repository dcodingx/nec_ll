#!/usr/bin/env bash
# ============================================================
# Start LLM_Q3_V1 via vLLM (OpenAI-compatible API)
#
# Prerequisites:
#   pip install vllm
#   Model present at ${LLM_MODEL_PATH}
#
# Environment variables (override via config/client.env):
#   LLM_MODEL_PATH   — path to the downloaded model
#   LLM_PORT         — vLLM server port (default: 8004)
#   LLM_GPU_MEM      — GPU memory utilisation (default: 0.90)
#   LLM_CUDA_DEVICE  — CUDA device index (default: 0)
# ============================================================

set -euo pipefail

MODEL="${LLM_MODEL_PATH:-/opt/voicebot/models/LLM_Q3_V1}"
SERVED_NAME="${LLM_MODEL_NAME:-LLM_Q3_V1}"
PORT="${LLM_PORT:-8004}"
GPU_MEM="${LLM_GPU_MEM:-0.90}"
export CUDA_VISIBLE_DEVICES="${LLM_CUDA_DEVICE:-0}"

echo "Starting ${SERVED_NAME} vLLM server on port ${PORT} …"
echo "  Model : ${MODEL}"
echo "  GPU   : cuda:${CUDA_VISIBLE_DEVICES}"
echo "  Mem   : ${GPU_MEM}"

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len 8192 \
    --dtype bfloat16
