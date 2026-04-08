#!/usr/bin/env bash
# start_qwen35_h100.sh
# Manually start Q3_LLM_V2 vLLM on H100 (port 8004)
# Single GPU, bfloat16, OpenAI-compatible API

set -euo pipefail

# Load config if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for env_file in "${SCRIPT_DIR}/client.env" "${SCRIPT_DIR}/config/client.env"; do
    [[ -f "${env_file}" ]] && set -a && source "${env_file}" && set +a && break
done

MODEL="${LLM_MODEL_PATH:-/home/models/Q3_LLM_V2}"
MODEL_NAME="${LLM_MODEL_NAME:-Q3_LLM_V2}"
PORT="${LLM_PORT:-8004}"
GPU_MEM="${LLM_GPU_MEM:-0.72}"
MAX_LEN="${LLM_MAX_LEN:-32768}"
VENV="${SCRIPT_DIR}/.venv"
[[ ! -d "${VENV}" ]] && VENV="/opt/nec_ll/.venv"

export CUDA_VISIBLE_DEVICES="${LLM_CUDA_DEVICE:-0}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Starting Q3_LLM_V2 vLLM — Single H100                    ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Model        : ${MODEL}"
echo "║  Served as    : ${MODEL_NAME}"
echo "║  Port         : ${PORT}"
echo "║  GPU          : CUDA:${CUDA_VISIBLE_DEVICES}"
echo "║  GPU Memory   : ${GPU_MEM}"
echo "║  Max Tokens   : ${MAX_LEN}"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

[[ ! -d "${VENV}" ]] && echo "❌ venv not found at ${VENV}. Run setup.sh first." && exit 1
[[ ! -d "${MODEL}" ]] && echo "❌ Model not found at ${MODEL}. Run setup.sh first." && exit 1

source "${VENV}/bin/activate"
echo "✅ venv activated"

vllm serve "${MODEL}" \
    --served-model-name "${MODEL_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len "${MAX_LEN}" \
    --dtype bfloat16 \
    --disable-log-requests \
    2>&1
