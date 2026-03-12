#!/usr/bin/env bash
# ============================================================
# run_local.sh — No-sudo local runner for LLM_Q3_V1
#
# Use this when you do NOT have sudo/root access.
# Installs everything under ~/shisa-llm-deploy and starts
# vLLM + wrapper as background processes (no systemd).
#
# Usage:
#   cd <repo> && bash run_local.sh
#
# To stop everything afterwards:
#   bash run_local.sh --stop
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${HOME}/shisa-llm-deploy"
LOG="${HOME}/shisa_llm_run.log"
VLLM_PID_FILE="${INSTALL_DIR}/.vllm.pid"
WRAPPER_PID_FILE="${INSTALL_DIR}/.wrapper.pid"

# ── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "${LOG}"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "${LOG}"; }
error() { echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG}"; exit 1; }

# ── Stop mode ────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping vLLM and wrapper …"
    if [[ -f "${VLLM_PID_FILE}" ]]; then
        kill "$(cat "${VLLM_PID_FILE}")" 2>/dev/null && echo "  vLLM stopped." || echo "  vLLM already stopped."
        rm -f "${VLLM_PID_FILE}"
    fi
    if [[ -f "${WRAPPER_PID_FILE}" ]]; then
        kill "$(cat "${WRAPPER_PID_FILE}")" 2>/dev/null && echo "  Wrapper stopped." || echo "  Wrapper already stopped."
        rm -f "${WRAPPER_PID_FILE}"
    fi
    exit 0
fi

exec > >(tee -a "${LOG}") 2>&1
echo "============================================================"
echo "  LLM_Q3_V1 — Local Run (no sudo)"
echo "============================================================"
info "Log: ${LOG}"

# ── Load config ──────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/config/client.env"
if [[ -f "${ENV_FILE}" ]]; then
    info "Loading config from ${ENV_FILE}"
    set -a; source "${ENV_FILE}"; set +a
else
    warn "No config/client.env found — using defaults."
    warn "Copy config/client.env.example → config/client.env and edit it."
fi

HF_MODEL_ID="${HF_MODEL_ID:-shisa-ai/LLM_Q3_V1}"
# Default model path to home dir (no sudo needed)
LLM_MODEL_PATH="${LLM_MODEL_PATH:-${HOME}/voicebot/models/LLM_Q3_V1}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-LLM_Q3_V1}"
LLM_PORT="${LLM_PORT:-8004}"
LLM_API_PORT="${LLM_API_PORT:-8005}"
LLM_CUDA_DEVICE="${LLM_CUDA_DEVICE:-0}"
LLM_GPU_MEM="${LLM_GPU_MEM:-0.90}"
HF_TOKEN="${HF_TOKEN:-}"

# ━━━ 1/4  Install files to ~/shisa-llm-deploy ━━━━━━━━━━━━━━
echo ""
info "━━━ 1/4  Copying files to ${INSTALL_DIR} ━━━"
mkdir -p "${INSTALL_DIR}"
cp -r "${SCRIPT_DIR}/." "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/start_vllm_llm_v2.sh"

# Write resolved .env
cat > "${INSTALL_DIR}/client.env" <<EOF
HF_MODEL_ID=${HF_MODEL_ID}
LLM_MODEL_PATH=${LLM_MODEL_PATH}
LLM_MODEL_NAME=${LLM_MODEL_NAME}
LLM_PORT=${LLM_PORT}
LLM_API_PORT=${LLM_API_PORT}
LLM_CUDA_DEVICE=${LLM_CUDA_DEVICE}
LLM_GPU_MEM=${LLM_GPU_MEM}
LLM_VLLM_BASE_URL=http://localhost:${LLM_PORT}/v1
LLM_API_BASE_URL=http://localhost:${LLM_API_PORT}
EOF
info "Config written to ${INSTALL_DIR}/client.env"

# ━━━ 2/4  Python venv + dependencies ━━━━━━━━━━━━━━━━━━━━━━━
echo ""
info "━━━ 2/4  Installing Python venv + vLLM (first run takes ~10 min) ━━━"
VENV="${INSTALL_DIR}/.venv"
if [[ ! -d "${VENV}" ]]; then
    python3 -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
pip install --upgrade pip --quiet
pip install vllm --quiet
pip install fastapi "uvicorn[standard]" httpx psutil pydantic \
            requests matplotlib Pillow huggingface_hub --quiet
deactivate
info "venv ready: ${VENV}"

# ━━━ 3/4  Download model (skipped if already present) ━━━━━━━
echo ""
info "━━━ 3/4  Downloading model from HuggingFace ━━━"
if [[ -f "${LLM_MODEL_PATH}/config.json" ]]; then
    info "Model already present at ${LLM_MODEL_PATH} — skipping download."
else
    info "Downloading ${HF_MODEL_ID} → ${LLM_MODEL_PATH}"
    info "This may take 10–30 minutes …"
    HF_MODEL_ID="${HF_MODEL_ID}" \
    LLM_MODEL_PATH="${LLM_MODEL_PATH}" \
    HF_TOKEN="${HF_TOKEN}" \
        "${VENV}/bin/python" "${INSTALL_DIR}/download_model.py" \
        || error "Model download failed. Check ${LOG}."
fi
info "Model ready: ${LLM_MODEL_PATH}"

# ━━━ 4/4  Start vLLM + wrapper as background processes ━━━━━━
echo ""
info "━━━ 4/4  Starting vLLM and wrapper ━━━"

# Kill any existing processes on these ports
for PORT in "${LLM_PORT}" "${LLM_API_PORT}"; do
    PID=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
    if [[ -n "${PID}" ]]; then
        warn "Port ${PORT} already in use (PID ${PID}) — killing existing process."
        kill "${PID}" 2>/dev/null || true
        sleep 2
    fi
done

# Start vLLM
info "Starting vLLM on port ${LLM_PORT} …"
CUDA_VISIBLE_DEVICES="${LLM_CUDA_DEVICE}" \
    nohup "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "${LLM_MODEL_PATH}" \
        --served-model-name "${LLM_MODEL_NAME}" \
        --port "${LLM_PORT}" \
        --gpu-memory-utilization "${LLM_GPU_MEM}" \
        --max-model-len 8192 \
        --dtype bfloat16 \
    >> "${HOME}/vllm.log" 2>&1 &
echo $! > "${VLLM_PID_FILE}"
info "vLLM started (PID $(cat "${VLLM_PID_FILE}")) — logs: ~/vllm.log"

# Wait for vLLM to be ready
info "Waiting for vLLM to be ready (up to 300s) …"
MAX_WAIT=300; INTERVAL=5; elapsed=0
while true; do
    if curl -sf "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
        info "vLLM is ready! (${elapsed}s)"
        break
    fi
    if (( elapsed >= MAX_WAIT )); then
        error "vLLM did not respond after ${MAX_WAIT}s. Check ~/vllm.log"
    fi
    sleep "${INTERVAL}"; elapsed=$(( elapsed + INTERVAL ))
    echo -n "."
done

# Start wrapper
info "Starting API wrapper on port ${LLM_API_PORT} …"
source "${INSTALL_DIR}/client.env"
export LLM_MODEL_NAME LLM_MODEL_PATH LLM_VLLM_BASE_URL LLM_API_PORT
nohup "${VENV}/bin/python" "${INSTALL_DIR}/llm_api_wrapper_v2.py" \
    >> "${HOME}/wrapper.log" 2>&1 &
echo $! > "${WRAPPER_PID_FILE}"
info "Wrapper started (PID $(cat "${WRAPPER_PID_FILE}")) — logs: ~/wrapper.log"

# Give wrapper a few seconds to bind
sleep 5

# ── Run healthcheck ──────────────────────────────────────────
echo ""
info "Running healthcheck …"
LLM_PORT="${LLM_PORT}" LLM_API_PORT="${LLM_API_PORT}" LLM_MODEL_NAME="${LLM_MODEL_NAME}" \
    bash "${INSTALL_DIR}/healthcheck.sh" || warn "Some health checks failed — check logs."

# ── Run test suite ───────────────────────────────────────────
echo ""
info "Running test suite …"
LLM_API_BASE_URL="http://localhost:${LLM_API_PORT}" \
    "${VENV}/bin/python" "${INSTALL_DIR}/test_llm_v2.py"

echo ""
info "All done! Services are running in the background."
info "  vLLM logs    : ~/vllm.log"
info "  Wrapper logs : ~/wrapper.log"
info "  Stop both    : bash run_local.sh --stop"
