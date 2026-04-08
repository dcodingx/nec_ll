#!/usr/bin/env bash
# ============================================================
# nec_ll — Setup Script (H100, single GPU)
#
# Run ONCE on the client H100 server after cloning the repo:
#
#   git clone https://github.com/dcodingx/nec_ll.git
#   cd nec_ll && sudo bash setup.sh
#
# What this script does:
#   1. Loads config from config/client.env
#   2. Creates Python venv and installs vLLM
#   3. Downloads Qwen3.5-27B from HuggingFace (no token required)
#   4. Renames downloaded model folder to Q3_LLM_V2
#   5. Registers and starts systemd service:
#        voicebot-llm  — vLLM server (port 8004, auto-starts on boot)
#
# Prerequisites:
#   - NVIDIA H100 GPU with CUDA 12.x drivers
#   - Python 3.10+
#   - sudo access
# ============================================================
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "[ERROR] Run with sudo: sudo bash setup.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/nec_ll"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
LOG="/tmp/nec_ll_setup.log"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "${LOG}"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "${LOG}"; }
error() { echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG}"; exit 1; }

exec > >(tee -a "${LOG}") 2>&1
info "Setup log: ${LOG}"
echo "============================================================"
echo "  Q3_LLM_V2 — vLLM Setup (H100)"
echo "============================================================"

# ── Load config ──────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/config/client.env"
if [[ -f "${ENV_FILE}" ]]; then
    info "Loading config from ${ENV_FILE}"
    set -a; source "${ENV_FILE}"; set +a
else
    warn "No config/client.env found — using defaults."
    warn "Copy config/client.env.example → config/client.env and edit first."
fi

HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3.5-27B}"
LLM_DOWNLOAD_PATH="${LLM_DOWNLOAD_PATH:-/home/models/Qwen3.5-27B}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-Q3_LLM_V2}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/home/models/Q3_LLM_V2}"
LLM_PORT="${LLM_PORT:-8004}"
LLM_CUDA_DEVICE="${LLM_CUDA_DEVICE:-0}"
LLM_GPU_MEM="${LLM_GPU_MEM:-0.72}"
LLM_MAX_LEN="${LLM_MAX_LEN:-32768}"
HF_TOKEN="${HF_TOKEN:-}"

# ━━━ 1/5  Install to ${INSTALL_DIR} ━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
info "━━━ 1/5  Installing to ${INSTALL_DIR} ━━━"
mkdir -p "${INSTALL_DIR}"
cp -r "${SCRIPT_DIR}/." "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/start_qwen35_h100.sh"

# Write resolved .env
tee "${INSTALL_DIR}/client.env" > /dev/null <<EOF
HF_MODEL_ID=${HF_MODEL_ID}
LLM_DOWNLOAD_PATH=${LLM_DOWNLOAD_PATH}
LLM_MODEL_NAME=${LLM_MODEL_NAME}
LLM_MODEL_PATH=${LLM_MODEL_PATH}
LLM_PORT=${LLM_PORT}
LLM_CUDA_DEVICE=${LLM_CUDA_DEVICE}
LLM_GPU_MEM=${LLM_GPU_MEM}
LLM_MAX_LEN=${LLM_MAX_LEN}
EOF
info "Config written to ${INSTALL_DIR}/client.env"

# ━━━ 2/5  Python venv + vLLM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
info "━━━ 2/5  Installing Python venv + vLLM (~10 min) ━━━"

PYTHON_BIN=""
for py in python3.11 python3.10 python3.12 python3; do
    if command -v "${py}" > /dev/null 2>&1; then
        PYTHON_BIN="$(command -v "${py}")"; break
    fi
done
[[ -z "${PYTHON_BIN}" ]] && error "No python3 found. Install Python 3.10+ first."
PYTHON_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
info "Using Python ${PYTHON_VER} at ${PYTHON_BIN}"

if ! "${PYTHON_BIN}" -c 'import ensurepip' 2>/dev/null; then
    apt-get update -qq
    apt-get install -y "python${PYTHON_VER}-venv" python3-pip
fi

VENV="${INSTALL_DIR}/.venv"
if [[ -d "${VENV}" && ! -f "${VENV}/bin/activate" ]]; then
    rm -rf "${VENV}"
fi
if [[ ! -d "${VENV}" ]]; then
    info "Creating venv at ${VENV} …"
    "${PYTHON_BIN}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
pip install --upgrade pip --quiet
pip install vllm --quiet
pip install "huggingface_hub>=0.34.0,<1.0" --quiet
deactivate
info "venv installed at ${VENV}"

# ━━━ 3/5  Download model from HuggingFace ━━━━━━━━━━━━━━━━━━
echo ""
info "━━━ 3/5  Downloading model from HuggingFace ━━━"
if [[ -f "${LLM_MODEL_PATH}/config.json" ]]; then
    info "Model already present at ${LLM_MODEL_PATH} — skipping download."
elif [[ -f "${LLM_DOWNLOAD_PATH}/config.json" ]]; then
    info "Downloaded model found at ${LLM_DOWNLOAD_PATH} — renaming to ${LLM_MODEL_NAME} …"
    mv "${LLM_DOWNLOAD_PATH}" "${LLM_MODEL_PATH}"
    info "Model renamed: ${LLM_MODEL_PATH}"
else
    info "Downloading ${HF_MODEL_ID} → ${LLM_DOWNLOAD_PATH}"
    info "This may take 10–30 minutes …"
    HF_MODEL_ID="${HF_MODEL_ID}" \
    LLM_MODEL_PATH="${LLM_DOWNLOAD_PATH}" \
    HF_TOKEN="${HF_TOKEN}" \
        "${VENV}/bin/python" "${INSTALL_DIR}/download_model.py" \
        || error "Model download failed. Check ${LOG} and re-run setup.sh."
    info "Renaming model folder: ${LLM_DOWNLOAD_PATH} → ${LLM_MODEL_PATH}"
    mv "${LLM_DOWNLOAD_PATH}" "${LLM_MODEL_PATH}"
    info "Model ready as ${LLM_MODEL_NAME}: ${LLM_MODEL_PATH}"
fi

# ━━━ 4/5  Cleanup — remove unwanted source files ━━━━━━━━━━━
echo ""
info "━━━ 4/5  Cleaning up unwanted files ━━━"
for f in download.py run_local.sh llm_api_wrapper_v2.py test_inference.py; do
    [[ -f "${INSTALL_DIR}/${f}" ]] && rm -f "${INSTALL_DIR}/${f}" && info "  Removed: ${f}"
done
info "Cleanup complete."

# ━━━ 5/5  Systemd service ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
info "━━━ 5/5  Installing systemd service ━━━"

tee /etc/systemd/system/voicebot-llm.service > /dev/null <<EOF
[Unit]
Description=Q3_LLM_V2 vLLM Server — OpenAI-compatible API (port ${LLM_PORT})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/client.env
Environment=CUDA_VISIBLE_DEVICES=${LLM_CUDA_DEVICE}
ExecStart=${VENV}/bin/python -m vllm.entrypoints.openai.api_server \
    --model ${LLM_MODEL_PATH} \
    --served-model-name ${LLM_MODEL_NAME} \
    --port ${LLM_PORT} \
    --gpu-memory-utilization ${LLM_GPU_MEM} \
    --max-model-len ${LLM_MAX_LEN} \
    --dtype bfloat16 \
    --disable-log-requests
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable voicebot-llm
systemctl start voicebot-llm

echo ""
echo "============================================================"
info "Setup complete!"
info "  vLLM API   : http://localhost:${LLM_PORT}/v1"
info "  Model      : ${LLM_MODEL_NAME} at ${LLM_MODEL_PATH}"
echo ""
info "Service management:"
info "  sudo systemctl status voicebot-llm"
info "  sudo systemctl restart voicebot-llm"
info "  sudo journalctl -u voicebot-llm -f"
echo ""
info "Health check:"
info "  curl http://localhost:${LLM_PORT}/health"
info "  curl http://localhost:${LLM_PORT}/v1/models"
echo "============================================================"
