#!/usr/bin/env bash
# ============================================================
# shisa-llm-deploy — Client Setup Script (H100)
#
# Run ONCE on the client H100 server after cloning the repo:
#
#   git clone <your-repo-url> shisa-llm-deploy
#   cd shisa-llm-deploy && sudo bash setup.sh
#
# What this script does:
#   1. Loads config from config/client.env
#   2. Downloads LLM_Q3_V1 from HuggingFace (once)
#   3. Creates a Python venv and installs vLLM + dependencies
#   4. Registers and starts two systemd services:
#        voicebot-llm         — vLLM server        (port 8004)
#        voicebot-llm-wrapper — FastAPI wrapper     (port 8005)
#
# Prerequisites on the H100 server:
#   - NVIDIA H100 GPU with CUDA 12.x drivers
#   - Python 3.10+
#   - sudo access
#   - Internet access for HuggingFace download (or a pre-downloaded model)
# ============================================================
set -euo pipefail

# ── Root / sudo check ────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "[ERROR] This script must be run with sudo or as root."
  echo "        Please re-run:"
  echo ""
  echo "          sudo bash setup.sh"
  echo ""
  echo "        If your account lacks sudo privileges, ask a system admin to"
  echo "        run this script, or run: sudo usermod -aG sudo \$USER && newgrp sudo"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/shisa-llm-deploy"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
LOG="/tmp/shisa_llm_setup.log"

# ── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "${LOG}"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "${LOG}"; }
error() { echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG}"; exit 1; }

exec > >(tee -a "${LOG}") 2>&1
info "Setup log: ${LOG}"
echo "============================================================"
echo "  LLM_Q3_V1 — Client Setup"
echo "============================================================"

# ── Load config ──────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/config/client.env"
if [[ -f "${ENV_FILE}" ]]; then
    info "Loading config from ${ENV_FILE}"
    set -a; source "${ENV_FILE}"; set +a
else
    warn "No config/client.env found — using defaults."
    warn "Copy config/client.env.example → config/client.env and edit it first."
fi

HF_MODEL_ID="${HF_MODEL_ID:-shisa-ai/LLM_Q3_V1}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/opt/voicebot/models/LLM_Q3_V1}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-LLM_Q3_V1}"
LLM_PORT="${LLM_PORT:-8004}"
LLM_API_PORT="${LLM_API_PORT:-8005}"
LLM_CUDA_DEVICE="${LLM_CUDA_DEVICE:-0}"
LLM_GPU_MEM="${LLM_GPU_MEM:-0.90}"
HF_TOKEN="${HF_TOKEN:-}"  # LLM_Q3_V1 is public — no token needed

# ━━━ 1/5  Install to ${INSTALL_DIR} ━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
info "━━━ 1/5  Installing to ${INSTALL_DIR} ━━━"
sudo mkdir -p "${INSTALL_DIR}"
sudo cp -r "${SCRIPT_DIR}/." "${INSTALL_DIR}/"
sudo chmod +x "${INSTALL_DIR}/start_vllm_llm_v2.sh"

# Write resolved .env
sudo tee "${INSTALL_DIR}/client.env" > /dev/null <<EOF
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

# ━━━ 2/5  Python venv + vLLM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
info "━━━ 2/5  Installing Python venv + vLLM (this takes ~10 min) ━━━"

# Detect the Python binary (prefer 3.11 → 3.10 → 3.12 → python3)
PYTHON_BIN=""
for py in python3.11 python3.10 python3.12 python3; do
    if command -v "${py}" > /dev/null 2>&1; then
        PYTHON_BIN="$(command -v "${py}")"
        break
    fi
done
[[ -z "${PYTHON_BIN}" ]] && error "No python3 found. Install Python 3.10+ first."
PYTHON_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
info "Using Python ${PYTHON_VER} at ${PYTHON_BIN}"

# Ensure python3-venv/ensurepip is available
if ! "${PYTHON_BIN}" -c 'import ensurepip' 2>/dev/null; then
    info "ensurepip not found — installing python${PYTHON_VER}-venv via apt …"
    apt-get update -qq
    apt-get install -y "python${PYTHON_VER}-venv" python3-pip \
        || error "Failed to install python${PYTHON_VER}-venv. Run manually: apt install python${PYTHON_VER}-venv"
fi

VENV="${INSTALL_DIR}/.venv"
if [[ ! -d "${VENV}" ]]; then
    info "Creating venv at ${VENV} …"
    "${PYTHON_BIN}" -m venv "${VENV}" \
        || error "venv creation failed. Try manually: ${PYTHON_BIN} -m venv ${VENV}"
fi
[[ -f "${VENV}/bin/activate" ]] || error "venv created but activate script missing at ${VENV}/bin/activate"
source "${VENV}/bin/activate"
pip install --upgrade pip --quiet
# Install vLLM first (GPU binary, ~2GB)
pip install vllm --quiet
# Install remaining wrapper/test dependencies
pip install fastapi "uvicorn[standard]" httpx psutil pydantic \
            requests matplotlib Pillow huggingface_hub --quiet
deactivate
info "venv installed at ${VENV}"

# ━━━ 3/5  Download model from HuggingFace (uses venv Python) ━
echo ""
info "━━━ 3/5  Downloading model from HuggingFace ━━━"
if [[ -f "${LLM_MODEL_PATH}/config.json" ]]; then
    info "Model already present at ${LLM_MODEL_PATH} — skipping download."
else
    info "Downloading ${HF_MODEL_ID} → ${LLM_MODEL_PATH}"
    info "This may take 10–30 minutes …"
    HF_MODEL_ID="${HF_MODEL_ID}" \
    LLM_MODEL_PATH="${LLM_MODEL_PATH}" \
    HF_TOKEN="${HF_TOKEN}" \
        "${INSTALL_DIR}/.venv/bin/python" "${INSTALL_DIR}/download_model.py" \
        || error "Model download failed. Check ${LOG} and re-run setup.sh."
fi
info "Model ready: ${LLM_MODEL_PATH}"
echo ""
info "━━━ 4/5  Writing launcher scripts ━━━"

sudo tee "${INSTALL_DIR}/start_wrapper.sh" > /dev/null <<'WRAPPER_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${INSTALL_DIR}/client.env"
source "${INSTALL_DIR}/.venv/bin/activate"
export LLM_MODEL_NAME LLM_MODEL_PATH LLM_VLLM_BASE_URL LLM_API_PORT
exec python "${INSTALL_DIR}/llm_api_wrapper_v2.py"
WRAPPER_SCRIPT
sudo chmod +x "${INSTALL_DIR}/start_wrapper.sh"

# Readiness probe: polls vLLM /health until it responds, used as ExecStartPre
sudo tee "${INSTALL_DIR}/wait_for_vllm.sh" > /dev/null <<'PROBE_SCRIPT'
#!/usr/bin/env bash
# Polls vLLM health endpoint until it responds 200 or timeout is reached.
# Used as ExecStartPre in voicebot-llm-wrapper.service.
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${INSTALL_DIR}/client.env"
LLM_PORT="${LLM_PORT:-8004}"
MAX_WAIT=300   # seconds — H100 model load can take ~2 min
INTERVAL=5
elapsed=0
echo "[wait_for_vllm] Waiting for vLLM on port ${LLM_PORT} …"
while true; do
    if curl -sf "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
        echo "[wait_for_vllm] vLLM is ready (${elapsed}s elapsed). Starting wrapper."
        exit 0
    fi
    if (( elapsed >= MAX_WAIT )); then
        echo "[wait_for_vllm] Timeout after ${MAX_WAIT}s — vLLM not ready. Aborting."
        exit 1
    fi
    sleep "${INTERVAL}"
    elapsed=$(( elapsed + INTERVAL ))
done
PROBE_SCRIPT
sudo chmod +x "${INSTALL_DIR}/wait_for_vllm.sh"

# ━━━ 5/5  Systemd services ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
info "━━━ 5/5  Installing systemd services ━━━"

# vLLM service
sudo tee /etc/systemd/system/voicebot-llm.service > /dev/null <<EOF
[Unit]
Description=VoiceBot LLM vLLM Server (${LLM_MODEL_NAME})
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/client.env
ExecStart=${INSTALL_DIR}/.venv/bin/python -m vllm.entrypoints.openai.api_server \\
    --model ${LLM_MODEL_PATH} \\
    --served-model-name ${LLM_MODEL_NAME} \\
    --port ${LLM_PORT} \\
    --gpu-memory-utilization ${LLM_GPU_MEM} \\
    --max-model-len 8192 \\
    --dtype bfloat16
Environment=CUDA_VISIBLE_DEVICES=${LLM_CUDA_DEVICE}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Wrapper service
sudo tee /etc/systemd/system/voicebot-llm-wrapper.service > /dev/null <<EOF
[Unit]
Description=VoiceBot LLM API Wrapper (${LLM_MODEL_NAME})
After=voicebot-llm.service
Requires=voicebot-llm.service

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/client.env
# Poll vLLM /health before starting — avoids blind sleep
ExecStartPre=${INSTALL_DIR}/wait_for_vllm.sh
ExecStart=${INSTALL_DIR}/start_wrapper.sh
Environment=LLM_WRAPPER_PORT=${LLM_API_PORT}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable voicebot-llm voicebot-llm-wrapper
sudo systemctl start voicebot-llm

info "Waiting for vLLM to become ready before starting wrapper …"
LLM_PORT="${LLM_PORT}" bash "${INSTALL_DIR}/wait_for_vllm.sh" 2>/dev/null \
    || { warn "vLLM did not respond in time during setup — starting wrapper anyway."; }
sudo systemctl start voicebot-llm-wrapper

echo ""
echo "============================================================"
info "Setup complete!"
info "  vLLM server  : http://localhost:${LLM_PORT}  (model: ${LLM_MODEL_NAME})"
info "  API wrapper  : http://localhost:${LLM_API_PORT}"
echo ""
info "Check status:"
info "  sudo systemctl status voicebot-llm voicebot-llm-wrapper"
info "View logs:"
info "  sudo journalctl -u voicebot-llm -f"
info "  sudo journalctl -u voicebot-llm-wrapper -f"
info "Health check:"
info "  curl http://localhost:${LLM_PORT}/health"
info "Run test suite:"
info "  cd ${INSTALL_DIR} && source .venv/bin/activate"
info "  python test_llm_v2.py"
echo "============================================================"
