#!/usr/bin/env bash
# binary.sh
# Creates systemd service for Q3_LLM_V2 auto-start on boot
# and optionally obfuscates any Python scripts with pyarmor.
#
# Run AFTER setup.sh:
#   sudo bash binary.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] Run with sudo: sudo bash binary.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load config
for env_file in "${SCRIPT_DIR}/client.env" "${SCRIPT_DIR}/config/client.env"; do
    [[ -f "${env_file}" ]] && set -a && source "${env_file}" && set +a && break
done

INSTALL_DIR="${INSTALL_DIR:-/opt/nec_ll}"
VENV="${INSTALL_DIR}/.venv"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/home/models/Q3_LLM_V2}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-Q3_LLM_V2}"
LLM_PORT="${LLM_PORT:-8004}"
LLM_GPU_MEM="${LLM_GPU_MEM:-0.72}"
LLM_MAX_LEN="${LLM_MAX_LEN:-32768}"
LLM_CUDA_DEVICE="${LLM_CUDA_DEVICE:-0}"
SERVICE_NAME="voicebot-llm"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  binary.sh — Systemd Service + Obfuscation                ║"
echo "╚════════════════════════════════════════════════════════════╝"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1: Create / update systemd service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "▶ Step 1: Installing systemd service: ${SERVICE_NAME}"

tee "${SERVICE_FILE}" > /dev/null <<SVCEOF
[Unit]
Description=Q3_LLM_V2 vLLM Server — OpenAI-compatible API (port ${LLM_PORT})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
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
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
echo "✅ Service ${SERVICE_NAME} installed and enabled for auto-start on boot."

read -p "Start vLLM now? (y/n): " -n 1 -r; echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    systemctl start "${SERVICE_NAME}"
    sleep 3
    systemctl status "${SERVICE_NAME}" --no-pager | head -8
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2: Pyarmor obfuscation (optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
read -p "Obfuscate Python scripts with pyarmor? (y/n): " -n 1 -r; echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "▶ Step 2: Installing pyarmor …"
    "${VENV}/bin/pip" install pyarmor --quiet

    OBFUSC_OUT="${INSTALL_DIR}/.obfuscated"
    mkdir -p "${OBFUSC_OUT}"
    cd "${INSTALL_DIR}"

    # Obfuscate test scripts (service uses vllm binary directly — nothing to obfuscate there)
    for f in test_llm_v2.py; do
        if [[ -f "${INSTALL_DIR}/${f}" ]]; then
            echo "  Obfuscating ${f} …"
            "${VENV}/bin/pyarmor" gen --output "${OBFUSC_OUT}" "${INSTALL_DIR}/${f}"
            [[ -f "${OBFUSC_OUT}/${f}" ]] && cp "${OBFUSC_OUT}/${f}" "${INSTALL_DIR}/${f}"
            echo "  ✅ ${f} obfuscated"
        fi
    done
    echo "✅ Obfuscation complete."
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║ ✅ Done!                                                   ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Service : ${SERVICE_NAME}                                 ║"
echo "║  Check   : sudo systemctl status ${SERVICE_NAME}          ║"
echo "║  Restart : sudo systemctl restart ${SERVICE_NAME}         ║"
echo "║  Logs    : sudo journalctl -u ${SERVICE_NAME} -f          ║"
echo "║  API     : http://localhost:${LLM_PORT}/v1/models         ║"
echo "╚════════════════════════════════════════════════════════════╝"
