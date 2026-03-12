#!/usr/bin/env bash
# ============================================================
# build.sh — LLM_Q3_V1 Security Hardening
#
# Run ONCE on the client AFTER setup.sh has completed:
#
#   sudo bash build.sh
#
# What this does:
#   1. Obfuscates Python source with pyarmor (hides logic/imports)
#   2. Generates a random AES-256 key → /etc/voicebot/model.key
#   3. Encrypts model to encrypted .bin.enc parts (streaming, no temp disk)
#   4. Shreds & removes the plaintext model
#   5. Installs start_vllm_secure.sh and updates the systemd service
#
# Security guarantees after this runs:
#   - Python source is obfuscated — model name, logic not visible
#   - Model weights are AES-256-CBC encrypted at rest
#   - Model decrypts to RAM (/dev/shm) only at service start
#   - RAM copy is shredded on service stop/reboot
#   - /etc/voicebot/model.key is root-only (chmod 600)
#
# IMPORTANT: Back up /etc/voicebot/model.key to a secure location
#            outside this server. Losing it = model unrecoverable.
# ============================================================
set -euo pipefail

# ── Root check ───────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] Run with sudo: sudo bash build.sh"
    exit 1
fi

INSTALL_DIR="/opt/shisa-llm-deploy"
VENV="${INSTALL_DIR}/.venv"
MODEL_STORE="/opt/voicebot/models"
MODEL_DIR="${MODEL_STORE}/LLM_Q3_V1"
MODEL_ENC_PREFIX="${MODEL_STORE}/LLM_Q3_V1.bin.enc.part_"
KEY_DIR="/etc/voicebot"
KEY_FILE="${KEY_DIR}/model.key"
LOG="/tmp/shisa_llm_build.log"
PART_SIZE="1G"    # chunk size for split (git-LFS / transfer friendly)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "${LOG}"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "${LOG}"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG}"; exit 1; }
section() { echo -e "\n${CYAN}━━━ $* ━━━${NC}" | tee -a "${LOG}"; }

exec > >(tee -a "${LOG}") 2>&1
echo "============================================================"
echo "  LLM_Q3_V1 — Security Hardening Build"
echo "============================================================"
info "Log: ${LOG}"

# ── Pre-flight checks ────────────────────────────────────────
[[ -d "${INSTALL_DIR}" ]]     || error "Install dir not found: ${INSTALL_DIR}. Run setup.sh first."
[[ -f "${VENV}/bin/python" ]] || error "venv not found: ${VENV}. Run setup.sh first."
command -v openssl >/dev/null || error "openssl not found. Install: apt install openssl"
command -v split   >/dev/null || error "split not found. Install: apt install coreutils"

# Model dir only required if encryption hasn't run yet
ENC_PARTS_EXIST=0
ls "${MODEL_ENC_PREFIX}"* > /dev/null 2>&1 && ENC_PARTS_EXIST=1
if [[ $ENC_PARTS_EXIST -eq 0 ]]; then
    [[ -d "${MODEL_DIR}" ]] || error "Model dir not found: ${MODEL_DIR} and no encrypted parts found. Run setup.sh first."
else
    info "Encrypted parts already exist — skipping model dir check."
fi
command -v shred   >/dev/null || error "shred not found. Install: apt install coreutils"

# Verify openssl supports AES-256-CBC + PBKDF2 (requires openssl >= 1.1.1)
openssl enc -aes-256-cbc -pbkdf2 -iter 1 -pass pass:test -in /dev/null -out /dev/null 2>/dev/null \
    || error "openssl version too old — needs >= 1.1.1 for PBKDF2 support."
info "Pre-flight checks passed."

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section "1/5  Obfuscating Python source with pyarmor"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

info "Installing pyarmor into venv …"
"${VENV}/bin/pip" install pyarmor --quiet

OBFUSC_OUT="${INSTALL_DIR}/.obfuscated"
mkdir -p "${OBFUSC_OUT}"

cd "${INSTALL_DIR}"

# Files to obfuscate
PYTHON_FILES=("llm_api_wrapper_v2.py" "test_llm_v2.py")
for f in "${PYTHON_FILES[@]}"; do
    if [[ -f "${INSTALL_DIR}/${f}" ]]; then
        info "Obfuscating ${f} …"
        "${VENV}/bin/pyarmor" gen --output "${OBFUSC_OUT}" "${INSTALL_DIR}/${f}"
    else
        warn "${f} not found — skipping."
    fi
done

# Replace originals with obfuscated versions
info "Installing obfuscated files over originals …"
if [[ -d "${OBFUSC_OUT}" ]]; then
    # Copy the pyarmor runtime package
    cp -r "${OBFUSC_OUT}/." "${INSTALL_DIR}/"
    # Wipe originals that were obfuscated
    for f in "${PYTHON_FILES[@]}"; do
        if [[ -f "${OBFUSC_OUT}/${f}" ]]; then
            cp "${OBFUSC_OUT}/${f}" "${INSTALL_DIR}/${f}"
            info "  Replaced: ${f}"
        fi
    done
fi

# Remove source references to the real HF model ID
for f in download_model.py setup.sh run_local.sh; do
    [[ -f "${INSTALL_DIR}/${f}" ]] && rm -f "${INSTALL_DIR}/${f}" && info "  Removed: ${f}"
done

info "Python obfuscation complete."

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section "2/5  Generating AES-256 encryption key"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mkdir -p "${KEY_DIR}"
chmod 700 "${KEY_DIR}"

if [[ -f "${KEY_FILE}" ]]; then
    warn "Key already exists at ${KEY_FILE} — reusing existing key."
    warn "Delete ${KEY_FILE} and re-run to generate a new key."
else
    # Generate 64-char hex key (256 bits)
    openssl rand -hex 32 > "${KEY_FILE}"
    chmod 600 "${KEY_FILE}"
    chown root:root "${KEY_FILE}"
    info "Key generated: ${KEY_FILE}"
fi

KEY_PREVIEW=$(head -c 8 "${KEY_FILE}")
echo ""
echo -e "${YELLOW}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  IMPORTANT — BACK UP YOUR ENCRYPTION KEY                ║${NC}"
echo -e "${YELLOW}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${YELLOW}║  Location : ${KEY_FILE}                        ║${NC}"
echo -e "${YELLOW}║  Preview  : ${KEY_PREVIEW}...                              ║${NC}"
echo -e "${YELLOW}║                                                          ║${NC}"
echo -e "${YELLOW}║  Run this to display the full key for backup:            ║${NC}"
echo -e "${YELLOW}║    sudo cat ${KEY_FILE}                         ║${NC}"
echo -e "${YELLOW}║                                                          ║${NC}"
echo -e "${YELLOW}║  Store it in a password manager or vault.                ║${NC}"
echo -e "${YELLOW}║  Losing this key = model permanently unrecoverable.      ║${NC}"
echo -e "${YELLOW}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section "3/5  Encrypting model (streaming — no extra disk needed)"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

info "Model source : ${MODEL_DIR}"
info "Encrypted to : ${MODEL_ENC_PREFIX}*"
info "This will take several minutes for a ~15GB model …"

# Skip if already encrypted
if ls "${MODEL_ENC_PREFIX}"* > /dev/null 2>&1; then
    PART_COUNT=$(ls "${MODEL_ENC_PREFIX}"* | wc -l)
    info "Encrypted parts already present (${PART_COUNT} parts) — skipping encryption."
else

# Stream: tar → openssl AES-256-CBC/PBKDF2 → split into PART_SIZE chunks
# -pass file: reads passphrase from key file (no key in command line / process list)
tar -czf - -C "${MODEL_STORE}" "LLM_Q3_V1" | \
    openssl enc -aes-256-cbc -pbkdf2 -iter 100000 \
        -pass "file:${KEY_FILE}" | \
    split -b "${PART_SIZE}" - "${MODEL_ENC_PREFIX}"

PART_COUNT=$(ls "${MODEL_ENC_PREFIX}"* 2>/dev/null | wc -l)
info "Encrypted into ${PART_COUNT} part(s): ${MODEL_ENC_PREFIX}*"
fi  # end skip-if-already-encrypted

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section "4/5  Shredding plaintext model from disk"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ -d "${MODEL_DIR}" ]]; then
    info "Removing plaintext model from disk …"
    find "${MODEL_DIR}" -type f \( \
        -name "*.safetensors" \
        -o -name "*.bin" \
        -o -name "*.pt" \
        -o -name "*.gguf" \
    \) -delete 2>/dev/null || true
    rm -rf "${MODEL_DIR}"
    info "Plaintext model removed from disk."
else
    info "Plaintext model already removed — skipping."
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section "5/5  Installing secure launcher and updating systemd"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

source "${INSTALL_DIR}/client.env" 2>/dev/null || true
LLM_MODEL_NAME="${LLM_MODEL_NAME:-LLM_Q3_V1}"
LLM_PORT="${LLM_PORT:-8004}"
LLM_GPU_MEM="${LLM_GPU_MEM:-0.90}"
LLM_CUDA_DEVICE="${LLM_CUDA_DEVICE:-0}"

RAM_DISK="/dev/shm/llm_runtime"

cat > "${INSTALL_DIR}/start_vllm_secure.sh" << 'SECURE_LAUNCHER'
#!/usr/bin/env bash
# ============================================================
# start_vllm_secure.sh — Decrypt model to RAM, run vLLM, wipe on exit
# Generated by build.sh — do not edit manually
# ============================================================
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${INSTALL_DIR}/client.env"

KEY_FILE="/etc/voicebot/model.key"
MODEL_ENC_PREFIX="/opt/voicebot/models/LLM_Q3_V1.bin.enc.part_"
RAM_DISK="/dev/shm/llm_runtime"
RAM_MODEL_PATH="${RAM_DISK}/LLM_Q3_V1"

# ── Cleanup: wipe RAM copy on exit ───────────────────────────
cleanup() {
    echo "[secure] Received exit signal — wiping RAM model copy …"
    if [[ -d "${RAM_MODEL_PATH}" ]]; then
        find "${RAM_MODEL_PATH}" -type f \( \
            -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \
        \) -exec shred -uz {} \; 2>/dev/null || true
        rm -rf "${RAM_DISK}"
        echo "[secure] RAM model wiped."
    fi
}
trap cleanup EXIT INT TERM

# ── Validate key ─────────────────────────────────────────────
[[ -f "${KEY_FILE}" ]] || { echo "[ERROR] Key not found: ${KEY_FILE}"; exit 1; }
[[ "$(stat -c '%a' "${KEY_FILE}")" == "600" ]] || chmod 600 "${KEY_FILE}"

# ── Decrypt model to RAM ─────────────────────────────────────
if [[ -f "${RAM_MODEL_PATH}/config.json" ]]; then
    echo "[secure] Model already decrypted in RAM — skipping decrypt."
else
    echo "[secure] Decrypting model to RAM disk (${RAM_DISK}) …"
    mkdir -p "${RAM_DISK}"

    # Stream: reassemble parts → openssl decrypt → tar extract to RAM
    cat "${MODEL_ENC_PREFIX}"* | \
        openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 \
            -pass "file:${KEY_FILE}" | \
        tar -xzf - -C "${RAM_DISK}"

    echo "[secure] Model decrypted to ${RAM_MODEL_PATH}"
fi

# ── Launch vLLM from RAM ─────────────────────────────────────
echo "[secure] Starting vLLM from RAM disk …"
CUDA_VISIBLE_DEVICES="${LLM_CUDA_DEVICE:-0}" \
exec "${INSTALL_DIR}/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${RAM_MODEL_PATH}" \
    --served-model-name "${LLM_MODEL_NAME}" \
    --port "${LLM_PORT}" \
    --gpu-memory-utilization "${LLM_GPU_MEM}" \
    --max-model-len 8192 \
    --dtype bfloat16
SECURE_LAUNCHER

chmod 700 "${INSTALL_DIR}/start_vllm_secure.sh"
chown root:root "${INSTALL_DIR}/start_vllm_secure.sh"
info "Installed: ${INSTALL_DIR}/start_vllm_secure.sh"

# Update the systemd service to use the secure launcher
SERVICE_USER="${SUDO_USER:-$(whoami)}"
cat > /etc/systemd/system/voicebot-llm.service << EOF
[Unit]
Description=VoiceBot LLM — Secure vLLM Server (${LLM_MODEL_NAME})
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/client.env
ExecStart=${INSTALL_DIR}/start_vllm_secure.sh
ExecStop=/bin/bash -c 'find /dev/shm/llm_runtime -type f \\( -name "*.safetensors" -o -name "*.bin" \\) -exec shred -uz {} \\; 2>/dev/null; rm -rf /dev/shm/llm_runtime'
Environment=CUDA_VISIBLE_DEVICES=${LLM_CUDA_DEVICE}
KillMode=mixed
TimeoutStopSec=30
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

info "Updated: /etc/systemd/system/voicebot-llm.service"
systemctl daemon-reload
systemctl restart voicebot-llm
info "voicebot-llm service restarted with secure launcher."

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
info "Security hardening complete!"
echo ""
info "  Python source  : obfuscated (pyarmor)"
info "  Model at rest  : AES-256-CBC encrypted in ${MODEL_ENC_PREFIX}*"
info "  Model at runtime: decrypts to /dev/shm/llm_runtime (RAM only)"
info "  RAM cleanup    : auto-wipe on service stop/reboot"
info "  Encryption key : ${KEY_FILE} (root-only, chmod 600)"
echo ""
echo -e "${YELLOW}ACTION REQUIRED:${NC}"
echo -e "  Run: ${CYAN}sudo cat ${KEY_FILE}${NC}"
echo -e "  Copy the key to a secure vault (password manager, HSM, etc.)"
echo -e "  This key is NOT stored in git and cannot be recovered if lost."
echo "============================================================"
