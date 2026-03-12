#!/usr/bin/env bash
# ============================================================
# healthcheck.sh — Quick health verification for the LLM stack
#
# Checks:
#   1. vLLM server  (port 8004) — /health endpoint
#   2. API wrapper  (port 8005) — test query (English + Japanese)
#   3. GPU process  — confirms vLLM is holding GPU memory
#
# Usage:
#   bash healthcheck.sh             # uses defaults from client.env
#   LLM_PORT=8001 LLM_API_PORT=8005 bash healthcheck.sh   # custom ports
#
# Exit code:
#   0  — all checks passed
#   1  — one or more checks failed
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load config if present ────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/client.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a; source "${ENV_FILE}"; set +a
elif [[ -f "${SCRIPT_DIR}/config/client.env" ]]; then
    set -a; source "${SCRIPT_DIR}/config/client.env"; set +a
fi

LLM_PORT="${LLM_PORT:-8004}"
LLM_API_PORT="${LLM_API_PORT:-8005}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-shisa-v2-qwen2.5-7b}"
TIMEOUT=30   # seconds per request

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

pass() { echo -e "${GREEN}[PASS]${NC} $*"; (( PASS++ )) || true; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; (( FAIL++ )) || true; }
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

echo "============================================================"
echo "  ${LLM_MODEL_NAME} — Health Check"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── 1. vLLM /health ──────────────────────────────────────────
echo ""
info "Check 1/4 — vLLM server health (port ${LLM_PORT})"
VLLM_HEALTH=$(curl -sf --max-time "${TIMEOUT}" \
    "http://localhost:${LLM_PORT}/health" 2>&1) && rc=0 || rc=$?
if [[ ${rc} -eq 0 ]]; then
    pass "vLLM /health responded OK"
else
    fail "vLLM /health unreachable at http://localhost:${LLM_PORT}/health"
fi

# ── 2. vLLM /v1/models (model listed) ────────────────────────
echo ""
info "Check 2/4 — vLLM model listing"
MODELS_RESP=$(curl -sf --max-time "${TIMEOUT}" \
    "http://localhost:${LLM_PORT}/v1/models" 2>&1) && rc=0 || rc=$?
if [[ ${rc} -eq 0 ]]; then
    if echo "${MODELS_RESP}" | grep -q "${LLM_MODEL_NAME}"; then
        pass "Model '${LLM_MODEL_NAME}' is loaded and listed"
    else
        fail "Connected to vLLM but '${LLM_MODEL_NAME}' not in model list"
        echo "     Response: ${MODELS_RESP}"
    fi
else
    fail "Could not reach http://localhost:${LLM_PORT}/v1/models"
fi

# ── 3. Wrapper — English query ────────────────────────────────
echo ""
info "Check 3/4 — API wrapper English query (port ${LLM_API_PORT})"
EN_RESP=$(curl -sf --max-time "${TIMEOUT}" \
    -X POST "http://localhost:${LLM_API_PORT}/llm_testing" \
    -H "Content-Type: application/json" \
    -d '{"text":"My computer is not working.","language":"en","use_full_prompt":false}' \
    2>&1) && rc=0 || rc=$?
if [[ ${rc} -eq 0 ]]; then
    if echo "${EN_RESP}" | grep -q '"response"'; then
        pass "Wrapper returned a response for English query"
        SNIPPET=$(echo "${EN_RESP}" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('response','')[:100])" 2>/dev/null || true)
        [[ -n "${SNIPPET}" ]] && echo "     Preview: ${SNIPPET}…"
    elif echo "${EN_RESP}" | grep -q '"error"'; then
        fail "Wrapper returned an error for English query"
        echo "     ${EN_RESP}"
    else
        fail "Unexpected wrapper response for English query"
        echo "     ${EN_RESP}"
    fi
else
    fail "API wrapper unreachable at http://localhost:${LLM_API_PORT}/llm_testing"
fi

# ── 4. Wrapper — Japanese query ───────────────────────────────
echo ""
info "Check 4/4 — API wrapper Japanese query (port ${LLM_API_PORT})"
JA_RESP=$(curl -sf --max-time "${TIMEOUT}" \
    -X POST "http://localhost:${LLM_API_PORT}/llm_testing" \
    -H "Content-Type: application/json" \
    -d '{"text":"パソコンが動きません。","language":"ja","use_full_prompt":false}' \
    2>&1) && rc=0 || rc=$?
if [[ ${rc} -eq 0 ]]; then
    if echo "${JA_RESP}" | grep -q '"response"'; then
        pass "Wrapper returned a response for Japanese query"
        SNIPPET=$(echo "${JA_RESP}" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('response','')[:100])" 2>/dev/null || true)
        [[ -n "${SNIPPET}" ]] && echo "     Preview: ${SNIPPET}…"
    elif echo "${JA_RESP}" | grep -q '"error"'; then
        fail "Wrapper returned an error for Japanese query"
        echo "     ${JA_RESP}"
    else
        fail "Unexpected wrapper response for Japanese query"
        echo "     ${JA_RESP}"
    fi
else
    fail "API wrapper unreachable at http://localhost:${LLM_API_PORT}/llm_testing"
fi

# ── 5. GPU process check (optional — skipped if nvidia-smi absent) ──
echo ""
info "Bonus — GPU memory check (nvidia-smi)"
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null) && rc=0 || rc=$?
    if [[ ${rc} -eq 0 ]] && [[ -n "${GPU_INFO}" ]]; then
        while IFS=',' read -r idx used total util; do
            used=$(echo "${used}" | xargs); total=$(echo "${total}" | xargs)
            util=$(echo "${util}" | xargs); idx=$(echo "${idx}" | xargs)
            echo "     GPU${idx}: ${used}/${total} MB  ${util}% util"
        done <<< "${GPU_INFO}"
        pass "GPU info retrieved"
    else
        fail "nvidia-smi present but failed to query GPU"
    fi
else
    info "nvidia-smi not found — skipping GPU check"
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "============================================================"

if [[ ${FAIL} -gt 0 ]]; then
    echo ""
    echo "Troubleshooting tips:"
    echo "  sudo systemctl status voicebot-llm voicebot-llm-wrapper"
    echo "  sudo journalctl -u voicebot-llm -n 50"
    echo "  sudo journalctl -u voicebot-llm-wrapper -n 50"
    exit 1
fi

echo ""
echo "All checks passed — stack is healthy."
exit 0
