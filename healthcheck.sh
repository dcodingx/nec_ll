#!/usr/bin/env bash
# healthcheck.sh — Health verification for Q3_LLM_V2 vLLM stack
#
# Checks:
#   1. vLLM server (port 8004) — /health endpoint
#   2. /v1/models — confirms Q3_LLM_V2 is loaded
#   3. GPU — confirms vLLM is using GPU memory
#
# Usage:
#   bash healthcheck.sh
#   LLM_PORT=8004 bash healthcheck.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for env_file in "${SCRIPT_DIR}/client.env" "${SCRIPT_DIR}/config/client.env"; do
    [[ -f "${env_file}" ]] && set -a && source "${env_file}" && set +a && break
done

LLM_PORT="${LLM_PORT:-8004}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-Q3_LLM_V2}"
PASS=0; FAIL=0

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✅ PASS${NC}  $*"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}❌ FAIL${NC}  $*"; FAIL=$((FAIL+1)); }
info() { echo -e "  ${YELLOW}ℹ${NC}       $*"; }

echo "============================================================"
echo "  Q3_LLM_V2 Health Check"
echo "  vLLM port: ${LLM_PORT}"
echo "============================================================"

# ── Check 1: vLLM /health ─────────────────────────────────────
echo ""
echo "▶ Check 1: vLLM server health (port ${LLM_PORT})"
if curl -sf --max-time 10 "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
    ok "vLLM /health responded"
else
    fail "vLLM not responding on port ${LLM_PORT}"
    info "Start with: sudo systemctl start voicebot-llm"
    info "Logs: sudo journalctl -u voicebot-llm -f"
fi

# ── Check 2: /v1/models ───────────────────────────────────────
echo ""
echo "▶ Check 2: Model loaded (${LLM_MODEL_NAME})"
MODELS_RESP=$(curl -sf --max-time 10 "http://localhost:${LLM_PORT}/v1/models" 2>/dev/null || echo "")
if echo "${MODELS_RESP}" | grep -q "${LLM_MODEL_NAME}"; then
    ok "Model '${LLM_MODEL_NAME}' is loaded and served"
elif [[ -n "${MODELS_RESP}" ]]; then
    fail "Model '${LLM_MODEL_NAME}' not found. Loaded: $(echo ${MODELS_RESP} | python3 -c 'import sys,json; d=json.load(sys.stdin); print([m["id"] for m in d.get("data",[])])' 2>/dev/null)"
else
    fail "/v1/models did not respond"
fi

# ── Check 3: GPU memory ───────────────────────────────────────
echo ""
echo "▶ Check 3: GPU memory usage"
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [[ "${GPU_MEM_USED}" -gt 1000 ]]; then
        ok "GPU memory in use: ${GPU_MEM_USED} MiB / ${GPU_MEM_TOTAL} MiB"
    else
        fail "GPU memory is low (${GPU_MEM_USED} MiB) — vLLM may not have loaded model"
    fi
else
    info "nvidia-smi not found — skipping GPU check"
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "  Results: ${GREEN}${PASS} passed${NC} / ${RED}${FAIL} failed${NC}"
echo "============================================================"

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
