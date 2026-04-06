#!/usr/bin/env bash
# binary.sh
# Creates:
#  1. Systemd service to auto-start vLLM on boot
#  2. PyInstaller binary packaging for Vocode application
#  3. Overwrites git repo
#
# Run: bash binary.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/home/venv-qwen35"
SERVICE_NAME="qwen35-vllm"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  binary.sh — Service + Binary Packaging                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1: Create systemd service for vLLM auto-start
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

create_systemd_service() {
    echo "▶ Creating systemd service: ${SERVICE_NAME}"
    echo ""
    
    # Create service file content
    SERVICE_CONTENT="[Unit]
Description=Qwen3.5-27B vLLM Server (OpenAI-compatible API on port 8004)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${SCRIPT_DIR}
Environment=\"PATH=${VENV_PATH}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\"
Environment=\"CUDA_VISIBLE_DEVICES=0\"
Environment=\"QWEN35_MODEL=/home/models/Qwen3.5-27B\"
Environment=\"QWEN35_PORT=8004\"
Environment=\"QWEN35_GPU_MEM=0.72\"
Environment=\"QWEN35_MAX_LEN=32768\"
ExecStart=${SCRIPT_DIR}/start_qwen35_h100.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"
    
    echo "$SERVICE_CONTENT" | sudo tee "$SERVICE_FILE" > /dev/null
    
    if [ -f "$SERVICE_FILE" ]; then
        echo "✅ Service file created: $SERVICE_FILE"
    else
        echo "❌ Failed to create service file"
        return 1
    fi
    
    # Reload systemd
    echo "▶ Reloading systemd daemon..."
    sudo systemctl daemon-reload
    
    # Enable service
    echo "▶ Enabling service to start on boot..."
    sudo systemctl enable "$SERVICE_NAME"
    echo "✅ Service enabled"
    
    # Optional: start service now
    read -p "Start vLLM now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "▶ Starting service..."
        sudo systemctl start "$SERVICE_NAME"
        sleep 3
        sudo systemctl status "$SERVICE_NAME" --no-pager | head -5
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2: PyInstaller binary packaging (Vocode application)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

create_pyinstaller_binary() {
    local target_app="$1"
    local output_name="$2"
    
    echo ""
    echo "▶ PyInstaller binary packaging"
    echo "  Source: $target_app"
    echo "  Output: $output_name"
    echo ""
    
    # Check if PyInstaller is installed
    if ! "${VENV_PATH}/bin/pip" show pyinstaller > /dev/null 2>&1; then
        echo "▶ Installing PyInstaller..."
        "${VENV_PATH}/bin/pip" install pyinstaller --quiet
    fi
    
    # Create build directory
    mkdir -p "${SCRIPT_DIR}/build"
    mkdir -p "${SCRIPT_DIR}/dist"
    
    # Run PyInstaller
    echo "▶ Running PyInstaller..."
    "${VENV_PATH}/bin/pyinstaller" \
        --name "${output_name}" \
        --onefile \
        --distpath "${SCRIPT_DIR}/dist" \
        --buildpath "${SCRIPT_DIR}/build" \
        --specpath "${SCRIPT_DIR}" \
        --hidden-import=vllm \
        --hidden-import=fastapi \
        --hidden-import=uvicorn \
        --collect-all=vllm \
        "${target_app}"
    
    if [ -f "${SCRIPT_DIR}/dist/${output_name}" ]; then
        echo "✅ Binary created: ${SCRIPT_DIR}/dist/${output_name}"
        ls -lh "${SCRIPT_DIR}/dist/${output_name}"
    else
        echo "⚠ Binary creation may have warnings, check build logs"
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3: Git sync (push current state)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sync_git() {
    echo ""
    echo "▶ Syncing with git repository"
    echo "  Repo: https://github.com/dcodingx/nec_ll.git"
    echo ""
    
    cd "${SCRIPT_DIR}"
    
    # Check git status
    if [ ! -d ".git" ]; then
        echo "⚠ Not a git repository. Initializing..."
        git init
        git remote add origin https://github.com/dcodingx/nec_ll.git
    fi
    
    # Add all files
    echo "▶ Staging files..."
    git add -A
    
    # Show status
    git status --short
    
    # Ask to commit
    read -p "Commit and push? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        git commit -m "Client deployment: Qwen3.5-27B vLLM setup ($TIMESTAMP)" || echo "⚠ No changes to commit"
        git push -u origin main || git push -u origin master || echo "⚠ Push may have failed"
        echo "✅ Git sync complete"
    else
        echo "Skipped git push"
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

main() {
    # Check prerequisites
    if [ ! -d "${VENV_PATH}" ]; then
        echo "❌ Error: venv not found at ${VENV_PATH}"
        exit 1
    fi
    
    echo "Prerequisites: ✅"
    echo ""
    
    # Step 1: Create systemd service
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ Step 1: Systemd Service Setup                             ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    create_systemd_service
    
    # Step 2: PyInstaller binary (optional, for Vocode)
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ Step 2: PyInstaller Binary Packaging (Optional)           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    read -p "Package application as binary? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Check if we have a main app to package
        if [ -f "llm_api_wrapper_v2.py" ]; then
            create_pyinstaller_binary "llm_api_wrapper_v2.py" "qwen35-api"
        elif [ -f "main.py" ]; then
            create_pyinstaller_binary "main.py" "qwen35-app"
        else
            echo "⚠ No main application found for packaging"
        fi
    fi
    
    # Step 3: Git sync
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ Step 3: Git Repository Sync                               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    sync_git
    
    # Final summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ ✅ Deployment Complete!                                   ║"
    echo "╠════════════════════════════════════════════════════════════╣"
    echo "║ Service: ${SERVICE_NAME}                                   ║"
    echo "║ Check:   sudo systemctl status ${SERVICE_NAME}            ║"
    echo "║          sudo systemctl restart ${SERVICE_NAME}           ║"
    echo "║                                                            ║"
    echo "║ Logs:    journalctl -u ${SERVICE_NAME} -f                 ║"
    echo "║                                                            ║"
    echo "║ API:     http://localhost:8004/v1/chat/completions       ║"
    echo "║ Docs:    http://localhost:8004/docs                       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
}

main "$@"
