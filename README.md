# Client H100 Deployment — Qwen3.5-27B vLLM

**Objective**: Deploy Qwen3.5-27B model on client's single H100 GPU with OpenAI-compatible API on port 8004.

---

## 📋 Overview

| Component | Details |
|-----------|---------|
| **Model** | Qwen/Qwen3.5-27B (~16 GB bfloat16) |
| **Server** | vLLM OpenAI-compatible API |
| **Port** | 8004 |
| **GPU** | Single H100 (no tensor parallel) |
| **Storage** | `/home/models/Qwen3.5-27B` |
| **venv** | `/home/venv-qwen35` |
| **Service** | `qwen35-vllm` (systemd auto-start) |

---

## 🚀 Quick Start (5 Steps)

### Step 1: Environment Setup
```bash
python3 setup.py
```
Creates Python venv with vLLM and all dependencies.

**Output**: `/home/venv-qwen35/` ready for use

---

### Step 2: Download Model
```bash
python3 download.py
```
Downloads Qwen3.5-27B from Hugging Face Hub → `/home/models/`

⏱ **Duration**: 5-15 minutes (depends on network)  
✓ **Resumable**: Can interrupt and resume safely

**After successful download:**
```bash
rm download.py  # Cleanup
```

---

### Step 3: Start vLLM Server
```bash
bash start_qwen35_h100.sh
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8004
INFO:     Application startup complete
```

🟢 **Status**: Server is running and ready

---

### Step 4: Health Check
In another terminal:
```bash
python3 test_inference.py
```

**Expected output:**
```
✅ Response received!
Response: I'm doing well, thank you for asking!
Tokens: Input: 7 | Output: 12 | Total: 19
```

---

### Step 5: Setup Auto-Start Service + Binary
```bash
bash binary.sh
```

This will:
1. ✅ Create systemd service `qwen35-vllm`
2. ✅ Enable auto-start on boot
3. ⚙️ Optionally package application as PyInstaller binary
4. 📤 Sync with git repository

---

## 📁 File Structure

```
/home/ubuntu/nec_ll/
├── setup.py                  # Environment setup (venv + deps)
├── download.py              # Download model from HF
├── start_qwen35_h100.sh      # Start vLLM server (port 8004)
├── test_inference.py         # Health check
├── binary.sh                 # Systemd service + PyInstaller
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .git/                     # Git repository
└── dist/                     # (Created by binary.sh) PyInstaller output
```

---

## 🔧 Detailed Commands

### Environment Setup
```bash
# Create new venv (or reuse existing)
python3 setup.py

# Manually verify CUDA
/home/venv-qwen35/bin/python3 -c "import torch; print(torch.cuda.is_available())"
```

### Model Download
```bash
# Download (resumable if interrupted)
python3 download.py

# Verify model exists
ls -lah /home/models/Qwen3.5-27B/

# Check model files
ls /home/models/Qwen3.5-27B/ | head -10
```

### Start Server

**Option A: Manual (for testing)**
```bash
bash start_qwen35_h100.sh
# Ctrl+C to stop
```

**Option B: Background (with nohup)**
```bash
nohup bash start_qwen35_h100.sh > vllm_server.log 2>&1 &
```

**Option C: Systemd service (auto-restart)**
```bash
sudo systemctl start qwen35-vllm
sudo systemctl status qwen35-vllm
sudo journalctl -u qwen35-vllm -f
```

### Health Checks
```bash
# Test inference
python3 test_inference.py

# Check GPU memory
nvidia-smi

# Check port usage
netstat -tlnp | grep 8004

# Direct curl test
curl -X POST http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.3,
    "max_tokens": 50
  }'

# View API docs
curl -s http://localhost:8004/docs | head -20
```

---

## 🛠 Troubleshooting

### Problem: "Connection refused" when running test_inference.py

**Diagnosis:**
```bash
# Check if vLLM is running
ps aux | grep vllm

# Check port
netstat -tlnp | grep 8004

# Check logs
sudo journalctl -u qwen35-vllm --no-pager | tail -20
```

**Solution:**
1. Make sure `bash start_qwen35_h100.sh` is running
2. Wait 2-3 minutes for model to load (first time)
3. Check GPU memory: `nvidia-smi`

---

### Problem: "Out of memory" error

**Diagnosis:**
```bash
nvidia-smi  # Check available VRAM
```

**Solution:**
- Reduce GPU memory utilization (default 0.72)
- Edit environment before starting server:
  ```bash
  export QWEN35_GPU_MEM=0.5
  bash start_qwen35_h100.sh
  ```
- Or reduce max_model_len (default 32768)

---

### Problem: Model download fails

**Diagnosis:**
```bash
# Check internet connectivity
ping huggingface.co

# Check Hugging Face credentials
cat ~/.huggingface/token
```

**Solution:**
1. Resume download: `python3 download.py`
2. Check `/home/models/` directory exists and is writable:
   ```bash
   ls -ld /home/models/
   touch /home/models/test.txt
   ```

---

## 📊 Performance Expectations

| Metric | Value |
|--------|-------|
| Model size | ~16 GB (bfloat16) |
| GPU memory usage | ~18 GB (with overhead) |
| Startup time (cold) | 2-3 minutes |
| Startup time (warm) | <5 seconds |
| Inference latency | 50-200 ms (50 tokens) |
| Throughput | ~50-100 tokens/sec |

---

## 🔐 Security & Best Practices

1. **Firewall**: Only expose port 8004 internally (not to internet)
   ```bash
   sudo ufw allow from 192.168.0.0/16 to any port 8004
   sudo ufw default deny incoming
   ```

2. **API Key** (optional): Add authentication via reverse proxy or
   ```bash
   # Start with --api-key
   vllm serve /home/models/Qwen3.5-27B --api-key YOUR_KEY
   ```

3. **Logs**: Monitor vLLM logs for errors
   ```bash
   sudo journalctl -u qwen35-vllm -f
   ```

4. **Backups**: Keep model checksum
   ```bash
   sha256sum /home/models/Qwen3.5-27B/model.safetensors.index.json
   ```

---

## 🔄 Service Management

### Start/Stop/Restart
```bash
sudo systemctl start qwen35-vllm
sudo systemctl stop qwen35-vllm
sudo systemctl restart qwen35-vllm
```

### View Logs
```bash
# Real-time logs
sudo journalctl -u qwen35-vllm -f

# Last 100 lines
sudo journalctl -u qwen35-vllm -n 100

# Errors only
sudo journalctl -u qwen35-vllm -p err
```

### Auto-start on Boot
```bash
# Enable
sudo systemctl enable qwen35-vllm

# Disable
sudo systemctl disable qwen35-vllm

# Check status
systemctl is-enabled qwen35-vllm
```

---

## 📤 Git Repository

All deployment files are tracked in git:
```bash
# View changes
git status
git diff

# Commit updates
git add -A
git commit -m "Update deployment config"

# Push to GitHub
git push origin main
```

Repository: https://github.com/dcodingx/nec_ll.git

---

## 📝 Cleanup

After successful deployment, remove temporary files:
```bash
# Remove model download script (if no longer needed)
rm download.py

# Clear build artifacts (optional)
rm -rf build/ __pycache__/
```

---

## ✅ Checklist

- [ ] Python venv created at `/home/venv-qwen35/`
- [ ] Model downloaded to `/home/models/Qwen3.5-27B/`
- [ ] vLLM server starts and listens on port 8004
- [ ] `test_inference.py` passes successfully
- [ ] Systemd service `qwen35-vllm` enabled and running
- [ ] Logs show no errors: `sudo journalctl -u qwen35-vllm`
- [ ] Health check: `curl http://localhost:8004/docs`
- [ ] Git repository synced with changes

---

## 📞 Support

If issues arise:
1. Check logs: `sudo journalctl -u qwen35-vllm -f`
2. Verify GPU: `nvidia-smi`
3. Test manually: `python3 test_inference.py`
4. Review this README troubleshooting section

---

**Last Updated**: 2026-04-06  
**Deployment Status**: Ready for production
