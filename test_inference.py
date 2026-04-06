#!/usr/bin/env python3
"""
test_inference.py — Health check for vLLM server

Tests that Qwen3.5-27B is running and responding correctly on port 8004.
Run: python3 test_inference.py
"""

import sys
import json
import time
import requests
from datetime import datetime

VLLM_URL = "http://localhost:8004/v1/chat/completions"
MODEL_NAME = "Qwen3.5-27B"
TIMEOUT = 30

def test_inference():
    """Test vLLM inference with a simple prompt."""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.3,
        "max_tokens": 50,
        "top_p": 0.9,
    }
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  vLLM Health Check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
╠════════════════════════════════════════════════════════════╣
║  URL    : {VLLM_URL}
║  Model  : {MODEL_NAME}
║  Timeout: {TIMEOUT}s
╚════════════════════════════════════════════════════════════╝
    """)
    
    try:
        print("▶ Sending test request...")
        response = requests.post(
            VLLM_URL,
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        
        # Extract response
        if "choices" not in result or not result["choices"]:
            print(f"❌ Invalid response format: {json.dumps(result, indent=2)}")
            return False
        
        message = result["choices"][0].get("message", {}).get("content", "")
        finish_reason = result["choices"][0].get("finish_reason", "unknown")
        tokens_used = result.get("usage", {})
        
        print(f"""
✅ Response received!

Response:
  {message}

Finish reason: {finish_reason}
Tokens:
  - Input:  {tokens_used.get('prompt_tokens', '?')}
  - Output: {tokens_used.get('completion_tokens', '?')}
  - Total:  {tokens_used.get('total_tokens', '?')}

Model: {result.get('model', 'unknown')}
    """)
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {VLLM_URL}")
        print("   Is vLLM running? Try: bash start_qwen35_h100.sh")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout after {TIMEOUT}s")
        print("   Model may still be loading. Wait a moment and try again.")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(1, max_retries + 1):
        print(f"\n{'─'*60}")
        print(f"Attempt {attempt}/{max_retries}")
        print(f"{'─'*60}\n")
        
        if test_inference():
            print(f"""
╔════════════════════════════════════════════════════════════╗
║  ✅ vLLM Server is Healthy!                               ║
╠════════════════════════════════════════════════════════════╣
║  Ready for production use.                                 ║
║                                                            ║
║  Next: Delete download.py (cleanup)                        ║
║        Run: bash binary.sh (create service + binary)       ║
╚════════════════════════════════════════════════════════════╝
            """)
            return 0
        
        if attempt < max_retries:
            print(f"\nℹ Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  ❌ vLLM Server Health Check Failed                       ║
╠════════════════════════════════════════════════════════════╣
║  Troubleshooting:                                          ║
║  1. Is vLLM running?                                       ║
║     Check: ps aux | grep vllm                             ║
║  2. Is model loaded?                                       ║
║     Check: nvidia-smi (GPU memory usage)                   ║
║  3. Port 8004 open?                                        ║
║     Check: netstat -tlnp | grep 8004                       ║
║  4. Restart vLLM: bash start_qwen35_h100.sh               ║
╚════════════════════════════════════════════════════════════╝
    """)
    return 1

if __name__ == "__main__":
    sys.exit(main())
