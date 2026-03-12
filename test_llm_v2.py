#!/usr/bin/env python3
"""
LLM Test Suite for LLM_Q3_V1
Tests performance, latency, memory usage, and accuracy

Model: LLM_Q3_V1
Context: IT Support Voice Assistant
The LLM helps users troubleshoot computer problems:
- English: Power issues, error messages, blue screen errors
- Japanese: 電源問題、エラーメッセージ、ブルースクリーン
"""

import requests
import time
import psutil
import os
import json
import importlib
import concurrent.futures
import subprocess
from datetime import datetime

# Configuration
MODEL_NAME = "LLM_Q3_V1"
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://localhost:8005")
LLM_QUERY_ENDPOINT = f"{LLM_API_BASE_URL}/llm_testing"
_default_output = f"llm_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
OUTPUT_FILE = os.getenv("OUTPUT_FILE", _default_output)
METRICS_DIR = os.getenv("METRICS_DIR", "llm_metrics_artifacts")

# Test Configuration
USE_FULL_PROMPTS = True  # Set to True to use full IT Support system prompts
CONCURRENT_WORKERS = 5   # Number of concurrent requests for load testing
STABILITY_TEST_DURATION = 2100  # Duration in seconds (300 = 5 minutes, 1800 = 30 minutes, 2100 = 35 minutes)
STABILITY_TEST_INTERVAL = 10   # Interval between requests in seconds
CONVERSATION_SESSION_WORKERS = 5  # Concurrent multi-turn sessions for conversation load testing
SINGLE_REQUEST_SAMPLE_COUNT = 3  # Keep 2-3 single request samples for quick baseline checks
CONCURRENT_REQUEST_SAMPLE_COUNT = 3  # Keep a few one-shot concurrent requests for comparison
TERMINAL_DETAILED_PER_HIT = False  # Keep terminal concise; full per-hit details are written to txt
TERMINAL_STABILITY_LOG_EVERY = 5  # Print every N stability hits (and always print failures)

# Test queries based on IT Support context from main.py
# The LLM is trained to handle computer troubleshooting scenarios
TEST_QUERIES = [
    {
        "text": "My computer is not working.",
        "language": "en",
        "expected_keywords": ["computer", "condition", "power", "error", "blue screen"],
        "description": "English - Initial computer issue report"
    },
    {
        "text": "The power does not turn on at all.",
        "language": "en",
        "expected_keywords": ["disconnect", "devices", "power adapter", "mouse", "power button"],
        "description": "English - Power issue troubleshooting"
    },
    {
        "text": "An English error message is displayed on my screen.",
        "language": "en",
        "expected_keywords": ["error message", "restart", "content", "computer"],
        "description": "English - Error message scenario"
    },
    {
        "text": "A blue screen is displayed on my computer.",
        "language": "en",
        "expected_keywords": ["blue screen", "restart", "power adapter", "mouse"],
        "description": "English - Blue screen error"
    },
    {
        "text": "パソコンが動きません。",
        "language": "ja",
        "expected_keywords": ["パソコン", "状態", "電源", "エラー", "ブルースクリーン"],
        "description": "Japanese - Initial computer problem report"
    },
    {
        "text": "電源がまったく入らない。",
        "language": "ja",
        "expected_keywords": ["電源", "アダプター", "マウス", "機器", "ボタン"],
        "description": "Japanese - Power does not turn on"
    },
    {
        "text": "英語のエラーメッセージが表示されています。",
        "language": "ja",
        "expected_keywords": ["エラーメッセージ", "再起動", "内容", "パソコン"],
        "description": "Japanese - Error message displayed"
    },
    {
        "text": "ブルースクリーンが表示されます。",
        "language": "ja",
        "expected_keywords": ["ブルースクリーン", "再起動", "電源", "修復"],
        "description": "Japanese - Blue screen appears"
    },
]

# Concurrent conversation sessions (multi-turn) to simulate real chat traffic.
TEST_CONVERSATION_SESSIONS = [
    {
        "session_name": "en_power_flow",
        "turns": [
            {"text": "My computer is not working.", "language": "en"},
            {"text": "The power light is not turning on. What should I check first?", "language": "en"},
            {"text": "I removed external devices, still same issue. Next step?", "language": "en"},
        ],
    },
    {
        "session_name": "en_error_followup",
        "turns": [
            {"text": "An English error message appears after startup.", "language": "en"},
            {"text": "Should I restart immediately or check something before restart?", "language": "en"},
            {"text": "After restart the same error comes back. How to proceed?", "language": "en"},
        ],
    },
    {
        "session_name": "ja_blue_screen_flow",
        "turns": [
            {"text": "ブルースクリーンが表示されます。", "language": "ja"},
            {"text": "再起動しても改善しません。次に何を確認すべきですか？", "language": "ja"},
            {"text": "電源アダプターと周辺機器を確認しました。ほかに方法は？", "language": "ja"},
        ],
    },
    {
        "session_name": "en_blue_screen_full",
        "turns": [
            {"text": "A blue screen appears every time I start my computer.", "language": "en"},
            {"text": "I restarted with only the power adapter. The blue screen is still there.", "language": "en"},
            {"text": "I turned it on and off several times but no repair screen appeared.", "language": "en"},
        ],
    },
    {
        "session_name": "ja_power_full",
        "turns": [
            {"text": "電源がまったく入りません。", "language": "ja"},
            {"text": "電源アダプターとマウス以外を取り外しましたが、起動しませんでした。", "language": "ja"},
            {"text": "充電ランプは赤く点灯しています。30秒長押しをしましたが改善しませんでした。", "language": "ja"},
        ],
    },
]


def estimate_tokens_from_text(text, language):
    """Estimate token count from text length for coarse throughput/input sizing."""
    char_per_token = 2 if language == "ja" else 4
    return len(text) // char_per_token if text else 0


def normalize_ttft_ms(response_payload, latency_s):
    """Resolve TTFT from known response keys; fallback to latency when unavailable."""
    ttft_candidates = [
        response_payload.get("ttft_ms"),
        response_payload.get("first_token_ms"),
        response_payload.get("time_to_first_token_ms"),
    ]

    metrics_obj = response_payload.get("metrics")
    if isinstance(metrics_obj, dict):
        ttft_candidates.extend([
            metrics_obj.get("ttft_ms"),
            metrics_obj.get("first_token_ms"),
            metrics_obj.get("time_to_first_token_ms"),
        ])

    for candidate in ttft_candidates:
        if candidate is None:
            continue
        try:
            value = float(candidate)
            if value >= 0:
                return value, False
        except (TypeError, ValueError):
            continue

    return max(latency_s * 1000.0, 0.0), True


def compute_gpu_delta(gpu_before, gpu_after):
    """Compute per-GPU and total memory delta in MB between two snapshots."""
    before_map = {g["index"]: g for g in gpu_before}
    after_map = {g["index"]: g for g in gpu_after}
    all_indices = sorted(set(before_map.keys()) | set(after_map.keys()))

    per_gpu = []
    total_delta = 0
    for idx in all_indices:
        b = before_map.get(idx, {}).get("used", 0)
        a = after_map.get(idx, {}).get("used", 0)
        d = a - b
        total_delta += d
        per_gpu.append({
            "index": idx,
            "before_used_mb": b,
            "after_used_mb": a,
            "delta_used_mb": d,
        })
    return per_gpu, total_delta


def format_gpu_snapshot(gpus):
    """Human-readable GPU snapshot formatter."""
    if not gpus:
        return "N/A"
    return " | ".join(
        f"GPU{g['index']}: {g['used']}/{g['total']}MB ({g['utilization']:.1f}% util)"
        for g in gpus
    )


def format_gpu_delta(gpu_delta_per_gpu):
    """Human-readable GPU delta formatter."""
    if not gpu_delta_per_gpu:
        return "N/A"
    return " | ".join(
        f"GPU{g['index']}: {g['delta_used_mb']}MB"
        for g in gpu_delta_per_gpu
    )


def total_gpu_used_mb(gpus):
    """Total used GPU memory in MB across all visible GPUs for a snapshot."""
    return sum(g.get("used", 0) for g in gpus) if gpus else 0


def save_metrics_screenshot(summary, all_results, stability_results):
    """Save metrics charts as PNG for easy screenshot sharing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(METRICS_DIR, exist_ok=True)

    artifact_base = os.path.join(METRICS_DIR, f"llm_metrics_{timestamp}")
    png_path = f"{artifact_base}.png"
    json_path = f"{artifact_base}.json"

    latencies = [r.get("latency", 0.0) for r in all_results if r.get("success")]
    throughputs = [r.get("throughput_tps", 0.0) for r in all_results if r.get("success")]
    tokens = [r.get("tokens_generated", 0) for r in all_results if r.get("success")]
    stability_latency = [r.get("latency", 0.0) for r in stability_results if r.get("success")]

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(
            {
                "timestamp": timestamp,
                "summary": summary,
                "samples": {
                    "latency": latencies,
                    "throughput_tps": throughputs,
                    "tokens_generated": tokens,
                    "stability_latency": stability_latency,
                },
            },
            jf,
            indent=2,
            ensure_ascii=False,
        )

    # Preferred renderer: matplotlib (rich charts).
    try:
        plt = importlib.import_module("matplotlib.pyplot")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"LLM Metrics Snapshot - {MODEL_NAME}", fontsize=16, fontweight="bold")

        # Chart 1: Latency trend across successful requests
        ax1 = axes[0, 0]
        if latencies:
            ax1.plot(range(1, len(latencies) + 1), latencies, marker="o", linewidth=1.5)
            ax1.set_title("Latency Per Request")
            ax1.set_xlabel("Request Index")
            ax1.set_ylabel("Latency (s)")
            ax1.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No successful requests", ha="center", va="center")
            ax1.set_title("Latency Per Request")

        # Chart 2: Throughput distribution
        ax2 = axes[0, 1]
        throughput_non_zero = [t for t in throughputs if t > 0]
        if throughput_non_zero:
            ax2.hist(throughput_non_zero, bins=min(10, len(throughput_non_zero)), edgecolor="black")
            ax2.set_title("Throughput Distribution")
            ax2.set_xlabel("Tokens/sec")
            ax2.set_ylabel("Frequency")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No throughput data", ha="center", va="center")
            ax2.set_title("Throughput Distribution")

        # Chart 3: Stability latency trend
        ax3 = axes[1, 0]
        if stability_latency:
            ax3.plot(range(1, len(stability_latency) + 1), stability_latency, marker=".", linewidth=1.2)
            ax3.set_title("Stability Test Latency Trend")
            ax3.set_xlabel("Stability Iteration")
            ax3.set_ylabel("Latency (s)")
            ax3.grid(alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No stability samples", ha="center", va="center")
            ax3.set_title("Stability Test Latency Trend")

        # Chart 4: Summary text + success ratio
        ax4 = axes[1, 1]
        ax4.axis("off")
        success_count = summary.get("successful", 0)
        failed_count = summary.get("failed", 0)
        total_count = summary.get("total_requests", 0)
        success_rate = summary.get("success_rate", 0.0)
        avg_latency = summary.get("avg_latency", 0.0)
        avg_tokens = summary.get("avg_tokens", 0.0)
        avg_throughput = summary.get("avg_throughput", 0.0)

        summary_text = (
            f"Total Requests: {total_count}\\n"
            f"Successful: {success_count}\\n"
            f"Failed: {failed_count}\\n"
            f"Success Rate: {success_rate:.1%}\\n"
            f"Avg Latency: {avg_latency:.3f}s\\n"
            f"Avg Tokens: {avg_tokens:.0f}\\n"
            f"Avg Throughput: {avg_throughput:.2f} tok/s"
        )
        ax4.text(0.02, 0.98, summary_text, va="top", fontsize=11, family="monospace")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(png_path, dpi=160)
        plt.close(fig)

        return {
            "png_path": png_path,
            "json_path": json_path,
            "renderer": "matplotlib",
        }

    except Exception:
        # Fallback renderer: Pillow text panel to always produce a PNG artifact.
        try:
            from PIL import Image, ImageDraw, ImageFont

            image = Image.new("RGB", (1600, 900), color=(248, 249, 251))
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()

            lines = [
                f"LLM Metrics Snapshot - {MODEL_NAME}",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Total Requests: {summary.get('total_requests', 0)}",
                f"Successful: {summary.get('successful', 0)}",
                f"Failed: {summary.get('failed', 0)}",
                f"Success Rate: {summary.get('success_rate', 0.0):.1%}",
                "",
                f"Average Latency: {summary.get('avg_latency', 0.0):.3f}s",
                f"Min Latency: {summary.get('min_latency', 0.0):.3f}s",
                f"Max Latency: {summary.get('max_latency', 0.0):.3f}s",
                f"Average Tokens: {summary.get('avg_tokens', 0.0):.0f}",
                f"Total Tokens: {summary.get('total_tokens', 0)}",
                f"Average Throughput: {summary.get('avg_throughput', 0.0):.2f} tok/s",
                f"Average Response Length: {summary.get('avg_response_length', 0.0):.0f} chars",
                "",
                "Details: JSON artifact saved alongside this image.",
            ]

            y = 40
            for line in lines:
                draw.text((40, y), line, fill=(20, 20, 20), font=font)
                y += 28

            image.save(png_path)
            return {
                "png_path": png_path,
                "json_path": json_path,
                "renderer": "pillow",
            }
        except Exception as e:
            return {
                "png_path": None,
                "json_path": json_path,
                "renderer": "none",
                "error": str(e),
            }


def get_gpu_memory():
    """Query GPU memory usage"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,memory.total,memory.used,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpus = []
        for line in lines:
            parts = line.split(',')
            idx, total, used, util = map(str.strip, parts)
            gpus.append({
                'index': int(idx),
                'total': int(total),
                'used': int(used),
                'utilization': float(util)
            })
        return gpus
    except Exception as e:
        print(f"Could not query GPU memory: {e}")
        return []


def query_llm(query_data, use_full_prompt=USE_FULL_PROMPTS):
    """
    Send a query to the LLM API and measure performance
    
    Args:
        query_data: Dict with 'text' and 'language' keys
        use_full_prompt: Whether to use full IT support prompts
    
    Returns:
        Dict with response, latency, memory consumption, and metrics
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    text = query_data["text"]
    language = query_data["language"]
    
    # Capture GPU state BEFORE request
    gpu_before = get_gpu_memory()

    start_time = time.perf_counter()
    
    try:
        payload = {
            "text": text,
            "language": language,
            "use_full_prompt": use_full_prompt
        }
        if query_data.get("conversation_id") is not None:
            payload["conversation_id"] = query_data["conversation_id"]
        if query_data.get("turn_id") is not None:
            payload["turn_id"] = query_data["turn_id"]

        response = requests.post(
            LLM_QUERY_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        
    except requests.exceptions.Timeout:
        end_time = time.perf_counter()
        gpu_after = get_gpu_memory()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        gpu_delta_per_gpu, gpu_delta_total = compute_gpu_delta(gpu_before, gpu_after)
        input_tokens_est = estimate_tokens_from_text(text, language)
        return {
            "success": False,
            "error": "Timeout",
            "latency": end_time - start_time,
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "memory_consumed": mem_after - mem_before,
            "input_chars": len(text),
            "input_bytes": len(text.encode("utf-8")),
            "input_tokens_est": input_tokens_est,
            "tokens_generated": 0,
            "response_length": 0,
            "throughput_tps": 0,
            "ttft_ms": (end_time - start_time) * 1000.0,
            "ttft_is_estimated": True,
            "first_token_speed_per_s": (1.0 / (end_time - start_time)) if (end_time - start_time) > 0 else 0.0,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "gpu_delta_per_gpu": gpu_delta_per_gpu,
            "gpu_delta_total_mb": gpu_delta_total,
        }
    except requests.exceptions.RequestException as e:
        end_time = time.perf_counter()
        gpu_after = get_gpu_memory()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        gpu_delta_per_gpu, gpu_delta_total = compute_gpu_delta(gpu_before, gpu_after)
        input_tokens_est = estimate_tokens_from_text(text, language)
        return {
            "success": False,
            "error": str(e),
            "latency": end_time - start_time,
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "memory_consumed": mem_after - mem_before,
            "input_chars": len(text),
            "input_bytes": len(text.encode("utf-8")),
            "input_tokens_est": input_tokens_est,
            "tokens_generated": 0,
            "response_length": 0,
            "throughput_tps": 0,
            "ttft_ms": (end_time - start_time) * 1000.0,
            "ttft_is_estimated": True,
            "first_token_speed_per_s": (1.0 / (end_time - start_time)) if (end_time - start_time) > 0 else 0.0,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "gpu_delta_per_gpu": gpu_delta_per_gpu,
            "gpu_delta_total_mb": gpu_delta_total,
        }
    
    end_time = time.perf_counter()
    
    # Capture GPU state AFTER request
    gpu_after = get_gpu_memory()
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    latency = end_time - start_time
    mem_consumed = mem_after - mem_before
    
    # Extract response
    llm_response = result.get("response", "")
    has_error = "error" in result
    
    input_tokens_est = estimate_tokens_from_text(text, language)
    tokens_generated = estimate_tokens_from_text(llm_response, language) if llm_response else 0
    throughput_tps = tokens_generated / latency if latency > 0 and tokens_generated > 0 else 0
    ttft_ms, ttft_is_estimated = normalize_ttft_ms(result, latency)
    first_token_speed_per_s = (1000.0 / ttft_ms) if ttft_ms > 0 else 0.0
    
    success = not has_error
    
    # Check for expected keywords
    expected_keywords = query_data.get("expected_keywords", [])
    keywords_found = []
    keywords_missing = []
    
    if expected_keywords and not has_error:
        for keyword in expected_keywords:
            if keyword.lower() in llm_response.lower():
                keywords_found.append(keyword)
            else:
                keywords_missing.append(keyword)
        
        keyword_accuracy = len(keywords_found) / len(expected_keywords) if expected_keywords else 0
    else:
        keyword_accuracy = None
    
    gpu_delta_per_gpu, gpu_delta_total = compute_gpu_delta(gpu_before, gpu_after)

    return {
        "success": success,
        "query": text,
        "language": language,
        "response": llm_response,
        "latency": latency,
        "latency_ms": latency * 1000.0,
        "ttft_ms": ttft_ms,
        "ttft_is_estimated": ttft_is_estimated,
        "first_token_speed_per_s": first_token_speed_per_s,
        "input_chars": len(text),
        "input_bytes": len(text.encode("utf-8")),
        "input_tokens_est": input_tokens_est,
        "tokens_generated": tokens_generated,
        "throughput_tps": throughput_tps,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_consumed": mem_consumed,
        "response_length": len(llm_response),
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing,
        "keyword_accuracy": keyword_accuracy,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_delta_per_gpu": gpu_delta_per_gpu,
        "gpu_delta_total_mb": gpu_delta_total,
        "error": result.get("error") if has_error else None
    }


def run_concurrent_queries(queries, max_workers=CONCURRENT_WORKERS):
    """Run multiple queries concurrently"""
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(query_llm, q, USE_FULL_PROMPTS) for q in queries]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"success": False, "error": str(e)})
    
    total_time = time.time() - start_time
    return results, total_time


def run_concurrent_conversation_sessions(sessions, max_workers=CONVERSATION_SESSION_WORKERS):
    """Run concurrent multi-turn conversation sessions."""

    def execute_single_session(session_idx, session_payload):
        session_name = session_payload.get("session_name", f"session_{session_idx + 1}")
        turns = session_payload.get("turns", [])
        session_results = []

        for turn_idx, turn in enumerate(turns, start=1):
            turn_query = {
                "text": turn["text"],
                "language": turn["language"],
                "description": f"{session_name} turn {turn_idx}",
                "conversation_id": session_name,
                "turn_id": turn_idx,
            }
            turn_result = query_llm(turn_query, USE_FULL_PROMPTS)
            turn_result["session_name"] = session_name
            turn_result["session_index"] = session_idx + 1
            turn_result["session_turn"] = turn_idx
            turn_result["session_total_turns"] = len(turns)
            session_results.append(turn_result)

        return session_results

    start_time = time.time()
    all_turn_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(execute_single_session, idx, s)
            for idx, s in enumerate(sessions)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                all_turn_results.extend(future.result())
            except Exception as e:
                all_turn_results.append({
                    "success": False,
                    "error": f"Session execution error: {e}",
                    "query": "",
                    "language": "unknown",
                })

    total_time = time.time() - start_time
    return all_turn_results, total_time




if __name__ == "__main__":
    print("="*80)
    print(f"LLM Matrix Test Suite - {MODEL_NAME}")
    print("="*80)
    print(f"API Endpoint: {LLM_QUERY_ENDPOINT}")
    print(f"Full Prompts: {USE_FULL_PROMPTS}")
    print(f"Concurrent Workers: {CONCURRENT_WORKERS}")
    print(f"Single Request Samples: {SINGLE_REQUEST_SAMPLE_COUNT}")
    print(f"Concurrent One-shot Samples: {CONCURRENT_REQUEST_SAMPLE_COUNT}")
    print(f"Conversation Session Workers: {CONVERSATION_SESSION_WORKERS}")
    print(f"Conversation Sessions: {len(TEST_CONVERSATION_SESSIONS)}")
    print(f"Terminal Detailed Per-hit Logs: {TERMINAL_DETAILED_PER_HIT}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = []
    hit_counter = [0]

    def attach_hit_meta(result, phase, **kwargs):
        hit_counter[0] += 1
        result["hit_id"] = hit_counter[0]
        result["phase"] = phase
        result.update(kwargs)
        return result
    
    # ===================================================================
    # 9.1 System Idle State
    # ===================================================================
    print("\n" + "="*80)
    print("9.1 SYSTEM IDLE STATE")
    print("="*80)
    
    gpus_idle = get_gpu_memory()
    cpu_idle = psutil.cpu_percent(interval=1)
    mem_idle = psutil.virtual_memory()
    
    print(f"CPU Usage (Idle): {cpu_idle:.2f}%")
    print(f"RAM Usage (Idle): {mem_idle.used/1024**3:.2f} GB / {mem_idle.total/1024**3:.2f} GB")
    if gpus_idle:
        for gpu in gpus_idle:
            print(f"GPU {gpu['index']} (Idle): {gpu['used']}/{gpu['total']} MB "
                  f"({gpu['utilization']:.1f}% util)")
    
    # ===================================================================
    # 9.2 Active State - Single Request Test
    # ===================================================================
    print("\n" + "="*80)
    print("9.2 ACTIVE STATE - SINGLE REQUEST TEST")
    print("="*80)
    
    single_results = []
    single_test_queries = TEST_QUERIES[:max(1, SINGLE_REQUEST_SAMPLE_COUNT)]
    for idx, query in enumerate(single_test_queries, 1):
        print(f"\n--- Test {idx}/{len(single_test_queries)}: {query.get('description', 'N/A')} ---")
        print(f"Query: {query['text'][:80]}{'...' if len(query['text']) > 80 else ''}")
        print(f"Language: {query['language']}")
        
        result = query_llm(query, USE_FULL_PROMPTS)
        result = attach_hit_meta(result, "single_request", description=query.get("description", ""))
        single_results.append(result)
        all_results.append(result)
        
        # Print result
        if result['success']:
            if TERMINAL_DETAILED_PER_HIT:
                print(f"Success")
                print(f"  Latency: {result['latency']:.3f}s")
                print(f"  TTFT: {result['ttft_ms']:.2f} ms")
                print(f"  First-token Speed: {result['first_token_speed_per_s']:.2f} 1/s")
                print(f"  Tokens: {result['tokens_generated']}")
                print(f"  Throughput: {result['throughput_tps']:.2f} tokens/sec")
                print(f"  Input Size: {result['input_chars']} chars, {result['input_tokens_est']} tok(est), {result['input_bytes']} bytes")
                print(f"  Response Length: {result['response_length']} chars")
                print(f"  RAM Delta: {result['memory_consumed']:+.2f} MB")
                print(f"  GPU Delta Total: {result.get('gpu_delta_total_mb', 0):+d} MB")
                if result['keyword_accuracy'] is not None:
                    print(f"  Keyword Accuracy: {result['keyword_accuracy']:.1%}")
                    if result['keywords_missing']:
                        print(f"   Missing keywords: {', '.join(result['keywords_missing'])}")
            else:
                print(
                    f"Success | Latency={result['latency']:.3f}s | TTFT={result['ttft_ms']:.2f}ms | "
                    f"Tok={result['tokens_generated']} | RAM={result['memory_consumed']:+.2f}MB | "
                    f"GPU={result.get('gpu_delta_total_mb', 0):+d}MB"
                )
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")
        
        if idx < len(single_test_queries):
            time.sleep(1)  # Small delay between tests
    
    # ===================================================================
    # 9.3 Concurrent Request Test
    # ===================================================================
    print("\n" + "="*80)
    print(f"9.3 CONCURRENT REQUEST TEST ({CONCURRENT_WORKERS} workers)")
    print("="*80)
    
    concurrent_test_queries = TEST_QUERIES[:max(1, CONCURRENT_REQUEST_SAMPLE_COUNT)]
    concurrent_results, concurrent_time = run_concurrent_queries(concurrent_test_queries, CONCURRENT_WORKERS)
    for r in concurrent_results:
        attach_hit_meta(r, "concurrent_requests")
    all_results.extend(concurrent_results)
    
    concurrent_success = [r for r in concurrent_results if r['success']]
    print(f"\nTotal Queries: {len(concurrent_results)}")
    print(f"Total Time: {concurrent_time:.2f} seconds")
    print(f"Successful: {len(concurrent_success)}")
    print(f"Failed: {len(concurrent_results) - len(concurrent_success)}")
    if concurrent_success:
        avg_latency = sum(r['latency'] for r in concurrent_success) / len(concurrent_success)
        print(f"Average Latency: {avg_latency:.3f} seconds")

    # ===================================================================
    # 9.4 Concurrent Conversation Sessions Test
    # ===================================================================
    print("\n" + "="*80)
    print(f"9.4 CONCURRENT CONVERSATION SESSIONS ({CONVERSATION_SESSION_WORKERS} workers)")
    print("="*80)

    conversation_results, conversation_time = run_concurrent_conversation_sessions(
        TEST_CONVERSATION_SESSIONS,
        CONVERSATION_SESSION_WORKERS,
    )
    for r in conversation_results:
        attach_hit_meta(r, "concurrent_conversation_sessions")
    all_results.extend(conversation_results)

    conversation_success = [r for r in conversation_results if r.get("success")]
    print(f"\nTotal Session Turns: {len(conversation_results)}")
    print(f"Total Time: {conversation_time:.2f} seconds")
    print(f"Successful: {len(conversation_success)}")
    print(f"Failed: {len(conversation_results) - len(conversation_success)}")
    if conversation_success:
        avg_conv_latency = sum(r["latency"] for r in conversation_success) / len(conversation_success)
        avg_conv_ttft = sum(r.get("ttft_ms", 0.0) for r in conversation_success) / len(conversation_success)
        print(f"Average Latency: {avg_conv_latency:.3f} seconds")
        print(f"Average TTFT: {avg_conv_ttft:.2f} ms")
    
    # ===================================================================
    # 9.5 Stability Test (Long-Running)
    # ===================================================================
    print("\n" + "="*80)
    print(f"9.5 STABILITY TEST")
    print(f"Duration: {STABILITY_TEST_DURATION}s ({STABILITY_TEST_DURATION/60:.1f}min), "
          f"Interval: {STABILITY_TEST_INTERVAL}s")
    print("="*80)
    
    stability_results = []
    start_time = time.time()
    end_time = start_time + STABILITY_TEST_DURATION
    request_count = 0
    
    while time.time() < end_time:
        # Pick query using round-robin
        query = TEST_QUERIES[request_count % len(TEST_QUERIES)]
        
        elapsed = time.time() - start_time
        remaining = end_time - time.time()

        should_log_stability = ((request_count + 1) % TERMINAL_STABILITY_LOG_EVERY == 0)
        if should_log_stability:
            print(
                f"\n[Stability] Request #{request_count+1} | "
                f"Elapsed: {elapsed/60:.1f}min | Remaining: {remaining/60:.1f}min"
            )
            if TERMINAL_DETAILED_PER_HIT:
                print(f"Query: {query['text'][:60]}...")
        
        result = query_llm(query, USE_FULL_PROMPTS)
        result['iteration'] = request_count + 1
        result = attach_hit_meta(result, "stability_test", iteration=request_count + 1)
        stability_results.append(result)
        all_results.append(result)
        
        if result['success']:
            if should_log_stability:
                print(
                    f"Latency: {result['latency']:.3f}s, TTFT: {result['ttft_ms']:.2f}ms, "
                    f"Tokens: {result['tokens_generated']}, RAM Delta: {result['memory_consumed']:+.2f}MB"
                )
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")
        
        request_count += 1
        
        # Check if we have time for another request
        if time.time() + STABILITY_TEST_INTERVAL < end_time:
            time.sleep(STABILITY_TEST_INTERVAL)
        else:
            print("Time limit reached, ending stability test")
            break
    
    total_duration = time.time() - start_time
    print(f"\nStability Test Completed:")
    print(f"  Total Requests: {request_count}")
    print(f"  Total Duration: {total_duration/60:.1f} minutes")
    print(f"  Average Rate: {request_count/(total_duration/60):.2f} requests/minute")
    
    # ===================================================================
    # Post-Test System State
    # ===================================================================
    print("\n" + "="*80)
    print("POST-TEST SYSTEM STATE")
    print("="*80)
    
    gpus_after = get_gpu_memory()
    cpu_after = psutil.cpu_percent(interval=1)
    mem_after = psutil.virtual_memory()
    
    print(f"CPU Usage (After): {cpu_after:.2f}%")
    print(f"RAM Usage (After): {mem_after.used/1024**3:.2f} GB / {mem_after.total/1024**3:.2f} GB")
    if gpus_after:
        for gpu in gpus_after:
            print(f"GPU {gpu['index']} (After): {gpu['used']}/{gpu['total']} MB "
                  f"({gpu['utilization']:.1f}% util)")
    
    # ===================================================================
    # Summary Statistics
    # ===================================================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    successful = [r for r in all_results if r.get('success')]
    failed = [r for r in all_results if not r.get('success')]
    latencies = [r['latency'] for r in successful]
    tokens_list = [r['tokens_generated'] for r in successful]
    throughputs = [r['throughput_tps'] for r in successful if r['throughput_tps'] > 0]
    response_lengths = [r['response_length'] for r in successful]
    ttft_list = [r.get('ttft_ms', 0.0) for r in successful if r.get('ttft_ms') is not None]
    input_chars_list = [r.get('input_chars', 0) for r in all_results]
    input_tokens_list = [r.get('input_tokens_est', 0) for r in all_results]
    input_bytes_list = [r.get('input_bytes', 0) for r in all_results]
    ram_delta_list = [r.get('memory_consumed', 0.0) for r in all_results]
    gpu_delta_total_list = [r.get('gpu_delta_total_mb', 0) for r in all_results]
    gpu_used_before_list = [total_gpu_used_mb(r.get('gpu_before', [])) for r in all_results]
    gpu_used_after_list = [total_gpu_used_mb(r.get('gpu_after', [])) for r in all_results]
    gpu_peak_used_per_hit = [max([g.get('used', 0) for g in r.get('gpu_after', [])], default=0) for r in all_results]
    
    print(f"Total Requests: {len(all_results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(all_results):.1%})")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nPerformance Metrics:")
        print(f"  Average Latency: {sum(latencies)/len(latencies):.3f}s")
        print(f"  Min Latency: {min(latencies):.3f}s")
        print(f"  Max Latency: {max(latencies):.3f}s")
        if ttft_list:
            print(f"  Average TTFT: {sum(ttft_list)/len(ttft_list):.2f} ms")
            print(f"  Min TTFT: {min(ttft_list):.2f} ms")
            print(f"  Max TTFT: {max(ttft_list):.2f} ms")
        print(f"  Average Tokens: {sum(tokens_list)/len(tokens_list):.0f}")
        print(f"  Total Tokens: {sum(tokens_list)}")
        if throughputs:
            print(f"  Average Throughput: {sum(throughputs)/len(throughputs):.2f} tokens/sec")
        print(f"  Average Response Length: {sum(response_lengths)/len(response_lengths):.0f} chars")

    print(f"\nInput/Frame Size Metrics:")
    print(f"  Average Input Size: {sum(input_chars_list)/len(input_chars_list):.1f} chars")
    print(f"  Average Input Tokens (est): {sum(input_tokens_list)/len(input_tokens_list):.1f}")
    print(f"  Average Input Bytes: {sum(input_bytes_list)/len(input_bytes_list):.1f}")

    print(f"\nMemory Metrics (All Hits):")
    print(f"  Total RAM Delta: {sum(ram_delta_list):+.2f} MB")
    print(f"  Avg RAM Delta/Hit: {sum(ram_delta_list)/len(ram_delta_list):+.2f} MB")
    print(f"  Peak RAM Increase/Hit: {max(ram_delta_list):+.2f} MB")
    print(f"  Peak RAM Decrease/Hit: {min(ram_delta_list):+.2f} MB")
    print(f"  Total GPU Delta: {sum(gpu_delta_total_list):+d} MB")
    print(f"  Avg GPU Delta/Hit: {sum(gpu_delta_total_list)/len(gpu_delta_total_list):+.2f} MB")
    print(f"  Peak GPU Increase/Hit: {max(gpu_delta_total_list):+d} MB")
    print(f"  Peak GPU Decrease/Hit: {min(gpu_delta_total_list):+d} MB")
    print(f"  Peak GPU Used (aggregate across GPUs): {max(gpu_used_after_list)} MB")
    print(f"  Peak GPU Used (single GPU in a hit): {max(gpu_peak_used_per_hit)} MB")

    phase_stats = {}
    for r in all_results:
        phase = r.get("phase", "unknown")
        phase_stats.setdefault(phase, {"count": 0, "success": 0, "ram_delta": 0.0, "gpu_delta": 0})
        phase_stats[phase]["count"] += 1
        phase_stats[phase]["success"] += 1 if r.get("success") else 0
        phase_stats[phase]["ram_delta"] += r.get("memory_consumed", 0.0)
        phase_stats[phase]["gpu_delta"] += r.get("gpu_delta_total_mb", 0)

    if phase_stats:
        print(f"\nPhase-wise Consumption:")
        for phase_name, ps in phase_stats.items():
            rate = (ps["success"] / ps["count"]) if ps["count"] else 0.0
            print(
                f"  {phase_name}: hits={ps['count']}, success={rate:.1%}, "
                f"ram_delta={ps['ram_delta']:+.2f}MB, gpu_delta={ps['gpu_delta']:+d}MB"
            )
    
    if failed:
        print(f"\nError Analysis:")
        error_types = {}
        for r in failed:
            error = r.get('error', 'Unknown')
            error_types[error] = error_types.get(error, 0) + 1
        for error, count in error_types.items():
            print(f"  {error}: {count}")
    
    # Write results to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"LLM Matrix Test Results - {MODEL_NAME}\n")
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("TEST CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"API Endpoint: {LLM_QUERY_ENDPOINT}\n")
        f.write(f"Full Prompts: {USE_FULL_PROMPTS}\n")
        f.write(f"Concurrent Workers: {CONCURRENT_WORKERS}\n")
        f.write(f"Single Request Samples: {SINGLE_REQUEST_SAMPLE_COUNT}\n")
        f.write(f"Concurrent One-shot Samples: {CONCURRENT_REQUEST_SAMPLE_COUNT}\n")
        f.write(f"Conversation Session Workers: {CONVERSATION_SESSION_WORKERS}\n")
        f.write(f"Conversation Sessions: {len(TEST_CONVERSATION_SESSIONS)}\n")
        f.write(f"Stability Test Duration: {STABILITY_TEST_DURATION}s\n")
        f.write(f"Stability Test Interval: {STABILITY_TEST_INTERVAL}s\n\n")
        
        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Requests: {len(all_results)}\n")
        f.write(f"Successful: {len(successful)} ({len(successful)/len(all_results):.1%})\n")
        f.write(f"Failed: {len(failed)}\n")
        
        if successful:
            f.write(f"\nPerformance Metrics:\n")
            f.write(f"  Average Latency: {sum(latencies)/len(latencies):.3f}s\n")
            f.write(f"  Min Latency: {min(latencies):.3f}s\n")
            f.write(f"  Max Latency: {max(latencies):.3f}s\n")
            if ttft_list:
                f.write(f"  Average TTFT: {sum(ttft_list)/len(ttft_list):.2f} ms\n")
                f.write(f"  Min TTFT: {min(ttft_list):.2f} ms\n")
                f.write(f"  Max TTFT: {max(ttft_list):.2f} ms\n")
            f.write(f"  Average Tokens: {sum(tokens_list)/len(tokens_list):.0f}\n")
            f.write(f"  Total Tokens: {sum(tokens_list)}\n")
            if throughputs:
                f.write(f"  Average Throughput: {sum(throughputs)/len(throughputs):.2f} tokens/sec\n")
            f.write(f"  Average Response Length: {sum(response_lengths)/len(response_lengths):.0f} chars\n")

        f.write(f"\nInput/Frame Size Metrics:\n")
        f.write(f"  Average Input Size: {sum(input_chars_list)/len(input_chars_list):.1f} chars\n")
        f.write(f"  Average Input Tokens (est): {sum(input_tokens_list)/len(input_tokens_list):.1f}\n")
        f.write(f"  Average Input Bytes: {sum(input_bytes_list)/len(input_bytes_list):.1f}\n")

        f.write(f"\nMemory Metrics (All Hits):\n")
        f.write(f"  Total RAM Delta: {sum(ram_delta_list):+.2f} MB\n")
        f.write(f"  Avg RAM Delta/Hit: {sum(ram_delta_list)/len(ram_delta_list):+.2f} MB\n")
        f.write(f"  Peak RAM Increase/Hit: {max(ram_delta_list):+.2f} MB\n")
        f.write(f"  Peak RAM Decrease/Hit: {min(ram_delta_list):+.2f} MB\n")
        f.write(f"  Total GPU Delta: {sum(gpu_delta_total_list):+d} MB\n")
        f.write(f"  Avg GPU Delta/Hit: {sum(gpu_delta_total_list)/len(gpu_delta_total_list):+.2f} MB\n")
        f.write(f"  Peak GPU Increase/Hit: {max(gpu_delta_total_list):+d} MB\n")
        f.write(f"  Peak GPU Decrease/Hit: {min(gpu_delta_total_list):+d} MB\n")
        f.write(f"  Peak GPU Used (aggregate across GPUs): {max(gpu_used_after_list)} MB\n")
        f.write(f"  Peak GPU Used (single GPU in a hit): {max(gpu_peak_used_per_hit)} MB\n")

        if phase_stats:
            f.write(f"\nPhase-wise Consumption:\n")
            for phase_name, ps in phase_stats.items():
                rate = (ps["success"] / ps["count"]) if ps["count"] else 0.0
                f.write(
                    f"  {phase_name}: hits={ps['count']}, success={rate:.1%}, "
                    f"ram_delta={ps['ram_delta']:+.2f}MB, gpu_delta={ps['gpu_delta']:+d}MB\n"
                )
        
        if failed:
            f.write(f"\nError Analysis:\n")
            for error, count in error_types.items():
                f.write(f"  {error}: {count}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("CUMULATIVE CONSUMPTION BY HIT COUNT\n")
        f.write("-"*80 + "\n")
        cumulative_ram = 0.0
        cumulative_gpu = 0
        for r in sorted(all_results, key=lambda x: x.get("hit_id", 0)):
            cumulative_ram += r.get("memory_consumed", 0.0)
            cumulative_gpu += r.get("gpu_delta_total_mb", 0)
            f.write(
                f"Hit {r.get('hit_id', 0):03d}: "
                f"CumRAM={cumulative_ram:+.2f}MB, "
                f"CumGPU={cumulative_gpu:+d}MB\n"
            )

        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED HIT LOG (COMPACT, ALL REQUESTS/FRAMES)\n")
        f.write("-"*80 + "\n")
        f.write(
            "hit|phase|session|turn|ok|lang|lat_s|ttft_ms|first_tok_s|thr_tok_s|"
            "in_ch|in_tok|in_b|out_ch|out_tok|ram_b|ram_a|ram_d|gpu_b|gpu_a|gpu_d|"
            "kw_acc|error\n"
        )
        for r in sorted(all_results, key=lambda x: x.get("hit_id", 0)):
            kw_acc = r.get("keyword_accuracy")
            kw_acc_text = f"{kw_acc:.3f}" if kw_acc is not None else "NA"
            err = (r.get("error") or "").replace("\n", " ").replace("|", "/")[:120]
            session_name = r.get("session_name", "-")
            turn_info = "-"
            if r.get("session_turn") is not None and r.get("session_total_turns") is not None:
                turn_info = f"{r.get('session_turn')}/{r.get('session_total_turns')}"
            line = (
                f"{r.get('hit_id', 0):03d}|{r.get('phase', 'unknown')}|{session_name}|{turn_info}|"
                f"{int(bool(r.get('success')))}|{r.get('language', 'unknown')}|{r.get('latency', 0.0):.3f}|"
                f"{r.get('ttft_ms', 0.0):.2f}|{r.get('first_token_speed_per_s', 0.0):.2f}|"
                f"{r.get('throughput_tps', 0.0):.2f}|{r.get('input_chars', 0)}|"
                f"{r.get('input_tokens_est', 0)}|{r.get('input_bytes', 0)}|"
                f"{r.get('response_length', 0)}|{r.get('tokens_generated', 0)}|"
                f"{r.get('memory_before_mb', 0.0):.2f}|{r.get('memory_after_mb', 0.0):.2f}|"
                f"{r.get('memory_consumed', 0.0):+.2f}|{total_gpu_used_mb(r.get('gpu_before', []))}|"
                f"{total_gpu_used_mb(r.get('gpu_after', []))}|{r.get('gpu_delta_total_mb', 0):+d}|"
                f"{kw_acc_text}|{err}"
            )
            f.write(line + "\n")
            q = (r.get("query", "") or "").replace("\n", " ")
            if q:
                f.write(f"  query: {q[:200]}\n")
            if r.get("keywords_found") is not None:
                found = ", ".join(r.get("keywords_found", [])) if r.get("keywords_found") else "None"
                missing = ", ".join(r.get("keywords_missing", [])) if r.get("keywords_missing") else "None"
                f.write(f"  kw_found: {found}\n")
                f.write(f"  kw_missing: {missing}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TEST COMPLETED\n")
        f.write("="*80 + "\n")
    
    print(f"\nTest completed successfully!")
    print(f"Results written to: {OUTPUT_FILE}")

    summary_payload = {
        "total_requests": len(all_results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": (len(successful) / len(all_results)) if all_results else 0.0,
        "avg_latency": (sum(latencies) / len(latencies)) if successful else 0.0,
        "min_latency": min(latencies) if successful else 0.0,
        "max_latency": max(latencies) if successful else 0.0,
        "avg_tokens": (sum(tokens_list) / len(tokens_list)) if successful else 0.0,
        "total_tokens": sum(tokens_list) if successful else 0,
        "avg_throughput": (sum(throughputs) / len(throughputs)) if throughputs else 0.0,
        "avg_response_length": (sum(response_lengths) / len(response_lengths)) if successful else 0.0,
        "avg_ttft_ms": (sum(ttft_list) / len(ttft_list)) if ttft_list else 0.0,
        "avg_input_chars": (sum(input_chars_list) / len(input_chars_list)) if input_chars_list else 0.0,
        "total_ram_delta_mb": sum(ram_delta_list),
        "total_gpu_delta_mb": sum(gpu_delta_total_list),
    }
    screenshot_artifacts = save_metrics_screenshot(summary_payload, all_results, stability_results)
    if screenshot_artifacts.get("png_path"):
        print(f"Metrics screenshot saved: {screenshot_artifacts['png_path']}")
    else:
        print("Metrics screenshot could not be generated (JSON artifact still saved).")
    print(f"Metrics JSON saved: {screenshot_artifacts['json_path']}")

    print("="*80)
