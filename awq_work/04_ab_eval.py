"""A/B TTFT benchmark: measures warm-turn Time-To-First-Token against
the live vLLM endpoint (localhost:5000, model=mira-cbt).

Runs N_ROUNDS chat completions with a fixed system+user prompt, records
TTFT for each, and prints median / p95 / min / max.

Usage:
    python 04_ab_eval.py [--rounds 20] [--tag AWQ]
"""

import argparse
import statistics
import time
import json
import urllib.request

BASE_URL = "http://localhost:5000"
MODEL = "mira-cbt"

SYSTEM_PROMPT = (
    "You are Mira, a compassionate Thai-language CBT counsellor. "
    "Respond in Thai, keeping replies under 60 words."
)

USER_PROMPTS = [
    "สวัสดีครับ ผมรู้สึกเครียดมากเลย ช่วยได้ไหม",
    "ฉันไม่สามารถนอนหลับได้เลย มีวิธีแนะนำไหม",
    "ความวิตกกังวลทำให้ฉันทำงานไม่ได้ ต้องทำอย่างไร",
    "ฉันรู้สึกเหนื่อยและหมดแรง ช่วยฉันด้วย",
    "วันนี้ทุกอย่างดูแย่มาก อยากพูดคุย",
]


def chat_ttft(prompt: str, timeout: int = 60) -> float:
    """Send a streaming chat completion and return TTFT in ms."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
        "stream": True,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    first_token_ms = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                first_token_ms = (time.perf_counter() - t_start) * 1000
                break  # TTFT measured — drain rest silently

    if first_token_ms is None:
        raise RuntimeError("No token received in response")
    return first_token_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--tag", type=str, default="AWQ")
    args = parser.parse_args()

    print(f"[04_ab_eval] tag={args.tag}  rounds={args.rounds}  model={MODEL}")
    print(f"[04_ab_eval] endpoint: {BASE_URL}/v1/chat/completions")

    # Warm-up (1 request, not counted)
    print("[04_ab_eval] warm-up request...")
    try:
        wt = chat_ttft(USER_PROMPTS[0])
        print(f"  warm-up TTFT: {wt:.0f} ms  (discarded)")
    except Exception as e:
        print(f"  warm-up FAILED: {e}")
        raise

    ttfts = []
    for i in range(args.rounds):
        prompt = USER_PROMPTS[i % len(USER_PROMPTS)]
        try:
            ms = chat_ttft(prompt)
            ttfts.append(ms)
            print(f"  round {i+1:3d}: {ms:7.1f} ms   prompt={prompt[:30]!r}")
        except Exception as e:
            print(f"  round {i+1:3d}: ERROR — {e}")

    if not ttfts:
        print("[04_ab_eval] No successful rounds — aborting")
        return

    ttfts_sorted = sorted(ttfts)
    p50 = statistics.median(ttfts)
    p95 = ttfts_sorted[int(len(ttfts_sorted) * 0.95)]
    print()
    print(f"{'='*50}")
    print(f"[{args.tag}] TTFT results  (n={len(ttfts)})")
    print(f"  min    : {min(ttfts):.1f} ms")
    print(f"  median : {p50:.1f} ms")
    print(f"  p95    : {p95:.1f} ms")
    print(f"  max    : {max(ttfts):.1f} ms")
    print(f"{'='*50}")

    result = {
        "tag": args.tag,
        "n": len(ttfts),
        "min_ms": round(min(ttfts), 1),
        "median_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "max_ms": round(max(ttfts), 1),
    }
    out_path = f"/workspace/awq_work/ttft_{args.tag.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[04_ab_eval] results written to {out_path}")


if __name__ == "__main__":
    main()
