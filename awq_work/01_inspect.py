"""Inspect Ultravox model structure and check prerequisites."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_PATH = "/workspace/models/ultravox/merged-full-v2"
OUTPUT_PATH = "/workspace/models/ultravox/merged-full-v2-awq"

# 1. Verify model dir exists
assert Path(MODEL_PATH).is_dir(), f"Model dir missing: {MODEL_PATH}"
print(f"[OK] Model dir found: {MODEL_PATH}")

# 2. Check disk space (need ~30GB for AWQ output)
stat = shutil.disk_usage("/workspace")
free_gb = stat.free / (1024**3)
print(f"[i] /workspace free: {free_gb:.1f} GB (need >= 30 GB)")
assert free_gb >= 30, f"Insufficient disk: {free_gb:.1f}GB < 30GB"

# 3. Check CUDA
import torch
assert torch.cuda.is_available(), "CUDA required"
print(f"[OK] CUDA: {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB total")

# 4. Inspect Ultravox model structure (without loading full weights)
from transformers import AutoConfig, AutoModel
print("[i] Loading model config...")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"    Architecture: {config.architectures}")
print(f"    Model type: {config.model_type}")

# Load model via from_config (avoids meta device anti-pattern)
print("[i] Loading model structure via from_config...")
model = AutoModel.from_config(config, trust_remote_code=True)

# 5. Identify modules to skip from AWQ
modules_to_skip = []
language_model_prefix = None
for name, _ in model.named_modules():
    lname = name.lower()
    if any(k in lname for k in ["audio_tower", "whisper", "multi_modal_projector", "audio_projector"]):
        modules_to_skip.append(name)
    if "language_model" in lname and language_model_prefix is None:
        language_model_prefix = name.split(".")[0]

# Always skip lm_head
if "lm_head" not in modules_to_skip:
    modules_to_skip.append("lm_head")

print(f"\n[OK] Modules to SKIP from quantization:")
for m in modules_to_skip[:20]:
    print(f"    - {m}")
if len(modules_to_skip) > 20:
    print(f"    ... and {len(modules_to_skip) - 20} more")
print(f"[i] Total skip count: {len(modules_to_skip)}")
print(f"[i] Language model prefix: {language_model_prefix}")

# 6. Check vLLM status (we'll need to stop it)
try:
    result = subprocess.run(
        ["pgrep", "-f", "vllm"],
        capture_output=True, text=True, timeout=5
    )
    vllm_pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
    if vllm_pids:
        print(f"\n[!] vLLM running (PID: {vllm_pids}) -- must stop in Phase 3")
    else:
        print(f"\n[i] vLLM not running -- proceed")
except Exception as e:
    print(f"[?] Could not check vLLM status: {e}")

# 7. Save inspection result
output = {
    "model_path": MODEL_PATH,
    "output_path": OUTPUT_PATH,
    "architectures": config.architectures,
    "model_type": config.model_type,
    "modules_to_skip": modules_to_skip,
    "language_model_prefix": language_model_prefix,
    "free_disk_gb": free_gb,
}
Path("/workspace/awq_work").mkdir(parents=True, exist_ok=True)
with open("/workspace/awq_work/inspection.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\n[OK] Inspection saved to /workspace/awq_work/inspection.json")
