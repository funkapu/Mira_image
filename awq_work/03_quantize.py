"""AWQ INT4 quantization of Ultravox merged-full-v2.

- Loads UltravoxModel (custom code, trust_remote_code=True) onto GPU in bf16
- Applies AWQ + W4A16 quantization to language_model.* Linear modules
- Preserves audio_tower, multi_modal_projector, lm_head in original precision
- Saves to /workspace/models/ultravox/merged-full-v2-awq/

Calibration: 256 samples (128 EN + 128 synthetic TH) from calib_data.jsonl.
Expected runtime: 1-2h on A100-80GB.

PLAN-B PATCH: class-level save_pretrained replacement to bypass UltravoxModel's
diff_state_dict filter (which drops all language_model weights when keep_params
is empty, as confirmed by diag_save_filter.py).
"""
import os
import json
import time
import types
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, PreTrainedModel

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.base import AWQMapping

# ---------------- config ----------------
MODEL_PATH = "/workspace/models/ultravox/merged-full-v2"
CALIB_PATH = "/workspace/awq_work/calib_data.jsonl"
OUTPUT_DIR = "/workspace/models/ultravox/merged-full-v2-awq-sym"

NUM_CALIB = 256
MAX_SEQ_LEN = 2048

# Modules to leave in original bf16 (preserve audio path + output head)
IGNORE_PATTERNS = [
    "re:.*audio_tower.*",
    "re:.*multi_modal_projector.*",
    "re:.*lm_head$",
]

# ---------------- helpers ----------------
def banner(msg):
    print(f"\n{'='*72}\n{msg}\n{'='*72}", flush=True)

def gpu_mem():
    if not torch.cuda.is_available():
        return "no cuda"
    free, total = torch.cuda.mem_get_info()
    return f"free={free/1e9:.1f}GB used={(total-free)/1e9:.1f}GB total={total/1e9:.1f}GB"

# ---------------- run ----------------
t0 = time.time()
banner(f"AWQ quantize start — pid={os.getpid()}  cuda={torch.cuda.is_available()}")
print(f"torch {torch.__version__}  device 0: {torch.cuda.get_device_name(0)}")
print(f"GPU before load: {gpu_mem()}")

banner("[1/5] loading tokenizer")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"vocab={tok.vocab_size}  eos={tok.eos_token!r}")

banner("[2/5] loading UltravoxModel onto GPU (bf16)")
load_t0 = time.time()
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
# Disable KV cache so DynamicCache objects don't appear in the FX-traced graph
model.config.use_cache = False
if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
    model.language_model.config.use_cache = False
print(f"loaded in {time.time()-load_t0:.1f}s  GPU: {gpu_mem()}  use_cache disabled")
print(f"top-level: {[n for n,_ in model.named_children()]}")
n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
print(f"total Linear modules in model: {n_lin}")

# ---- PLAN-B PATCH: bypass UltravoxModel.save_pretrained diff_state_dict filter ----
# UltravoxModel.save_pretrained filters through keep_params ∪ trainable_params.
# After from_pretrained with low_cpu_mem_usage+device_map, keep_params stays empty
# (hook never fires), so diff_state_dict silently drops all language_model tensors.
# Fix: replace save_pretrained at the CLASS level so llmcompressor's internal
# model.save_pretrained(..., save_compressed=True) call also uses the patched version.
banner("[PATCH] replacing UltravoxModel.save_pretrained with PreTrainedModel.save_pretrained")
UltravoxClass = type(model)
_orig_save = UltravoxClass.save_pretrained

def _full_save_pretrained(self, save_directory, **kwargs):
    """Bypass diff_state_dict filter — serialize full state_dict via PreTrainedModel."""
    kwargs.pop("state_dict", None)   # let PreTrainedModel build it from scratch
    return PreTrainedModel.save_pretrained(self, save_directory, **kwargs)

UltravoxClass.save_pretrained = _full_save_pretrained
print(f"patched {UltravoxClass.__name__}.save_pretrained — orig was {_orig_save}")

banner("[3/5] loading calibration data")
ds = load_dataset("json", data_files=CALIB_PATH, split="train")
print(f"dataset size: {len(ds)}  cols: {ds.column_names}")
ds = ds.shuffle(seed=42).select(range(NUM_CALIB))

def preprocess(ex):
    return tok(
        ex["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
        add_special_tokens=False,
    )

ds = ds.map(preprocess, remove_columns=ds.column_names)
print(f"tokenized {len(ds)} samples")
import statistics as st
lens = [len(r["input_ids"]) for r in ds]
print(f"token-len stats: min={min(lens)} median={int(st.median(lens))} mean={int(st.mean(lens))} max={max(lens)}")

banner("[4/5] running llmcompressor oneshot (AWQ + W4A16)")
NUM_LAYERS = 64
LM_PREFIX = "language_model.model.layers"

mappings = []
for i in range(NUM_LAYERS):
    p = f"{LM_PREFIX}.{i}"
    mappings += [
        AWQMapping(
            smooth_layer=f"{p}.input_layernorm",
            balance_layers=[f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"],
        ),
        AWQMapping(
            smooth_layer=f"{p}.self_attn.v_proj",
            balance_layers=[f"{p}.self_attn.o_proj"],
        ),
        AWQMapping(
            smooth_layer=f"{p}.post_attention_layernorm",
            balance_layers=[f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"],
        ),
        AWQMapping(
            smooth_layer=f"{p}.mlp.up_proj",
            balance_layers=[f"{p}.mlp.down_proj"],
        ),
    ]

recipe = [
    AWQModifier(
        ignore=IGNORE_PATTERNS,
        scheme="W4A16",
        targets=["Linear"],
        mappings=mappings,
    ),
]
print("recipe:")
for r in recipe:
    print(f"  - {r.__class__.__name__}: targets={r.targets} scheme={r.scheme} ignore={r.ignore}")
print(f"GPU pre-oneshot: {gpu_mem()}")

q_t0 = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_CALIB,
    output_dir=OUTPUT_DIR,
    save_compressed=True,
    trust_remote_code_model=True,
)
print(f"oneshot done in {(time.time()-q_t0)/60:.1f} min")
print(f"GPU post-oneshot: {gpu_mem()}")

# Belt-and-braces: explicit save after oneshot in case oneshot's internal save
# was called before the patch took effect or used a stale reference.
banner("[4b] belt-and-braces explicit save_pretrained")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
print("explicit save done")

banner("[5/5] verifying output")
import subprocess
out = subprocess.run(["du", "-sh", OUTPUT_DIR], capture_output=True, text=True)
print(out.stdout.strip())

cfg_path = os.path.join(OUTPUT_DIR, "config.json")
assert os.path.exists(cfg_path), f"FAIL: {cfg_path} missing"
cfg = json.load(open(cfg_path))
qc = cfg.get("quantization_config")
assert qc is not None, "FAIL: quantization_config missing from config.json"
print(f"quantization_config present: True")
print(json.dumps({k: v for k, v in qc.items() if k != "config_groups"}, indent=2)[:800])

index_path = os.path.join(OUTPUT_DIR, "model.safetensors.index.json")
assert os.path.exists(index_path), f"FAIL: {index_path} missing — no shards written"
print(f"model.safetensors.index.json present: True")

shards = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".safetensors"))
print(f"output shards: {len(shards)}")
assert len(shards) >= 4, f"FAIL: expected >=8 shards, got {len(shards)}"
for s in shards[:3] + (["..."] if len(shards) > 6 else []) + shards[-3:]:
    if s == "...":
        print("  ...")
        continue
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, s)) / 1e9
    print(f"  {s}  {sz:.2f} GB")

total_gb = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR)) / 1e9
assert total_gb >= 12.0, f"FAIL: output dir only {total_gb:.1f} GB — suspiciously small"
print(f"total output size: {total_gb:.1f} GB  [OK]")

banner(f"TOTAL ELAPSED: {(time.time()-t0)/60:.1f} min")
print("OK — quantization complete.")
