"""Phase-C diagnostic: empirically confirm whether diff_state_dict drops
quantized tensor names (the hypothesized cause of save failure).

Does NOT run oneshot. Loads UltravoxModel exactly as 03_quantize.py does,
then inspects keep_params, trainable_params, and the filter behavior on
representative quantized tensor name patterns.

Cheap memory-wise: never materializes full state_dict tensors; works
only on key sets.
"""
import os, sys, time, json
import torch
from collections import Counter
from transformers import AutoModel

MODEL_PATH = "/workspace/models/ultravox/merged-full-v2"

def banner(m): print(f"\n{'='*72}\n{m}\n{'='*72}", flush=True)

banner(f"DIAG pid={os.getpid()}  cuda={torch.cuda.is_available()}")
t0 = time.time()

banner("[1/4] load UltravoxModel (same args as 03_quantize.py)")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"loaded in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------
banner("[2/4] inspect keep_params")
kp = getattr(model, "keep_params", None)
print(f"hasattr keep_params: {kp is not None}")
print(f"type(keep_params):   {type(kp).__name__}")
print(f"len(keep_params):    {len(kp) if kp is not None else 'N/A'}")

if kp:
    samples = list(kp)[:10]
    print("first 10 keep_params keys:")
    for k in samples:
        print(f"  {k}")

    # prefix breakdown
    def prefix(k):
        for p in ("language_model.", "audio_tower.", "multi_modal_projector.", "lm_head"):
            if k.startswith(p): return p.rstrip(".")
        return "OTHER"
    cnt = Counter(prefix(k) for k in kp)
    print(f"keep_params prefix breakdown: {dict(cnt)}")
else:
    print("keep_params is empty or missing — THEORY-CONFIRMING signal #1")

# -----------------------------------------------------------------
banner("[3/4] inspect trainable_params (requires_grad)")
trainable = [n for n,p in model.named_parameters() if p.requires_grad]
print(f"len(trainable_params): {len(trainable)}")
if trainable:
    print("first 10:")
    for n in trainable[:10]:
        print(f"  {n}")
else:
    print("0 trainable params (expected in eval/inference) — THEORY-CONFIRMING signal #2")

# -----------------------------------------------------------------
banner("[4/4] mock filter test: would a quantized tensor name survive diff_state_dict?")

# Pick a real Linear name and simulate llmcompressor's quantized output names.
# llmcompressor W4A16_ASYM produces these renamed children on each quantized Linear:
#   <name>.weight_packed        (int4 packed)
#   <name>.weight_scale         (per-group scale)
#   <name>.weight_zero_point    (per-group zp)
# The original .weight key is REMOVED.
real_linear_name = "language_model.model.layers.0.self_attn.q_proj"
candidate_quant_keys = [
    f"{real_linear_name}.weight_packed",
    f"{real_linear_name}.weight_scale",
    f"{real_linear_name}.weight_zero_point",
    f"{real_linear_name}.weight",          # baseline: original key
]

# Reproduce diff_state_dict logic without materializing tensors
trainable_set = {k.replace("_fsdp_wrapped_module.", "") for k in trainable}
keep_set = set(kp) if kp else set()

print(f"\nFilter check: would key survive `k in keep_params or k in trainable_params`?")
for k in candidate_quant_keys:
    in_keep = k in keep_set
    in_train = k in trainable_set
    survives = in_keep or in_train
    print(f"  {'KEEP' if survives else 'DROP'}  {k}    (keep={in_keep}, train={in_train})")

# Also check: is the original .weight in keep_params?
print(f"\nIs '{real_linear_name}.weight' in keep_params? {real_linear_name + '.weight' in keep_set}")

# Sanity: count how many language_model.*.weight keys are in keep_params today
lm_weight_keys = [k for k in keep_set if k.startswith("language_model.") and k.endswith(".weight")]
print(f"language_model.*.weight in keep_params: {len(lm_weight_keys)}")
audio_weight_keys = [k for k in keep_set if k.startswith("audio_tower.") and k.endswith(".weight")]
print(f"audio_tower.*.weight     in keep_params: {len(audio_weight_keys)}")
proj_weight_keys = [k for k in keep_set if k.startswith("multi_modal_projector.")]
print(f"multi_modal_projector.*  in keep_params: {len(proj_weight_keys)}")

banner(f"DIAG DONE in {(time.time()-t0)/60:.1f} min")
print("VERDICT:")
if len(lm_weight_keys) == 0 and len(audio_weight_keys) == 0:
    print("  THEORY CONFIRMED — keep_params has zero language_model/audio_tower weights.")
    print("  diff_state_dict will drop ALL quantized language_model tensors.")
    print("  Proceed to plan B: monkey-patch + belt-and-braces save.")
elif len(lm_weight_keys) > 0:
    print("  THEORY PARTIALLY WRONG — keep_params DOES contain language_model weights.")
    print("  Likely root cause: llmcompressor renames .weight -> .weight_packed,")
    print("  and the new name is not in keep_params. Same fix applies but reasoning differs.")
else:
    print("  UNEXPECTED state — investigate manually.")
