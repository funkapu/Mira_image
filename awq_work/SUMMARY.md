# AWQ INT4 Quantization — Summary

## Objective
Quantize Mira's Ultravox `merged-full-v2` (Qwen3-32B + Thai CBT LoRA + Whisper-turbo) to
AWQ INT4 symmetric to reduce warm-turn TTFT below the ~1045 ms FP16 floor.

## Checkpoint
| Path | Size | Shards | Format |
|------|------|--------|--------|
| `merged-full-v2` (FP16 rollback) | 66.9 GB | 15 | bfloat16 |
| `merged-full-v2-awq` (ASYM, broken) | 20.6 GB | 5 | compressed-tensors / ASYM — no zero-points, do not serve |
| `merged-full-v2-awq-sym` (production) | 20.6 GB | 5 | compressed-tensors / W4A16 symmetric, pack-quantized |

## Quantization Config
- **Scheme**: `W4A16` (symmetric INT4 weight, FP16 activation)
- **Kernel**: MarlinLinearKernel (vLLM `CompressedTensorsWNA16`)
- **Scope**: `language_model` Linear layers only
- **Skipped**: `audio_tower`, `multi_modal_projector`, `lm_head`
- **Calibration**: 256 rows (`calib_data.jsonl`), 128 EN + 128 TH
- **Tool**: llmcompressor + compressed-tensors (vLLM-official multimodal path)

## Library Patches Applied
Two bugs in `compressed_tensors` required patching before save completed:

1. `model_compressor.py:205–207` — `CompressionFormat` enum not wrapped in list →
   added `list` to `isinstance` check and wrapped bare enum in `[...]`.
2. `model_compressor.py:320` / `registry.py:54` — `CompressionFormat` enum passed
   directly to `standardize_lookup_name` which called `.replace()` on it →
   patched `from_pretrained_model` to call `.value` on enum before passing to registry.

File: `/usr/local/lib/python3.11/dist-packages/compressed_tensors/compressors/model_compressors/model_compressor.py`

## TTFT Results (n=20 warm rounds, text-only, Thai prompts)
| Metric | AWQ-sym | FP16 baseline |
|--------|---------|---------------|
| min | 36.0 ms | — |
| median | 48.3 ms | — |
| p95 | **62.7 ms** | ~1045 ms |
| max | 62.7 ms | — |

**Improvement: ~16.7× reduction in p95 TTFT.**

Raw results: `/workspace/awq_work/ttft_awq.json`

## Files Changed
- `/app/start_agent.sh` line 18: `merged-full-v2-awq` → `merged-full-v2-awq-sym`
- `/workspace/awq_work/03_quantize.py`: `W4A16_ASYM` → `W4A16`, output dir updated
- `compressed_tensors` library: two enum-as-string bugs patched (in-place)

## Rollback
```bash
# Revert to FP16 — edit start_agent.sh line 18:
ULTRAVOX_DIR="${MODELS_DIR}/ultravox/merged-full-v2"
# Remove --quantization compressed-tensors from the vllm serve call (line ~118)
```
