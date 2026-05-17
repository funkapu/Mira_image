"""Whisper Thai STT — Modal A10G, faster-whisper (CTranslate2)

First cold start: downloads biodatlab/distill-whisper-th-large-v3,
converts to CT2 int8 via ct2-transformers-converter CLI, saves to volume.
Subsequent starts: loads from volume in ~2s.

Deploy:
    modal deploy voice/modal_stt.py

Set in .env:
    WHISPER_STT_URL=https://funkfuze--mira-whisper-stt-serve.modal.run
"""
import modal

APP_NAME = "mira-whisper-stt"
HF_MODEL_ID = "biodatlab/distill-whisper-th-large-v3"

whisper_volume = modal.Volume.from_name("mira-whisper-cache", create_if_missing=True)
CACHE_PATH = "/cache"
CT2_MODEL_DIR = f"{CACHE_PATH}/whisper-th-ct2"

stt_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper>=1.0.3",
        "ctranslate2>=4.0.0",
        "transformers>=4.46.0",
        "torch>=2.0.0",
        "huggingface_hub>=0.20.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "fastapi",
    )
    .env({"HF_HOME": f"{CACHE_PATH}/hf"})
)

app = modal.App(APP_NAME, image=stt_image)

_model = None


def _ensure_ct2_model():
    """Convert HF Whisper model → CTranslate2 int8 on first run, cache in volume."""
    import shutil
    import subprocess
    from pathlib import Path
    from huggingface_hub import snapshot_download

    ct2_path = Path(CT2_MODEL_DIR)
    if (ct2_path / "model.bin").exists():
        print("[STT] CT2 model already cached")
        return

    print(f"[STT] Downloading {HF_MODEL_ID}...")
    hf_path = snapshot_download(repo_id=HF_MODEL_ID)
    print(f"[STT] Downloaded to {hf_path}")

    print(f"[STT] Converting to CTranslate2 int8 → {CT2_MODEL_DIR}")
    ct2_path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ct2-transformers-converter",
            "--model", hf_path,
            "--output_dir", CT2_MODEL_DIR,
            "--quantization", "int8",
            "--force",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ct2-transformers-converter failed:\n{result.stderr}")

    # Copy feature extractor config — ct2 converter skips these but
    # faster-whisper needs preprocessor_config.json to know mel bin count (128 for large-v3).
    # Without it, faster-whisper defaults to 80 bins → shape mismatch crash.
    for fname in [
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "normalizer.json",
    ]:
        src = Path(hf_path) / fname
        if src.exists():
            shutil.copy2(src, ct2_path / fname)
            print(f"[STT] Copied {fname}")

    whisper_volume.commit()
    print("[STT] Conversion done, volume committed")


def _patch_mel_config(ct2_dir: str) -> None:
    """Inject preprocessor_config.json with 128 mel bins if missing.
    large-v3 uses 128 bins; without this file faster-whisper defaults to 80 → crash.
    """
    import json
    from pathlib import Path
    cfg = Path(ct2_dir) / "preprocessor_config.json"
    if cfg.exists():
        return
    cfg.write_text(json.dumps({
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 128,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": False,
        "sampling_rate": 16000,
    }, indent=2))
    print("[STT] Injected preprocessor_config.json (feature_size=128)")


def _load_model():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel
    _ensure_ct2_model()
    _patch_mel_config(CT2_MODEL_DIR)
    print("[STT] Loading faster-whisper...")
    _model = WhisperModel(CT2_MODEL_DIR, device="cuda", compute_type="int8")
    print("[STT] Ready")
    return _model


@app.function(
    gpu="A10G",
    volumes={CACHE_PATH: whisper_volume},
    timeout=600,
    scaledown_window=300,
    min_containers=0,
    max_containers=3,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def serve():
    import asyncio
    import numpy as np
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    _load_model()

    web_app = FastAPI(title="Mira Whisper STT")

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": HF_MODEL_ID, "backend": "faster-whisper"}

    @web_app.post("/transcribe")
    async def transcribe(request: Request):
        raw = await request.body()
        sample_rate = int(request.headers.get("x-sample-rate", "16000"))

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if sample_rate != 16000:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, 16000, sample_rate).astype(np.float32)

        model = _load_model()
        segments, _ = await asyncio.to_thread(
            model.transcribe,
            audio,
            language="th",
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            beam_size=1,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return JSONResponse({"text": text})

    return web_app
