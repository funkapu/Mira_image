"""Whisper Turbo Thai STT — biodatlab/distill-whisper-th-large-v3

Two modes:
  - Remote (default): calls Modal STT service via WHISPER_STT_URL env var
  - Local fallback: loads model in-process (slow on CPU, fast on GPU)
"""
import asyncio
import os
import time
import numpy as np


class WhisperTurboThaiSTT:
    """Thai STT — remote Modal endpoint or local HF pipeline."""

    MODEL_ID = "biodatlab/distill-whisper-th-large-v3"

    def __init__(self, device: str | None = None):
        self._remote_url = os.environ.get("WHISPER_STT_URL", "").rstrip("/")
        self.pipe = None

        if self._remote_url:
            print(f"[STT] Remote mode -> {self._remote_url}")
        else:
            import torch
            from transformers import pipeline as hf_pipeline

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            print(f"[STT] Local mode — loading {self.MODEL_ID} on {device}...")
            t0 = time.time()
            self.pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=self.MODEL_ID,
                device=device,
                dtype=torch.float16,
                generate_kwargs={
                    "language": "th",
                    "task": "transcribe",
                    "max_new_tokens": 256,
                },
            )
            print(f"[STT] Loaded in {time.time() - t0:.1f}s")

    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        t0 = time.time()

        if self._remote_url:
            text = await self._transcribe_remote(audio_data, sample_rate)
        else:
            text = await self._transcribe_local(audio_data, sample_rate)

        print(f"[STT] {(time.time() - t0) * 1000:.0f}ms -> '{text[:80]}'")
        return text

    async def _transcribe_remote(self, audio_data: np.ndarray, sample_rate: int) -> str:
        import httpx

        pcm_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{self._remote_url}/transcribe",
                content=pcm_bytes,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Sample-Rate": str(sample_rate),
                },
            )
            r.raise_for_status()
            return (r.json().get("text") or "").strip()

    async def _transcribe_local(self, audio_data: np.ndarray, sample_rate: int) -> str:
        result = await asyncio.to_thread(
            self.pipe,
            {"array": audio_data, "sampling_rate": sample_rate},
        )
        return (result.get("text") or "").strip()
