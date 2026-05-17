"""faster-whisper STT — runs locally in voice-agent process, same GPU, ~100ms.

Two public classes:
  - FastWhisperSTT: low-level singleton that owns the CT2 WhisperModel and
    exposes `transcribe(audio, sample_rate)`.
  - StreamingFastWhisperSTT (v5): LiveKit `STT` plugin advertising
    `streaming=True`. Uses an injected Silero VAD to segment user speech and
    emits interim transcripts every ~1s during long utterances, plus a final
    transcript on end_of_speech. Distil-Whisper has no native streaming, so
    interim transcripts are produced by re-transcribing the partial audio.
"""
from __future__ import annotations

import asyncio
import os
import time

import numpy as np

# CT2 model cached here (Modal volume or local dir)
# NOTE: resolved at call time (not module import) so IPC subprocesses
# that inherit a live WHISPER_CACHE_DIR env var always use the right path.
_DEFAULT_CACHE_DIR = "/cache/whisper-th-ct2"
_HF_MODEL_ID = "biodatlab/distill-whisper-th-large-v3"


def _get_cache_dir() -> str:
    return os.environ.get("WHISPER_CACHE_DIR", _DEFAULT_CACHE_DIR)


def _ensure_ct2_model(cache_dir: str) -> str:
    """Return path to CT2 model, converting from HF on first run."""
    import shutil
    import subprocess
    from pathlib import Path
    from huggingface_hub import snapshot_download

    ct2_path = Path(cache_dir)
    if (ct2_path / "model.bin").exists():
        print(f"[STT] CT2 model cached at {cache_dir}")
        _ensure_preprocessor_config(ct2_path)
        return cache_dir

    print(f"[STT] Downloading {_HF_MODEL_ID}...")
    hf_path = snapshot_download(repo_id=_HF_MODEL_ID)

    print(f"[STT] Converting to CTranslate2 → {cache_dir}")
    ct2_path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ct2-transformers-converter", "--model", hf_path,
         "--output_dir", cache_dir, "--quantization", "float16", "--force"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ct2-transformers-converter failed:\n{result.stderr}")

    # Copy feature extractor configs (ct2 converter skips them)
    for fname in ["preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                  "vocab.json", "merges.txt", "special_tokens_map.json", "normalizer.json"]:
        src = Path(hf_path) / fname
        if src.exists():
            shutil.copy2(src, ct2_path / fname)

    _ensure_preprocessor_config(ct2_path)
    print("[STT] Conversion done")
    return cache_dir


def _ensure_preprocessor_config(ct2_path) -> None:
    """Inject preprocessor_config.json with 128 mel bins if missing (large-v3 needs 128)."""
    import json
    from pathlib import Path
    cfg = Path(ct2_path) / "preprocessor_config.json"
    if cfg.exists():
        # Verify feature_size is 128
        try:
            data = json.loads(cfg.read_text())
            if data.get("feature_size") == 128:
                return
        except Exception:
            pass
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
    print("[STT] Patched preprocessor_config.json (feature_size=128)")


class FastWhisperSTT:
    MODEL_ID = _HF_MODEL_ID
    _instance: "FastWhisperSTT | None" = None

    def __init__(self, device: str = "cuda", compute_type: str = "float16"):
        from faster_whisper import WhisperModel
        model_path = _ensure_ct2_model(_get_cache_dir())
        print(f"[STT] Loading faster-whisper ({device}/{compute_type})...")
        t0 = time.time()
        self.device = device
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)
        print(f"[STT] Ready in {time.time()-t0:.1f}s")

    @classmethod
    def get(cls, device: str = "cuda", compute_type: str | None = None) -> "FastWhisperSTT":
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        if cls._instance is None or cls._instance.device != device:
            cls._instance = cls(device=device, compute_type=compute_type)
        return cls._instance

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        start = time.time()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        # Trim leading/trailing silence
        nonzero = np.where(np.abs(audio) > 0.01)[0]
        if len(nonzero) > 0:
            pad = int(sample_rate * 0.05)  # 50ms padding
            audio = audio[max(0, nonzero[0] - pad): nonzero[-1] + pad]
        if len(audio) < int(sample_rate * 0.1):  # < 100ms after trim → skip
            return ""

        segments, _ = await asyncio.to_thread(
            self.model.transcribe, audio,
            language="th", task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=1.8,
            log_prob_threshold=-0.5,
            repetition_penalty=1.2,
            word_timestamps=True,
        )

        segments_list = []
        for seg in segments:
            # Loosened from -0.7 to -1.05 — Thai with code-switching ("NSC", "ICT")
            # and proper nouns often scores -0.9..-1.05 even when intelligible.
            # Strict threshold caused Mira to miss "ไปก่อน" / goodbye intent.
            if seg.avg_logprob < -1.05:
                print(f"  [STT] skip low-conf segment: '{seg.text[:40]}' (logprob={seg.avg_logprob:.2f})")
                continue
            if seg.no_speech_prob > 0.5:
                print(f"  [STT] skip no-speech segment: '{seg.text[:40]}' (no_speech={seg.no_speech_prob:.2f})")
                continue
            segments_list.append(seg.text.strip())

        text = " ".join(segments_list).strip()
        elapsed_ms = (time.time() - start) * 1000

        text = self._clean_hallucination(text)

        if text:
            print(f"[STT] {elapsed_ms:.0f}ms → '{text[:80]}'")

        return text

    @staticmethod
    def _clean_hallucination(text: str) -> str:
        """Remove Whisper hallucination patterns from Thai STT output."""
        if not text:
            return ""

        import re

        # 1. Collapse repeated substrings: "ว่าว่าว่า" → "ว่า"
        text = re.sub(r'(.{2,30}?)\1{2,}', r'\1', text)

        # 2. Collapse repeated words with spaces: "อ่ะ อ่ะ อ่ะ" → "อ่ะ"
        words = text.split()
        if len(words) > 3:
            cleaned = [words[0]]
            repeat_count = 0
            for i in range(1, len(words)):
                if words[i] == words[i - 1]:
                    repeat_count += 1
                    if repeat_count >= 2:
                        continue  # skip 3rd+ consecutive repeat
                else:
                    repeat_count = 0
                cleaned.append(words[i])
            text = " ".join(cleaned)

        # 3. Detect gibberish — many short unique words from likely short audio
        words = text.split()
        if len(words) > 10:
            short_words = [w for w in words if len(w) <= 6]
            unique_ratio = len(set(short_words)) / max(len(short_words), 1)
            if unique_ratio > 0.85 and len(words) > 12:
                text = " ".join(words[:5])
                print(f"  [STT] cleaned gibberish (unique ratio {unique_ratio:.0%}, {len(words)} words)")

        # 4. Known Thai hallucination patterns — collapse to single instance
        _PATTERNS = [
            r'(ว่า){3,}',
            r'(อ่ะ\s*){3,}',
            r'(ครับ\s*){3,}',
            r'(ค่ะ\s*){3,}',
            r'(นะ\s*){3,}',
            r'(ๆ\s*){3,}',
            r'(ที่จะ){2,}',
            r'(ได้รับการ){2,}',
        ]
        for pattern in _PATTERNS:
            text = re.sub(pattern, lambda m: m.group(1), text)

        # 5. Final repetition ratio check → truncate
        words = text.split()
        if len(words) > 5:
            unique = set(words)
            ratio = 1 - (len(unique) / len(words))
            if ratio > 0.4:
                text = " ".join(words[:5])
                print(f"  [STT] cleaned high repetition ({ratio:.0%})")

        return text.strip()


# ─────────────────────────────────────────────────────────────────────────
# v5 Streaming STT plugin
# ─────────────────────────────────────────────────────────────────────────
#
# StreamingFastWhisperSTT advertises streaming=True / interim_results=True to
# LiveKit. It owns a Silero VAD stream and the FastWhisperSTT singleton, and:
#   - emits START_OF_SPEECH when VAD detects speech
#   - emits INTERIM_TRANSCRIPT every ~_INTERIM_INTERVAL_S seconds while the
#     user keeps talking (re-transcribing the accumulated frames)
#   - emits END_OF_SPEECH + FINAL_TRANSCRIPT when VAD says speech ended
#
# Interim transcripts are NOT used by the LLM today (the agent only acts on
# FINAL_TRANSCRIPT), but emitting them lets us log a true "stt_first_chunk"
# latency event before silence detection completes. The final-transcript
# path remains the source of truth for response generation.

from livekit import rtc as _lk_rtc
from livekit.agents import stt as _lk_stt
from livekit.agents import vad as _lk_vad
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS as _DEFAULT_CONN,
    NOT_GIVEN as _NOT_GIVEN,
    APIConnectOptions as _APIConnectOptions,
    NotGivenOr as _NotGivenOr,
)
from livekit.agents import utils as _lk_utils

_INTERIM_INTERVAL_S = 1.0   # min seconds between interim transcribe runs
_INTERIM_OVERLAP_S = 0.3    # head-overlap kept between windows (informational)


class StreamingFastWhisperSTT(_lk_stt.STT):
    """LiveKit STT plugin with streaming + interim transcripts.

    Wraps the singleton FastWhisperSTT and an injected Silero VAD. Falls back
    to non-streaming `_recognize_impl` (delegated to FastWhisperSTT) when
    callers don't use the streaming path.
    """

    def __init__(self, vad: _lk_vad.VAD, *, device: str = "cuda", compute_type: str | None = None):
        super().__init__(
            capabilities=_lk_stt.STTCapabilities(streaming=True, interim_results=True)
        )
        self._vad = vad
        self._fast = FastWhisperSTT.get(device=device, compute_type=compute_type)
        self.last_audio_b64: str | None = None
        # Latest final transcript timing — read by the agent for telemetry.
        self.last_speech_end_t: float | None = None
        # Timestamp of last TTS output completion — used to suppress STT echo.
        self._tts_done_at: float = 0.0

    def notify_tts_done(self) -> None:
        """Call after each TTS utterance is queued to start the echo-mute window."""
        self._tts_done_at = time.perf_counter()

    @property
    def model(self) -> str:
        return FastWhisperSTT.MODEL_ID

    @property
    def provider(self) -> str:
        return "faster-whisper-ct2"

    async def _recognize_impl(
        self,
        buffer: _lk_utils.AudioBuffer,
        *,
        language: _NotGivenOr[str] = _NOT_GIVEN,
        conn_options: _APIConnectOptions,
    ) -> _lk_stt.SpeechEvent:
        # Non-streaming fallback — same behaviour as voice/mira_agent_v4.py:ThaiSTT
        frames: list[_lk_rtc.AudioFrame] = buffer if isinstance(buffer, list) else [buffer]
        text = await _transcribe_frames(self._fast, frames)
        return _lk_stt.SpeechEvent(
            type=_lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[_lk_stt.SpeechData(language="th", text=text, confidence=1.0)],
        )

    def stream(
        self,
        *,
        language: _NotGivenOr[str] = _NOT_GIVEN,
        conn_options: _APIConnectOptions = _DEFAULT_CONN,
    ) -> _lk_stt.RecognizeStream:
        return _StreamingRecognizeStream(
            stt=self,
            vad=self._vad,
            fast=self._fast,
            language=language,
            conn_options=conn_options,
        )


async def _transcribe_frames(fast: FastWhisperSTT, frames: list[_lk_rtc.AudioFrame]) -> str:
    """Decode AudioFrame list → 16kHz float32 mono → Whisper text."""
    if not frames:
        return ""
    sr = frames[0].sample_rate or 16000
    parts = [
        np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
        for f in frames
    ]
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if sr != 16000:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 16000, sr).astype(np.float32)

    # Reject silent/echo audio
    duration_s = len(audio) / 16000
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
    if duration_s < 0.4 or rms < 0.002:
        return ""

    return await fast.transcribe(audio, 16000)



def _frames_to_wav_b64(frames: list) -> str:
    """Encode AudioFrame list → 16kHz mono WAV → base64 string for Ultravox audio tower."""
    import base64, io, wave
    import numpy as np
    if not frames:
        return ""
    sr = frames[0].sample_rate or 16000
    parts = [
        np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
        for f in frames
    ]
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if sr != 16000:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    pcm16 = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(16000)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


class _StreamingRecognizeStream(_lk_stt.RecognizeStream):
    """RecognizeStream that runs Silero VAD locally and re-transcribes the
    accumulated frames every _INTERIM_INTERVAL_S seconds.
    """

    def __init__(
        self,
        *,
        stt: StreamingFastWhisperSTT,
        vad: _lk_vad.VAD,
        fast: FastWhisperSTT,
        language: _NotGivenOr[str],
        conn_options: _APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._streaming_stt = stt
        self._vad = vad
        self._fast = fast
        self._language = language

    async def _metrics_monitor_task(self, event_aiter):  # noqa: ANN001 — base class signature
        async for _ in event_aiter:
            pass

    async def _run(self) -> None:
        vad_stream = self._vad.stream()
        SE = _lk_stt.SpeechEventType
        VE = _lk_vad.VADEventType

        async def _forward_input() -> None:
            async for ev in self._input_ch:
                if isinstance(ev, self._FlushSentinel):
                    vad_stream.flush()
                    continue
                vad_stream.push_frame(ev)
            vad_stream.end_input()

        async def _recognize() -> None:
            collected: list[_lk_rtc.AudioFrame] = []
            last_interim = 0.0
            interim_in_flight: asyncio.Task | None = None

            async def _maybe_emit_interim() -> None:
                nonlocal interim_in_flight
                if interim_in_flight and not interim_in_flight.done():
                    return  # one at a time — drop overlapping interim runs
                snapshot = list(collected)
                interim_in_flight = asyncio.create_task(_run_interim(snapshot))

            async def _run_interim(frames: list[_lk_rtc.AudioFrame]) -> None:
                try:
                    text = await _transcribe_frames(self._fast, frames)
                    if not text:
                        return
                    self._event_ch.send_nowait(
                        _lk_stt.SpeechEvent(
                            type=SE.INTERIM_TRANSCRIPT,
                            alternatives=[
                                _lk_stt.SpeechData(language="th", text=text, confidence=0.5)
                            ],
                        )
                    )
                except Exception:  # noqa: BLE001 — interim is best-effort
                    pass

            async for ev in vad_stream:
                if ev.type == VE.START_OF_SPEECH:
                    collected = list(ev.frames or [])
                    last_interim = time.perf_counter()
                    self._event_ch.send_nowait(_lk_stt.SpeechEvent(SE.START_OF_SPEECH))

                elif ev.type == VE.INFERENCE_DONE:
                    if ev.frames:
                        collected.extend(ev.frames)
                    if ev.speaking and (time.perf_counter() - last_interim) >= _INTERIM_INTERVAL_S:
                        last_interim = time.perf_counter()
                        await _maybe_emit_interim()

                elif ev.type == VE.END_OF_SPEECH:
                    self._streaming_stt.last_speech_end_t = time.perf_counter()

                    # Echo-mute: drop STT that fires within 1.2 s of Mira's last TTS.
                    _since_tts = time.perf_counter() - self._streaming_stt._tts_done_at
                    if _since_tts < 5.0:
                        print(f"  [STT] muted (echo window {_since_tts:.2f}s after TTS)")
                        collected = []
                        if interim_in_flight and not interim_in_flight.done():
                            interim_in_flight.cancel()
                        continue

                    self._event_ch.send_nowait(_lk_stt.SpeechEvent(SE.END_OF_SPEECH))

                    # Final transcript on the merged buffer.
                    final_frames = list(ev.frames) if ev.frames else collected
                    # Store raw audio for Ultravox audio tower (MIRA_USE_AUDIO_TOWER).
                    try:
                        self._streaming_stt.last_audio_b64 = _frames_to_wav_b64(final_frames)
                    except Exception:
                        self._streaming_stt.last_audio_b64 = None
                    text = await _transcribe_frames(self._fast, final_frames)
                    if text:
                        self._event_ch.send_nowait(
                            _lk_stt.SpeechEvent(
                                type=SE.FINAL_TRANSCRIPT,
                                alternatives=[
                                    _lk_stt.SpeechData(language="th", text=text, confidence=1.0)
                                ],
                            )
                        )
                    collected = []
                    if interim_in_flight and not interim_in_flight.done():
                        interim_in_flight.cancel()

        tasks = [
            asyncio.create_task(_forward_input(), name="streaming_stt.forward_input"),
            asyncio.create_task(_recognize(), name="streaming_stt.recognize"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await _lk_utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()
