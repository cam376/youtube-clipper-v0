"""
Audio extraction (FFmpeg) and transcription (faster-whisper).

A transcript is a list of words with timestamps:
    [{"start": 1.23, "end": 1.61, "word": "Hello"}, ...]
Whisper's own segments are also returned, as they give natural sentence breaks.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

from faster_whisper import WhisperModel

# "small" is a good speed/accuracy trade-off on CPU. Set WHISPER_MODEL to
# "base" for speed or "medium" for accuracy (noticeably better in French,
# roughly 2-3x slower on CPU). Read when a transcription starts, so the
# variable only has to be set before launching the server.
DEFAULT_WHISPER_MODEL = "small"

log = logging.getLogger("clipper.transcription")

_model = None
_model_key = None


def whisper_settings() -> tuple[str, str, str]:
    return (
        os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL).strip() or DEFAULT_WHISPER_MODEL,
        os.environ.get("WHISPER_DEVICE", "cpu"),
        os.environ.get("WHISPER_COMPUTE", "int8"),
    )


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract mono 16 kHz WAV (what Whisper wants) from the video."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return audio_path


def _get_model() -> WhisperModel:
    global _model, _model_key
    key = whisper_settings()
    if _model is None or _model_key != key:
        name, device, compute = key
        log.info("loading faster-whisper model %s (device=%s, compute=%s)", name, device, compute)
        _model = WhisperModel(name, device=device, compute_type=compute)
        _model_key = key
    return _model


def transcribe(audio_path: Path) -> dict:
    """
    Return {"language", "language_probability", "model", "duration_seconds",
            "audio_seconds", "segments": [...], "words": [...]}.

    segments: [{"start", "end", "text"}]
    words:    [{"start", "end", "word"}]
    """
    t0 = time.monotonic()
    model = _get_model()
    model_name = whisper_settings()[0]
    seg_iter, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )

    segments, words = [], []
    for seg in seg_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        for w in seg.words or []:
            token = w.word.strip()
            if token:
                words.append({"start": float(w.start), "end": float(w.end), "word": token})

    elapsed = time.monotonic() - t0
    log.info("transcribed %.0f s of audio in %.0f s with %s (language=%s, p=%.2f)",
             info.duration, elapsed, model_name, info.language, info.language_probability)
    return {
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "model": model_name,
        "duration_seconds": round(elapsed, 1),
        "audio_seconds": round(float(info.duration), 1),
        "segments": segments,
        "words": words,
    }
