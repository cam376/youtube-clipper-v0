"""
Audio extraction (FFmpeg) and transcription (faster-whisper).

A transcript is a list of words with timestamps:
    [{"start": 1.23, "end": 1.61, "word": "Hello"}, ...]
Whisper's own segments are also returned, as they give natural sentence breaks.
"""

import os
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

# "small" is a good speed/accuracy trade-off on CPU. Set WHISPER_MODEL to
# "base" for speed or "medium" for accuracy.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")

_model = None


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
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    return _model


def transcribe(audio_path: Path) -> dict:
    """
    Return {"language": str, "segments": [...], "words": [...]}.

    segments: [{"start", "end", "text"}]
    words:    [{"start", "end", "word"}]
    """
    model = _get_model()
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

    return {"language": info.language, "segments": segments, "words": words}
