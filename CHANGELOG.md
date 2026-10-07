# Changelog

## v0.2.0 — 2026-10-07 — multi-person vertical framing

- New `app/framing.py`: per-clip face detection (OpenCV YuNet, Haar fallback),
  track linking across sampled frames, one layout per clip.
- `TWO_PERSON`: top/bottom split-screen, each speaker cropped independently
  (1080x960 halves, aspect preserved), captions moved to the centre seam.
- `SINGLE_PERSON`: the 9:16 crop now follows the detected face instead of the
  geometric centre.
- `CENTER_CROP`: unchanged behaviour when no reliable face is found, and the
  fallback for any analysis error.
- Crop smoothing: moving average + dead zone + glide, no frame-by-frame jitter.
- `DEBUG_FACES=true` writes a diagnostic contact sheet and JSON per clip.
- New dependencies: opencv-python-headless, numpy.
- Ranking, transcription, captions styling and YouTube import are untouched.

## v0.1.0 — 2026-10-07 — first working checkpoint

Verified end to end on a real YouTube video on Windows.

- Single page: YouTube URL -> Generate Clips -> status -> 3-5 vertical clips with preview and download.
- Pipeline: yt-dlp import, FFmpeg audio extraction, faster-whisper transcript (word timestamps),
  20-60 s candidate windows scored by Ollama + `qwen2.5:3b`, greedy non-overlapping selection,
  FFmpeg cut + 1080x1920 centre crop + burned ASS subtitles, H.264/AAC export.
- Heuristic ranking fallback when Ollama is not reachable (reported in the status line).
- Exact dependency versions recorded in `requirements-lock.txt`.
- Windows setup documented in `WINDOWS_SETUP.md`.

The clipping engine (`app/clipping.py`, `app/ranking.py`, `app/transcription.py`,
`app/video.py`, `app/youtube.py`) is frozen at this tag. Later work should build on
top of it, not rewrite it.
