# Changelog

## v0.2.2 — 2026-10-07 — two-person robustness, Whisper model switch

- Framing: a second face is now "persistent" when it is detected in >= 25 % of
  samples AND its detections span >= 70 % of the clip, in addition to the old
  >= 40 % coverage rule. The dominance rule (second face < 0.5 x first) no
  longer applies to a face that spans the clip. Track fragments of one person
  (same seat, never overlapping in time) are merged before classification.
  Thresholds for solo videos, the centre-crop fallback, crop smoothing and
  the split-screen layout are unchanged.
- Debug output (`DEBUG_FACES=true`) now records per track coverage, span,
  longest gap and fragment count, plus separation, strength ratio, sample
  count and all thresholds under `classification` in `clip_N_faces.json`.
  The same numbers appear in each clip's `layout_note`.
- Whisper: `WHISPER_MODEL` is read when a transcription starts (set it before
  launching the server); the model is reloaded if it changes. The server logs
  model, detected language, language probability and transcription duration,
  and the page shows them in the status line at the end.
- New tests: `tests/test_framing_classification.py`.

## v0.2.1 — 2026-10-07 — FFmpeg 9 compatibility

- Face-framed clips are rendered with `-/filter_complex FILE` on FFmpeg 7+
  (FFmpeg 9 removed `-filter_complex_script`); FFmpeg < 7 keeps the old
  option. The version is read from `ffmpeg -version`, and a rejected option
  is retried with the other form, so git builds without a numeric version
  also work. The filtergraph itself and the side-file approach are unchanged.
- ffmpeg errors on face-framed clips now surface ffmpeg's stderr in the job error.
- `tests/test_ffmpeg_filter_script.py`: regression test, run with
  `python -m unittest discover -s tests -v` against the ffmpeg on PATH.

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
