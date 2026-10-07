# Changelog

## v0.2.1 — 2026-10-07 — release candidate, engine frozen

Validated on real videos (Windows, CPU):

| Source | Whisper | Result | Wall time |
|---|---|---|---|
| 27 min two-person interview | `small` | 5/5 clips publishable, 5/5 split-screen, clean framing | ~20 min |
| 23m30 French video | `medium` | 5/5 clips publishable, captions only slightly better than `small` | ~50 min |

Decision: `WHISPER_MODEL=small` is the production default. `medium` stays
available as an optional high-accuracy mode via the environment variable.
The clipping engine (ingestion, transcription, ranking, framing, captions,
rendering) is frozen at this version.

Engine fixes since v0.2.0:
- FFmpeg 9: face-framed clips use `-/filter_complex FILE` on FFmpeg 7+
  (`-filter_complex_script` was removed in 9); FFmpeg < 7 keeps the old
  option, and a rejected option is retried with the other form.
- Framing: a second face counts as persistent when detected in >= 25 % of
  samples with detections spanning >= 70 % of the clip (in addition to the
  >= 40 % coverage rule); the dominance rule no longer demotes a face that
  spans the clip; track fragments of one person are merged. Fixed the one
  interview clip that fell to SINGLE_PERSON.
- `DEBUG_FACES=true` records coverage, span, longest gap, fragments,
  separation, strength ratio, sample count and thresholds per clip.
- `WHISPER_MODEL` is read when a transcription starts; model, detected
  language and transcription time are logged and shown on the page.
- Tests: `tests/test_ffmpeg_filter_script.py`, `tests/test_framing_classification.py`.

Deployment (web layer only, engine untouched):
- `Dockerfile`, `docker-compose.yml` (app + Ollama), `DEPLOYMENT.md`.
- `HOST`, `PORT`, `OUTPUT_DIR`, `MAX_CONCURRENT_JOBS` environment variables;
  jobs beyond the limit wait with "Waiting for a free slot...".
- `GET /health`.
- Only `clip_N.mp4` files are served from job folders; debug sheets,
  transcripts and source videos are not exposed.

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
