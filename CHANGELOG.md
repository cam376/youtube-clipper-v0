# Changelog

## v0.3.0 — 2026-10-08 — client pilot (branch feature/v0.3-client-pilot)

Built on the frozen v0.2.1 engine. YouTube import, Whisper (default
`small`), face detection, framing, crop smoothing, FFmpeg compatibility,
caption timing source and output resolution are unchanged.

- **Dynamic clip count.** `select_clips()` replaces the fixed top-5: every
  candidate with score >= `MIN_CLIP_SCORE` (default 7 on the unchanged 0-10
  ranking scale) is kept unless it overlaps a stronger kept clip by more than
  `DEDUP_MAX_OVERLAP` (20 % of the shorter clip). There is no minimum
  count: 0 qualifying moments give 0 clips. Candidates are no longer sampled down to 40:
  Ollama scores them in batches of 40 with the same prompt.
  `output/<job>/ranking.json` records every candidate and score.
- **Persistent job manifest** `output/<job_id>/job.json` (source, title,
  clips with timestamps, score, framing plan, caption cues, style, font,
  edited / in_library flags, file names, glossary). Jobs reopen after a
  restart. Rendering is sequential; rerenders go through one worker.
- **Caption editor** per cue (fixed start/end, editable text) with
  deterministic retiming; **Find & replace** across all clips with a preview
  of affected clips; only touched clips rerender from the stored plan.
- **Subtitle presets** CLEAN (= v0.2.1 look), BOLD, KARAOKE, MINIMAL and
  **font choices** Clean Sans, Heavy Sans, Condensed, Classic with explicit
  fallback chains detected from installed fonts (never silent).
- **Review UI**: dynamic clip cards with score, duration, layout, style,
  font, badges, Edit captions, style/font Apply, Add to client library,
  Download HD, Rerender; Jobs list; Glossary.
- **Static client library export** to `client_libraries/<id>/` with 540x960
  previews (optional "KIVRO PREVIEW" watermark, `CLIENT_PREVIEW_WATERMARK`
  default true), posters, a mobile-first page with selection persisted in
  localStorage, a sticky "N clips selected" bar, clipboard summary and
  optional WhatsApp link. HD masters untouched; no internal paths exported.
- New env: `LIBRARIES_DIR`, `CLIENT_PREVIEW_WATERMARK`, `MIN_CLIP_SCORE`,
  `DEDUP_MAX_OVERLAP`.
- 53 new tests (71 total). See PILOT_GUIDE.md for the operator workflow.

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
