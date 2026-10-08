# Kivro (YouTube Clipper) v0.3

Paste a YouTube URL, click **Generate Clips**, get every strong vertical
(1080x1920) subtitled short the video contains, polish captions and style per
clip, pick the ones for the client, export a private static client library.
Everything runs locally. See **PILOT_GUIDE.md** for the operator workflow.

Only use this on videos you own or have permission to process. The importer
uses plain, unauthenticated yt-dlp: no cookies, no login, no bypasses.

## Pipeline

```
YouTube URL
  -> yt-dlp download (app/youtube.py)
  -> FFmpeg audio extraction + faster-whisper transcript (app/transcription.py)
  -> 20-60 s candidate windows, scored by Ollama + Qwen, kept when score >= MIN_CLIP_SCORE (app/ranking.py)
  -> face analysis per clip: one person / two people / none (app/framing.py)
  -> FFmpeg cut, 9:16 reframe, burned subtitles, H.264/AAC (app/video.py)
  -> clips + job.json manifest in output/<job_id>/, shown on the page
  -> edit captions / style / font per clip -> that clip rerenders (app/clipping.py)
  -> Export client library -> client_libraries/<id>/ static site (app/library.py)
```

## Clip selection (v0.3)

The ranking prompt and criteria are unchanged (0-10 per candidate). Selection:

| Setting | Default | Rule |
|---|---|---|
| `MIN_CLIP_SCORE` | `7` | a candidate is kept when its score is >= this |
| `DEDUP_MAX_OVERLAP` | `0.2` | dropped when it overlaps a stronger kept clip by more than 20 % of the shorter one |

No top-K and no minimum: 0 strong moments give 0 clips, 3 give 3, 40 give 40. Clips render
one at a time. `output/<job_id>/ranking.json` holds every candidate's score.

## Captions, styles, fonts

Caption cues (3-4 words, Whisper word timing) live in `job.json`. Editing a
cue keeps its start/end; same token count keeps each word's timing, a
different count spreads words evenly over the cue. Styles: CLEAN (v0.2.1
look), BOLD, KARAOKE (spoken word in electric blue via ASS \k tags),
MINIMAL. Fonts: Clean Sans (Arial > Liberation Sans > DejaVu Sans),
Heavy Sans (Arial Black > Impact > Liberation Sans), Condensed (Franklin
Gothic Medium > Arial Narrow > Liberation Sans Narrow > DejaVu Sans
Condensed > Arial), Classic (Georgia > Times New Roman > Liberation Serif >
DejaVu Serif). The first installed family is used; a fallback is flagged in
the UI and in `job.json`.

If Ollama is not running, the app falls back to a simple heuristic ranking and
says so in the status line, so the pipeline still completes.

## Vertical framing

Each clip is analysed with OpenCV's YuNet face detector (model bundled in
`app/models/`, Haar cascade fallback) on 2 frames per second, faces are
linked into tracks, and one layout is chosen for the whole clip:

| Layout          | When                                           | Result                                  |
|-----------------|------------------------------------------------|-----------------------------------------|
| `SINGLE_PERSON` | one face present in >= 30 % of samples        | 9:16 crop centred on the face           |
| `TWO_PERSON`    | two persistent faces, clearly apart. Persistent = detected in >= 40 % of samples, or >= 25 % with detections spanning >= 70 % of the clip | top/bottom split, 1080x960 each, captions on the seam |
| `CENTER_CROP`   | no reliable face                               | the original centre crop                |

Crop positions are smoothed (moving average + dead zone) so they stay still
unless a person really moves. Set `DEBUG_FACES=true` to also write
`clip_N_faces.jpg` (face boxes, track ids, crop regions, layout) and
`clip_N_faces.json` next to each clip.

## Requirements

- Python 3.10+
- FFmpeg on your PATH (`ffmpeg -version` must work)
- [Ollama](https://ollama.com) running locally

## Run

```bash
pip install -r requirements.txt
ollama pull qwen2.5:3b
python app/main.py
```

Open http://localhost:8000 and paste a URL.

First run downloads the Whisper model (`small`, about 500 MB) from Hugging Face.

## Deployment

`DEPLOYMENT.md` covers the single-server private beta: Dockerfile,
`docker compose up -d` with Ollama, volumes, HTTPS via Caddy, `GET /health`.

## Tests

```bash
python -m unittest discover -s tests -v
```
71 tests: FFmpeg filter-script compatibility, framing classification,
dynamic selection, captions, styles and fonts, manifest and rerender,
client library export. Render tests use the ffmpeg on PATH; the static
client page test runs only if `playwright` is installed.

## Tuning (environment variables)

| Variable         | Default                  | Notes                                     |
|------------------|--------------------------|-------------------------------------------|
| `WHISPER_MODEL`  | `small`                  | `base` is faster, `medium` is more accurate (French in particular), about 2-3x slower on CPU |
| `WHISPER_DEVICE` | `cpu`                    | `cuda` if you have a GPU                  |
| `WHISPER_COMPUTE`| `int8`                   | `float16` on GPU                          |
| `OLLAMA_MODEL`   | `qwen2.5:3b`             | any Qwen model you have pulled            |
| `OLLAMA_URL`     | `http://localhost:11434` |                                           |
| `DEBUG_FACES`    | unset                    | `true` writes face/crop diagnostics per clip |
| `HOST` / `PORT`  | `127.0.0.1` / `8000`     | bind address and port                     |
| `OUTPUT_DIR`     | `./output`               | where jobs and clips are written          |
| `MAX_CONCURRENT_JOBS` | `1`                 | jobs processed at once; others wait       |
| `LIBRARIES_DIR`  | `./client_libraries`     | where client libraries are exported       |
| `CLIENT_PREVIEW_WATERMARK` | `true`         | default for the "KIVRO PREVIEW" mark on exported previews |
| `MIN_CLIP_SCORE` / `DEDUP_MAX_OVERLAP` | `7` / `0.2` | see Clip selection |

## Whisper model policy (v0.2.1)

`small` is the production default. `medium` is an optional high-accuracy
mode, never the default. Real measurements on the Windows test machine:

| Source | Model | Clips | Wall time |
|---|---|---|---|
| 27 min two-person interview | `small` | 5/5 publishable | ~20 min |
| 23m30 French video | `medium` | 5/5 publishable, captions only slightly better | ~50 min |

## Trying a larger Whisper model

```bash
WHISPER_MODEL=medium python app/main.py          # macOS / Linux
```
```powershell
$env:WHISPER_MODEL = "medium"; python app\main.py   # Windows PowerShell
```
The first run downloads the model (about 1.5 GB for `medium`). The console
prints the model, detected language and transcription time; the page shows
the same in the status line when the job is done. Compare
`output/<job_id>/transcript.json` between runs.

## Layout

```
app/
  main.py           FastAPI server, routes, render worker
  clipping.py       pipeline for one job + rerender of one clip
  manifest.py       output/<job_id>/job.json read/write
  captions.py       caption cues, edit retiming, find & replace
  styles.py         CLEAN / BOLD / KARAOKE / MINIMAL ASS presets
  fonts.py          font choices and fallback detection
  library.py        static client library export (previews, page)
  templates/client_library/   the client page (index.html, style.css, app.js)
  youtube.py        isolated YouTube import provider
  transcription.py  FFmpeg audio extraction + faster-whisper
  ranking.py        candidate windows, Ollama scoring, selection
  framing.py        face detection, tracking, layout choice, crop smoothing
  video.py          FFmpeg cut / reframe / subtitles / export
  models/           YuNet face detection model (ONNX, Apache-2.0, from opencv_zoo)
static/
  index.html, app.js, editor.js, export.js, style.css
output/             one folder per job (source, transcript, job.json, clips)
client_libraries/   exported static client libraries
```
