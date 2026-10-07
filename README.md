# YouTube Clipper v0

Paste a YouTube URL, click **Generate Clips**, get 3-5 vertical (1080x1920)
subtitled MP4 shorts you can preview and download. Everything runs locally.

Only use this on videos you own or have permission to process. The importer
uses plain, unauthenticated yt-dlp: no cookies, no login, no bypasses.

## Pipeline

```
YouTube URL
  -> yt-dlp download (app/youtube.py)
  -> FFmpeg audio extraction + faster-whisper transcript (app/transcription.py)
  -> 20-60 s candidate windows, scored by Ollama + Qwen (app/ranking.py)
  -> face analysis per clip: one person / two people / none (app/framing.py)
  -> FFmpeg cut, 9:16 reframe, burned subtitles, H.264/AAC (app/video.py)
  -> clips served from output/<job_id>/ and shown on the page
```

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

## Tests

```bash
python -m unittest discover -s tests -v
```
Renders synthetic clips with the ffmpeg on PATH and checks the complex
filtergraph side-file option matches that ffmpeg (FFmpeg 6 through 9).

## Tuning (environment variables)

| Variable         | Default                  | Notes                                     |
|------------------|--------------------------|-------------------------------------------|
| `WHISPER_MODEL`  | `small`                  | `base` is faster, `medium` is more accurate (French in particular), about 2-3x slower on CPU |
| `WHISPER_DEVICE` | `cpu`                    | `cuda` if you have a GPU                  |
| `WHISPER_COMPUTE`| `int8`                   | `float16` on GPU                          |
| `OLLAMA_MODEL`   | `qwen2.5:3b`             | any Qwen model you have pulled            |
| `OLLAMA_URL`     | `http://localhost:11434` |                                           |
| `DEBUG_FACES`    | unset                    | `true` writes face/crop diagnostics per clip |

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
  main.py           FastAPI server, job table, routes
  clipping.py       orchestrates the pipeline for one job
  youtube.py        isolated YouTube import provider
  transcription.py  FFmpeg audio extraction + faster-whisper
  ranking.py        candidate windows, Ollama scoring, selection
  framing.py        face detection, tracking, layout choice, crop smoothing
  video.py          FFmpeg cut / reframe / subtitles / export
  models/           YuNet face detection model (ONNX, Apache-2.0, from opencv_zoo)
static/
  index.html, app.js, style.css
output/             one folder per job (source, transcript, clips)
```
