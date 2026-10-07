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
  -> FFmpeg cut, 9:16 centre crop, burned subtitles, H.264/AAC (app/video.py)
  -> clips served from output/<job_id>/ and shown on the page
```

If Ollama is not running, the app falls back to a simple heuristic ranking and
says so in the status line, so the pipeline still completes.

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

## Tuning (environment variables)

| Variable         | Default                  | Notes                                     |
|------------------|--------------------------|-------------------------------------------|
| `WHISPER_MODEL`  | `small`                  | `base` is faster, `medium` more accurate  |
| `WHISPER_DEVICE` | `cpu`                    | `cuda` if you have a GPU                  |
| `WHISPER_COMPUTE`| `int8`                   | `float16` on GPU                          |
| `OLLAMA_MODEL`   | `qwen2.5:3b`             | any Qwen model you have pulled            |
| `OLLAMA_URL`     | `http://localhost:11434` |                                           |

## Layout

```
app/
  main.py           FastAPI server, job table, routes
  clipping.py       orchestrates the pipeline for one job
  youtube.py        isolated YouTube import provider
  transcription.py  FFmpeg audio extraction + faster-whisper
  ranking.py        candidate windows, Ollama scoring, selection
  video.py          FFmpeg cut / crop / subtitles / export
static/
  index.html, app.js, style.css
output/             one folder per job (source, transcript, clips)
```
