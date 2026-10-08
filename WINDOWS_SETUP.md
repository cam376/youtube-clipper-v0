# Windows setup — v0.1.0 (known-working)

This is the setup the v0.1.0 checkpoint runs on. Follow it in order on a fresh
Windows 10/11 (64-bit) machine. Every command is for **PowerShell** unless noted.
The dependency versions are the exact ones recorded in `requirements-lock.txt`.

## 1. Python 3.11 or 3.12 (64-bit)

1. Download from https://www.python.org/downloads/windows/ (the "Windows installer (64-bit)").
2. In the installer tick **Add python.exe to PATH**, then *Install Now*.
3. Check:
   ```powershell
   py -3 --version
   ```
   Expect `Python 3.11.x` or `3.12.x`. 3.13 also works, 3.10 is the minimum.

The Microsoft Store Python also works, but the python.org build is what this
checkpoint was validated with.

## 2. FFmpeg (with libass, needed for burned subtitles)

Option A, winget (simplest):
```powershell
winget install --id Gyan.FFmpeg -e
```
Close and reopen PowerShell afterwards so PATH is refreshed.

Option B, manual: download `ffmpeg-release-essentials.zip` from
https://www.gyan.dev/ffmpeg/builds/, unzip to `C:\ffmpeg`, then add
`C:\ffmpeg\bin` to your user PATH (Settings > System > About > Advanced
system settings > Environment Variables).

Check:
```powershell
ffmpeg -version
ffmpeg -filters | Select-String "^ ... ass "
```
The second command must print a line containing `ass` (the libass subtitle
renderer). Gyan.dev builds include it. Builds without libass will fail at
the "Creating clips..." stage.

## 3. Ollama + Qwen

1. Install from https://ollama.com/download/windows (`OllamaSetup.exe`).
2. Ollama runs as a tray app and listens on `http://localhost:11434`.
3. Pull the model used by the ranker:
   ```powershell
   ollama pull qwen2.5:3b
   ```
4. Check:
   ```powershell
   ollama list
   curl.exe http://localhost:11434/api/tags
   ```

`qwen2.5:3b` needs about 2 GB of RAM/VRAM. Ollama must be running when you
click *Generate Clips*, otherwise the app falls back to a heuristic ranker and
says so in the status line.

## 4. Project

```powershell
git clone https://github.com/cam376/youtube-clipper-v0.git
cd youtube-clipper-v0
git checkout v0.1.0
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

If `Activate.ps1` is blocked ("running scripts is disabled on this system"):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
then run the activate line again. In `cmd.exe` use `.venv\Scripts\activate.bat` instead.

## 5. Run

```powershell
python app\main.py
```
Open http://localhost:8000, paste a YouTube URL for a video you own or have
permission to process, click *Generate Clips*.

Or double-click `start-windows.bat`, which creates the venv, installs the
pinned dependencies on first run, warns if Ollama is down, and starts the server.

First run downloads the Whisper `small` model (about 480 MB) into
`%USERPROFILE%\.cache\huggingface\hub`. Later runs are offline except for the
YouTube download itself.

## 6. What a successful run looks like

| Stage shown on page    | Typical duration (20 min 1080p video, CPU only) |
|------------------------|-------------------------------------------------|
| Importing video...     | 10 s to 2 min, depends on your connection        |
| Transcribing...        | 3 to 8 min with `small` on a 4-8 core CPU        |
| Finding best moments...| 10 to 60 s with `qwen2.5:3b`                     |
| Creating clips...      | 15 to 40 s per clip                              |
| Done.                  | 3 to 5 clip cards with players and Download      |

Everything for a run is in `output\<job_id>\`: `source.mp4`, `audio.wav`,
`transcript.json`, `selection.json`, `clip_N.ass`, `clip_N.mp4`. Delete the
folder when you no longer need it; nothing else is stored.

## 7. Optional speed-ups

Set environment variables in the same PowerShell window before `python app\main.py`:

```powershell
$env:WHISPER_MODEL = "medium"    # optional high-accuracy mode; measured ~50 min for a 23m30 French video vs ~20 min for 27 min with small, captions only slightly better (default stays: small)
$env:WHISPER_MODEL = "base"      # faster, less accurate transcription
$env:OLLAMA_MODEL  = "qwen2.5:7b" # better ranking if you have the RAM (default: qwen2.5:3b)
```

To A/B the captions on one video:
```powershell
$env:WHISPER_MODEL = "medium"
python app\main.py
```
Run the same URL, then compare `output\<job_id>\transcript.json` with the
`small` run. The console line `transcribed ... with medium (language=fr ...)`
confirms which model ran; the page shows it in the status line at the end.
`medium` downloads about 1.5 GB on first use and needs about 3 GB of RAM.

NVIDIA GPU for Whisper:
```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
$env:WHISPER_DEVICE  = "cuda"
$env:WHISPER_COMPUTE = "float16"
```
This needs a recent NVIDIA driver. If you get a DLL error, go back to `cpu`.

## 7b. Face-aware framing (v0.2.0)

No extra install: OpenCV comes from `requirements-lock.txt` and the face model
is in the repo. To see what the framing decided for each clip:

```powershell
$env:DEBUG_FACES = "true"
python app\main.py
```
Then open `output\<job_id>\clip_N_faces.jpg`. The title line shows the layout
(`TWO_PERSON`, `SINGLE_PERSON`, `CENTER_CROP`) and why; coloured boxes are
face tracks with their id and coverage, yellow/cyan rectangles are the crops.

## 7c. Testing v0.3 (client pilot branch)

```powershell
cd youtube-clipper-v0
git fetch origin
git checkout feature/v0.3-client-pilot
git pull
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt      # nothing new, confirms the pins
python -m unittest discover -s tests -v   # expect 71 tests OK (70 if playwright is not installed)
python app\main.py
```
Open http://localhost:8000, generate, edit a caption, change a style, mark
clips, **Export client library**, then preview the folder:
```powershell
python -m http.server 8080 --directory "client_libraries\<library_id>"
```
Fonts on this machine: Arial, Arial Black, Franklin Gothic Medium and
Georgia are all part of Windows 10/11, so no font fallback badge should
appear. The full workflow is in PILOT_GUIDE.md.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ffmpeg` not recognized | PATH not refreshed | reopen PowerShell; verify with `ffmpeg -version` |
| Error at "Creating clips..." mentioning `ass` or `No such filter` | FFmpeg built without libass | install the Gyan.dev build (step 2) |
| Error at "Importing video..." from yt-dlp | YouTube changed something | `pip install -U yt-dlp` and retry |
| yt-dlp says video is private / members-only / needs sign-in | not an authorized public video | this prototype does not bypass that; use a video you own that is public or unlisted |
| Status says "Ollama unavailable" | Ollama tray app not running or model not pulled | start Ollama, run `ollama pull qwen2.5:3b`, click again |
| Transcription extremely slow | antivirus scanning model files, or `medium` model | exclude `.cache\huggingface` from Defender, or use `base` |
| Windows Firewall prompt on first start | uvicorn opening port 8000 | allow on private networks; the server binds 127.0.0.1 only |
| Clips play but no subtitles | Whisper found no words in that window | check `output\<job_id>\clip_N.ass` is non-empty |
| Port 8000 already in use | another app | change the port at the bottom of `app\main.py` |
| `Unrecognized option 'filter_complex_script'` | FFmpeg 9 removed that option | fixed in v0.2.1; run `python -m unittest discover -s tests -v` to confirm your ffmpeg renders |
| Two-person interview still centre-cropped | faces too small, turned away, or detected in < 40 % of samples | run with `DEBUG_FACES=true` and check coverage in the title line |
| `ImportError: DLL load failed` from cv2 | missing Visual C++ runtime | install the Microsoft Visual C++ 2015-2022 Redistributable (x64) |

## 9. Record of the machine this was validated on

Fill this in once so the checkpoint is reproducible:

- Windows version: 
- CPU / RAM / GPU: 
- `py -3 --version`: 
- `ffmpeg -version` first line: 
- `ollama --version`: 
- Test video length and time to "Done.": 
