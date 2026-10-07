"""
FastAPI app: one page, one POST to start a job, one GET to poll it.

    python app/main.py   ->   http://localhost:8000

Environment (all optional, defaults suit local development):
    HOST                 bind address (default 127.0.0.1; 0.0.0.0 in Docker)
    PORT                 port (default 8000)
    OUTPUT_DIR           where jobs and clips are written (default ./output)
    MAX_CONCURRENT_JOBS  jobs processed at the same time (default 1); others wait
    WHISPER_MODEL, OLLAMA_MODEL, OLLAMA_URL, DEBUG_FACES: see README
"""

import os
import re
import sys
import threading
import uuid
from pathlib import Path

VERSION = "0.2.1"

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = ROOT_DIR / "static"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR") or ROOT_DIR / "output").resolve()
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MAX_CONCURRENT_JOBS", "1")))

# Allow `python app/main.py` from anywhere: make sibling modules importable.
sys.path.insert(0, str(APP_DIR))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from clipping import run_job  # noqa: E402
from ranking import OLLAMA_URL  # noqa: E402
from transcription import whisper_settings  # noqa: E402
from framing import debug_enabled  # noqa: E402

app = FastAPI(title="YouTube Clipper v0", version=VERSION)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job table. Fine for a local, single-user prototype.
JOBS: dict[str, dict] = {}

# Jobs beyond MAX_CONCURRENT_JOBS wait here instead of competing for the CPU.
JOB_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
RUNNING = {"count": 0}
RUNNING_LOCK = threading.Lock()

CLIP_NAME = re.compile(r"clip_\d+\.mp4")  # the only files served from a job folder


def _run_queued(job: dict, url: str, job_dir: Path, public_prefix: str) -> None:
    with JOB_SLOTS:
        with RUNNING_LOCK:
            RUNNING["count"] += 1
        try:
            run_job(job, url, job_dir, public_prefix)
        finally:
            with RUNNING_LOCK:
                RUNNING["count"] -= 1


class GenerateRequest(BaseModel):
    url: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/generate")
def generate(req: GenerateRequest):
    url = req.url.strip()
    if "youtube.com/" not in url and "youtu.be/" not in url:
        raise HTTPException(status_code=400, detail="Please paste a YouTube URL.")

    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "url": url, "stage": "queued", "status": "Waiting for a free slot...",
           "detail": "", "clips": [], "error": None}
    JOBS[job_id] = job

    job_dir = OUTPUT_DIR / job_id
    thread = threading.Thread(
        target=_run_queued, args=(job, url, job_dir, f"/output/{job_id}"), daemon=True
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/health")
def health():
    """Liveness + a few facts for monitoring. Always 200 when the app is up."""
    ollama = "unreachable"
    try:
        import requests
        if requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok:
            ollama = "ok"
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok",
        "version": VERSION,
        "whisper_model": whisper_settings()[0],
        "ollama": ollama,
        "debug_faces": debug_enabled(),
        "jobs_running": RUNNING["count"],
        "jobs_known": len(JOBS),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "output_dir": str(OUTPUT_DIR),
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {k: v for k, v in job.items() if k != "traceback"}


@app.api_route("/output/{job_id}/{filename}", methods=["GET", "HEAD"])
def download(job_id: str, filename: str, download: bool = False):
    # Only the rendered clips are public. Source video, transcript, filter
    # scripts and DEBUG_FACES sheets stay on disk for the operator.
    if not CLIP_NAME.fullmatch(filename):
        raise HTTPException(status_code=404, detail="File not found")
    path = (OUTPUT_DIR / job_id / filename).resolve()
    if not path.is_file() or OUTPUT_DIR not in path.parents:
        raise HTTPException(status_code=404, detail="File not found")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if download else None
    return FileResponse(path, media_type="video/mp4", headers=headers)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    uvicorn.run(app, host=HOST, port=PORT)
