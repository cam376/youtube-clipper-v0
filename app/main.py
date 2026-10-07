"""
FastAPI app: one page, one POST to start a job, one GET to poll it.

    python app/main.py   ->   http://localhost:8000
"""

import sys
import threading
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = ROOT_DIR / "static"
OUTPUT_DIR = ROOT_DIR / "output"

# Allow `python app/main.py` from anywhere: make sibling modules importable.
sys.path.insert(0, str(APP_DIR))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from clipping import run_job  # noqa: E402

app = FastAPI(title="YouTube Clipper v0")
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job table. Fine for a local, single-user prototype.
JOBS: dict[str, dict] = {}


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
    job = {"id": job_id, "url": url, "stage": "importing", "status": "Importing video...",
           "detail": "", "clips": [], "error": None}
    JOBS[job_id] = job

    job_dir = OUTPUT_DIR / job_id
    thread = threading.Thread(
        target=run_job, args=(job, url, job_dir, f"/output/{job_id}"), daemon=True
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {k: v for k, v in job.items() if k != "traceback"}


@app.api_route("/output/{job_id}/{filename}", methods=["GET", "HEAD"])
def download(job_id: str, filename: str, download: bool = False):
    path = (OUTPUT_DIR / job_id / filename).resolve()
    if not path.is_file() or OUTPUT_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="File not found")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if download else None
    return FileResponse(path, media_type="video/mp4", headers=headers)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000)
