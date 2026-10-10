"""
FastAPI app: one page, a job pipeline, and the v0.3 review/edit/export API.

    python app/main.py   ->   http://localhost:8000

Environment (all optional, defaults suit local development):
    HOST                 bind address (default 127.0.0.1; 0.0.0.0 in Docker)
    PORT                 port (default 8000)
    OUTPUT_DIR           where jobs and clips are written (default ./output)
    LIBRARIES_DIR        where client libraries are exported (default ./client_libraries)
    ROOT_PATH            URL prefix when served under a sub-path behind a reverse proxy that
                         strips it (e.g. "/vezly.ai" for https://settermonster.com/vezly.ai/).
                         All links the app emits are relative, so this only feeds FastAPI's
                         docs/openapi URLs; leave empty for http://localhost:8000.
    MAX_CONCURRENT_JOBS  jobs processed at the same time (default 1); others wait
    CLIENT_PREVIEW_WATERMARK  default for the export watermark checkbox (default true)
    MIN_CLIP_SCORE, DEDUP_MAX_OVERLAP: see ranking.py
    WHISPER_MODEL, OLLAMA_MODEL, OLLAMA_URL, DEBUG_FACES: see README
"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

VERSION = "0.3.0"

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = ROOT_DIR / "static"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR") or ROOT_DIR / "output").resolve()
LIBRARIES_DIR = Path(os.environ.get("LIBRARIES_DIR") or ROOT_DIR / "client_libraries").resolve()
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MAX_CONCURRENT_JOBS", "1")))
PREVIEW_WATERMARK_DEFAULT = os.environ.get("CLIENT_PREVIEW_WATERMARK", "true").strip().lower() in ("1", "true", "yes", "on")
ROOT_PATH = "/" + os.environ.get("ROOT_PATH", "").strip("/") if os.environ.get("ROOT_PATH", "").strip("/") else ""

# Allow `python app/main.py` from anywhere: make sibling modules importable.
sys.path.insert(0, str(APP_DIR))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import manifest as mf  # noqa: E402
from clipping import run_job, rerender_clip  # noqa: E402
from captions import set_cue_text, find_in_cues, replace_in_cues  # noqa: E402
from styles import style_names, normalize_style  # noqa: E402
from fonts import font_choices, FONT_CHOICES  # noqa: E402
from library import export_library  # noqa: E402
from ranking import OLLAMA_URL, MIN_CLIP_SCORE, DEDUP_MAX_OVERLAP  # noqa: E402
from transcription import whisper_settings  # noqa: E402
from framing import debug_enabled  # noqa: E402

app = FastAPI(title="Kivro", version=VERSION, root_path=ROOT_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)

# Live state of running jobs (status text, detail). Everything durable is in job.json.
JOBS: dict[str, dict] = {}
JOB_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
RUNNING = {"count": 0}
RUNNING_LOCK = threading.Lock()

# One ffmpeg at a time for rerenders and exports (bounded rendering).
RENDER_QUEUE: "queue.Queue[tuple]" = queue.Queue()
RENDER_LOCK = threading.Lock()
EXPORTS: dict[str, dict] = {}

CLIP_NAME = re.compile(r"clip_\d+\.mp4")  # the only files served from a job folder
JOB_ID = re.compile(r"[0-9a-f]{12}")


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
def _run_queued(job: dict, url: str, job_dir: Path, public_prefix: str) -> None:
    with JOB_SLOTS:
        with RUNNING_LOCK:
            RUNNING["count"] += 1
        try:
            run_job(job, url, job_dir, public_prefix)
        finally:
            with RUNNING_LOCK:
                RUNNING["count"] -= 1


def _render_worker() -> None:
    while True:
        job_id, clip_id = RENDER_QUEUE.get()
        try:
            job_dir = OUTPUT_DIR / job_id
            m = mf.load(job_dir)
            clip = mf.find_clip(m, clip_id) if m else None
            if clip is None:
                continue
            with RENDER_LOCK:
                rerender_clip(job_dir, m, clip)
        except Exception as exc:  # noqa: BLE001
            print(f"[rerender] {job_id}/{clip_id} failed: {exc}", flush=True)
        finally:
            RENDER_QUEUE.task_done()


threading.Thread(target=_render_worker, daemon=True, name="rerender").start()


def _queue_rerender(job_id: str, clip_id: str) -> None:
    mf.update_clip(OUTPUT_DIR / job_id, clip_id, {"render_state": "queued"})
    RENDER_QUEUE.put((job_id, clip_id))


def _modify(job_id: str, fn) -> dict:
    d = _job_dir(job_id)
    try:
        return mf.modify(d, fn)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job has no manifest")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _job_dir(job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    d = OUTPUT_DIR / job_id
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="Unknown job")
    return d


def _manifest(job_id: str) -> tuple[Path, dict]:
    d = _job_dir(job_id)
    m = mf.load(d)
    if m is None:
        raise HTTPException(status_code=404, detail="Job has no manifest")
    return d, m


def _clip(m: dict, clip_id: str) -> dict:
    c = mf.find_clip(m, clip_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Unknown clip")
    return c


def _view(job_id: str, m: dict) -> dict:
    view = mf.public_view(m, f"output/{job_id}")
    live = JOBS.get(job_id)
    if live:
        for k in ("stage", "status", "detail", "error", "transcription_note", "ranking_note", "framing_warning"):
            if k in live:
                view[k] = live[k]
    view["libraries"] = m.get("libraries", [])
    return view


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    url: str


class CaptionsUpdate(BaseModel):
    cues: list[dict]            # [{"text": "..."}] same length and order as the clip's cues
    rerender: bool = True


class StyleUpdate(BaseModel):
    style: Optional[str] = None
    font: Optional[str] = None
    rerender: bool = True


class FindReplace(BaseModel):
    find: str
    replace: str = ""
    apply: bool = False
    case_sensitive: bool = False
    rerender: bool = True


class LibraryFlag(BaseModel):
    selected: bool


class Glossary(BaseModel):
    terms: list[str]


class ExportRequest(BaseModel):
    client_name: str
    title: Optional[str] = None
    show_price: bool = False
    price_per_clip: Optional[float] = None
    currency: str = "EUR"
    whatsapp_number: Optional[str] = None
    watermark: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Routes: page, jobs
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    ollama = "unreachable"
    try:
        import requests
        if requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok:
            ollama = "ok"
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok", "version": VERSION, "whisper_model": whisper_settings()[0], "ollama": ollama,
        "debug_faces": debug_enabled(), "jobs_running": RUNNING["count"], "jobs_known": len(mf.load_all(OUTPUT_DIR)),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS, "rerenders_queued": RENDER_QUEUE.qsize(),
        "output_dir": str(OUTPUT_DIR), "libraries_dir": str(LIBRARIES_DIR), "root_path": ROOT_PATH,
        "selection": {"min_clip_score": MIN_CLIP_SCORE, "dedup_max_overlap": DEDUP_MAX_OVERLAP},
    }


@app.get("/api/options")
def options():
    return {
        "styles": style_names(),
        "fonts": font_choices(),
        "preview_watermark_default": PREVIEW_WATERMARK_DEFAULT,
        "selection": {"min_clip_score": MIN_CLIP_SCORE, "dedup_max_overlap": DEDUP_MAX_OVERLAP},
    }


@app.get("/api/jobs")
def list_jobs():
    out = []
    for m in mf.load_all(OUTPUT_DIR):
        live = JOBS.get(m["job_id"], {})
        out.append({
            "job_id": m["job_id"], "title": m["source"].get("title"), "url": m["source"].get("url"),
            "created_at": m.get("created_at"), "stage": live.get("stage", m.get("stage")),
            "status": live.get("status", m.get("status")),
            "clip_count": len(m.get("clips", [])),
            "library_count": sum(1 for c in m.get("clips", []) if c.get("in_library")),
            "libraries": len(m.get("libraries", [])),
        })
    return out


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
    threading.Thread(target=_run_queued, args=(job, url, job_dir, f"output/{job_id}"), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    live = JOBS.get(job_id)
    d = OUTPUT_DIR / job_id
    m = mf.load(d) if JOB_ID.fullmatch(job_id) and d.is_dir() else None
    if m is None:
        if live is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return {k: v for k, v in live.items() if k != "traceback"}
    return _view(job_id, m)


# --------------------------------------------------------------------------- #
# Routes: editing
# --------------------------------------------------------------------------- #
@app.put("/api/jobs/{job_id}/clips/{clip_id}/captions")
def update_captions(job_id: str, clip_id: str, body: CaptionsUpdate):
    result = {"changed": 0, "index": 0}

    def apply(m):
        clip = _clip(m, clip_id)
        if len(body.cues) != len(clip["cues"]):
            raise HTTPException(status_code=400, detail="Cue count mismatch; reload the clip and try again.")
        new_cues, changed = [], 0
        for old, upd in zip(clip["cues"], body.cues):
            text = str(upd.get("text", old["text"]))
            if text.strip() != old["text"].strip():
                changed += 1
                new_cues.append(set_cue_text(old, text))
            else:
                new_cues.append(old)
        clip["cues"] = new_cues
        if changed:
            clip["edited"] = True
        result.update(changed=changed, index=clip["index"])

    m = _modify(job_id, apply)
    if result["changed"] and body.rerender:
        _queue_rerender(job_id, clip_id)
        m = mf.load(_job_dir(job_id))
    return {"clip": _view(job_id, m)["clips"][result["index"] - 1], "changed_cues": result["changed"]}


@app.put("/api/jobs/{job_id}/clips/{clip_id}/style")
def update_style(job_id: str, clip_id: str, body: StyleUpdate):
    if body.font is not None and body.font not in FONT_CHOICES:
        raise HTTPException(status_code=400, detail="Unknown font choice")
    index = {"i": 0}

    def apply(m):
        clip = _clip(m, clip_id)
        if body.style is not None:
            clip["style"] = normalize_style(body.style)
        if body.font is not None:
            clip["font"] = body.font
        index["i"] = clip["index"]

    m = _modify(job_id, apply)
    if body.rerender:
        _queue_rerender(job_id, clip_id)
        m = mf.load(_job_dir(job_id))
    return {"clip": _view(job_id, m)["clips"][index["i"] - 1]}


@app.post("/api/jobs/{job_id}/clips/{clip_id}/rerender")
def rerender(job_id: str, clip_id: str):
    d, m = _manifest(job_id)
    clip = _clip(m, clip_id)
    _queue_rerender(job_id, clip_id)
    m = mf.load(d)
    return {"clip": _view(job_id, m)["clips"][clip["index"] - 1]}


@app.post("/api/jobs/{job_id}/find-replace")
def find_replace(job_id: str, body: FindReplace):
    d, m = _manifest(job_id)
    if not body.find.strip():
        raise HTTPException(status_code=400, detail="Nothing to find")
    matches = []
    for clip in m["clips"]:
        idx = find_in_cues(clip["cues"], body.find, body.case_sensitive)
        if idx:
            matches.append({"clip_id": clip["id"], "label": clip["label"], "cue_count": len(idx),
                            "examples": [clip["cues"][i]["text"] for i in idx[:3]]})
    replaced = 0
    if body.apply:
        touched: list[str] = []

        def apply(mm):
            nonlocal replaced
            for clip in mm["clips"]:
                new_cues, n = replace_in_cues(clip["cues"], body.find, body.replace, body.case_sensitive)
                if n:
                    clip["cues"] = new_cues
                    clip["edited"] = True
                    replaced += n
                    touched.append(clip["id"])

        _modify(job_id, apply)
        if body.rerender:
            for cid in touched:
                _queue_rerender(job_id, cid)
    return {"matches": matches, "affected_clips": len(matches), "replaced": replaced,
            "rerender_queued": bool(body.apply and body.rerender and matches)}


@app.put("/api/jobs/{job_id}/clips/{clip_id}/library")
def set_library_flag(job_id: str, clip_id: str, body: LibraryFlag):
    def apply(m):
        _clip(m, clip_id)["in_library"] = bool(body.selected)
    m = _modify(job_id, apply)
    return {"clip_id": clip_id, "in_library": bool(body.selected),
            "library_count": sum(1 for c in m["clips"] if c.get("in_library"))}


@app.put("/api/jobs/{job_id}/glossary")
def set_glossary(job_id: str, body: Glossary):
    terms: list[str] = []
    for t in body.terms:
        t = t.strip()
        if t and t not in terms:
            terms.append(t)

    def apply(m):
        m["glossary"] = terms
    _modify(job_id, apply)
    return {"glossary": terms}


# --------------------------------------------------------------------------- #
# Routes: client library export
# --------------------------------------------------------------------------- #
def _export_worker(job_id: str, export_id: str, options: dict) -> None:
    state = EXPORTS[export_id]
    try:
        d, m = _manifest(job_id)
        with RENDER_LOCK:
            summary = export_library(d, m, options, LIBRARIES_DIR,
                                     progress=lambda msg: state.update({"detail": msg}))
        mf.modify(d, lambda mm: mm.setdefault("libraries", []).append(summary))
        state.update({"stage": "done", "detail": "", "library": summary,
                      "preview_url": f"libraries/{summary['library_id']}/index.html"})
    except Exception as exc:  # noqa: BLE001
        state.update({"stage": "error", "error": f"{exc.__class__.__name__}: {exc}"})


@app.post("/api/jobs/{job_id}/export-library")
def start_export(job_id: str, body: ExportRequest):
    d, m = _manifest(job_id)
    if not any(c.get("in_library") for c in m["clips"]):
        raise HTTPException(status_code=400, detail="Mark at least one clip with 'Add to client library' first.")
    if not body.client_name.strip():
        raise HTTPException(status_code=400, detail="Client name is required")
    export_id = uuid.uuid4().hex[:8]
    options = body.model_dump()
    if options.get("watermark") is None:
        options["watermark"] = PREVIEW_WATERMARK_DEFAULT
    EXPORTS[export_id] = {"export_id": export_id, "job_id": job_id, "stage": "exporting", "detail": "starting", "error": None}
    threading.Thread(target=_export_worker, args=(job_id, export_id, options), daemon=True).start()
    return EXPORTS[export_id]


@app.get("/api/exports/{export_id}")
def export_status(export_id: str):
    st = EXPORTS.get(export_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Unknown export")
    return st


@app.get("/api/libraries")
def list_libraries():
    out = []
    for d in sorted(LIBRARIES_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if LIBRARIES_DIR.is_dir() else []:
        lj = d / "library.json"
        if lj.is_file():
            try:
                import json
                data = json.loads(lj.read_text(encoding="utf-8"))
                out.append({"library_id": data["library_id"], "title": data["title"], "client_name": data["client_name"],
                            "clip_count": len(data["clips"]), "created_at": data["created_at"],
                            "preview_url": f"libraries/{data['library_id']}/index.html", "path": str(d)})
            except (OSError, ValueError, KeyError):
                continue
    return out


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
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
app.mount("/libraries", StaticFiles(directory=LIBRARIES_DIR), name="libraries")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    uvicorn.run(app, host=HOST, port=PORT)
