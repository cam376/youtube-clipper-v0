"""
Persistent job manifest: output/<job_id>/job.json

Everything needed to reopen, edit and export a job after the server
restarts. Plain JSON, written atomically. No database.

Framing plans are stored as plain dicts and rebuilt into framing.FramePlan
objects with plan_from_dict(), so a rerender reuses the exact crop keyframes.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from framing import FramePlan, Region

MANIFEST_NAME = "job.json"
MANIFEST_VERSION = 1

_LOCK = threading.RLock()


def manifest_path(job_dir: Path) -> Path:
    return job_dir / MANIFEST_NAME


def plan_to_dict(plan: FramePlan | None) -> dict | None:
    if plan is None:
        return None
    return {
        "layout": plan.layout,
        "note": plan.note,
        "diagnostics": plan.diagnostics,
        "regions": [
            {"w": r.w, "h": r.h, "x_keys": [list(k) for k in r.x_keys], "y_keys": [list(k) for k in r.y_keys],
             "track_id": r.track_id}
            for r in plan.regions
        ],
    }


def plan_from_dict(d: dict | None) -> FramePlan | None:
    if not d:
        return None
    regions = [
        Region(w=int(r["w"]), h=int(r["h"]),
               x_keys=[(float(t), float(v)) for t, v in r["x_keys"]],
               y_keys=[(float(t), float(v)) for t, v in r["y_keys"]],
               track_id=r.get("track_id"))
        for r in d.get("regions", [])
    ]
    return FramePlan(d["layout"], regions, d.get("note", ""), d.get("diagnostics") or {})


def new_manifest(job_id: str, url: str) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "job_id": job_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": None,
        "source": {"url": url, "title": None, "duration": None, "uploader": None},
        "stage": "queued",
        "status": "Waiting for a free slot...",
        "error": None,
        "transcription": {},
        "ranking": {},
        "selection": {},
        "glossary": [],
        "clips": [],
    }


def save(job_dir: Path, manifest: dict) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path(job_dir).with_suffix(".json.tmp")
    with _LOCK:
        tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, manifest_path(job_dir))


def modify(job_dir: Path, fn) -> dict:
    """
    Load the manifest, apply fn(manifest) and save, all under one lock, so a
    render worker finishing in the background and an edit from the UI never
    overwrite each other's changes.
    """
    with _LOCK:
        m = load(job_dir)
        if m is None:
            raise FileNotFoundError(f"no manifest in {job_dir}")
        fn(m)
        save(job_dir, m)
        return m


def update_clip(job_dir: Path, clip_id: str, fields: dict) -> dict:
    """Atomically merge `fields` into one clip record. Returns the manifest."""
    def apply(m):
        c = find_clip(m, clip_id)
        if c is None:
            raise KeyError(clip_id)
        c.update(fields)
    return modify(job_dir, apply)


def load(job_dir: Path) -> dict | None:
    p = manifest_path(job_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_all(output_dir: Path) -> list[dict]:
    """Every job on disk, newest first."""
    out = []
    if not output_dir.is_dir():
        return out
    for d in output_dir.iterdir():
        if d.is_dir():
            m = load(d)
            if m:
                out.append(m)
    out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return out


def find_clip(manifest: dict, clip_id: str) -> dict | None:
    for c in manifest.get("clips", []):
        if c["id"] == clip_id:
            return c
    return None


def public_view(manifest: dict, public_prefix: str) -> dict:
    """What the local UI receives: the manifest plus a URL per rendered clip."""
    view = dict(manifest)
    clips = []
    for c in manifest.get("clips", []):
        v = {k: val for k, val in c.items() if k not in ("plan",)}
        v["url"] = f"{public_prefix}/{c['files']['mp4']}?v={c.get('render_version', 0)}"
        v["filename"] = c["files"]["mp4"]
        clips.append(v)
    view["clips"] = clips
    view["clip_count"] = len(clips)
    view["library_count"] = sum(1 for c in clips if c.get("in_library"))
    return view
