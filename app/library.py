"""
Static client library export.

export_library() takes the clips marked in_library in a job manifest and
writes a self-contained folder:

    client_libraries/<library_id>/
        index.html        the client page (library data inlined, works from
                          a plain static host or a local http.server)
        library.json      the same client-facing data, for the record
        previews/cNN.mp4  540x960 web previews (optional "Kivro Preview" mark)
        previews/cNN.jpg  poster frames
        assets/           style.css, app.js

HD masters in output/<job_id>/ are never modified. The mapping from library
clip ids back to job clips is written to output/<job_id>/export_<library_id>.json
(operator-only) and never into the library folder.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from fonts import resolve_font
from video import _ffmpeg_filter_path

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "client_library"
PREVIEW_W, PREVIEW_H = 540, 960
WATERMARK_TEXT = "KIVRO PREVIEW"


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return s[:32] or "client"


def _watermark_ass(path: Path) -> Path:
    font = resolve_font("clean_sans")["family"]
    content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PREVIEW_W}
PlayResY: {PREVIEW_H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: WM,{font},22,&H55FFFFFF,&H55FFFFFF,&H55000000,&H00000000,-1,0,0,0,100,100,2,0,1,1,0,9,20,20,28,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,9:59:59.00,WM,,0,0,0,,{WATERMARK_TEXT}
"""
    path.write_text(content, encoding="utf-8")
    return path


def render_preview(hd_path: Path, out_path: Path, watermark_ass: Path | None) -> Path:
    vf = f"scale={PREVIEW_W}:{PREVIEW_H}:flags=lanczos,setsar=1"
    if watermark_ass is not None:
        vf += f",ass='{_ffmpeg_filter_path(watermark_ass)}'"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(hd_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-profile:v", "main", "-level", "4.0",
        "-maxrate", "1600k", "-bufsize", "3200k", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-ac", "2",
        "-movflags", "+faststart",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"preview render failed: {res.stderr.strip()[-1500:]}")
    return out_path


def render_poster(preview_path: Path, out_path: Path) -> Path:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "1", "-i", str(preview_path),
           "-frames:v", "1", "-q:v", "4", str(out_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # very short clip: take the first frame instead
        cmd[cmd.index("-ss") + 1] = "0"
        subprocess.run(cmd, check=True)
    return out_path


def build_index_html(library: dict) -> str:
    html = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    data = json.dumps(library, ensure_ascii=False).replace("</", "<\\/")
    return (html.replace("{{TITLE}}", _h(library["title"]))
                .replace("{{CLIENT}}", _h(library["client_name"]))
                .replace("{{LIBRARY_JSON}}", data))


def _h(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def export_library(job_dir: Path, manifest: dict, options: dict, libraries_root: Path,
                   progress=None) -> dict:
    """
    options: client_name, title, show_price (bool), price_per_clip (float|None),
             currency (str), whatsapp_number (str|None), watermark (bool)
    Returns the library summary (also appended to manifest["libraries"]).
    """
    selected = [c for c in manifest.get("clips", []) if c.get("in_library")]
    if not selected:
        raise ValueError("No clips are marked for the client library.")
    for c in selected:
        if not (job_dir / c["files"]["mp4"]).is_file():
            raise FileNotFoundError(f"{c['files']['mp4']} is missing")

    client_name = (options.get("client_name") or "Client").strip()
    title = (options.get("title") or f"{client_name}'s Content Library").strip()
    library_id = f"{slugify(client_name)}-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    lib_dir = libraries_root / library_id
    (lib_dir / "previews").mkdir(parents=True, exist_ok=True)
    (lib_dir / "assets").mkdir(parents=True, exist_ok=True)

    watermark = bool(options.get("watermark", True))
    wm_ass = _watermark_ass(job_dir / "preview_watermark.ass") if watermark else None

    clips_out, export_map = [], []
    for n, c in enumerate(selected, start=1):
        if progress:
            progress(f"preview {n}/{len(selected)}")
        cid = f"c{n:02d}"
        hd = job_dir / c["files"]["mp4"]
        mp4 = render_preview(hd, lib_dir / "previews" / f"{cid}.mp4", wm_ass)
        jpg = render_poster(mp4, lib_dir / "previews" / f"{cid}.jpg")
        clips_out.append({
            "id": cid,
            "label": f"Clip {n:02d}",
            "preview": f"previews/{mp4.name}",
            "poster": f"previews/{jpg.name}",
            "duration": c.get("duration"),
        })
        export_map.append({"library_clip_id": cid, "label": f"Clip {n:02d}",
                           "job_clip_id": c["id"], "hd_file": c["files"]["mp4"]})

    show_price = bool(options.get("show_price", False))
    price = options.get("price_per_clip")
    library = {
        "library_id": library_id,
        "client_name": client_name,
        "title": title,
        "subtitle": f"{len(clips_out)} clip{'s' if len(clips_out) != 1 else ''} prepared for you",
        "created_at": time.strftime("%Y-%m-%d"),
        "show_price": show_price,
        "price_per_clip": float(price) if (show_price and price not in (None, "")) else None,
        "currency": (options.get("currency") or "EUR") if show_price else None,
        "whatsapp_number": re.sub(r"[^0-9]", "", options.get("whatsapp_number") or "") or None,
        "watermarked_previews": watermark,
        "clips": clips_out,
    }
    (lib_dir / "library.json").write_text(json.dumps(library, indent=1, ensure_ascii=False), encoding="utf-8")
    (lib_dir / "index.html").write_text(build_index_html(library), encoding="utf-8")
    for asset in ("style.css", "app.js"):
        shutil.copy(TEMPLATE_DIR / asset, lib_dir / "assets" / asset)

    (job_dir / f"export_{library_id}.json").write_text(json.dumps({
        "library_id": library_id, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "library_dir": str(lib_dir), "clips": export_map,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    summary = {"library_id": library_id, "path": str(lib_dir), "created_at": library["created_at"],
               "clip_count": len(clips_out), "client_name": client_name, "title": title}
    manifest.setdefault("libraries", []).append(summary)
    return summary
