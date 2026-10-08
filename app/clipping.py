"""
The end-to-end pipeline and the per-clip rerender path.

run_job():
    YouTube URL -> source video -> audio -> transcript -> candidates -> ranking
    -> threshold selection -> per clip: framing plan, caption cues, ASS, render.
    Every step is recorded in output/<job_id>/job.json (see manifest.py).

rerender_clip():
    Rebuild the ASS file from the clip's stored cues/style/font and run the
    same ffmpeg render with the stored framing plan. Nothing else runs: no
    download, no Whisper, no ranking, no face analysis.

`job` is a plain dict shared with the web layer for live status.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import manifest as mf
from youtube import download_youtube_video
from transcription import extract_audio, transcribe
from ranking import build_candidates, rank_candidates, select_clips, MIN_CLIP_SCORE, DEDUP_MAX_OVERLAP, MIN_CLIPS_FLOOR
from video import render_clip
from captions import build_cues
from styles import render_ass, DEFAULT_STYLE, normalize_style
from fonts import DEFAULT_FONT
from framing import analyze_clip, plan_layout, save_debug_sheet, debug_enabled, LAYOUT_SPLIT, LAYOUT_CENTER

STAGES = {
    "queued": "Waiting for a free slot...",
    "importing": "Importing video...",
    "transcribing": "Transcribing...",
    "ranking": "Finding best moments...",
    "clipping": "Creating clips...",
    "done": "Done.",
    "error": "Error.",
}


def _set(job: dict, stage: str, detail: str = "") -> None:
    job["stage"] = stage
    job["status"] = STAGES[stage]
    job["detail"] = detail


def _sync(job: dict, manifest: dict, job_dir: Path) -> None:
    """Mirror live status into the manifest and persist it."""
    for k in ("stage", "status", "detail", "error", "transcription_note", "ranking_note", "framing_warning"):
        if k in job:
            manifest[k] = job[k]
    mf.save(job_dir, manifest)


def _clip_record(i: int, c: dict, plan, cues: list[dict], ass_info: dict, out: Path, layout: str) -> dict:
    return {
        "id": f"c{i:03d}",
        "index": i,
        "label": f"Clip {i:02d}",
        "start": c["start"],
        "end": c["end"],
        "duration": round(c["end"] - c["start"], 1),
        "score": c.get("score"),
        "below_threshold": bool(c.get("below_threshold", False)),
        "text": c["text"],
        "layout": layout,
        "layout_note": plan.note if plan else "",
        "plan": mf.plan_to_dict(plan),
        "cues": cues,
        "style": ass_info["style"],
        "font": ass_info["font_choice"],
        "font_family": ass_info["font_family"],
        "font_note": ass_info["font_note"],
        "edited": False,
        "in_library": False,
        "render_state": "done",
        "render_error": None,
        "render_version": 1,
        "files": {"mp4": out.name, "ass": out.with_suffix(".ass").name},
    }


def run_job(job: dict, url: str, job_dir: Path, public_prefix: str) -> None:
    """Run the whole pipeline for one URL. Errors are stored on the job and manifest."""
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = mf.new_manifest(job["id"], url)
    mf.save(job_dir, manifest)
    try:
        # 1. Source video
        _set(job, "importing")
        _sync(job, manifest, job_dir)
        source = download_youtube_video(url, job_dir)
        try:
            info = json.loads((job_dir / "source.json").read_text(encoding="utf-8"))
            manifest["source"].update({k: info.get(k) for k in ("title", "duration", "uploader")})
        except (OSError, json.JSONDecodeError):
            pass
        job["title"] = manifest["source"].get("title")

        # 2-3. Audio + transcript
        _set(job, "transcribing")
        _sync(job, manifest, job_dir)
        audio = extract_audio(source, job_dir / "audio.wav")
        transcript = transcribe(audio)
        (job_dir / "transcript.json").write_text(json.dumps(transcript, indent=1, ensure_ascii=False), encoding="utf-8")
        job["transcription_note"] = (
            f"transcribed with whisper {transcript.get('model', '?')} "
            f"(language {transcript.get('language', '?')}, {transcript.get('duration_seconds', 0):.0f} s)"
        )
        manifest["transcription"] = {k: transcript.get(k) for k in
                                     ("language", "language_probability", "model", "duration_seconds", "audio_seconds")}
        if not transcript["segments"]:
            raise RuntimeError("No speech was detected in this video.")

        # 4-5. Candidates + ranking + threshold selection
        _set(job, "ranking")
        _sync(job, manifest, job_dir)
        cands = build_candidates(transcript["segments"])
        if not cands:
            raise RuntimeError("The video is too short to produce 20-60 s clips.")
        ranked, note = rank_candidates(cands)
        job["ranking_note"] = note
        chosen = select_clips(ranked, floor=MIN_CLIPS_FLOOR)
        scores = sorted((c["score"] for c in ranked), reverse=True)
        manifest["ranking"] = {"note": note, "candidates": len(ranked),
                               "score_max": scores[0] if scores else None,
                               "score_median": scores[len(scores) // 2] if scores else None}
        manifest["selection"] = {
            "min_clip_score": MIN_CLIP_SCORE,
            "dedup_max_overlap": DEDUP_MAX_OVERLAP,
            "min_clips_floor": MIN_CLIPS_FLOOR,
            "accepted": sum(1 for c in chosen if not c.get("below_threshold")),
            "floor_added": sum(1 for c in chosen if c.get("below_threshold")),
        }
        (job_dir / "ranking.json").write_text(json.dumps(
            sorted(ranked, key=lambda c: c["score"], reverse=True), indent=1, ensure_ascii=False), encoding="utf-8")
        (job_dir / "selection.json").write_text(json.dumps(chosen, indent=1, ensure_ascii=False), encoding="utf-8")

        # 6-9. One clip at a time: framing, cues, ASS, render (bounded memory)
        _set(job, "clipping")
        _sync(job, manifest, job_dir)
        clips = []
        for i, c in enumerate(chosen, start=1):
            _set(job, "clipping", f"clip {i}/{len(chosen)}")
            try:
                analysis = analyze_clip(source, c["start"], c["end"])
                plan = plan_layout(analysis)
                if debug_enabled():
                    save_debug_sheet(analysis, plan, job_dir / f"clip_{i}_faces.jpg")
            except Exception as exc:  # noqa: BLE001
                plan = None
                job["framing_warning"] = f"face analysis failed, used centre crop ({exc.__class__.__name__}: {exc})"
            layout = plan.layout if plan else LAYOUT_CENTER

            cues = build_cues(transcript["words"], c["start"], c["end"])
            ass_path = job_dir / f"clip_{i}.ass"
            ass_info = render_ass(cues, ass_path, style=DEFAULT_STYLE, font_choice=DEFAULT_FONT,
                                  split_screen=(layout == LAYOUT_SPLIT))
            out = render_clip(source, c["start"], c["end"], ass_path, job_dir / f"clip_{i}.mp4", plan)
            rec = _clip_record(i, c, plan, cues, ass_info, out, layout)
            clips.append(rec)
            manifest["clips"] = clips
            job["clips"] = [_public_clip(r, public_prefix) for r in clips]
            _sync(job, manifest, job_dir)

        # 10. Done
        _set(job, "done")
        _sync(job, manifest, job_dir)
    except Exception as exc:  # noqa: BLE001
        job["error"] = f"{exc.__class__.__name__}: {exc}"
        job["traceback"] = traceback.format_exc()
        _set(job, "error", job["error"])
        _sync(job, manifest, job_dir)


def _public_clip(rec: dict, public_prefix: str) -> dict:
    v = {k: val for k, val in rec.items() if k != "plan"}
    v["url"] = f"{public_prefix}/{rec['files']['mp4']}?v={rec.get('render_version', 0)}"
    v["filename"] = rec["files"]["mp4"]
    return v


# --------------------------------------------------------------------------- #
# Rerender one clip from its manifest record
# --------------------------------------------------------------------------- #
def rerender_clip(job_dir: Path, manifest: dict, clip: dict) -> dict:
    """
    Regenerate the ASS file from clip["cues"] / clip["style"] / clip["font"]
    and rerender clip["files"]["mp4"] with the stored framing plan.
    State changes (render_state, render_version, font fields) are merged into
    job.json atomically so concurrent edits to other clips are never lost.
    """
    source = job_dir / "source.mp4"
    if not source.is_file():
        raise FileNotFoundError("source.mp4 is missing; the job folder cannot be rerendered")
    plan = mf.plan_from_dict(clip.get("plan"))
    layout = plan.layout if plan else LAYOUT_CENTER
    ass_path = job_dir / clip["files"]["ass"]
    out_path = job_dir / clip["files"]["mp4"]

    mf.update_clip(job_dir, clip["id"], {"render_state": "rendering", "render_error": None})
    try:
        ass_info = render_ass(clip["cues"], ass_path, style=normalize_style(clip.get("style")),
                              font_choice=clip.get("font") or DEFAULT_FONT,
                              split_screen=(layout == LAYOUT_SPLIT))
        tmp_out = out_path.with_name(out_path.stem + ".rerender.mp4")
        render_clip(source, clip["start"], clip["end"], ass_path, tmp_out, plan)
        os.replace(tmp_out, out_path)
        tmp_filter = tmp_out.with_suffix(".filter")
        if tmp_filter.exists():
            os.replace(tmp_filter, out_path.with_suffix(".filter"))
        fields = {
            "style": ass_info["style"], "font": ass_info["font_choice"],
            "font_family": ass_info["font_family"], "font_note": ass_info["font_note"],
            "render_state": "done", "render_version": int(clip.get("render_version", 0)) + 1,
        }
        clip.update(fields)
        mf.update_clip(job_dir, clip["id"], fields)
    except Exception as exc:  # noqa: BLE001
        err = f"{exc.__class__.__name__}: {exc}"
        clip.update({"render_state": "error", "render_error": err})
        mf.update_clip(job_dir, clip["id"], {"render_state": "error", "render_error": err})
        raise
    return clip
