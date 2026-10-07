"""
The end-to-end pipeline:
    YouTube URL -> source video -> audio -> transcript -> candidates -> ranking
    -> cut / crop / subtitle / export -> list of MP4 files.

`run_job` mutates `job` (a plain dict) so the web layer can report status.
"""

import json
import traceback
from pathlib import Path

from youtube import download_youtube_video
from transcription import extract_audio, transcribe
from ranking import build_candidates, rank_candidates, select_best
from video import build_subtitles, render_clip
from framing import analyze_clip, plan_layout, save_debug_sheet, debug_enabled, LAYOUT_SPLIT, LAYOUT_CENTER

STAGES = {
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


def run_job(job: dict, url: str, job_dir: Path, public_prefix: str) -> None:
    """Run the whole pipeline for one URL. Errors are stored on the job."""
    try:
        job_dir.mkdir(parents=True, exist_ok=True)

        # 1. Source video
        _set(job, "importing")
        source = download_youtube_video(url, job_dir)

        # 2-3. Audio + transcript
        _set(job, "transcribing")
        audio = extract_audio(source, job_dir / "audio.wav")
        transcript = transcribe(audio)
        (job_dir / "transcript.json").write_text(json.dumps(transcript, indent=1), encoding="utf-8")
        job["transcription_note"] = (
            f"transcribed with whisper {transcript.get('model', '?')} "
            f"(language {transcript.get('language', '?')}, {transcript.get('duration_seconds', 0):.0f} s)"
        )
        if not transcript["segments"]:
            raise RuntimeError("No speech was detected in this video.")

        # 4-5. Candidates + ranking + selection
        _set(job, "ranking")
        cands = build_candidates(transcript["segments"])
        if not cands:
            raise RuntimeError("The video is too short to produce 20-60 s clips.")
        ranked, note = rank_candidates(cands)
        chosen = select_best(ranked)
        job["ranking_note"] = note
        (job_dir / "selection.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")

        # 6-9. Cut, crop, subtitle, export
        _set(job, "clipping")
        clips = []
        for i, c in enumerate(chosen, start=1):
            _set(job, "clipping", f"clip {i}/{len(chosen)}")

            # Face-aware framing: one layout per clip. Any failure in the
            # analysis falls back to the original centre crop.
            try:
                analysis = analyze_clip(source, c["start"], c["end"])
                plan = plan_layout(analysis)
                if debug_enabled():
                    save_debug_sheet(analysis, plan, job_dir / f"clip_{i}_faces.jpg")
            except Exception as exc:  # noqa: BLE001
                analysis, plan = None, None
                job["framing_warning"] = f"face analysis failed, used centre crop ({exc.__class__.__name__}: {exc})"
            layout = plan.layout if plan else LAYOUT_CENTER

            ass = build_subtitles(transcript["words"], c["start"], c["end"], job_dir / f"clip_{i}.ass",
                                  split_screen=(layout == LAYOUT_SPLIT))
            out = render_clip(source, c["start"], c["end"], ass, job_dir / f"clip_{i}.mp4", plan)
            clips.append({
                "layout": layout,
                "layout_note": plan.note if plan else "",
                "index": i,
                "start": c["start"],
                "end": c["end"],
                "duration": round(c["end"] - c["start"], 1),
                "score": c.get("score"),
                "text": c["text"],
                "url": f"{public_prefix}/{out.name}",
                "filename": out.name,
            })
        job["clips"] = clips

        # 10. Done
        _set(job, "done")
    except Exception as exc:  # noqa: BLE001
        job["error"] = f"{exc.__class__.__name__}: {exc}"
        job["traceback"] = traceback.format_exc()
        _set(job, "error", job["error"])
