"""
FFmpeg operations: cut a section, convert to 1080x1920 (centre crop), burn
subtitles, export H.264 / AAC MP4.
"""

import functools
import re
import subprocess
from pathlib import Path

OUT_W, OUT_H = 1080, 1920

# How a filtergraph file is handed to ffmpeg. FFmpeg 7.0 introduced the generic
# "-/option FILE" form (load the option value from a file) and deprecated
# "-filter_complex_script"; FFmpeg 9 removed the old option entirely.
FILTER_SCRIPT_MODERN = "-/filter_complex"
FILTER_SCRIPT_LEGACY = "-filter_complex_script"

# Subtitle look: big white text with a black outline, placed in the lower third.
# For split-screen clips the same style is used with Alignment 5 (middle-centre),
# which puts the text on the seam between the two speakers, away from both faces.
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,{alignment},60,60,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def build_subtitles(words: list[dict], clip_start: float, clip_end: float, ass_path: Path,
                    max_words: int = 4, split_screen: bool = False) -> Path:
    """
    Group word timestamps into short chunks (a few words each) and write an
    ASS file whose times are relative to the clip start.
    split_screen=True centres the captions vertically (safe zone between faces).
    """
    clip_words = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    lines = []
    chunk: list[dict] = []

    def flush():
        if not chunk:
            return
        start = chunk[0]["start"] - clip_start
        end = chunk[-1]["end"] - clip_start
        text = _escape_ass(" ".join(w["word"] for w in chunk))
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
        chunk.clear()

    for w in clip_words:
        if chunk and (len(chunk) >= max_words or w["start"] - chunk[-1]["end"] > 1.0):
            flush()
        chunk.append(w)
        if w["word"][-1:] in ".!?,":
            flush()
    flush()

    header = ASS_HEADER.format(OUT_W=OUT_W, OUT_H=OUT_H, alignment=5 if split_screen else 2)
    ass_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return ass_path


def _ffmpeg_filter_path(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument (Windows-safe)."""
    s = str(p.resolve()).replace("\\", "/")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    return s


@functools.lru_cache(maxsize=1)
def ffmpeg_major_version() -> int | None:
    """Major version of the ffmpeg on PATH, or None for git/unparseable builds."""
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.match(r"ffmpeg version n?(\d+)\.", out)
    return int(m.group(1)) if m else None


def filter_script_args(script: Path, major: int | None = None) -> list[str]:
    """
    Arguments that make ffmpeg read the complex filtergraph from `script`.
    FFmpeg < 7 only knows -filter_complex_script; 7 and 8 accept both;
    9+ only knows -/filter_complex. Unknown versions get the modern form,
    and render_clip retries with the other form if ffmpeg rejects it.
    """
    if major is None:
        major = ffmpeg_major_version()
    option = FILTER_SCRIPT_LEGACY if (major is not None and major < 7) else FILTER_SCRIPT_MODERN
    return [option, str(script)]


def _run_with_filter_script(script: Path, build_cmd) -> None:
    """Run build_cmd(filter_args); on 'Unrecognized option' retry with the other syntax."""
    primary = filter_script_args(script)
    other = FILTER_SCRIPT_LEGACY if primary[0] == FILTER_SCRIPT_MODERN else FILTER_SCRIPT_MODERN
    stderr = ""
    for args in (primary, [other, str(script)]):
        res = subprocess.run(build_cmd(args), capture_output=True, text=True)
        if res.returncode == 0:
            return
        stderr = res.stderr.strip()
        if "Unrecognized option" not in stderr:
            break
    raise RuntimeError(f"ffmpeg failed: {stderr[-2000:]}")


def render_clip(source: Path, start: float, end: float, ass_path: Path | None, out_path: Path,
                plan=None) -> Path:
    """
    Cut [start, end] from `source`, convert to 1080x1920, burn subtitles,
    encode H.264 + AAC.

    plan: optional framing.FramePlan. None or CENTER_CROP keeps the original
    centre-crop behaviour; SINGLE_PERSON / TWO_PERSON use face-based crops
    (the filtergraph is written to a side file so long expressions never hit
    the command-line length limit).
    """
    ass_filter = f"ass='{_ffmpeg_filter_path(ass_path)}'" if ass_path is not None else None

    def build_cmd(video_args: list[str]) -> list[str]:
        return [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(source),
            *video_args,
            "-r", "30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "+faststart",
            str(out_path),
        ]

    if plan is not None and plan.layout != "CENTER_CROP":
        from framing import build_filtergraph
        script = out_path.with_suffix(".filter")
        script.write_text(build_filtergraph(plan, ass_filter), encoding="utf-8")
        _run_with_filter_script(script, lambda fargs: build_cmd([*fargs, "-map", "[v]", "-map", "0:a?"]))
        return out_path

    vf = (
        f"scale=w={OUT_W}:h={OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},"
        "setsar=1"
    )
    if ass_filter:
        vf += f",{ass_filter}"
    subprocess.run(build_cmd(["-vf", vf]), check=True)
    return out_path
