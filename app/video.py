"""
FFmpeg operations: cut a section, convert to 1080x1920 (centre crop), burn
subtitles, export H.264 / AAC MP4.
"""

import subprocess
from pathlib import Path

OUT_W, OUT_H = 1080, 1920

# Subtitle look: big white text with a black outline, placed in the lower third.
ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,420,1

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
                    max_words: int = 4) -> Path:
    """
    Group word timestamps into short chunks (a few words each) and write an
    ASS file whose times are relative to the clip start.
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

    ass_path.write_text(ASS_HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return ass_path


def _ffmpeg_filter_path(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument (Windows-safe)."""
    s = str(p.resolve()).replace("\\", "/")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    return s


def render_clip(source: Path, start: float, end: float, ass_path: Path | None, out_path: Path) -> Path:
    """
    Cut [start, end] from `source`, scale+centre-crop to 1080x1920, burn
    subtitles, encode H.264 + AAC.
    """
    vf = (
        f"scale=w={OUT_W}:h={OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},"
        "setsar=1"
    )
    if ass_path is not None:
        vf += f",ass='{_ffmpeg_filter_path(ass_path)}'"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
        "-i", str(source),
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
