"""
YouTube import provider.

One isolated function: download a video from a YouTube URL into a local file.

This is intended only for videos you own or have permission to process.
It uses yt-dlp with its default, unauthenticated behaviour: no cookies,
no login, no DRM / private / paywall bypass. If YouTube refuses the download
(private video, members-only, age gate, etc.), the error is surfaced as-is.
"""

from pathlib import Path

import yt_dlp


def download_youtube_video(url: str, dest_dir: Path, max_height: int = 1080) -> Path:
    """Download `url` into `dest_dir` and return the path of the MP4 file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / "source.%(ext)s")

    ydl_opts = {
        # Best MP4 video up to max_height + best audio, merged into an MP4 by ffmpeg.
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
            f"/best[height<={max_height}][ext=mp4]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    candidates = sorted(dest_dir.glob("source.*"))
    mp4 = [p for p in candidates if p.suffix == ".mp4"]
    if mp4:
        return mp4[0]
    if candidates:
        return candidates[0]
    raise RuntimeError("yt-dlp finished but no source file was written")
