"""
Regression test: the complex filtergraph is handed to ffmpeg through a side
file, and the option used for that must match the installed ffmpeg.

    FFmpeg < 7 : -filter_complex_script FILE
    FFmpeg 7-8 : both forms accepted
    FFmpeg 9+  : -/filter_complex FILE   (old option removed)

Run from the repository root with the ffmpeg you want to validate on PATH:

    python -m unittest discover -s tests -v

The render tests use a synthetic source, so no YouTube access or face
detection is involved; they only exercise video.render_clip and the
filtergraph produced by framing.build_filtergraph.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import video  # noqa: E402
from framing import FramePlan, Region, LAYOUT_SPLIT, LAYOUT_SINGLE, LAYOUT_CENTER, build_filtergraph  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    streams = json.loads(out)["streams"]
    return {s["codec_type"]: s for s in streams}


def _make_source(path: Path, seconds: float = 3.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440",
         "-t", f"{seconds}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)],
        check=True,
    )
    return path


def _split_plan() -> FramePlan:
    # Two 9:8 windows whose x position moves between keyframes, like a real plan.
    top = Region(w=400, h=356, x_keys=[(0.0, 100.0), (1.0, 100.0), (2.0, 140.0), (3.0, 140.0)],
                 y_keys=[(0.0, 120.0)], track_id=0)
    bot = Region(w=400, h=356, x_keys=[(0.0, 760.0)], y_keys=[(0.0, 150.0), (3.0, 170.0)], track_id=1)
    return FramePlan(LAYOUT_SPLIT, [top, bot], "test")


def _single_plan() -> FramePlan:
    return FramePlan(LAYOUT_SINGLE, [Region(w=404, h=720, x_keys=[(0.0, 700.0), (3.0, 650.0)],
                                            y_keys=[(0.0, 0.0)], track_id=0)], "test")


class FilterScriptArgsTest(unittest.TestCase):
    def test_option_chosen_by_major_version(self):
        script = Path("x.filter")
        self.assertEqual(video.filter_script_args(script, major=6)[0], video.FILTER_SCRIPT_LEGACY)
        self.assertEqual(video.filter_script_args(script, major=7)[0], video.FILTER_SCRIPT_MODERN)
        self.assertEqual(video.filter_script_args(script, major=8)[0], video.FILTER_SCRIPT_MODERN)
        self.assertEqual(video.filter_script_args(script, major=9)[0], video.FILTER_SCRIPT_MODERN)
        with mock.patch.object(video, "ffmpeg_major_version", return_value=None):
            self.assertEqual(video.filter_script_args(script)[0], video.FILTER_SCRIPT_MODERN)
        self.assertEqual(video.filter_script_args(script, major=9), ["-/filter_complex", "x.filter"])

    def test_major_version_parsing(self):
        cases = {
            "ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026": 9,
            "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023": 6,
            "ffmpeg version n7.1 Copyright": 7,
            "ffmpeg version N-127233-g452820cba6-20261007 Copyright": None,
        }
        for banner, expected in cases.items():
            with self.subTest(banner=banner):
                video.ffmpeg_major_version.cache_clear()
                fake = subprocess.CompletedProcess(["ffmpeg"], 0, stdout=banner + "\n", stderr="")
                with mock.patch.object(video.subprocess, "run", return_value=fake):
                    self.assertEqual(video.ffmpeg_major_version(), expected)
        video.ffmpeg_major_version.cache_clear()


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class RenderWithInstalledFfmpegTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="clipper-ffmpeg-test-"))
        cls.source = _make_source(cls.tmp / "source.mp4")
        words = [{"start": i * 0.5, "end": i * 0.5 + 0.4, "word": f"w{i}"} for i in range(6)]
        cls.ass = video.build_subtitles(words, 0.0, 3.0, cls.tmp / "sub.ass", split_screen=True)
        video.ffmpeg_major_version.cache_clear()
        cls.major = video.ffmpeg_major_version()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _assert_vertical_mp4(self, path: Path):
        self.assertTrue(path.is_file() and path.stat().st_size > 0, path)
        streams = _probe(path)
        self.assertEqual((streams["video"]["width"], streams["video"]["height"]), (1080, 1920))
        self.assertEqual(streams["video"]["codec_name"], "h264")
        self.assertEqual(streams["audio"]["codec_name"], "aac")

    def test_split_screen_renders_via_side_file(self):
        plan = _split_plan()
        out = self.tmp / "split.mp4"
        video.render_clip(self.source, 0.0, 3.0, self.ass, out, plan)
        self._assert_vertical_mp4(out)
        script = out.with_suffix(".filter")
        self.assertTrue(script.is_file())
        ass_filter = f"ass='{video._ffmpeg_filter_path(self.ass)}'"
        self.assertEqual(script.read_text(encoding="utf-8"), build_filtergraph(plan, ass_filter))
        self.assertIn("vstack=inputs=2", script.read_text(encoding="utf-8"))

    def test_single_person_renders_via_side_file(self):
        out = self.tmp / "single.mp4"
        video.render_clip(self.source, 0.0, 3.0, self.ass, out, _single_plan())
        self._assert_vertical_mp4(out)
        self.assertTrue(out.with_suffix(".filter").is_file())

    def test_center_crop_path_unchanged(self):
        out = self.tmp / "center.mp4"
        video.render_clip(self.source, 0.0, 3.0, self.ass, out, None)
        self._assert_vertical_mp4(out)
        self.assertFalse(out.with_suffix(".filter").exists())
        out2 = self.tmp / "center2.mp4"
        video.render_clip(self.source, 0.0, 3.0, self.ass, out2, FramePlan(LAYOUT_CENTER, [], "test"))
        self._assert_vertical_mp4(out2)
        self.assertFalse(out2.with_suffix(".filter").exists())

    def test_wrong_syntax_falls_back_to_the_other_form(self):
        # Pretend the installed ffmpeg is from the other era so the first
        # attempt may be rejected; the retry must still produce the clip.
        wrong_major = 6 if (self.major is None or self.major >= 7) else 9
        out = self.tmp / "fallback.mp4"
        with mock.patch.object(video, "ffmpeg_major_version", return_value=wrong_major):
            video.render_clip(self.source, 0.0, 3.0, self.ass, out, _split_plan())
        self._assert_vertical_mp4(out)

    def test_bad_filtergraph_reports_ffmpeg_error(self):
        out = self.tmp / "bad.mp4"
        broken = FramePlan(LAYOUT_SINGLE, [Region(w=0, h=0, x_keys=[(0.0, 0.0)], y_keys=[(0.0, 0.0)])], "test")
        with self.assertRaises(RuntimeError) as ctx:
            video.render_clip(self.source, 0.0, 1.0, None, out, broken)
        self.assertIn("ffmpeg failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
