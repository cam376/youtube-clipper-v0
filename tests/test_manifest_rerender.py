"""
Persistent job manifest and the per-clip rerender path.

Proves: the manifest survives reload with edits/style/font; a rerender
regenerates only the edited clip, reuses the stored framing plan exactly,
and never calls download / Whisper / ranking / face analysis.
"""

import copy
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

import clipping  # noqa: E402
import manifest as mf  # noqa: E402
from captions import build_cues, set_cue_text  # noqa: E402
from framing import FramePlan, Region, LAYOUT_SINGLE, LAYOUT_SPLIT, build_filtergraph  # noqa: E402
from styles import render_ass  # noqa: E402
from video import render_clip, _ffmpeg_filter_path  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_source(path: Path, seconds=4):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-f", "lavfi", "-i", "sine=frequency=440",
                    "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)


def sample_words(start, n=8):
    return [{"start": start + i * 0.3, "end": start + i * 0.3 + 0.25, "word": f"mot{i}"} for i in range(n)]


def sample_plan(layout=LAYOUT_SINGLE):
    if layout == LAYOUT_SPLIT:
        return FramePlan(LAYOUT_SPLIT, [Region(400, 356, [(0.0, 100.0), (1.5, 130.0)], [(0.0, 120.0)], 0),
                                       Region(400, 356, [(0.0, 760.0)], [(0.0, 150.0)], 1)], "t")
    return FramePlan(LAYOUT_SINGLE, [Region(404, 720, [(0.0, 700.0), (1.0, 650.0)], [(0.0, 0.0)], 0)], "t")


class ManifestRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kivro-manifest-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_dict_round_trip(self):
        plan = sample_plan(LAYOUT_SPLIT)
        back = mf.plan_from_dict(mf.plan_to_dict(plan))
        self.assertEqual(back, plan)
        self.assertIsNone(mf.plan_from_dict(None))

    def test_manifest_survives_reload_with_edits(self):
        m = mf.new_manifest("abc123abc123", "https://youtu.be/x")
        cues = build_cues(sample_words(0.0), 0.0, 3.0)
        m["clips"].append({"id": "c001", "index": 1, "label": "Clip 01", "start": 0.0, "end": 3.0, "cues": cues,
                           "style": "CLEAN", "font": "clean_sans", "edited": False, "in_library": False,
                           "plan": mf.plan_to_dict(sample_plan()), "files": {"mp4": "clip_1.mp4", "ass": "clip_1.ass"},
                           "render_version": 1})
        mf.save(self.tmp, m)

        def edit(mm):
            c = mf.find_clip(mm, "c001")
            c["cues"][0] = set_cue_text(c["cues"][0], "Arnaud Angeli a dit")
            c["edited"] = True
            c["style"] = "BOLD"
            c["font"] = "heavy_sans"
            c["in_library"] = True
            mm["glossary"] = ["Arnaud Angeli"]
        mf.modify(self.tmp, edit)

        reloaded = mf.load(self.tmp)
        c = mf.find_clip(reloaded, "c001")
        self.assertEqual(c["cues"][0]["text"], "Arnaud Angeli a dit")
        self.assertEqual((c["style"], c["font"], c["edited"], c["in_library"]), ("BOLD", "heavy_sans", True, True))
        self.assertEqual(reloaded["glossary"], ["Arnaud Angeli"])
        self.assertEqual(mf.plan_from_dict(c["plan"]), sample_plan())
        self.assertIsNotNone(reloaded["updated_at"])
        self.assertEqual(json.loads((self.tmp / "job.json").read_text(encoding="utf-8"))["job_id"], "abc123abc123")

    def test_update_clip_merges_without_losing_other_edits(self):
        m = mf.new_manifest("abc123abc123", "u")
        m["clips"] = [{"id": "c001", "render_state": "done", "style": "CLEAN"},
                      {"id": "c002", "render_state": "done", "style": "CLEAN"}]
        mf.save(self.tmp, m)
        mf.update_clip(self.tmp, "c001", {"render_state": "rendering"})
        mf.modify(self.tmp, lambda mm: mf.find_clip(mm, "c002").update({"style": "BOLD"}))
        mf.update_clip(self.tmp, "c001", {"render_state": "done", "render_version": 2})
        final = mf.load(self.tmp)
        self.assertEqual(mf.find_clip(final, "c002")["style"], "BOLD")
        self.assertEqual(mf.find_clip(final, "c001")["render_version"], 2)

    def test_load_all_lists_jobs_newest_first(self):
        for jid, created in (("aaaaaaaaaaaa", "2026-01-01T00:00:00"), ("bbbbbbbbbbbb", "2026-02-01T00:00:00")):
            m = mf.new_manifest(jid, "u")
            m["created_at"] = created
            mf.save(self.tmp / jid, m)
        self.assertEqual([m["job_id"] for m in mf.load_all(self.tmp)], ["bbbbbbbbbbbb", "aaaaaaaaaaaa"])

    def test_public_view_hides_plan_and_adds_urls(self):
        m = mf.new_manifest("abc123abc123", "u")
        m["clips"] = [{"id": "c001", "plan": {"layout": "x"}, "files": {"mp4": "clip_1.mp4", "ass": "clip_1.ass"},
                       "render_version": 3, "in_library": True}]
        v = mf.public_view(m, "/output/abc123abc123")
        self.assertNotIn("plan", v["clips"][0])
        self.assertEqual(v["clips"][0]["url"], "/output/abc123abc123/clip_1.mp4?v=3")
        self.assertEqual((v["clip_count"], v["library_count"]), (1, 1))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class RerenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="kivro-rerender-"))
        make_source(cls.tmp / "source.mp4")
        m = mf.new_manifest("abc123abc123", "u")
        for i, layout in ((1, LAYOUT_SPLIT), (2, LAYOUT_SINGLE)):
            plan = sample_plan(layout)
            cues = build_cues(sample_words(0.0), 0.0, 3.0)
            ass = cls.tmp / f"clip_{i}.ass"
            info = render_ass(cues, ass, "CLEAN", "clean_sans", split_screen=(layout == LAYOUT_SPLIT))
            render_clip(cls.tmp / "source.mp4", 0.0, 3.0, ass, cls.tmp / f"clip_{i}.mp4", plan)
            m["clips"].append({"id": f"c00{i}", "index": i, "label": f"Clip 0{i}", "start": 0.0, "end": 3.0,
                               "cues": cues, "style": "CLEAN", "font": info["font_choice"], "layout": layout,
                               "plan": mf.plan_to_dict(plan), "files": {"mp4": f"clip_{i}.mp4", "ass": f"clip_{i}.ass"},
                               "render_state": "done", "render_version": 1, "edited": False, "in_library": False})
        mf.save(cls.tmp, m)
        cls.m = m

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_rerender_only_touches_the_edited_clip_and_reuses_the_plan(self):
        m = mf.load(self.tmp)
        clip = mf.find_clip(m, "c001")
        other = self.tmp / "clip_2.mp4"
        other_bytes = other.read_bytes()
        plan_before = copy.deepcopy(clip["plan"])
        mtime_before = (self.tmp / "clip_1.mp4").stat().st_mtime_ns
        clip["cues"][0] = set_cue_text(clip["cues"][0], "Arnaud Angeli ici")
        clip["style"] = "KARAOKE"
        clip["font"] = "classic"

        forbidden = {
            "download_youtube_video": mock.patch.object(clipping, "download_youtube_video", side_effect=AssertionError("download called")),
            "transcribe": mock.patch.object(clipping, "transcribe", side_effect=AssertionError("whisper called")),
            "rank_candidates": mock.patch.object(clipping, "rank_candidates", side_effect=AssertionError("ranking called")),
            "analyze_clip": mock.patch.object(clipping, "analyze_clip", side_effect=AssertionError("face analysis called")),
            "plan_layout": mock.patch.object(clipping, "plan_layout", side_effect=AssertionError("framing called")),
        }
        for p in forbidden.values():
            p.start()
        try:
            clipping.rerender_clip(self.tmp, m, clip)
        finally:
            for p in forbidden.values():
                p.stop()

        self.assertNotEqual((self.tmp / "clip_1.mp4").stat().st_mtime_ns, mtime_before)
        self.assertEqual(other.read_bytes(), other_bytes)          # untouched
        self.assertEqual(clip["plan"], plan_before)                 # plan reused unchanged
        self.assertEqual(clip["render_version"], 2)
        self.assertEqual(clip["render_state"], "done")
        ass_text = (self.tmp / "clip_1.ass").read_text(encoding="utf-8")
        self.assertIn("Arnaud", ass_text)
        self.assertIn("\\k", ass_text)
        # the filtergraph written for the rerender is exactly the stored plan's
        plan = mf.plan_from_dict(plan_before)
        expected = build_filtergraph(plan, f"ass='{_ffmpeg_filter_path(self.tmp / 'clip_1.ass')}'")
        self.assertEqual((self.tmp / "clip_1.filter").read_text(encoding="utf-8"), expected)
        self.assertFalse((self.tmp / "clip_1.rerender.mp4").exists())
        # persisted
        on_disk = mf.find_clip(mf.load(self.tmp), "c001")
        self.assertEqual((on_disk["render_version"], on_disk["style"], on_disk["font"]), (2, "KARAOKE", "classic"))
        self.assertEqual((on_disk["start"], on_disk["end"]), (0.0, 3.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
