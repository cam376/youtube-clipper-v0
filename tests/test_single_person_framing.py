"""
SINGLE_PERSON (face-centred) framing regression.

A speaker clearly on the right of a 16:9 source must pull the 9:16 crop to
the right; the crop must never sit at the geometric centre; a short
detection dropout must not snap the crop back to the centre; the primary
track must be the actual speaker, not a small always-present face.

The plan-level tests need no video. The end-to-end test builds a synthetic
source with a real face image and runs the bundled YuNet detector.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import framing  # noqa: E402
import manifest as mf  # noqa: E402
from framing import FaceAnalysis, Track, plan_layout, LAYOUT_SINGLE, LAYOUT_CENTER  # noqa: E402

try:
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    HAVE_CV = True
except ImportError:  # pragma: no cover
    HAVE_CV = False
HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
FACE_IMAGE = ROOT / "tests" / "data" / "face.jpg"

W, H = 1280, 720
CROP_W = 404                      # _even(720 * 9 / 16)
GEOMETRIC_X = (W - CROP_W) / 2    # 438


def _track(tid, cx, present, size=90.0, cy=300.0):
    t = Track(id=tid)
    for i in present:
        t.boxes[i] = (cx - size / 2, cy - size / 2, size, size)
    t.last_idx = max(present)
    return t


def _analysis(tracks, n):
    return FaceAnalysis(width=W, height=H, sample_fps=framing.SAMPLE_FPS, n_samples=n, tracks=tracks,
                        frames=[], scale=2.0, detector_name="test")


class PlanLevelTest(unittest.TestCase):
    def test_right_side_speaker_moves_crop_right(self):
        n = 60                                  # 30 s
        a = _analysis([_track(0, 1010, list(range(n)))], n)
        plan = plan_layout(a)
        self.assertEqual(plan.layout, LAYOUT_SINGLE)
        r = plan.regions[0]
        expected = 1010 - CROP_W / 2            # 808
        for t, x in r.x_keys:
            self.assertAlmostEqual(x, expected, delta=3, msg=f"crop x at {t}s")
            self.assertGreater(x, GEOMETRIC_X + 300)
        d = plan.diagnostics
        self.assertFalse(d["geometric_centre_used"])
        self.assertEqual(d["primary_track"], 0)
        self.assertAlmostEqual(d["median_face_x"], 1010, delta=1)
        self.assertEqual(d["fallback_samples"], 0)
        self.assertEqual(d["clamped_right_samples"], 0)
        self.assertAlmostEqual(d["face_in_crop_median"], 0.5, places=3)
        self.assertEqual(len(d["smoothed_crop_x"]), n)
        self.assertTrue(all(abs(x - expected) < 3 for x in d["smoothed_crop_x"]))

    def test_dropout_holds_position_instead_of_snapping_to_centre(self):
        n = 60
        present = [i for i in range(n) if not (20 <= i < 28)]   # 4 s gap at 10-14 s
        a = _analysis([_track(0, 1010, present)], n)
        plan = plan_layout(a)
        d = plan.diagnostics
        self.assertEqual(d["fallback_samples"], 8)
        expected = 1010 - CROP_W / 2
        for i in range(20, 28):
            self.assertAlmostEqual(d["smoothed_crop_x"][i], expected, delta=3, msg=f"sample {i} during dropout")
        for t, x in plan.regions[0].x_keys:
            self.assertGreater(x, GEOMETRIC_X + 300)

    def test_leading_gap_uses_first_detection_not_centre(self):
        n = 60
        a = _analysis([_track(0, 1010, list(range(6, n)))], n)   # speaker appears at 3 s
        d = plan_layout(a).diagnostics
        self.assertEqual(d["fallback_samples"], 6)
        self.assertAlmostEqual(d["smoothed_crop_x"][0], 1010 - CROP_W / 2, delta=3)

    def test_primary_is_the_large_speaker_not_a_small_constant_face(self):
        # "second face dominated" branch: a small face present in every sample
        # (logo, picture-in-picture) and the real speaker, larger, seen in 45 %
        # of samples confined to the first half (span < 70 %, so not a second
        # person). The crop must centre on the speaker, not the small face.
        n = 60
        logo = _track(1, 300, list(range(n)), size=40.0)
        speaker = _track(0, 1000, list(range(27)), size=140.0)                     # 45 %, span 45 %
        plan = plan_layout(_analysis([logo, speaker], n))
        self.assertEqual(plan.layout, LAYOUT_SINGLE, plan.note)
        self.assertIn("dominated", plan.note)
        self.assertEqual(plan.diagnostics["primary_track"], 0)
        self.assertGreater(plan.regions[0].x_keys[0][1], GEOMETRIC_X + 250)

    def test_primary_among_one_persistent_and_one_sporadic_face(self):
        n = 60
        speaker = _track(0, 1000, list(range(n)), size=140.0)
        passerby = _track(1, 300, list(range(10)), size=120.0)                     # 17 %
        plan = plan_layout(_analysis([passerby, speaker], n))
        self.assertEqual(plan.layout, LAYOUT_SINGLE, plan.note)
        self.assertEqual(plan.diagnostics["primary_track"], 0)

    # ---------------------------------------------------------------- handoff
    def _two_fragments(self, n=80, cut=20, xa=400.0, xb=1000.0, size_a=90.0, size_b=90.0):
        a = _track(1, xa, list(range(cut)), size=size_a)          # first section
        b = _track(0, xb, list(range(cut, n)), size=size_b)       # remainder (primary: longer)
        return _analysis([b, a], n)

    def test_handoff_follows_visible_face_before_the_cut(self):
        n, cut = 80, 20                                            # 40 s clip, edit at 10 s
        plan = plan_layout(self._two_fragments(n, cut))
        self.assertEqual(plan.layout, LAYOUT_SINGLE, plan.note)
        d = plan.diagnostics
        self.assertEqual(d["primary_track"], 0)
        self.assertEqual([h["track"] for h in d["handoff_tracks"]], [1])
        self.assertEqual(d["handoff_samples"], cut)
        self.assertEqual(d["interpolated_samples"], 0)
        self.assertEqual(d["handoff_boundaries"], [cut])
        for i in range(cut):
            self.assertAlmostEqual(d["smoothed_crop_x"][i], 400 - CROP_W / 2, delta=3, msg=f"sample {i}")
            self.assertEqual(d["position_source"][i], "handoff:1")
        for i in range(cut, n):
            self.assertAlmostEqual(d["smoothed_crop_x"][i], 1000 - CROP_W / 2, delta=3, msg=f"sample {i}")
            self.assertEqual(d["position_source"][i], "primary")
        # the crop steps at the cut instead of panning across it
        keys = dict(plan.regions[0].x_keys)
        self.assertAlmostEqual(keys[round(cut / framing.SAMPLE_FPS - 0.02, 3)], 400 - CROP_W / 2, delta=3)
        self.assertAlmostEqual(keys[cut / framing.SAMPLE_FPS], 1000 - CROP_W / 2, delta=3)
        self.assertIn("handoff to #1", plan.note)

    def test_handoff_rejects_copresent_second_person(self):
        n = 80
        primary = _track(0, 1000, list(range(n)), size=90.0)
        other = _track(1, 400, list(range(10, 40)), size=90.0)     # fully co-present, 37 % (not a second persistent face)
        plan = plan_layout(_analysis([primary, other], n))
        self.assertEqual(plan.layout, LAYOUT_SINGLE, plan.note)
        d = plan.diagnostics
        self.assertEqual(d["handoff_tracks"], [])
        self.assertIn("co-present", d["handoff_rejected"][0]["reason"])
        self.assertTrue(all(abs(x - (1000 - CROP_W / 2)) < 3 for x in d["smoothed_crop_x"]))

    def test_handoff_rejects_tiny_face_and_too_short_tracks(self):
        n, cut = 80, 20
        primary = _track(0, 1000, list(range(cut, n)), size=120.0)
        logo = _track(1, 400, list(range(cut)), size=24.0)         # 0.2x the primary
        blip = _track(2, 600, [3, 4, 5], size=120.0)               # 3 samples
        plan = plan_layout(_analysis([primary, logo, blip], n))
        d = plan.diagnostics
        self.assertEqual(d["handoff_tracks"], [])
        reasons = {r["track"]: r["reason"] for r in d["handoff_rejected"]}
        self.assertIn("face size", reasons[1])
        self.assertIn("fewer than", reasons[2])
        self.assertEqual(d["handoff_samples"], 0)
        self.assertEqual(d["interpolated_samples"], cut)
        self.assertIn("#1", d["other_visible"]["0"])                # visible but not followed, reported
        for i in range(cut):                                        # held at the primary's position, as before
            self.assertAlmostEqual(d["smoothed_crop_x"][i], 1000 - CROP_W / 2, delta=3)

    def test_solo_track_has_no_handoff_and_same_result(self):
        n = 60
        plan = plan_layout(_analysis([_track(0, 1010, list(range(n)))], n))
        d = plan.diagnostics
        self.assertEqual((d["handoff_tracks"], d["handoff_rejected"], d["handoff_boundaries"]), ([], [], []))
        self.assertEqual(set(d["position_source"]), {"primary"})

    def test_centre_crop_is_explicit_when_no_reliable_face(self):
        plan = plan_layout(_analysis([_track(0, 1010, list(range(5)))], 60))   # 8 % coverage
        self.assertEqual(plan.layout, LAYOUT_CENTER)
        self.assertTrue(plan.diagnostics["geometric_centre_used"])

    def test_diagnostics_survive_the_manifest(self):
        a = _analysis([_track(0, 1010, list(range(60)))], 60)
        plan = plan_layout(a)
        back = mf.plan_from_dict(json.loads(json.dumps(mf.plan_to_dict(plan))))
        self.assertEqual(back.diagnostics["primary_track"], 0)
        self.assertEqual(back.regions, plan.regions)


@unittest.skipUnless(HAVE_CV and HAVE_FFMPEG and FACE_IMAGE.is_file(), "needs OpenCV, ffmpeg and tests/data/face.jpg")
class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="kivro-single-"))
        face = cv2.resize(cv2.imread(str(FACE_IMAGE)), (220, 220))
        fps, seconds = 30, 24
        raw = cls.tmp / "raw.avi"
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        for k in range(seconds * fps):
            t = k / fps
            frame = np.full((H, W, 3), (70, 80, 90), np.uint8)
            visible = not (9.0 <= t < 12.0)          # 3 s dropout
            if visible:
                x0 = int(1010 - 110 + 20 * np.sin(t / 4))
                frame[200:420, x0:x0 + 220] = face
            vw.write(frame)
        vw.release()
        cls.source = cls.tmp / "source.mp4"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
                        "-f", "lavfi", "-i", "sine=frequency=330", "-t", str(seconds),
                        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        str(cls.source)], check=True)
        cls.analysis = framing.analyze_clip(cls.source, 0.0, float(seconds))
        cls.plan = plan_layout(cls.analysis)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_detector_finds_the_right_side_speaker(self):
        self.assertEqual(self.plan.layout, LAYOUT_SINGLE, self.plan.note)
        d = self.plan.diagnostics
        self.assertGreater(d["coverage"], 0.8)
        self.assertAlmostEqual(d["median_face_x"], 1010, delta=25)

    def test_crop_follows_the_face_and_holds_through_dropout(self):
        d = self.plan.diagnostics
        for i, x in enumerate(d["smoothed_crop_x"]):
            self.assertGreater(x, GEOMETRIC_X + 300, f"sample {i}: crop x {x} is near the geometric centre")
            self.assertAlmostEqual(x, 1010 - CROP_W / 2, delta=40, msg=f"sample {i}")
        self.assertGreaterEqual(d["fallback_samples"], 4)
        self.assertFalse(d["geometric_centre_used"])

    def test_rendered_output_keeps_the_face_near_centre(self):
        # The speaker stays on the right of the 16:9 source for the whole clip.
        # In the rendered 1080x1920 output the face must sit in the centre
        # band (420-660 px of 1080), with the average close to 540, at every
        # sampled time. Uses whatever ffmpeg is on PATH (FFmpeg 9 on the
        # Windows machine), so this also covers the -/filter_complex path.
        from video import render_clip
        out = render_clip(self.source, 0.0, 24.0, None, self.tmp / "out.mp4", self.plan)
        det = framing._Detector(1080, 1920)
        positions = []
        for tt in (1.0, 3.0, 6.0, 8.0, 13.0, 15.0, 18.0, 22.0):
            png = self.tmp / f"f{tt}.png"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(tt), "-i", str(out),
                            "-frames:v", "1", str(png)], check=True)
            faces = det.detect(cv2.imread(str(png)))
            self.assertTrue(faces, f"no face in output at {tt}s")
            cx = faces[0][0] + faces[0][2] / 2
            positions.append(cx)
            self.assertGreaterEqual(cx, 420, f"face too far left in output at {tt}s: {cx:.0f}")
            self.assertLessEqual(cx, 660, f"face too far right in output at {tt}s: {cx:.0f}")
        mean = sum(positions) / len(positions)
        self.assertAlmostEqual(mean, 540, delta=60, msg=f"mean face x {mean:.0f}, positions {positions}")
        d = self.plan.diagnostics
        self.assertAlmostEqual(d["face_in_crop_median"], 0.5, delta=0.08)
        self.assertEqual(d["clamped_right_samples"], 0)


@unittest.skipUnless(HAVE_CV and HAVE_FFMPEG and FACE_IMAGE.is_file(), "needs OpenCV, ffmpeg and tests/data/face.jpg")
class HandoffEndToEndTest(unittest.TestCase):
    """One person for 40 s; a source edit at 10 s moves the face from x=400 to x=1000."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="kivro-handoff-"))
        face = cv2.resize(cv2.imread(str(FACE_IMAGE)), (220, 220))
        fps, seconds, cls.cut = 30, 40, 10.0
        raw = cls.tmp / "raw.avi"
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        for k in range(seconds * fps):
            t = k / fps
            frame = np.full((H, W, 3), (70, 80, 90), np.uint8)
            cx = 400 if t < cls.cut else 1000
            x0 = int(cx - 110 + 12 * np.sin(t / 3))
            frame[200:420, x0:x0 + 220] = face
            vw.write(frame)
        vw.release()
        cls.source = cls.tmp / "source.mp4"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
                        "-f", "lavfi", "-i", "sine=frequency=330", "-t", str(seconds),
                        "-c:v", "libx264", "-preset", "veryfast", "-g", "240", "-bf", "3", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(cls.source)], check=True)
        cls.analysis = framing.analyze_clip(cls.source, 0.0, float(seconds))
        cls.plan = plan_layout(cls.analysis)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_detector_splits_the_person_into_two_tracks_and_plan_hands_off(self):
        self.assertEqual(self.plan.layout, LAYOUT_SINGLE, self.plan.note)
        d = self.plan.diagnostics
        self.assertGreaterEqual(len(self.analysis.tracks), 2)
        self.assertEqual(len(d["handoff_tracks"]), 1, d)
        self.assertGreaterEqual(d["handoff_samples"], 16)            # ~10 s at 2 fps, minus edge samples
        self.assertEqual(d["handoff_rejected"], [])
        before = d["smoothed_crop_x"][:16]
        after = d["smoothed_crop_x"][24:]
        for x in before:
            self.assertAlmostEqual(x, 400 - CROP_W / 2, delta=40, msg=f"before the cut: {x}")
        for x in after:
            self.assertAlmostEqual(x, 1000 - CROP_W / 2, delta=40, msg=f"after the cut: {x}")

    def test_rendered_face_is_centred_before_and_after_the_edit(self):
        from video import render_clip
        out = render_clip(self.source, 0.0, 40.0, None, self.tmp / "out.mp4", self.plan)
        det = framing._Detector(1080, 1920)
        for tt in (1.0, 4.0, 7.0, 9.0, 11.5, 14.0, 20.0, 30.0, 38.0):
            png = self.tmp / f"f{tt}.png"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(tt), "-i", str(out),
                            "-frames:v", "1", str(png)], check=True)
            faces = det.detect(cv2.imread(str(png)))
            self.assertTrue(faces, f"face missing from output at {tt}s (crop held at the wrong position)")
            cx = faces[0][0] + faces[0][2] / 2
            self.assertGreaterEqual(cx, 420, f"face too far left at {tt}s: {cx:.0f}")
            self.assertLessEqual(cx, 660, f"face too far right at {tt}s: {cx:.0f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
