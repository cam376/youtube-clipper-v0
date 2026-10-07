"""
Layout classification on synthetic face tracks (no video, no detector).

Covers the real-world case that produced a false SINGLE_PERSON on a
two-person interview: the second speaker turns away for part of the clip,
so their face is detected less often, but their detections span the whole
clip. Also checks that solo videos, walk-in faces and empty frames keep
their previous classification.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import framing  # noqa: E402
from framing import (  # noqa: E402
    FaceAnalysis, Track, plan_layout, classification_metrics, _merge_fragments,
    LAYOUT_SPLIT, LAYOUT_SINGLE, LAYOUT_CENTER,
)

W, H = 1280, 720
N = 100  # samples (50 s at 2 fps)


def _track(tid: int, cx: float, present: list[int], size: float = 80.0, cy: float = 300.0) -> Track:
    t = Track(id=tid)
    for i in present:
        t.boxes[i] = (cx - size / 2, cy - size / 2, size, size)
    t.last_idx = max(present) if present else -1
    return t


def _analysis(tracks: list[Track], n: int = N) -> FaceAnalysis:
    return FaceAnalysis(width=W, height=H, sample_fps=framing.SAMPLE_FPS, n_samples=n,
                        tracks=tracks, frames=[], scale=2.0, detector_name="test")


class ClassificationTest(unittest.TestCase):
    def test_two_fully_present_faces_split(self):
        a = _analysis([_track(0, 330, list(range(N))), _track(1, 950, list(range(N)))])
        plan = plan_layout(a)
        self.assertEqual(plan.layout, LAYOUT_SPLIT)
        self.assertEqual([r.track_id for r in plan.regions], [0, 1])  # left on top

    def test_second_speaker_turning_away_still_split(self):
        # Interview: speaker B detected in 45 % of samples, spread over the
        # whole clip (looks at A while listening). Old rule: 0.45 < 0.5 * 1.0
        # -> "dominated" -> SINGLE. Now the span keeps it a second person.
        present_b = [i for i in range(N) if i % 20 < 9]  # 45 %, span 100 %
        a = _analysis([_track(0, 330, list(range(N))), _track(1, 950, present_b)])
        plan = plan_layout(a)
        self.assertEqual(plan.layout, LAYOUT_SPLIT, plan.note)
        self.assertIn("ratio 0.45", plan.note)

    def test_second_speaker_below_old_threshold_but_spanning_clip_split(self):
        present_b = [i for i in range(N) if i % 10 < 3]  # 30 %, span ~ 93 %
        a = _analysis([_track(0, 330, list(range(N))), _track(1, 950, present_b)])
        self.assertEqual(plan_layout(a).layout, LAYOUT_SPLIT)

    def test_walk_in_face_stays_single(self):
        # Someone passes behind the speaker for the first 40 % of the clip
        # only: same coverage as the interview case but no span -> SINGLE.
        a = _analysis([_track(0, 330, list(range(N))), _track(1, 950, list(range(40)))])
        plan = plan_layout(a)
        self.assertEqual(plan.layout, LAYOUT_SINGLE, plan.note)
        self.assertEqual(plan.regions[0].track_id, 0)

    def test_sporadic_second_face_stays_single(self):
        present_b = [i for i in range(N) if i % 10 == 0]  # 10 %, span 91 %
        a = _analysis([_track(0, 330, list(range(N))), _track(1, 950, present_b)])
        self.assertEqual(plan_layout(a).layout, LAYOUT_SINGLE)

    def test_two_faces_too_close_single(self):
        a = _analysis([_track(0, 600, list(range(N))), _track(1, 700, list(range(N)))])
        plan = plan_layout(a)
        self.assertEqual(plan.layout, LAYOUT_SINGLE)
        self.assertIn("too close", plan.note)

    def test_solo_video_single(self):
        a = _analysis([_track(0, 640, list(range(N)))])
        plan = plan_layout(a)
        self.assertEqual(plan.layout, LAYOUT_SINGLE)
        self.assertEqual(plan.regions[0].w, 404)  # 9:16 crop of a 720 p frame

    def test_no_reliable_face_center(self):
        self.assertEqual(plan_layout(_analysis([])).layout, LAYOUT_CENTER)
        a = _analysis([_track(0, 640, list(range(10)))])  # 10 % coverage
        self.assertEqual(plan_layout(a).layout, LAYOUT_CENTER)

    def test_fragmented_track_is_merged_before_classification(self):
        # Speaker B's track dies after a long look-away and comes back with a
        # new id: two fragments of 30 % each, same seat.
        frag1 = _track(1, 950, list(range(0, 30)))
        frag2 = _track(2, 955, list(range(70, 100)))
        merged = _merge_fragments([_track(0, 330, list(range(N))), frag1, frag2])
        self.assertEqual(len(merged), 2)
        b = [t for t in merged if t.id == 1][0]
        self.assertEqual(len(b.boxes), 60)
        self.assertEqual(b.fragments, 2)
        self.assertAlmostEqual(b.longest_gap(framing.SAMPLE_FPS), 40 / framing.SAMPLE_FPS)  # samples 30..69 missing
        self.assertEqual(plan_layout(_analysis(merged)).layout, LAYOUT_SPLIT)

    def test_overlapping_tracks_are_not_merged(self):
        a = _track(0, 330, list(range(N)))
        b = _track(1, 360, list(range(N)))  # same place, same time: two detections, not fragments
        self.assertEqual(len(_merge_fragments([a, b])), 2)

    def test_metrics_present_in_debug_summary(self):
        present_b = [i for i in range(N) if i % 20 < 9]
        a = _analysis([_track(0, 330, list(range(N))), _track(1, 950, present_b)])
        m = classification_metrics(a)
        for key in ("samples", "separation", "strength_ratio", "second_span", "second_longest_gap_s", "thresholds"):
            self.assertIn(key, m)
        self.assertEqual(m["samples"], N)
        self.assertAlmostEqual(m["strength_ratio"], 0.45)
        for t in m["tracks"]:
            for key in ("coverage", "span", "longest_gap_s", "fragments"):
                self.assertIn(key, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
