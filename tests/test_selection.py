"""
Dynamic clip selection (v0.3): threshold acceptance, deduplication, no
minimum count, batched scoring. No Ollama, no video.

    python -m unittest discover -s tests -v
"""

import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import ranking  # noqa: E402
from ranking import select_clips, overlap_fraction  # noqa: E402


def cand(cid, start, end, score, text="x"):
    return {"id": cid, "start": float(start), "end": float(end), "score": float(score), "text": text}


def spaced(n, score=8.0, gap=50):
    """n non-overlapping 40 s windows."""
    return [cand(i, i * gap, i * gap + 40, score) for i in range(n)]


class SelectionTest(unittest.TestCase):
    def test_three_qualifying_moments_give_three_clips(self):
        cands = spaced(3, 8.0) + [cand(10 + i, 1000 + i * 50, 1040 + i * 50, 5.0) for i in range(4)]
        out = select_clips(cands, min_score=7.0)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(c["score"] >= 7.0 for c in out))

    def test_eight_qualifying_moments_give_eight_clips(self):
        out = select_clips(spaced(8, 8.5), min_score=7.0)
        self.assertEqual(len(out), 8)

    def test_forty_qualifying_moments_give_forty_clips(self):
        out = select_clips(spaced(40, 9.0), min_score=7.0)
        self.assertEqual(len(out), 40)

    def test_weak_moments_rejected(self):
        cands = spaced(4, 6.9) + [cand(99, 5000, 5040, 7.0)]
        out = select_clips(cands, min_score=7.0)
        self.assertEqual([c["id"] for c in out], [99])

    def test_zero_one_two_qualifying_give_zero_one_two(self):
        weak = [cand(10 + i, 1000 + i * 50, 1040 + i * 50, 6.9) for i in range(6)]
        self.assertEqual(select_clips(weak, min_score=7.0), [])
        self.assertEqual(len(select_clips(weak + spaced(1, 7.0), min_score=7.0)), 1)
        self.assertEqual(len(select_clips(weak + spaced(2, 7.5), min_score=7.0)), 2)

    def test_no_backfill_and_no_minimum(self):
        src = inspect.getsource(ranking)
        for token in ("floor", "below_threshold", "MIN_CLIPS"):
            self.assertNotIn(token, src)
        self.assertNotIn("floor", inspect.signature(select_clips).parameters)

    def test_near_duplicates_keep_the_stronger(self):
        a = cand(0, 0, 40, 9.0)
        b = cand(1, 5, 45, 8.0)       # overlap 35 s / 40 s = 0.875 -> same moment
        c = cand(2, 36, 70, 8.0)      # overlap 4 s / 34 s = 0.12 -> different moment
        out = select_clips([b, c, a], min_score=7.0, max_overlap=0.2)
        self.assertEqual([x["id"] for x in out], [0, 2])

    def test_result_in_video_order(self):
        cands = [cand(0, 300, 340, 7.5), cand(1, 0, 40, 9.0), cand(2, 150, 190, 8.0)]
        self.assertEqual([c["start"] for c in select_clips(cands, min_score=7.0)], [0.0, 150.0, 300.0])

    def test_overlap_fraction(self):
        self.assertEqual(overlap_fraction(cand(0, 0, 40, 1), cand(1, 40, 80, 1)), 0.0)
        self.assertAlmostEqual(overlap_fraction(cand(0, 0, 40, 1), cand(1, 20, 80, 1)), 0.5)
        self.assertAlmostEqual(overlap_fraction(cand(0, 0, 40, 1), cand(1, 10, 30, 1)), 1.0)

    def test_no_top_k_truncation_left_in_code(self):
        src = inspect.getsource(ranking)
        self.assertNotIn("max_clips", src)
        self.assertNotIn("select_best", src)
        self.assertNotIn("MAX_CANDIDATES", src)

    def test_scoring_is_batched_not_sampled(self):
        cands = spaced(95, 8.0)
        for c in cands:
            c["text"] = "candidate"
        seen = []

        def fake_batch(batch):
            seen.append(len(batch))
            return {c["id"]: 8.0 for c in batch}

        with mock.patch.object(ranking, "_ollama_scores_batch", side_effect=fake_batch):
            scores = ranking._ollama_scores(cands)
        self.assertEqual(len(scores), 95)
        self.assertEqual(seen, [40, 40, 15])

    def test_defaults(self):
        self.assertEqual(ranking.MIN_CLIP_SCORE, 7.0)
        self.assertEqual(ranking.DEDUP_MAX_OVERLAP, 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
