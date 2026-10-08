"""
Natural clip endings: the start never moves, the end lands on a real
transcript boundary where the idea finishes, within the configured policy.
No Ollama: the semantic choice is mocked; the fallback is exercised directly.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import endpoints  # noqa: E402
from endpoints import candidate_endpoints, choose_endpoint_heuristic, refine_end, text_between  # noqa: E402
from ranking import dedupe_overlaps  # noqa: E402


def transcript(sentences, start=100.0, wps=2.5, gap=0.3):
    """
    sentences: list of (text, pause_after_seconds). Words are spaced 1/wps
    apart; a sentence is one Whisper segment. Returns (words, segments).
    """
    words, segments, t = [], [], start
    for text, pause in sentences:
        toks = text.split()
        seg_start = t
        for tok in toks:
            words.append({"start": round(t, 2), "end": round(t + 0.9 / wps, 2), "word": tok})
            t += 1.0 / wps
        segments.append({"start": round(seg_start, 2), "end": words[-1]["end"], "text": text})
        t += pause
    return words, segments


# A hook, an explanation that concludes at ~27 s, then a new topic.
IDEA_DONE_AT_27 = [
    ("Voici pourquoi la plupart des gens échouent avec leur première offre.", 0.4),      # ~5.8 s
    ("Ils vendent une solution avant d'avoir compris le problème du client.", 0.3),      # ~11.6
    ("Alors ils écrivent des pages de vente que personne ne lit.", 0.3),                 # ~17
    ("La solution est simple: parlez à dix clients avant d'écrire une ligne.", 0.6),     # ~24
    ("C'est tout, c'est ça la différence.", 1.2),                                        # ~27  <- idea complete
    ("Bon, passons maintenant à la question des prix.", 0.3),                            # new topic ~31
    ("Beaucoup de gens me demandent combien facturer.", 0.3),
    ("Je réponds toujours la même chose, ça dépend de la valeur créée.", 0.3),          # ~40
    ("Et la valeur, elle se mesure en résultats pas en heures.", 0.3),                   # ~45
    ("Voilà pour les prix.", 0.5),
]


class CandidateEndpointTest(unittest.TestCase):
    def test_endpoints_are_real_boundaries_after_min_and_before_hard_max(self):
        words, segs = transcript(IDEA_DONE_AT_27)
        eps = candidate_endpoints(words, segs, start=100.0, min_seconds=15, target_max=60, hard_max=75)
        self.assertTrue(eps)
        word_ends = {w["end"] for w in words}
        for e in eps:
            self.assertIn(e["time"], word_ends)
            self.assertGreaterEqual(e["rel"], 15)
            self.assertLessEqual(e["rel"], 75)
        self.assertEqual([e["id"] for e in eps], [f"E{i}" for i in range(1, len(eps) + 1)])
        self.assertTrue(any(e["terminal"] and abs(e["rel"] - 27) < 2 for e in eps), [e["rel"] for e in eps])

    def test_only_first_sentence_end_beyond_target_max_is_offered(self):
        long = [(f"Phrase numéro {i} qui continue encore et encore.", 0.3) for i in range(30)]   # ~4 s each
        words, segs = transcript(long)
        eps = candidate_endpoints(words, segs, start=100.0, min_seconds=15, target_max=60, hard_max=75)
        beyond = [e for e in eps if e["rel"] > 60]
        self.assertEqual(len(beyond), 1)
        self.assertTrue(beyond[0]["terminal"])
        self.assertLessEqual(beyond[0]["rel"], 75)


class RefineEndTest(unittest.TestCase):
    def setUp(self):
        self.words, self.segs = transcript(IDEA_DONE_AT_27)

    def _endpoint_ending_with(self, suffix, start=100.0):
        eps = candidate_endpoints(self.words, self.segs, start)
        match = [e for e in eps if e["text"].endswith(suffix)]
        self.assertTrue(match, f"no endpoint ends with {suffix!r}: {[e['text'][-25:] for e in eps]}")
        return match[0]

    def _refine_with_model_choice(self, suffix, start=100.0, end=140.0):
        """Simulate Qwen choosing the endpoint whose transcript ends with `suffix`."""
        target = self._endpoint_ending_with(suffix, start)
        with mock.patch.object(endpoints, "_ollama_choose", return_value=(target["id"], "idea resolved")):
            return refine_end(start, end, self.words, self.segs), target

    def test_1_forty_second_candidate_ends_where_idea_concludes(self):
        rec, target = self._refine_with_model_choice("différence.")      # idea complete at ~27 s
        self.assertEqual(rec["method"], "semantic")
        self.assertEqual(rec["final_end"], target["time"])
        self.assertAlmostEqual(rec["final_duration"], target["rel"], places=2)
        self.assertLess(rec["final_duration"], 32)
        self.assertGreaterEqual(rec["final_duration"], 15)
        self.assertTrue(rec["endpoint_terminal"])
        self.assertEqual(rec["original_duration"], 40.0)

    def test_2_idea_continuing_past_forty_extends_to_its_conclusion(self):
        rec, target = self._refine_with_model_choice("pour les prix.")            # last sentence, ends after 40 s
        self.assertGreater(target["rel"], 40)
        self.assertEqual(rec["final_end"], target["time"])
        self.assertGreater(rec["final_duration"], 40)

    def test_3_sentence_crossing_forty_is_not_cut_mid_sentence(self):
        # The old window ended at exactly 40 s, inside "Je réponds ... créée."
        with mock.patch.object(endpoints, "_ollama_choose", side_effect=RuntimeError("no ollama")):
            rec = refine_end(100.0, 140.0, self.words, self.segs)
        self.assertEqual(rec["method"], "fallback")
        self.assertTrue(rec["endpoint_terminal"])
        last_word = [w for w in self.words if w["end"] <= rec["final_end"]][-1]["word"]
        self.assertTrue(last_word.endswith("."), last_word)
        self.assertNotEqual(rec["final_end"], 140.0)

    def test_4_short_complete_idea_is_not_padded(self):
        rec, target = self._refine_with_model_choice("ligne.")           # complete idea under 20 s
        self.assertEqual(rec["final_end"], target["time"])
        self.assertLess(rec["final_duration"], 22)
        self.assertGreaterEqual(rec["final_duration"], 15)

    def test_5_rambling_passage_is_capped_by_the_policy(self):
        rambling = [("et puis on continue sans jamais finir la phrase parce que ça parle " * 2, 0.2) for _ in range(20)]
        words, segs = transcript(rambling)
        with mock.patch.object(endpoints, "_ollama_choose", side_effect=RuntimeError("no ollama")):
            rec = refine_end(100.0, 140.0, words, segs)
        self.assertLessEqual(rec["final_duration"], 75)
        self.assertLessEqual(rec["final_duration"], 60 + 0.01)   # no sentence end at all -> last boundary before target max
        self.assertIn("no sentence end", rec["endpoint_reason"])

    def test_6_start_never_changes(self):
        for suffix in ("ligne.", "différence.", "pour les prix."):
            rec, _ = self._refine_with_model_choice(suffix)
            self.assertEqual(rec["final_start"], 100.0)
            self.assertEqual(rec["original_start"], 100.0)

    def test_7_clips_get_different_durations(self):
        durations = {self._refine_with_model_choice(s)[0]["final_duration"] for s in ("ligne.", "différence.", "pour les prix.")}
        self.assertEqual(len(durations), 3)

    def test_model_answer_outside_the_list_falls_back(self):
        with mock.patch.object(endpoints, "_ollama_choose", return_value=("E99", "made up")):
            rec = refine_end(100.0, 140.0, self.words, self.segs)
        self.assertEqual(rec["method"], "fallback")
        self.assertIn("not a listed endpoint", rec["endpoint_reason"])
        self.assertIn(rec["final_end"], [e["time"] for e in candidate_endpoints(self.words, self.segs, 100.0)])

    def test_no_boundary_in_range_keeps_original_end(self):
        words, segs = transcript([("Une seule phrase très courte.", 0.2)])
        rec = refine_end(100.0, 140.0, words, segs)
        self.assertEqual((rec["method"], rec["final_end"]), ("unchanged", 140.0))

    def test_text_follows_the_new_end(self):
        rec, _ = self._refine_with_model_choice("différence.")
        text = text_between(self.words, 100.0, rec["final_end"])
        self.assertTrue(text.endswith("différence."), text[-40:])
        self.assertNotIn("prix", text)


class HeuristicTest(unittest.TestCase):
    def test_prefers_last_sentence_end_before_target_max(self):
        eps = [{"id": "E1", "rel": 20, "terminal": True, "time": 120, "pause": 0.3},
               {"id": "E2", "rel": 35, "terminal": False, "time": 135, "pause": 0.1},
               {"id": "E3", "rel": 58, "terminal": True, "time": 158, "pause": 0.4},
               {"id": "E4", "rel": 66, "terminal": True, "time": 166, "pause": 0.9}]
        chosen, why = choose_endpoint_heuristic(eps, target_max=60)
        self.assertEqual(chosen["id"], "E3")
        self.assertIn("last complete sentence", why)

    def test_finishes_the_thought_after_target_max_when_needed(self):
        eps = [{"id": "E1", "rel": 30, "terminal": False, "time": 130, "pause": 0.1},
               {"id": "E2", "rel": 66, "terminal": True, "time": 166, "pause": 0.9}]
        chosen, why = choose_endpoint_heuristic(eps, target_max=60)
        self.assertEqual(chosen["id"], "E2")
        self.assertIn("finishing the thought", why)


class DedupeAfterRefinementTest(unittest.TestCase):
    def test_extended_clip_overlapping_a_weaker_one_drops_the_weaker(self):
        a = {"start": 100.0, "end": 158.0, "score": 9.0}     # extended past its old 40 s end
        b = {"start": 145.0, "end": 175.0, "score": 8.0}     # now overlaps a by 13 s / 30 s = 43 %
        c = {"start": 200.0, "end": 230.0, "score": 7.5}
        kept, dropped = dedupe_overlaps([a, b, c], max_overlap=0.2)
        self.assertEqual([k["start"] for k in kept], [100.0, 200.0])
        self.assertEqual([d["start"] for d in dropped], [145.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
