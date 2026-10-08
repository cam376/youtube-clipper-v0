"""
Per-candidate COMPLETE / CONTINUES judgement, continuity guards, strict
parsing and absence of positional bias. The model is simulated by a judge
function that reads the candidate text, exactly like Qwen would; no id is
ever chosen by the judge.
"""

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import endpoints  # noqa: E402
from endpoints import (  # noqa: E402
    candidate_endpoints, continuity_guard, judge_endpoints, parse_verdict, refine_end,
    COMPLETE, CONTINUES, UNPARSED, JUDGE_PROMPT,
)


def transcript(sentences, start=100.0, wps=2.2, gap=0.3):
    words, segments, t = [], [], start
    for item in sentences:
        text, pause = item if isinstance(item, tuple) else (item, gap)
        toks = text.split()
        seg_start = t
        for tok in toks:
            words.append({"start": round(t, 2), "end": round(t + 0.9 / wps, 2), "word": tok})
            t += 1.0 / wps
        segments.append({"start": round(seg_start, 2), "end": words[-1]["end"], "text": text})
        t += pause
    return words, segments


def content_judge(marker="[DONE]"):
    """A judge that says COMPLETE iff the text up to the cut contains the marker (the 'truth' of the test)."""
    def judge(hook, body, next_text, pause):
        return (COMPLETE if marker in body else CONTINUES), "simulated", '{"verdict": "..."}'
    return judge


class GuardTest(unittest.TestCase):
    def test_1_setup_sentence_is_continues(self):
        g = continuity_guard("Le vrai levier c'est la régularité. Laisse-moi te dire également une chose.",
                             "Les gens qui réussissent publient tous les jours.", terminal=True, pause=0.4)
        self.assertIsNotNone(g)
        self.assertIn("setup", g)

    def test_2_complete_explanation_then_new_topic_passes_guards(self):
        g = continuity_guard("Parle à dix clients avant d'écrire une ligne. C'est ça la différence.",
                             "Bon, passons maintenant à la question des prix.", terminal=True, pause=1.1)
        self.assertIsNone(g)

    def test_3_dangling_parce_que_is_continues(self):
        g = continuity_guard("Et la deuxième raison c'est parce que", "les gens n'ont pas de système.", terminal=False, pause=1.3)
        self.assertIsNotNone(g)
        self.assertIn("dangling", g)
        g2 = continuity_guard("Et la deuxième raison c'est parce que.", "les gens n'ont pas de système.", terminal=True, pause=0.2)
        self.assertIsNotNone(g2)

    def test_4_question_followed_by_its_answer_is_continues(self):
        g = continuity_guard("Alors pourquoi la plupart des gens échouent ?", "Parce qu'ils n'ont pas de système.", terminal=True, pause=0.5)
        self.assertIsNotNone(g)
        self.assertIn("question", g)

    def test_continuation_start_words_are_continues(self):
        for nxt in ("Et ensuite tu vends.", "Donc voilà.", "Par exemple, hier.", "Premièrement, le prix.", "Parce que sinon."):
            g = continuity_guard("Tu dois construire une audience.", nxt, terminal=True, pause=0.3)
            self.assertIsNotNone(g, nxt)
            self.assertIn("continue", g)

    def test_non_terminal_boundary_needs_a_real_pause(self):
        self.assertIsNotNone(continuity_guard("tu construis ton système", "et après tu vends", terminal=False, pause=0.9))
        self.assertIsNone(continuity_guard("tu construis ton système", "Bon maintenant les prix.", terminal=False, pause=1.5))


class ParsingTest(unittest.TestCase):
    def test_8_raw_response_cannot_default_to_an_endpoint(self):
        for raw in ("", "E4", '{"endpoint": "E4"}', "{}", "not json", '{"verdict": "E4"}', '{"verdict": ""}', None):
            verdict, _ = parse_verdict(raw)
            self.assertEqual(verdict, UNPARSED, raw)
        self.assertEqual(parse_verdict('{"verdict": "COMPLETE", "reason": "ok"}')[0], COMPLETE)
        self.assertEqual(parse_verdict('{"verdict": "continues"}')[0], CONTINUES)
        self.assertEqual(parse_verdict('some text {"verdict":"COMPLETE"} trailing')[0], COMPLETE)

    def test_prompt_contains_no_endpoint_id_and_no_verdict_bias(self):
        self.assertIsNone(re.search(r"\bE\d+\b", JUDGE_PROMPT))
        self.assertIn("<COMPLETE or CONTINUES>", JUDGE_PROMPT)
        self.assertNotIn('"verdict": "COMPLETE"', JUDGE_PROMPT)
        self.assertNotIn('"verdict": "CONTINUES"', JUDGE_PROMPT)

    def test_unparsed_verdicts_never_select(self):
        words, segs = transcript([("Un point. " * 3, 0.3)] * 8)
        eps = candidate_endpoints(words, segs, 100.0)
        chosen, records, calls = judge_endpoints(eps, "hook", judge=lambda *a: (UNPARSED, "x", "garbage"))
        self.assertIsNone(chosen)
        self.assertTrue(all(r["verdict"] in (UNPARSED, CONTINUES) for r in records))


class SelectionTest(unittest.TestCase):
    def test_5_true_completion_at_24s_is_kept_not_pushed_later(self):
        sentences = [
            ("Voici la seule chose qui compte quand tu lances une offre.", 0.3),
            ("Tu dois parler à dix clients avant d'écrire une seule ligne.", 0.3),
            ("Ils te diront exactement quoi écrire [DONE].", 1.0),            # ~24 s: complete
            ("Bon, maintenant parlons des prix.", 0.3),
            ("Beaucoup de gens me demandent combien facturer.", 0.3),
            ("Je réponds toujours que ça dépend de la valeur créée.", 0.3),
            ("Et la valeur se mesure en résultats, pas en heures.", 0.3),
            ("Voilà pour les prix.", 0.5),
        ]
        words, segs = transcript(sentences, wps=1.2)      # [DONE] sentence ends at ~24 s
        rec = refine_end(100.0, 140.0, words, segs, judge=content_judge())
        self.assertEqual(rec["method"], "semantic")
        self.assertLess(rec["final_duration"], 30)
        self.assertGreater(rec["final_duration"], 20)
        chosen_text = [e for e in candidate_endpoints(words, segs, 100.0) if e["time"] == rec["final_end"]][0]["text"]
        self.assertTrue(chosen_text.endswith("[DONE]."), chosen_text[-40:])

    def test_6_incomplete_at_42s_completes_at_51s(self):
        sentences = [
            ("Il y a trois raisons pour lesquelles ton contenu ne décolle pas.", 0.3),
            ("Premièrement, tu publies sans régularité et l'algorithme t'oublie.", 0.3),
            ("Deuxièmement, tes hooks sont trop longs et les gens scrollent.", 0.3),
            ("Troisièmement, tu ne proposes jamais d'étape suivante.", 0.3),
            ("Corrige ces trois choses et ton contenu commence à travailler pour toi [DONE].", 0.9),
            ("Bon, on passe à autre chose.", 0.3),
            ("La semaine prochaine on parle de ventes.", 0.3),
        ]
        words, segs = transcript(sentences, wps=1.2)      # slower speech so the completion lands past 40 s
        eps = candidate_endpoints(words, segs, 100.0)
        done = [e for e in eps if e["text"].endswith("[DONE].")][0]
        self.assertGreater(done["rel"], 40)
        rec = refine_end(100.0, 140.0, words, segs, judge=content_judge())
        self.assertEqual(rec["final_end"], done["time"])
        self.assertGreater(rec["final_duration"], 40)

    def test_7_no_positional_bias(self):
        # The correct answer sits at a different ordinal in each case; the
        # algorithm must find each one from content, not from position.
        filler = "Une phrase neutre de plus sur le sujet."
        cases = []
        for ordinal in (1, 2, 4, 6):
            sents = []
            for k in range(1, 9):
                text = (filler[:-1] + " [DONE].") if k == ordinal else filler
                sents.append((text, 1.0 if k == ordinal else 0.3))
            cases.append((ordinal, sents))
        picked = []
        for ordinal, sents in cases:
            words, segs = transcript(sents, wps=3.0)       # fast: each sentence ~2.7 s
            eps = candidate_endpoints(words, segs, 100.0, min_seconds=1)
            with mock.patch.object(endpoints, "CLIP_MIN_SECONDS", 1):
                rec = refine_end(100.0, 125.0, words, segs, judge=content_judge())
            chosen = [e for e in eps if e["time"] == rec["final_end"]][0]
            self.assertTrue(chosen["text"].endswith("[DONE]."), (ordinal, chosen["text"][-40:]))
            picked.append(chosen["id"])
        self.assertEqual(len(set(picked)), 4, picked)

    def test_guard_rejected_candidates_cost_no_model_call(self):
        words, segs = transcript([
            ("Laisse-moi te dire une chose.", 0.3),
            ("Le secret c'est la régularité.", 0.3),
            ("Et c'est tout ce qui compte [DONE].", 1.0),
            ("Bon, passons à la suite.", 0.3),
        ] * 2, wps=1.2)
        calls = {"n": 0}

        def judge(hook, body, next_text, pause):
            calls["n"] += 1
            return (COMPLETE if "[DONE]" in body else CONTINUES), "sim", "{}"
        chosen, records, n = judge_endpoints(candidate_endpoints(words, segs, 100.0), "hook", judge)
        guarded = [r for r in records if r["source"] == "guard"]
        self.assertTrue(guarded)
        self.assertEqual(n, calls["n"])
        self.assertLess(n, len(records))

    def test_nothing_complete_falls_back_to_last_clean_sentence(self):
        words, segs = transcript([("Une phrase qui ne conclut rien de spécial.", 0.3)] * 12)
        rec = refine_end(100.0, 140.0, words, segs, judge=lambda *a: (CONTINUES, "sim", "{}"))
        self.assertEqual(rec["method"], "fallback")
        self.assertIn("no candidate judged COMPLETE", rec["endpoint_reason"])
        self.assertTrue(rec["endpoint_terminal"])
        self.assertIsNone(rec["endpoint_guard"])

    def test_lookahead_prefers_longer_pause_within_four_seconds(self):
        words, segs = transcript([
            ("Le début de l'idée avec un hook fort.", 0.3),
            ("Une explication utile de l'idée.", 0.3),
            ("Première conclusion possible [DONE].", 0.2),      # complete, short pause
            ("Deuxième conclusion équivalente [DONE].", 1.5),   # complete, long pause, < 4 s later
            ("Nouveau sujet maintenant.", 0.3),
        ] * 2, wps=2.0)
        with mock.patch.object(endpoints, "CLIP_MIN_SECONDS", 5):
            rec = refine_end(100.0, 125.0, words, segs, judge=content_judge())
        self.assertGreaterEqual(rec["endpoint_pause"], 1.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
