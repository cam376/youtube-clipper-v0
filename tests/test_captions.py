"""Caption cues: building, deterministic retiming on edit, find & replace."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from captions import build_cues, set_cue_text, find_in_cues, replace_in_cues, tokenize  # noqa: E402


def words(text, start=10.0, step=0.4):
    out, t = [], start
    for w in text.split():
        out.append({"start": round(t, 3), "end": round(t + step * 0.9, 3), "word": w})
        t += step
    return out


class BuildCuesTest(unittest.TestCase):
    def test_groups_words_relative_to_clip(self):
        cues = build_cues(words("un deux trois quatre cinq six sept."), clip_start=10.0, clip_end=20.0)
        self.assertEqual([c["text"] for c in cues], ["un deux trois quatre", "cinq six sept."])
        self.assertEqual(cues[0]["start"], 0.0)
        self.assertEqual(len(cues[0]["words"]), 4)
        self.assertAlmostEqual(cues[1]["words"][0]["start"], 1.6, places=3)

    def test_splits_on_punctuation_and_gaps(self):
        w = words("bonjour. ça va") + [{"start": 20.0, "end": 20.3, "word": "pause"}]
        cues = build_cues(w, 10.0, 25.0)
        self.assertEqual([c["text"] for c in cues], ["bonjour.", "ça va", "pause"])

    def test_only_words_inside_the_clip(self):
        cues = build_cues(words("a b c d e f g h", start=0.0, step=1.0), clip_start=2.5, clip_end=5.5)
        self.assertEqual(" ".join(c["text"] for c in cues), "c d e f")


class EditTimingTest(unittest.TestCase):
    def setUp(self):
        self.cue = build_cues(words("J'ai rencontré Arnault Angeli"), 10.0, 20.0)[0]

    def test_same_token_count_keeps_word_timing(self):
        new = set_cue_text(self.cue, "J'ai rencontré Arnor Angeli")
        self.assertEqual(new["text"], "J'ai rencontré Arnor Angeli")
        self.assertEqual((new["start"], new["end"]), (self.cue["start"], self.cue["end"]))
        self.assertEqual([(w["start"], w["end"]) for w in new["words"]],
                         [(w["start"], w["end"]) for w in self.cue["words"]])
        self.assertEqual(new["words"][2]["word"], "Arnor")

    def test_different_token_count_distributes_evenly(self):
        new = set_cue_text(self.cue, "J'ai vu Arnor")
        self.assertEqual((new["start"], new["end"]), (self.cue["start"], self.cue["end"]))
        self.assertEqual(len(new["words"]), 3)
        slot = (self.cue["end"] - self.cue["start"]) / 3
        for i, w in enumerate(new["words"]):
            self.assertAlmostEqual(w["start"], self.cue["start"] + i * slot, places=3)
            self.assertAlmostEqual(w["end"], self.cue["start"] + (i + 1) * slot, places=3)

    def test_edit_is_deterministic(self):
        a = set_cue_text(self.cue, "un deux trois quatre cinq six")
        b = set_cue_text(self.cue, "un deux trois quatre cinq six")
        self.assertEqual(a, b)

    def test_empty_text_keeps_timing_and_drops_words(self):
        new = set_cue_text(self.cue, "   ")
        self.assertEqual(new["text"], "")
        self.assertEqual(new["words"], [])
        self.assertEqual(new["start"], self.cue["start"])

    def test_tokenize_collapses_whitespace(self):
        self.assertEqual(tokenize("  a   b\tc "), ["a", "b", "c"])


class FindReplaceTest(unittest.TestCase):
    def setUp(self):
        self.cues = build_cues(words("Arnault Angeli a fondé GoHighLevel. arnault angeli encore"), 10.0, 20.0)

    def test_find_is_case_insensitive_by_default(self):
        self.assertEqual(find_in_cues(self.cues, "arnault angeli"), [0, 2])
        self.assertEqual(find_in_cues(self.cues, "Arnault Angeli", case_sensitive=True), [0])
        self.assertEqual(find_in_cues(self.cues, ""), [])

    def test_replace_retimes_only_touched_cues(self):
        new, n = replace_in_cues(self.cues, "Arnault Angeli", "Arnor Angeli")
        self.assertEqual(n, 2)
        self.assertEqual(new[0]["text"], "Arnor Angeli a fondé")
        self.assertEqual(new[2]["text"], "Arnor Angeli encore")
        self.assertEqual(new[1], self.cues[1])
        self.assertEqual([w["start"] for w in new[0]["words"]], [w["start"] for w in self.cues[0]["words"]])

    def test_replace_does_not_mutate_input(self):
        before = [dict(c) for c in self.cues]
        replace_in_cues(self.cues, "Arnault", "Arnor")
        self.assertEqual(self.cues, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
