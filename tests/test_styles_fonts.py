"""Subtitle presets and font choices: ASS output, ffmpeg render, fallbacks."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import fonts  # noqa: E402
import styles  # noqa: E402
import video  # noqa: E402
from captions import build_cues  # noqa: E402
from framing import FramePlan, Region, LAYOUT_SPLIT  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

V021_CLEAN_STYLE = ",72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,420,1"


def sample_cues():
    w, t = [], 0.2
    for tok in "Avec Arnaud Angeli on parle de GoHighLevel. Et ça marche".split():
        w.append({"start": t, "end": t + 0.3, "word": tok})
        t += 0.4
    return build_cues(w, 0.0, 6.0)


class PresetAssTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kivro-styles-"))
        self.cues = sample_cues()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_four_presets_exist(self):
        self.assertEqual([s["id"] for s in styles.style_names()], ["CLEAN", "BOLD", "KARAOKE", "MINIMAL"])

    def test_clean_matches_v021_style_values(self):
        info = styles.render_ass(self.cues, self.tmp / "c.ass", style="CLEAN")
        text = (self.tmp / "c.ass").read_text(encoding="utf-8")
        self.assertIn(V021_CLEAN_STYLE, text)
        self.assertIn("Dialogue: 0,0:00:00.20,", text)
        self.assertEqual(info["style"], "CLEAN")

    def test_each_preset_writes_distinct_style_line(self):
        lines = {}
        for s in ("CLEAN", "BOLD", "KARAOKE", "MINIMAL"):
            styles.render_ass(self.cues, self.tmp / f"{s}.ass", style=s)
            text = (self.tmp / f"{s}.ass").read_text(encoding="utf-8")
            lines[s] = [ln for ln in text.splitlines() if ln.startswith("Style: Default")][0]
            self.assertIn("Dialogue:", text)
        self.assertEqual(len(set(lines.values())), 4)

    def test_karaoke_uses_word_timing_tags(self):
        styles.render_ass(self.cues, self.tmp / "k.ass", style="KARAOKE")
        text = (self.tmp / "k.ass").read_text(encoding="utf-8")
        self.assertIn("{\\k40}Avec", text)          # 0.4 s between word starts -> 40 cs
        self.assertIn("&H00FF6B2F", text)            # electric blue highlight

    def test_bold_is_uppercase(self):
        styles.render_ass(self.cues, self.tmp / "b.ass", style="BOLD")
        self.assertIn("AVEC ARNAUD ANGELI", (self.tmp / "b.ass").read_text(encoding="utf-8"))

    def test_split_screen_centres_captions(self):
        styles.render_ass(self.cues, self.tmp / "s.ass", style="MINIMAL", split_screen=True)
        self.assertIn(",1,2,0,5,60,60,360,1", (self.tmp / "s.ass").read_text(encoding="utf-8"))

    def test_unknown_style_falls_back_to_clean(self):
        self.assertEqual(styles.normalize_style("fancy"), "CLEAN")
        self.assertEqual(styles.normalize_style("bold"), "BOLD")


class FontTest(unittest.TestCase):
    def test_four_choices(self):
        self.assertEqual(list(fonts.FONT_CHOICES), ["clean_sans", "heavy_sans", "condensed", "classic"])

    def test_first_installed_family_wins(self):
        with mock.patch.object(fonts, "installed_families", return_value=frozenset({"arial black", "arial"})):
            r = fonts.resolve_font("heavy_sans")
        self.assertEqual((r["family"], r["fallback"], r["note"]), ("Arial Black", False, ""))

    def test_fallback_is_explicit_not_silent(self):
        with mock.patch.object(fonts, "installed_families", return_value=frozenset({"liberation sans"})):
            r = fonts.resolve_font("heavy_sans")
        self.assertEqual(r["family"], "Liberation Sans")
        self.assertTrue(r["fallback"])
        self.assertIn("Arial Black not installed", r["note"])

    def test_nothing_installed_is_flagged(self):
        with mock.patch.object(fonts, "installed_families", return_value=frozenset({"comic sans ms"})):
            r = fonts.resolve_font("classic")
        self.assertTrue(r["fallback"])
        self.assertEqual(r["family"], "DejaVu Serif")
        self.assertIn("none of", r["note"])

    def test_unknown_choice_uses_default(self):
        self.assertEqual(fonts.resolve_font("papyrus")["choice"], "clean_sans")

    def test_font_switch_changes_ass_fontname(self):
        tmp = Path(tempfile.mkdtemp(prefix="kivro-fonts-"))
        try:
            with mock.patch.object(fonts, "installed_families", return_value=frozenset({"arial", "georgia"})):
                styles.render_ass(sample_cues(), tmp / "a.ass", style="CLEAN", font_choice="clean_sans")
                styles.render_ass(sample_cues(), tmp / "b.ass", style="CLEAN", font_choice="classic")
            self.assertIn("Style: Default,Arial,", (tmp / "a.ass").read_text(encoding="utf-8"))
            self.assertIn("Style: Default,Georgia,", (tmp / "b.ass").read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class PresetRenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="kivro-render-"))
        cls.source = cls.tmp / "source.mp4"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-f", "lavfi", "-i", "sine=frequency=440",
                        "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.source)], check=True)
        cls.plan = FramePlan(LAYOUT_SPLIT, [
            Region(400, 356, [(0.0, 100.0)], [(0.0, 120.0)], 0),
            Region(400, 356, [(0.0, 760.0)], [(0.0, 150.0)], 1)], "test")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_all_four_presets_render_with_every_font(self):
        cues = sample_cues()
        for style in ("CLEAN", "BOLD", "KARAOKE", "MINIMAL"):
            for font in ("clean_sans", "heavy_sans", "condensed", "classic"):
                ass = self.tmp / f"{style}_{font}.ass"
                styles.render_ass(cues, ass, style=style, font_choice=font, split_screen=True)
                out = video.render_clip(self.source, 0.0, 2.0, ass, self.tmp / f"{style}_{font}.mp4", self.plan)
                dims = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height", "-of", "csv=p=0", str(out)],
                                      capture_output=True, text=True, check=True).stdout.strip()
                self.assertEqual(dims, "1080,1920", f"{style}/{font}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
