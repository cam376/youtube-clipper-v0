"""
Static client library export.

Proves: only clips marked in_library are exported, previews are separate
540x960 files, HD masters are untouched, the client JSON/HTML carry no
internal paths, and (when Playwright is installed) the page works without a
backend, persists selections in localStorage and builds the summary text.
"""

import hashlib
import http.server
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import manifest as mf  # noqa: E402
from library import export_library  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
try:
    from playwright.sync_api import sync_playwright  # noqa: E402
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False

FORBIDDEN_IN_CLIENT_FILES = ("output/", "output\\", "source.mp4", "transcript", "job_id", "plan", "c001", "clip_1",
                             "debug", "faces", ".filter", "/home/", "C:\\", "hd_file")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_hd(path: Path, seconds=2):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30", "-f", "lavfi", "-i", "sine=frequency=330",
                    "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)


def probe_dims(path: Path) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="kivro-export-"))
        cls.job = cls.tmp / "job"
        cls.libs = cls.tmp / "client_libraries"
        cls.job.mkdir()
        m = mf.new_manifest("abc123abc123", "https://youtu.be/x")
        for i in range(1, 4):
            make_hd(cls.job / f"clip_{i}.mp4")
            m["clips"].append({"id": f"c00{i}", "index": i, "label": f"Clip 0{i}", "start": 0.0, "end": 2.0, "duration": 2.0,
                               "cues": [], "style": "CLEAN", "font": "clean_sans", "in_library": i != 2,
                               "plan": None, "files": {"mp4": f"clip_{i}.mp4", "ass": f"clip_{i}.ass"}, "render_version": 1})
        mf.save(cls.job, m)
        cls.m = m
        cls.hd_hashes = {i: sha(cls.job / f"clip_{i}.mp4") for i in range(1, 4)}
        cls.summary = export_library(cls.job, cls.m, {"client_name": "Arnor", "whatsapp_number": "+33 6 12 34 56 78",
                                                      "watermark": True}, cls.libs)
        cls.lib_dir = Path(cls.summary["path"])
        cls.library = json.loads((cls.lib_dir / "library.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_only_selected_clips_exported(self):
        self.assertEqual(self.summary["clip_count"], 2)
        self.assertEqual([c["label"] for c in self.library["clips"]], ["Clip 01", "Clip 02"])
        self.assertEqual(sorted(p.name for p in (self.lib_dir / "previews").iterdir()),
                         ["c01.jpg", "c01.mp4", "c02.jpg", "c02.mp4"])

    def test_previews_are_separate_540x960_files(self):
        for cid in ("c01", "c02"):
            self.assertEqual(probe_dims(self.lib_dir / "previews" / f"{cid}.mp4"), "540,960")
            self.assertTrue((self.lib_dir / "previews" / f"{cid}.jpg").stat().st_size > 0)
        self.assertTrue((self.job / "preview_watermark.ass").is_file())

    def test_hd_masters_unchanged(self):
        self.assertEqual({i: sha(self.job / f"clip_{i}.mp4") for i in range(1, 4)}, self.hd_hashes)

    def test_client_json_has_no_internal_data(self):
        blob = json.dumps(self.library)
        for bad in FORBIDDEN_IN_CLIENT_FILES:
            self.assertNotIn(bad, blob, bad)
        self.assertEqual(set(self.library), {"library_id", "client_name", "title", "subtitle", "created_at", "show_price",
                                             "price_per_clip", "currency", "whatsapp_number", "watermarked_previews", "clips"})
        self.assertEqual(set(self.library["clips"][0]), {"id", "label", "preview", "poster", "duration"})
        self.assertEqual(self.library["whatsapp_number"], "33612345678")
        self.assertFalse(self.library["show_price"])
        self.assertIsNone(self.library["price_per_clip"])
        self.assertEqual(self.library["title"], "Arnor's Content Library")
        self.assertEqual(self.library["subtitle"], "2 clips prepared for you")

    def test_index_html_is_self_contained(self):
        html = (self.lib_dir / "index.html").read_text(encoding="utf-8")
        for bad in FORBIDDEN_IN_CLIENT_FILES:
            self.assertNotIn(bad, html, bad)
        self.assertIn('id="library-data"', html)
        self.assertIn("assets/app.js", html)
        self.assertTrue((self.lib_dir / "assets" / "app.js").is_file() and (self.lib_dir / "assets" / "style.css").is_file())
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))  # no backend calls

    def test_operator_map_stays_in_job_folder(self):
        maps = list(self.job.glob("export_*.json"))
        self.assertEqual(len(maps), 1)
        data = json.loads(maps[0].read_text(encoding="utf-8"))
        self.assertEqual([(c["library_clip_id"], c["job_clip_id"]) for c in data["clips"]], [("c01", "c001"), ("c02", "c003")])
        self.assertFalse(list(self.lib_dir.rglob("export_*.json")))

    def test_export_without_selection_fails(self):
        m = json.loads(json.dumps(self.m))
        for c in m["clips"]:
            c["in_library"] = False
        with self.assertRaises(ValueError):
            export_library(self.job, m, {"client_name": "X"}, self.libs)

    def test_no_watermark_option_skips_watermark_file(self):
        job2 = self.tmp / "job2"
        shutil.copytree(self.job, job2, ignore=shutil.ignore_patterns("preview_watermark.ass", "export_*.json"))
        m = mf.load(job2)
        export_library(job2, m, {"client_name": "B", "watermark": False, "show_price": True, "price_per_clip": 15}, self.libs)
        self.assertFalse((job2 / "preview_watermark.ass").exists())
        lib = json.loads((Path(m["libraries"][-1]["path"]) / "library.json").read_text(encoding="utf-8"))
        self.assertEqual((lib["show_price"], lib["price_per_clip"], lib["currency"]), (True, 15.0, "EUR"))

    @unittest.skipUnless(HAVE_PLAYWRIGHT, "playwright not installed (pip install playwright)")
    def test_static_page_works_without_backend(self):
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(self.lib_dir))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                import os
                exe = os.environ.get("PLAYWRIGHT_CHROMIUM") or ("/opt/pw-browsers/chromium" if Path("/opt/pw-browsers/chromium").exists() else None)
                browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else p.chromium.launch()
                page = browser.new_page(viewport={"width": 390, "height": 844})
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(f"http://127.0.0.1:{port}/index.html")
                page.wait_for_selector(".card")
                self.assertEqual(page.locator(".card").count(), 2)
                self.assertTrue(page.evaluate("document.getElementById('bar').hidden"))
                page.locator(".btn.select").nth(1).click()
                page.locator(".btn.select").nth(0).click()
                self.assertEqual(page.inner_text("#count"), "2 clips selected")
                page.reload()
                page.wait_for_selector(".card")
                self.assertEqual(page.locator(".card.selected").count(), 2)
                stored = json.loads(page.evaluate(f"localStorage.getItem('kivro-selection-{self.library['library_id']}')"))
                self.assertEqual(sorted(stored), ["c01", "c02"])
                summary = page.evaluate("window.kivroBuildSummary(['c02'])")
                self.assertEqual(summary.splitlines()[:4], ["Hi, here are the clips I'd like:", "", "Clip 02", ""])
                self.assertIn("Total selected: 1", summary)
                page.click("#send")
                self.assertIn("Clip 01\nClip 02", page.input_value("#summary"))
                href = page.get_attribute("#whatsapp", "href")
                self.assertTrue(href.startswith("https://wa.me/33612345678?text="))
                self.assertEqual(errors, [])
                browser.close()
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
