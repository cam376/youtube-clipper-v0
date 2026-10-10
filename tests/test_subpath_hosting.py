"""
The app must work when served under a URL prefix behind a reverse proxy that
strips it (https://settermonster.com/vezly.ai/ -> app sees /), and still at
the root (http://localhost:8000/). That holds only if every URL the frontend
and the API emit is relative to the page.

The Playwright test mounts the real app under /vezly.ai on a throwaway
server (same behaviour as Caddy's handle_path) and drives the page.
"""

import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False


class RelativeUrlTest(unittest.TestCase):
    def test_frontend_uses_no_root_absolute_urls(self):
        bad = []
        for f in ("app.js", "editor.js", "export.js", "index.html"):
            for i, line in enumerate((STATIC / f).read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r'["`\']/(api|static|output|libraries)/', line):
                    bad.append(f"{f}:{i}: {line.strip()[:80]}")
        self.assertEqual(bad, [])

    def test_api_payloads_use_relative_urls(self):
        sys.path.insert(0, str(ROOT / "app"))
        import manifest as mf
        m = mf.new_manifest("abc123abc123", "u")
        m["clips"] = [{"id": "c001", "files": {"mp4": "clip_1.mp4", "ass": "clip_1.ass"}, "render_version": 1}]
        v = mf.public_view(m, "output/abc123abc123")
        self.assertFalse(v["clips"][0]["url"].startswith("/"))
        src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        self.assertNotIn('f"/output/', src)
        self.assertNotIn('f"/libraries/', src)
        self.assertIn("root_path=ROOT_PATH", src)


@unittest.skipUnless(HAVE_PLAYWRIGHT, "playwright not installed (pip install playwright)")
class SubPathPlaywrightTest(unittest.TestCase):
    PREFIX = "/vezly.ai"

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="kivro-subpath-"))
        os.environ["OUTPUT_DIR"] = str(cls.tmp / "output")
        os.environ["LIBRARIES_DIR"] = str(cls.tmp / "libs")
        os.environ["ROOT_PATH"] = cls.PREFIX
        sys.path.insert(0, str(ROOT / "app"))
        for mod in ("main",):
            sys.modules.pop(mod, None)
        import main  # noqa: WPS433
        import manifest as mf
        import uvicorn
        from fastapi import FastAPI

        # a finished job on disk so the page has something to list and a clip URL to resolve
        job = cls.tmp / "output" / "abc123abc123"
        job.mkdir(parents=True)
        (job / "clip_1.mp4").write_bytes(b"\x00" * 64)
        m = mf.new_manifest("abc123abc123", "https://youtu.be/x")
        m.update(stage="done", status="Done.")
        m["source"]["title"] = "Sub-path test"
        m["clips"] = [{"id": "c001", "index": 1, "label": "Clip 01", "start": 0.0, "end": 20.0, "duration": 20.0,
                       "score": 8.0, "text": "t", "layout": "CENTER_CROP", "layout_note": "", "plan": None, "cues": [],
                       "style": "CLEAN", "font": "clean_sans", "font_family": "Arial", "font_note": "", "edited": False,
                       "in_library": False, "render_state": "done", "render_error": None, "render_version": 1,
                       "files": {"mp4": "clip_1.mp4", "ass": "clip_1.ass"}}]
        mf.save(job, m)

        outer = FastAPI()
        outer.mount(cls.PREFIX, main.app)          # like Caddy handle_path: the app sees paths without the prefix
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]
        cls.server = uvicorn.Server(uvicorn.Config(outer, host="127.0.0.1", port=cls.port, log_level="warning"))
        threading.Thread(target=cls.server.run, daemon=True).start()
        time.sleep(1.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        for k in ("OUTPUT_DIR", "LIBRARIES_DIR", "ROOT_PATH"):
            os.environ.pop(k, None)

    def test_page_api_and_clip_resolve_under_the_prefix(self):
        base = f"http://127.0.0.1:{self.port}{self.PREFIX}/"
        with sync_playwright() as p:
            exe = "/opt/pw-browsers/chromium" if Path("/opt/pw-browsers/chromium").exists() else None
            browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else p.chromium.launch()
            page = browser.new_page()
            failed, errors = [], []
            page.on("requestfailed", lambda r: failed.append(r.url))
            page.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base)
            page.wait_for_selector("#jobs-list li")
            page.click("#jobs-list li")
            page.wait_for_selector(".clip")
            src = page.get_attribute(".clip video", "src")
            self.assertTrue(src.startswith("output/abc123abc123/clip_1.mp4"), src)
            resolved = page.evaluate("document.querySelector('.clip video').currentSrc || new URL(document.querySelector('.clip video').getAttribute('src'), location.href).href")
            self.assertTrue(resolved.startswith(base + "output/"), resolved)
            href = page.evaluate("new URL(document.querySelector('a.download').getAttribute('href'), location.href).href")
            self.assertTrue(href.startswith(base + "output/abc123abc123/clip_1.mp4"), href)
            self.assertEqual(errors, [])
            self.assertEqual([f for f in failed if "favicon" not in f], [])
            # the API itself answers under the prefix and emits relative links
            data = page.evaluate(f"fetch('{base}api/jobs/abc123abc123').then(r => r.json())")
            self.assertEqual(data["clips"][0]["url"].split("?")[0], "output/abc123abc123/clip_1.mp4")
            health = page.evaluate(f"fetch('{base}health').then(r => r.json())")
            self.assertEqual(health["root_path"], self.PREFIX)
            browser.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
