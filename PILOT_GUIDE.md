# Kivro v0.3 — client pilot workflow

The path from a client's YouTube URL to the list of clips they will post.
Everything runs locally on Windows; the only thing that leaves your machine
is a static folder you host for the client.

## 1. Generate

1. `python app\main.py`, open http://localhost:8000.
2. Paste the client's YouTube URL, click **Generate Clips**.
3. Kivro scores every 20-60 s candidate window and keeps **all** that reach
   the quality threshold (default `MIN_CLIP_SCORE=7` on the 0-10 ranking
   scale). A 27-minute video can return 3 clips or 30. The status line says
   "N clips ready"; the summary line shows how many candidates were scored,
   the max and median score, and the threshold.
4. There is no minimum: a video with no moment above the threshold returns
   0 clips. Lower `MIN_CLIP_SCORE` only if you judge the rejected candidates
   in `ranking.json` to be genuinely strong.
5. Jobs are persistent: close the server, reopen later, click the job in the
   **Jobs** list. Everything (clips, edits, styles, selection) is in
   `output\<job_id>\job.json`.

Tuning for a specific client (PowerShell, before starting the server):
```powershell
$env:MIN_CLIP_SCORE = "8"     # stricter
$env:MIN_CLIP_SCORE = "6.5"   # more permissive
```
`output\<job_id>\ranking.json` lists every candidate with its score, sorted,
so you can see where the threshold falls for this speaker.

## 2. Review and polish

Each clip card shows: preview, Clip NN, Score (0-100), duration, layout
(split-screen / face-centred / centre crop), subtitle style, font, badges.

- **Edit captions**: one row per caption cue with fixed start/end times.
  Fix the text, **Save & rerender**. Only that clip rerenders (15-40 s);
  nothing is re-downloaded, re-transcribed, re-ranked or re-analysed.
- **Find & replace in all clips**: fix a name once for the whole job.
  **Preview affected clips** lists which clips and cues match before anything
  changes; **Replace & rerender** applies it and rerenders only those clips.
- **Style + Font dropdowns + Apply**: CLEAN / BOLD / KARAOKE / MINIMAL and
  Clean Sans / Heavy Sans / Condensed / Classic, per clip. Rerenders that clip.
  A **font fallback** badge means the first-choice family is not installed on
  this machine and tells you which one was used instead.
- **Glossary**: names, brands, acronyms for this client, kept with the job.
  It is the reference spelling for your corrections; Whisper does not read it
  (see "Whisper and names" below).
- **Download HD**: the 1080x1920 master.
- **Rerender**: rebuild the clip from its current captions/style/font.

Rerenders run one at a time in the background; cards show "queued…" /
"rendering…" and refresh themselves.

## 3. Select and export

1. Click **Add to client library** on the clips the client should see. You
   can generate 30 and show 22.
2. **Export client library**: client name, title (prefilled
   "<Client>'s Content Library"), optional WhatsApp number (digits with
   country code, e.g. `33612345678`), watermark checkbox (default on), and an
   optional price per clip (off by default for the pilot).
3. Kivro writes `client_libraries\<client>-<date>-<id>\` with:
   - `index.html` — the client page, self-contained
   - `library.json` — the same client-facing data
   - `previews\cNN.mp4` — 540x960 web previews with the "KIVRO PREVIEW" mark
   - `previews\cNN.jpg` — poster frames
   - `assets\` — style.css, app.js
   The HD masters in `output\<job_id>\` are not touched. The mapping from
   library clip numbers to your job clips is in
   `output\<job_id>\export_<library_id>.json`, never in the exported folder.
4. Preview it right away from the result box, or standalone:
   ```powershell
   python -m http.server 8080 --directory "client_libraries\<library_id>"
   ```
   then open http://localhost:8080 on your phone too (same Wi-Fi, use your PC's IP).

## 4. Put it online for the client

The folder is a plain static site. Any of these works, no backend needed:

- **Netlify Drop** (https://app.netlify.com/drop): drag the folder, get a
  URL in seconds. Rename the site to something like `arnor-kivro`.
- **Cloudflare Pages**: create a project, "Upload assets", drop the folder.
- **GitHub Pages**: push the folder to a repo, enable Pages.
- Any web host / S3 bucket with static hosting: upload the folder as is.

The page has `noindex`; keep the URL private. Send it to the client with one
line: "Watch, tap Select on the ones you'd post, then Send my selection."

## 5. Receive the selection

The client taps **Send my selection**. The page shows the summary and copies it:

```
Hi, here are the clips I'd like:

Clip 02
Clip 04
Clip 07

Total selected: 3

Library: Arnor's Content Library (arnor-20261008-3ad7c2)
```

If you entered a WhatsApp number, a **Send on WhatsApp** button opens a chat
to you with that text. Selections are kept in the client's browser
(localStorage) so a refresh does not lose them. Map the clip numbers back to
your HD files with `output\<job_id>\export_<library_id>.json`.

## Checking a face-centred clip

Each clip's layout note (in `job.json` under `layout_note`, and in the
console log line `clip N framing:`) states, for a face-centred clip, the
primary track id, its coverage and span, the median face x in the source,
the crop x range actually used, the geometric-centre x it did not use, and
how many samples were held or interpolated. `job.json` also stores the
per-sample target and smoothed crop x under `plan.diagnostics`. With
`DEBUG_FACES=true`, `clip_N_faces.jpg` draws the primary track in a thick
box labelled PRIMARY, the crop window and its centre line; a grey box
labelled GEOMETRIC CENTRE appears only on centre-crop clips.

The note also gives the face position inside the rendered crop as a
percentage of the crop width (50 % = centred) and says if the crop was
clamped at a source edge, which happens when the face is closer to the edge
of the 16:9 frame than half the crop width: there are no more pixels to show
on that side, so the face cannot be centred without zooming in.

For a second-by-second view of one clip:
```powershell
python app\framing_report.py output\<job_id>\job.json c001
```
It prints, per sample, the target crop x, the rendered crop x, the face
position inside the crop, where that position came from (`primary`,
`handoff:#id` when another track of the same person was followed after a
source edit, `interpolated` when nothing was visible) and flags for
CLAMPED-LEFT / CLAMPED-RIGHT / OFF-CENTRE / PRIMARY ABSENT with the visible
face that was deliberately not followed and why (co-present second person,
face size outside 0.5x-2x the primary, fewer than 2 s of detections). Paste
that output when reporting a framing problem.

## Whisper and names

Whisper `small` stays the default. faster-whisper supports an
`initial_prompt` that can bias spelling toward a list of names, but it can
also make the model repeat or invent text on some audio, so v0.3 does not
enable it. The glossary is stored for that future step; today the reliable
tools are the caption editor and Find & replace.
