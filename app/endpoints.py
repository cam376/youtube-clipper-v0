"""
Natural clip endings: refine ONLY the end of a selected clip.

The ranker picks a strong start (hook) on a ~40 s window. This module keeps
that start and chooses where the idea actually finishes, using the
timestamped Whisper transcript:

1. candidate endpoints after the start: every word that ends a sentence
   (terminal punctuation . ! ? ...) and every Whisper segment end, within
   [start + CLIP_MIN_SECONDS, start + CLIP_HARD_MAX_SECONDS]; each carries
   the pause that follows it (gap to the next word);
2. semantic choice: Qwen (Ollama) gets the passage with [E1]..[En] markers
   inserted at the candidate endpoints and returns the id of the EARLIEST
   endpoint at which the idea introduced at the start is fully delivered;
   it can only answer with an id from the list, so the end always maps to a
   real transcript boundary;
3. fallback when Ollama is unavailable or answers with an unknown id: the
   last sentence end before CLIP_TARGET_MAX_SECONDS; if there is none, the
   first sentence end after it (still under CLIP_HARD_MAX_SECONDS); if there
   is none, the last transcript boundary before the target maximum.

Endpoints past CLIP_TARGET_MAX_SECONDS are offered only to finish a
sentence or thought that is already under way (the first terminal boundary
after the target max), never to keep talking. Nothing can exceed
CLIP_HARD_MAX_SECONDS.
"""

from __future__ import annotations

import json
import os
import re

import requests

from ranking import OLLAMA_MODEL, OLLAMA_URL

CLIP_MIN_SECONDS = float(os.environ.get("CLIP_MIN_SECONDS", "15"))
CLIP_TARGET_MAX_SECONDS = float(os.environ.get("CLIP_TARGET_MAX_SECONDS", "60"))
CLIP_HARD_MAX_SECONDS = float(os.environ.get("CLIP_HARD_MAX_SECONDS", "75"))
ENDPOINT_MAX_CANDIDATES = int(os.environ.get("ENDPOINT_MAX_CANDIDATES", "14"))

TERMINAL = re.compile(r"[.!?…]+[\"'»)]*$")


# --------------------------------------------------------------------------- #
# Candidate endpoints
# --------------------------------------------------------------------------- #
def candidate_endpoints(words: list[dict], segments: list[dict], start: float,
                        min_seconds: float | None = None, target_max: float | None = None,
                        hard_max: float | None = None) -> list[dict]:
    """
    Endpoints after `start`, time-ordered, each:
      {"id", "time", "rel", "terminal", "pause", "text"}
    `text` is the transcript from `start` up to the endpoint. Only the first
    terminal endpoint beyond target_max is kept (to finish a sentence).
    """
    min_seconds = CLIP_MIN_SECONDS if min_seconds is None else min_seconds
    target_max = CLIP_TARGET_MAX_SECONDS if target_max is None else target_max
    hard_max = CLIP_HARD_MAX_SECONDS if hard_max is None else hard_max

    ws = [w for w in words if w["end"] > start]
    seg_ends = {round(s["end"], 2) for s in segments if s["end"] > start}
    points: dict[float, dict] = {}
    for k, w in enumerate(ws):
        t = round(float(w["end"]), 2)
        nxt = ws[k + 1]["start"] if k + 1 < len(ws) else None
        pause = round(float(nxt - w["end"]), 2) if nxt is not None else 9.99
        terminal = bool(TERMINAL.search(w["word"]))
        is_seg_end = t in seg_ends
        if not (terminal or is_seg_end or pause >= 0.8):
            continue
        rel = t - start
        if rel < min_seconds or rel > hard_max:
            continue
        text = " ".join(x["word"] for x in ws[: k + 1])
        points[t] = {"time": t, "rel": round(rel, 2), "terminal": terminal, "pause": pause, "text": text}

    ordered = [points[t] for t in sorted(points)]
    # Beyond the target max keep only the first terminal boundary (finish the thought).
    within = [e for e in ordered if e["rel"] <= target_max]
    beyond = [e for e in ordered if e["rel"] > target_max and e["terminal"]][:1]
    ordered = within + beyond

    # Bound the list for the model: keep terminal/pause boundaries first.
    if len(ordered) > ENDPOINT_MAX_CANDIDATES:
        strong = [e for e in ordered if e["terminal"] or e["pause"] >= 0.5]
        if len(strong) >= 3:
            ordered = strong
        if len(ordered) > ENDPOINT_MAX_CANDIDATES:
            step = len(ordered) / ENDPOINT_MAX_CANDIDATES
            keep = sorted({int(k * step) for k in range(ENDPOINT_MAX_CANDIDATES)} | {len(ordered) - 1})
            ordered = [ordered[i] for i in keep]
    for i, e in enumerate(ordered, start=1):
        e["id"] = f"E{i}"
    return ordered


# --------------------------------------------------------------------------- #
# Semantic choice (Ollama + Qwen)
# --------------------------------------------------------------------------- #
PROMPT = """You are editing a short-form video clip. The clip starts with a strong hook and
must END exactly where the idea introduced at the start is fully delivered: the
explanation is complete, the conclusion or punchline has landed, and the last
sentence is finished.

Below is the transcript from the clip start. Possible endpoints are marked
inline as [E1], [E2], ... Each marker sits right after the last word of the
clip if it ended there.

Choose the EARLIEST endpoint that fully resolves the idea. Do not stop right
after a point is introduced but before it is explained. Do not continue into
a new, different topic. Prefer endpoints after a complete sentence or a pause.

Endpoints (id: seconds from the clip start, pause after it):
{listing}

Transcript:
{passage}

Respond with ONLY JSON like {{"endpoint": "E4", "reason": "short reason"}}.
"""


def _passage_with_markers(endpoints: list[dict]) -> str:
    out, prev_len = [], 0
    for e in endpoints:
        chunk = e["text"][prev_len:].strip()
        out.append(f"{chunk} [{e['id']}]")
        prev_len = len(e["text"])
    return " ".join(out)


def _ollama_choose(endpoints: list[dict]) -> tuple[str | None, str]:
    listing = "\n".join(f'{e["id"]}: {e["rel"]:.1f}s, pause {e["pause"]:.1f}s'
                        + (", sentence end" if e["terminal"] else "") for e in endpoints)
    body = {
        "model": OLLAMA_MODEL,
        "prompt": PROMPT.format(listing=listing, passage=_passage_with_markers(endpoints)),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=300)
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    eid = str(data.get("endpoint", "")).strip().upper()
    reason = str(data.get("reason", "")).strip()[:200]
    return (eid or None), reason


# --------------------------------------------------------------------------- #
# Fallback
# --------------------------------------------------------------------------- #
def choose_endpoint_heuristic(endpoints: list[dict], target_max: float | None = None) -> tuple[dict, str]:
    target_max = CLIP_TARGET_MAX_SECONDS if target_max is None else target_max
    within = [e for e in endpoints if e["rel"] <= target_max]
    terminal = [e for e in within if e["terminal"]]
    if terminal:
        return terminal[-1], "fallback: last complete sentence before the target maximum"
    beyond = [e for e in endpoints if e["rel"] > target_max and e["terminal"]]
    if beyond:
        return beyond[0], "fallback: first sentence end after the target maximum (finishing the thought)"
    if within:
        return within[-1], "fallback: last transcript boundary before the target maximum (no sentence end found)"
    return endpoints[-1], "fallback: only boundary available"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def refine_end(start: float, end: float, words: list[dict], segments: list[dict],
               use_llm: bool = True) -> dict:
    """
    Return a refinement record:
      {"original_start", "original_end", "final_start", "final_end",
       "original_duration", "final_duration", "endpoint_time", "endpoint_id",
       "endpoint_reason", "method": "semantic" | "fallback" | "unchanged",
       "candidates": n, "candidate_times": [...]}
    The start is never changed. If no endpoint exists in range, the original
    end is kept (method "unchanged").
    """
    rec = {"original_start": start, "original_end": end, "final_start": start,
           "original_duration": round(end - start, 2)}
    endpoints = candidate_endpoints(words, segments, start)
    rec["candidates"] = len(endpoints)
    rec["candidate_times"] = [e["time"] for e in endpoints]
    if not endpoints:
        rec.update(final_end=end, final_duration=round(end - start, 2), endpoint_time=end, endpoint_id=None,
                   endpoint_reason="no transcript boundary in the allowed range; original end kept",
                   method="unchanged")
        return rec

    chosen, reason, method = None, "", "fallback"
    if use_llm:
        try:
            eid, llm_reason = _ollama_choose(endpoints)
            match = next((e for e in endpoints if e["id"] == eid), None)
            if match is not None:
                chosen, reason, method = match, llm_reason or "chosen by the model", "semantic"
            else:
                reason = f"model answered '{eid}', not a listed endpoint; "
        except Exception as exc:  # noqa: BLE001
            reason = f"Ollama unavailable ({exc.__class__.__name__}); "
    if chosen is None:
        chosen, why = choose_endpoint_heuristic(endpoints)
        reason = reason + why
        method = "fallback"

    rec.update(final_end=chosen["time"], final_duration=round(chosen["time"] - start, 2),
               endpoint_time=chosen["time"], endpoint_id=chosen["id"], endpoint_reason=reason, method=method,
               endpoint_terminal=chosen["terminal"], endpoint_pause=chosen["pause"])
    return rec


def text_between(words: list[dict], start: float, end: float) -> str:
    return " ".join(w["word"] for w in words if w["end"] > start and w["start"] < end)
