"""
Candidate segment generation + ranking with Ollama (Qwen).

1. build_candidates(): group Whisper segments into windows of ~20-60 s that
   start and end on sentence boundaries.
2. rank_candidates(): ask a local Qwen model (via Ollama) to score each window.
   If Ollama is unreachable, fall back to a simple heuristic so the pipeline
   still finishes.
3. select_best(): pick the top N non-overlapping windows, in video order.
"""

import json
import os
import re

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

MIN_LEN = 20.0
MAX_LEN = 60.0
TARGET_LEN = 40.0
MAX_CANDIDATES = 40  # keep the prompt small enough for a 3B model


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def build_candidates(segments: list[dict]) -> list[dict]:
    """
    Slide over Whisper sentence segments and emit every window whose duration
    is within [MIN_LEN, MAX_LEN]. Windows are then thinned so neighbouring
    windows don't start at nearly the same time.
    """
    cands = []
    n = len(segments)
    for i in range(n):
        start = segments[i]["start"]
        texts = []
        for j in range(i, n):
            texts.append(segments[j]["text"])
            end = segments[j]["end"]
            dur = end - start
            if dur > MAX_LEN:
                break
            if dur >= MIN_LEN:
                cands.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": " ".join(texts),
                })
                # Prefer windows near TARGET_LEN: stop once we pass it.
                if dur >= TARGET_LEN:
                    break

    # Thin: keep at most one candidate per ~8 s of start time (the one closest
    # to TARGET_LEN), so the model doesn't see dozens of near-duplicates.
    bucketed: dict[int, dict] = {}
    for c in cands:
        key = int(c["start"] // 8)
        best = bucketed.get(key)
        if best is None or abs((c["end"] - c["start"]) - TARGET_LEN) < abs((best["end"] - best["start"]) - TARGET_LEN):
            bucketed[key] = c
    cands = sorted(bucketed.values(), key=lambda c: c["start"])

    # Cap the count by sampling evenly across the video.
    if len(cands) > MAX_CANDIDATES:
        step = len(cands) / MAX_CANDIDATES
        cands = [cands[int(k * step)] for k in range(MAX_CANDIDATES)]

    for idx, c in enumerate(cands):
        c["id"] = idx
    return cands


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
PROMPT = """You are an expert short-form video editor. Below are candidate passages
from a video transcript, each 20-60 seconds long. Rate each passage from 0 to 10
on how well it would work as a standalone vertical short (TikTok / YouTube Short).

Prefer passages that:
- start strongly (a hook, a bold claim, a question, the start of a story);
- make sense on their own without the rest of the video;
- contain an interesting insight, story or opinion;
- have a clear ending or payoff;
- are self-contained and don't rely on visuals or earlier context.

Penalise passages that start or end mid-thought, are mostly filler, or are
intros/outros/sponsor reads.

Respond with ONLY a JSON array, no commentary, like:
[{{"id": 0, "score": 7}}, {{"id": 1, "score": 3}}]

Candidates:
{candidates}
"""


def _ollama_scores(cands: list[dict]) -> dict[int, float]:
    listing = "\n\n".join(
        f'[id={c["id"]}] ({c["end"] - c["start"]:.0f}s) {c["text"]}' for c in cands
    )
    body = {
        "model": OLLAMA_MODEL,
        "prompt": PROMPT.format(candidates=listing),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 16384},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=600)
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    return _parse_scores(raw)


def _parse_scores(raw: str) -> dict[int, float]:
    """Be tolerant: the model may wrap the array in an object or add text."""
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        # e.g. {"scores": [...]} or {"0": 7, "1": 3}
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
        else:
            data = [{"id": k, "score": v} for k, v in data.items()]
    scores: dict[int, float] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                scores[int(item["id"])] = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
    return scores


def _heuristic_scores(cands: list[dict]) -> dict[int, float]:
    """Fallback when Ollama is unavailable: favour dense speech near 40 s."""
    scores = {}
    for c in cands:
        dur = c["end"] - c["start"]
        words = len(c["text"].split())
        density = words / max(dur, 1)
        closeness = 1 - abs(dur - TARGET_LEN) / TARGET_LEN
        ends_clean = 1.0 if c["text"].rstrip()[-1:] in ".!?" else 0.5
        scores[c["id"]] = density * 2 + closeness * 3 + ends_clean * 2
    return scores


def rank_candidates(cands: list[dict]) -> tuple[list[dict], str]:
    """Return candidates with a 'score' field, plus a note about which ranker ran."""
    if not cands:
        return [], "no candidates"
    try:
        scores = _ollama_scores(cands)
        if not scores:
            raise ValueError("Ollama returned no parsable scores")
        note = f"ranked by {OLLAMA_MODEL}"
    except Exception as exc:  # noqa: BLE001
        scores = _heuristic_scores(cands)
        note = f"Ollama unavailable ({exc.__class__.__name__}); used heuristic ranking"
    for c in cands:
        c["score"] = scores.get(c["id"], 0.0)
    return cands, note


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def select_best(cands: list[dict], min_clips: int = 3, max_clips: int = 5) -> list[dict]:
    """Greedy pick by score, skipping anything that overlaps an already chosen clip."""
    chosen: list[dict] = []
    for c in sorted(cands, key=lambda c: c["score"], reverse=True):
        if any(c["start"] < o["end"] and o["start"] < c["end"] for o in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= max_clips:
            break
    chosen.sort(key=lambda c: c["start"])
    return chosen
