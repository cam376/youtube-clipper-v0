"""
Natural clip endings: refine ONLY the end of a selected clip.

The ranker picks a strong start (hook) on a ~40 s window. This module keeps
that start and chooses where the idea actually finishes, using the
timestamped Whisper transcript:

1. candidate endpoints after the start: every word that ends a sentence
   (terminal punctuation . ! ? ...), every Whisper segment end, and every
   pause >= 0.8 s, within [start + CLIP_MIN_SECONDS, start + CLIP_HARD_MAX_SECONDS];
   each carries the pause that follows it and the words that follow it;
2. deterministic continuity guards reject candidates that obviously do not
   end the thought: a dangling connector at the cut, a question whose answer
   follows, a setup sentence ("laisse-moi te dire une chose", "voici
   pourquoi", "il y a deux raisons"...), or following words that start with
   a continuation ("et", "donc", "parce que", "par exemple", "premièrement"...);
3. semantic judgement, one candidate at a time in time order: Qwen (Ollama)
   sees the hook, the text up to the candidate and the next ~25 words, and
   answers COMPLETE or CONTINUES. It never picks an endpoint id. The first
   COMPLETE candidate that passed the guards wins (with one look-ahead
   preferring a longer pause within 4 s);
4. fallback when Ollama is unavailable or nothing is judged complete: the
   last guard-passing sentence end before CLIP_TARGET_MAX_SECONDS, then the
   last sentence end, then the first sentence end after the target max
   (never past CLIP_HARD_MAX_SECONDS), then the last boundary.

Nothing here prefers a duration: a 22 s complete thought ends at 22 s.
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
ENDPOINT_LOOKAHEAD_SECONDS = 4.0
NEXT_WORDS = 25

TERMINAL = re.compile(r"[.!?…]+[\"'»)]*$")

COMPLETE, CONTINUES, UNPARSED = "COMPLETE", "CONTINUES", "UNPARSED"

# --------------------------------------------------------------------------- #
# Deterministic continuity guards (French first, a few English equivalents)
# --------------------------------------------------------------------------- #
CONTINUATION_STARTS = [
    "et ", "et,", "donc", "mais", "parce que", "parce qu'", "car ", "alors", "ensuite", "en fait", "c'est-à-dire",
    "c'est à dire", "par exemple", "notamment", "premièrement", "deuxièmement", "troisièmement", "deuxième", "troisième",
    "si ", "s'il", "ce qui", "ce que", "pour ", "puis ", "du coup", "aussi", "également", "d'abord", "enfin", "surtout",
    "sauf que", "même si", "c'est pour ça", "c'est pourquoi", "ça veut dire", "autrement dit", "en plus", "après ",
    "and ", "so ", "but ", "because", "for example", "which means", "first", "second", "third", "then ", "also ",
]
DANGLING_ENDS = [
    "parce que", "parce qu'", "et", "mais", "donc", "car", "que", "qui", "de", "du", "des", "à", "au", "aux", "pour", "avec",
    "sans", "sur", "dans", "en", "comme", "si", "ou", "où", "quand", "le", "la", "les", "un", "une", "ce", "cette", "ces",
    "c'est", "il y a", "and", "but", "because", "so", "the", "a", "to", "of", "with", "if", "that", "which",
]
SETUP_PATTERNS = [
    r"laisse[-\s]?moi (te|vous) (dire|expliquer|montrer|raconter)", r"laissez[-\s]?moi (vous )?(dire|expliquer|montrer)",
    r"je vais (te|vous) (dire|expliquer|montrer|raconter|donner|partager)", r"je vais (t'|vous )expliquer",
    r"voici pourquoi", r"voilà pourquoi", r"voici comment", r"voilà comment", r"voici ce que", r"voilà ce que",
    r"il y a (deux|trois|quatre|cinq|\d+) (raisons|choses|étapes|points|erreurs|façons|manières|règles)",
    r"(la |le )?premi(ère|er) (étape|chose|raison|point|règle)", r"une (seule )?chose", r"deux choses",
    r"la question (c'est|est)", r"le (truc|secret|problème) c'est", r"ce qui (se passe|est intéressant)",
    r"(tu|vous) (sais|savez) (ce que|pourquoi|quoi)", r"let me (tell|explain|show)", r"here'?s why", r"here'?s how",
    r"there are (two|three|\d+)", r"the first (thing|step|reason)", r"one thing",
]
_SETUP_RE = re.compile("|".join(f"(?:{p})" for p in SETUP_PATTERNS), re.IGNORECASE)


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").strip()
    return re.sub(r"^[\s\"'«»(\[-]+", "", s)


def last_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return parts[-1] if parts else text


def continuity_guard(text_at_candidate: str, next_text: str, terminal: bool, pause: float) -> str | None:
    """
    Return a reason string when the candidate obviously does NOT end the
    thought, else None. Pure text rules, no model.
    """
    tail = _norm(text_at_candidate)
    tail_words = re.sub(r"[\"'»)]+$", "", tail)
    if not terminal:
        if pause < 1.0:
            return "no sentence end and no clear pause at the cut"
    stripped = re.sub(r"[.!?…\"'»)]+$", "", tail_words).strip()
    for d in sorted(DANGLING_ENDS, key=len, reverse=True):
        if stripped == d or stripped.endswith(" " + d):
            return f"cut ends on a dangling word ('{d}')"
    sent = last_sentence(text_at_candidate)
    nxt = _norm(next_text)
    if sent.rstrip().endswith("?") and nxt:
        return "cut ends on a question whose answer follows"
    if _SETUP_RE.search(sent):
        return f"cut ends on a setup sentence ('{sent.strip()[:60]}')"
    if sent.rstrip().endswith(":"):
        return "cut ends on a colon (something is about to be listed)"
    for c in CONTINUATION_STARTS:
        if nxt.startswith(c):
            return f"following words continue the thought ('{nxt[:40]}')"
    return None


# --------------------------------------------------------------------------- #
# Candidate endpoints
# --------------------------------------------------------------------------- #
def candidate_endpoints(words: list[dict], segments: list[dict], start: float,
                        min_seconds: float | None = None, target_max: float | None = None,
                        hard_max: float | None = None) -> list[dict]:
    """
    Endpoints after `start`, time-ordered, each:
      {"id", "time", "rel", "terminal", "pause", "text", "next_text", "guard"}
    `text` is the transcript from `start` up to the endpoint; `next_text` the
    following words; `guard` a rejection reason from continuity_guard or None.
    Only the first terminal endpoint beyond target_max is kept.
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
        next_text = " ".join(x["word"] for x in ws[k + 1: k + 1 + NEXT_WORDS])
        points[t] = {"time": t, "rel": round(rel, 2), "terminal": terminal, "pause": pause,
                     "text": text, "next_text": next_text,
                     "guard": continuity_guard(text, next_text, terminal, pause)}

    ordered = [points[t] for t in sorted(points)]
    within = [e for e in ordered if e["rel"] <= target_max]
    beyond = [e for e in ordered if e["rel"] > target_max and e["terminal"]][:1]
    ordered = within + beyond

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
# Semantic judgement: one candidate, COMPLETE or CONTINUES (no id choice)
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """You are editing a short-form video clip from a spoken transcript.

The clip opens with this hook:
"{hook}"

Here is the transcript from the clip start up to a possible cut point:
"{body}"

If the clip stopped exactly there, these are the words the speaker says next:
"{next_text}"

Pause after the cut: {pause:.1f} seconds.

Question: if the clip stops exactly at the cut point, has the main idea promised
by the hook been COMPLETELY delivered (explanation finished, conclusion or
punchline landed, nothing essential left hanging)? Or do the following words
CONTINUE or explain that same idea (an example, a reason, a second point, the
answer to a question, the actual advice after a setup)?

Answer with ONLY JSON: {{"verdict": "<COMPLETE or CONTINUES>", "reason": "<one short sentence>"}}
"""


def parse_verdict(raw: str) -> tuple[str, str]:
    """
    Strictly parse a model answer. Anything other than an explicit COMPLETE
    or CONTINUES verdict is UNPARSED, which never counts as complete and never
    maps to any endpoint.
    """
    data = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*?\}", raw or "", re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return UNPARSED, "no JSON object in the answer"
    v = str(data.get("verdict", "")).strip().upper()
    reason = str(data.get("reason", "")).strip()[:200]
    if v.startswith("COMPLETE"):
        return COMPLETE, reason
    if v.startswith("CONTINUE"):
        return CONTINUES, reason
    return UNPARSED, f"verdict field was '{v[:30]}'"


def _ollama_judge(hook: str, body: str, next_text: str, pause: float) -> tuple[str, str, str]:
    """Return (verdict, reason, raw_response)."""
    body_words = body.split()
    if len(body_words) > 140:
        body = "... " + " ".join(body_words[-140:])
    req = {
        "model": OLLAMA_MODEL,
        "prompt": JUDGE_PROMPT.format(hook=hook, body=body, next_text=next_text, pause=pause),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_ctx": 4096},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=req, timeout=180)
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    verdict, reason = parse_verdict(raw)
    return verdict, reason, raw[:300]


def hook_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    first = parts[0] if parts else text
    return " ".join(first.split()[:30])


# --------------------------------------------------------------------------- #
# Fallback
# --------------------------------------------------------------------------- #
def choose_endpoint_heuristic(endpoints: list[dict], target_max: float | None = None) -> tuple[dict, str]:
    target_max = CLIP_TARGET_MAX_SECONDS if target_max is None else target_max
    within = [e for e in endpoints if e["rel"] <= target_max]
    clean = [e for e in within if e["terminal"] and not e.get("guard")]
    if clean:
        return clean[-1], "fallback: last complete sentence before the target maximum that passes the continuity guards"
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
# Selection
# --------------------------------------------------------------------------- #
def judge_endpoints(endpoints: list[dict], hook: str, judge=None) -> tuple[dict | None, list[dict], int]:
    """
    Walk candidates in time order. Guard-rejected ones are CONTINUES without a
    model call. The first COMPLETE wins, unless a COMPLETE candidate within
    ENDPOINT_LOOKAHEAD_SECONDS has a longer pause. Returns (chosen or None,
    judgement records, model calls made). `judge` raises on Ollama failure.
    """
    judge = judge or _ollama_judge
    records, calls = [], 0
    chosen = None
    for e in endpoints:
        rec = {"id": e["id"], "rel": e["rel"], "terminal": e["terminal"], "pause": e["pause"], "guard": e.get("guard")}
        if e.get("guard"):
            rec.update(verdict=CONTINUES, reason=e["guard"], raw=None, source="guard")
            records.append(rec)
            continue
        if chosen is not None and e["rel"] - chosen["rel"] > ENDPOINT_LOOKAHEAD_SECONDS:
            break
        verdict, reason, raw = judge(hook, e["text"], e["next_text"], e["pause"])
        calls += 1
        rec.update(verdict=verdict, reason=reason, raw=raw, source="model")
        records.append(rec)
        if verdict == COMPLETE:
            if chosen is None:
                chosen = e
            elif e["pause"] > chosen["pause"] + 0.3:
                chosen = e          # same thought boundary within the look-ahead, cleaner pause
                break
            else:
                break
    return chosen, records, calls


def refine_end(start: float, end: float, words: list[dict], segments: list[dict],
               use_llm: bool = True, judge=None) -> dict:
    """
    Return a refinement record:
      {"original_start", "original_end", "final_start", "final_end",
       "original_duration", "final_duration", "endpoint_time", "endpoint_id",
       "endpoint_reason", "method": "semantic" | "fallback" | "unchanged",
       "candidates", "candidate_times", "judgements", "llm_calls"}
    The start is never changed. If no endpoint exists in range, the original
    end is kept (method "unchanged").
    """
    rec = {"original_start": start, "original_end": end, "final_start": start,
           "original_duration": round(end - start, 2)}
    endpoints = candidate_endpoints(words, segments, start)
    rec["candidates"] = len(endpoints)
    rec["candidate_times"] = [e["time"] for e in endpoints]
    rec["judgements"] = []
    rec["llm_calls"] = 0
    if not endpoints:
        rec.update(final_end=end, final_duration=round(end - start, 2), endpoint_time=end, endpoint_id=None,
                   endpoint_reason="no transcript boundary in the allowed range; original end kept",
                   method="unchanged")
        return rec

    hook = hook_sentence(endpoints[0]["text"])
    chosen, reason, method = None, "", "fallback"
    if use_llm:
        try:
            chosen, records, calls = judge_endpoints(endpoints, hook, judge)
            rec["judgements"], rec["llm_calls"] = records, calls
            if chosen is not None:
                j = next(r for r in records if r["id"] == chosen["id"])
                reason, method = f"judged COMPLETE: {j['reason']}", "semantic"
            else:
                reason = "no candidate judged COMPLETE; "
        except Exception as exc:  # noqa: BLE001
            reason = f"Ollama unavailable ({exc.__class__.__name__}); "
    if chosen is None:
        chosen, why = choose_endpoint_heuristic(endpoints)
        reason = reason + why
        method = "fallback"

    rec.update(final_end=chosen["time"], final_duration=round(chosen["time"] - start, 2),
               endpoint_time=chosen["time"], endpoint_id=chosen["id"], endpoint_reason=reason, method=method,
               endpoint_terminal=chosen["terminal"], endpoint_pause=chosen["pause"],
               endpoint_guard=chosen.get("guard"))
    return rec


def text_between(words: list[dict], start: float, end: float) -> str:
    return " ".join(w["word"] for w in words if w["end"] > start and w["start"] < end)
