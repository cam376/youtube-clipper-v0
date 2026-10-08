"""
Caption cues: the editable unit between Whisper words and the ASS file.

A cue is {"start", "end", "text", "words": [{"start", "end", "word"}]} with
times relative to the clip start. Cues are built once from Whisper word
timestamps (the same grouping v0.2.1 used) and stored in job.json; editing
changes only the text.

Timing strategy when text is edited (no re-transcription, no LLM):
  * the cue's start/end never change;
  * if the edited text has the same number of tokens as the original words,
    each token keeps the original word's start/end (a misspelling fix keeps
    perfect karaoke timing);
  * otherwise the tokens are spread evenly over the cue duration, each token
    getting an equal share, deterministically.
"""

from __future__ import annotations

import re

MAX_WORDS_PER_CUE = 4
MAX_GAP_SECONDS = 1.0


def build_cues(words: list[dict], clip_start: float, clip_end: float,
               max_words: int = MAX_WORDS_PER_CUE) -> list[dict]:
    """Group transcript words into short cues, times relative to clip_start."""
    clip_words = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    cues: list[dict] = []
    chunk: list[dict] = []

    def flush():
        if not chunk:
            return
        rel = [{"start": round(w["start"] - clip_start, 3), "end": round(w["end"] - clip_start, 3),
                "word": w["word"]} for w in chunk]
        cues.append({
            "start": rel[0]["start"],
            "end": rel[-1]["end"],
            "text": " ".join(w["word"] for w in rel),
            "words": rel,
        })
        chunk.clear()

    for w in clip_words:
        if chunk and (len(chunk) >= max_words or w["start"] - chunk[-1]["end"] > MAX_GAP_SECONDS):
            flush()
        chunk.append(w)
        if w["word"][-1:] in ".!?,":
            flush()
    flush()
    return cues


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def set_cue_text(cue: dict, new_text: str) -> dict:
    """Return a copy of `cue` with the new text and retimed words (see module doc)."""
    tokens = tokenize(new_text)
    start, end = float(cue["start"]), float(cue["end"])
    original = cue.get("words") or []
    if tokens and len(tokens) == len(original):
        words = [{"start": w["start"], "end": w["end"], "word": tok} for w, tok in zip(original, tokens)]
    elif tokens:
        slot = (end - start) / len(tokens)
        words = [{"start": round(start + i * slot, 3), "end": round(start + (i + 1) * slot, 3), "word": tok}
                 for i, tok in enumerate(tokens)]
    else:
        words = []
    return {"start": start, "end": end, "text": " ".join(tokens), "words": words}


def find_in_cues(cues: list[dict], find: str, case_sensitive: bool = False) -> list[int]:
    """Indexes of cues whose text contains `find`."""
    if not find:
        return []
    hay = (lambda s: s) if case_sensitive else (lambda s: s.lower())
    needle = hay(find)
    return [i for i, c in enumerate(cues) if needle in hay(c["text"])]


def replace_in_cues(cues: list[dict], find: str, replace: str, case_sensitive: bool = False) -> tuple[list[dict], int]:
    """Apply find/replace to every matching cue (text retimed per set_cue_text). Returns (new cues, count)."""
    if not find:
        return [dict(c) for c in cues], 0
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(find), flags)
    out, count = [], 0
    for c in cues:
        new_text, n = pattern.subn(lambda m: replace, c["text"])
        if n:
            count += n
            out.append(set_cue_text(c, new_text))
        else:
            out.append(dict(c))
    return out, count


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"
