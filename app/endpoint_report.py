"""
Read-only report on the natural-ending decisions of a job.

    python app/endpoint_report.py output/<job_id>/job.json            # all clips
    python app/endpoint_report.py output/<job_id>/job.json c003       # one clip, verbose

Per clip: original and final duration, chosen endpoint, semantic/fallback,
reason, the candidate endpoints that were available (recomputed from
transcript.json with the current policy), whether the final end equals the
original ~40 s end and why, punctuation density of the transcript in the
window, and the words just before and after the final end so a cut can be
judged without opening the video. Nothing is modified.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from endpoints import candidate_endpoints, CLIP_MIN_SECONDS, CLIP_TARGET_MAX_SECONDS, CLIP_HARD_MAX_SECONDS, TERMINAL  # noqa: E402


def words_around(words, t, before=12, after=10):
    idx = [i for i, w in enumerate(words) if w["end"] <= t + 1e-6]
    if not idx:
        return "", ""
    k = idx[-1]
    pre = " ".join(w["word"] for w in words[max(0, k - before + 1): k + 1])
    post = " ".join(w["word"] for w in words[k + 1: k + 1 + after])
    return pre, post


def punctuation_stats(words, start, end):
    ws = [w for w in words if w["end"] > start and w["start"] < end]
    term = sum(1 for w in ws if TERMINAL.search(w["word"]))
    return len(ws), term


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    job_path = Path(argv[1])
    only = argv[2] if len(argv) > 2 else None
    m = json.loads(job_path.read_text(encoding="utf-8"))
    tpath = job_path.parent / "transcript.json"
    transcript = json.loads(tpath.read_text(encoding="utf-8")) if tpath.is_file() else None
    words = transcript["words"] if transcript else []
    segments = transcript["segments"] if transcript else []

    sel = m.get("selection", {})
    er = sel.get("end_refinement", {})
    print(f"job {m.get('job_id')}  clips {len(m.get('clips', []))}  policy min {CLIP_MIN_SECONDS} target {CLIP_TARGET_MAX_SECONDS} hard {CLIP_HARD_MAX_SECONDS}")
    print(f"recorded at generation: {er}")
    if sel.get("dropped_after_end_refinement"):
        print(f"dropped after refinement: {sel['dropped_after_end_refinement']}")
    if transcript:
        n_words = len(words)
        n_term = sum(1 for w in words if TERMINAL.search(w["word"]))
        print(f"transcript: {n_words} words, {n_term} with terminal punctuation ({n_term / max(n_words, 1):.1%}), "
              f"{len(segments)} segments, model {transcript.get('model')}, language {transcript.get('language')}")
    else:
        print("transcript.json not found: candidate recomputation and context words unavailable")

    methods = Counter()
    print()
    hdr = f"{'clip':<6}{'orig':>6}{'final':>7}  {'method':<10}{'cand':>5} {'term':>5} {'=orig':>6}  endpoint / reason"
    print(hdr)
    print("-" * len(hdr))
    for c in m.get("clips", []):
        if only and c["id"] != only:
            continue
        r = c.get("refinement") or {}
        methods[r.get("method", "none")] += 1
        orig_end = r.get("original_end", c["end"])
        final_end = r.get("final_end", c["end"])
        equals = abs(final_end - orig_end) < 0.05
        cands = candidate_endpoints(words, segments, c["start"]) if transcript else []
        n_term = sum(1 for e in cands if e["terminal"])
        line = (f"{c['id']:<6}{r.get('original_duration', c['end'] - c['start']):>6.1f}{r.get('final_duration', c['end'] - c['start']):>7.1f}  "
                f"{r.get('method', 'none'):<10}{len(cands):>5} {n_term:>5} {'YES' if equals else 'no':>6}  "
                f"{r.get('endpoint_id') or '-'} @ {final_end:.1f}: {(r.get('endpoint_reason') or '')[:90]}")
        print(line)
        if equals:
            why = []
            if r.get("method") == "unchanged":
                why.append("method=unchanged: refine_end found NO candidate endpoint in range at generation")
            if not cands:
                why.append("recomputed now: still no candidate (no sentence end / segment end / pause >= 0.8 s between "
                           f"+{CLIP_MIN_SECONDS:.0f}s and +{CLIP_HARD_MAX_SECONDS:.0f}s)")
            elif r.get("method") == "unchanged":
                why.append(f"recomputed now: {len(cands)} candidates exist -> the policy or transcript differed at generation")
            if r.get("method") == "semantic":
                why.append("model chose the endpoint that coincides with the original window end")
            if r.get("method") == "fallback":
                why.append("fallback landed on the original window end")
            print(f"{'':6}  why 40s: " + "; ".join(why))
        if only or equals or r.get("method") != "semantic" or r.get("llm_calls"):
            if transcript:
                nw, nt = punctuation_stats(words, c["start"], c["start"] + CLIP_HARD_MAX_SECONDS)
                print(f"{'':6}  window words {nw}, with terminal punctuation {nt}")
                for e in cands:
                    mark = "<== chosen" if abs(e["time"] - final_end) < 0.05 else ""
                    print(f"{'':6}    {e['id']:<4} +{e['rel']:5.1f}s  {'sentence-end' if e['terminal'] else 'boundary    '}  pause {e['pause']:4.1f}  ...{e['text'][-50:]} {mark}")
                for j in r.get("judgements") or []:
                    print(f"{'':6}    judged {j['id']:<4} {j['verdict']:<9} [{j['source']}] {str(j.get('reason', ''))[:70]}"
                          + (f"  raw={str(j.get('raw'))[:60]!r}" if j.get("raw") else ""))
                pre, post = words_around(words, final_end)
                print(f"{'':6}  ends with: ...{pre}")
                print(f"{'':6}  next words: {post}...")
        print()
    print("methods:", dict(methods))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
