"""
Face-aware vertical reframing (V1, multi-person).

Flow for one clip:
    analyze_clip()   sample frames with ffmpeg, detect faces (OpenCV YuNet),
                     link detections into tracks across samples
    plan_layout()    classify SINGLE / SPLIT / CENTER and compute smoothed
                     crop keyframes per speaker
    build_filtergraph()  turn the plan into an ffmpeg filter_complex string
    save_debug_sheet()   optional contact sheet (DEBUG_FACES=true)

Only one layout is chosen per clip. Crops are smoothed and dead-zoned so
they stay still unless a person really moves.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:  # silence OpenCV's DNN backend chatter on newer builds
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

OUT_W, OUT_H = 1080, 1920
HALF_H = OUT_H // 2  # 960

SAMPLE_FPS = 2.0            # frames analysed per second of clip
DETECT_WIDTH = 640          # frames are downscaled to this width for detection
MIN_FACE_FRAC = 0.035       # ignore faces narrower than 3.5 % of frame width
SCORE_THRESHOLD = 0.6

MATCH_DIST_FACES = 1.0      # link detection to track if within 1 face-width
MAX_GAP_SECONDS = 8.0       # a track survives this long without detections

PERSISTENT_COVERAGE = 0.40  # fraction of samples a track must be present in
SINGLE_MIN_COVERAGE = 0.30
DOMINANCE_RATIO = 0.5       # 2nd face weaker than this * 1st -> single person
MIN_SEPARATION_FRAC = 0.15  # two faces must be this far apart horizontally
# A face that is detected less often (it turns away, profile view) still
# counts as persistent when its detections span most of the clip: the person
# is in their seat the whole time, the detector just misses them at times.
SPAN_COVERAGE = 0.25        # minimum coverage for the span rule
SPAN_FRAC = 0.70            # first-to-last detection must cover this much of the clip
MERGE_DIST_FACES = 1.0      # fragments of one person: same place within 1 face-width

# SINGLE_PERSON handoff: when the primary track is absent, follow another
# track that is visible instead of holding the primary's position, provided
# that track is temporally complementary (not a co-present second person)
# and has a plausible face size (not a logo or a tiny face in a graphic).
HANDOFF_MAX_COPRESENCE = 0.2   # fraction of the candidate's samples that overlap the primary
HANDOFF_SIZE_RANGE = (0.5, 2.0)  # candidate face height relative to the primary's
HANDOFF_MIN_SAMPLES = 4        # 2 s of detections before a track can be followed

SMOOTH_SECONDS = 1.5        # moving-average half-window
DEADZONE_FRAC = 0.06        # ignore moves smaller than 6 % of crop size
GLIDE_SECONDS = 1.0         # ramp length after a dead-zone step

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"

LAYOUT_SINGLE = "SINGLE_PERSON"
LAYOUT_SPLIT = "TWO_PERSON"
LAYOUT_CENTER = "CENTER_CROP"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    id: int
    boxes: dict = field(default_factory=dict)   # sample index -> (x, y, w, h) in source px
    last_idx: int = -1

    def coverage(self, n_samples: int) -> float:
        return len(self.boxes) / max(n_samples, 1)

    def median_box(self) -> tuple[float, float, float, float]:
        arr = np.array(list(self.boxes.values()), dtype=float)
        return tuple(np.median(arr, axis=0))

    def center_x(self) -> float:
        x, _, w, _ = self.median_box()
        return x + w / 2

    def span(self, n_samples: int) -> float:
        """Fraction of the clip between the first and last detection."""
        if not self.boxes or n_samples <= 1:
            return 0.0
        return (max(self.boxes) - min(self.boxes) + 1) / n_samples

    def longest_gap(self, sample_fps: float) -> float:
        """Longest run of samples without a detection, in seconds."""
        idxs = sorted(self.boxes)
        if len(idxs) < 2:
            return 0.0
        return max(b - a - 1 for a, b in zip(idxs, idxs[1:])) / sample_fps

    def metrics(self, n_samples: int, sample_fps: float) -> dict:
        return {
            "id": self.id,
            "coverage": round(self.coverage(n_samples), 3),
            "span": round(self.span(n_samples), 3),
            "longest_gap_s": round(self.longest_gap(sample_fps), 1),
            "fragments": getattr(self, "fragments", 1),
            "median_box": [round(float(v), 1) for v in self.median_box()],
        }


@dataclass
class FaceAnalysis:
    width: int
    height: int
    sample_fps: float
    n_samples: int
    tracks: list[Track]
    frames: list[np.ndarray]        # downscaled BGR frames (for debug only)
    scale: float                    # source px / detection px
    detector_name: str


@dataclass
class Region:
    """A crop window of fixed size whose position follows keyframes."""
    w: int
    h: int
    x_keys: list[tuple[float, float]]   # (t_seconds, x)
    y_keys: list[tuple[float, float]]
    track_id: int | None = None


@dataclass
class FramePlan:
    layout: str
    regions: list[Region]          # 1 for SINGLE, 2 for SPLIT (top, bottom), 0 for CENTER
    note: str = ""
    diagnostics: dict = field(default_factory=dict)   # SINGLE_PERSON: see _single_region


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
class _Detector:
    """YuNet (OpenCV DNN) when the model file is present, Haar cascade otherwise."""

    def __init__(self, det_w: int, det_h: int):
        self.name = "haar"
        self.yunet = None
        if MODEL_PATH.is_file() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    str(MODEL_PATH), "", (det_w, det_h), SCORE_THRESHOLD, 0.3, 50
                )
                self.name = "yunet"
            except cv2.error:
                self.yunet = None
        if self.yunet is None:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        self.min_w = max(int(det_w * MIN_FACE_FRAC), 12)

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Return [(x, y, w, h, score)] in detection-frame pixels, best first."""
        out = []
        if self.yunet is not None:
            _, faces = self.yunet.detect(frame)
            if faces is not None:
                for f in faces:
                    x, y, w, h, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
                    if w >= self.min_w:
                        out.append((x, y, w, h, score))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(self.min_w, self.min_w))
            for (x, y, w, h) in rects:
                out.append((float(x), float(y), float(w), float(h), 1.0))
        out.sort(key=lambda d: d[4], reverse=True)
        return out


def _probe_size(source: Path) -> tuple[int, int]:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(source)],
        check=True, capture_output=True, text=True,
    )
    st = json.loads(res.stdout)["streams"][0]
    return int(st["width"]), int(st["height"])


def _sample_frames(source: Path, start: float, end: float, det_w: int, det_h: int) -> list[np.ndarray]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(source),
        "-vf", f"fps={SAMPLE_FPS},scale={det_w}:{det_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    frame_bytes = det_w * det_h * 3
    n = len(raw) // frame_bytes
    buf = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(n, det_h, det_w, 3)
    return [buf[i].copy() for i in range(n)]


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
def _link_tracks(per_sample: list[list[tuple]], scale: float) -> list[Track]:
    """Greedy nearest-centre association. Boxes are stored in source pixels."""
    tracks: list[Track] = []
    max_gap = int(MAX_GAP_SECONDS * SAMPLE_FPS)
    for idx, dets in enumerate(per_sample):
        pairs = []
        for ti, tr in enumerate(tracks):
            if idx - tr.last_idx > max_gap:
                continue
            lx, ly, lw, lh = tr.boxes[tr.last_idx]
            lcx, lcy = lx + lw / 2, ly + lh / 2
            for di, (x, y, w, h, _s) in enumerate(dets):
                cx, cy = (x + w / 2) * scale, (y + h / 2) * scale
                dist = np.hypot(cx - lcx, cy - lcy) / max(lw, w * scale)
                if dist <= MATCH_DIST_FACES:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            x, y, w, h, _s = dets[di]
            tracks[ti].boxes[idx] = (x * scale, y * scale, w * scale, h * scale)
            tracks[ti].last_idx = idx
            used_t.add(ti)
            used_d.add(di)
        for di, (x, y, w, h, _s) in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(id=len(tracks))
            tr.boxes[idx] = (x * scale, y * scale, w * scale, h * scale)
            tr.last_idx = idx
            tracks.append(tr)
    return tracks


def _merge_fragments(tracks: list[Track]) -> list[Track]:
    """
    A person whose track died (no detection for > MAX_GAP_SECONDS) comes back
    as a new track id. Merge tracks that never overlap in time and sit at the
    same place, so coverage is computed per person, not per fragment.
    """
    merged: list[Track] = []
    for tr in sorted(tracks, key=lambda t: min(t.boxes)):
        x, y, w, h = tr.median_box()
        cx, cy = x + w / 2, y + h / 2
        target = None
        for m in merged:
            if set(m.boxes) & set(tr.boxes):
                continue
            mx, my, mw, mh = m.median_box()
            dist = np.hypot(cx - (mx + mw / 2), cy - (my + mh / 2)) / max(w, mw)
            if dist <= MERGE_DIST_FACES:
                target = m
                break
        if target is None:
            tr.fragments = 1
            merged.append(tr)
        else:
            target.boxes.update(tr.boxes)
            target.last_idx = max(target.last_idx, tr.last_idx)
            target.fragments = getattr(target, "fragments", 1) + 1
    return merged


def analyze_clip(source: Path, start: float, end: float) -> FaceAnalysis:
    width, height = _probe_size(source)
    det_w = min(DETECT_WIDTH, width)
    det_w -= det_w % 2
    det_h = int(round(height * det_w / width))
    det_h -= det_h % 2
    scale = width / det_w

    frames = _sample_frames(source, start, end, det_w, det_h)
    detector = _Detector(det_w, det_h)
    per_sample = [detector.detect(f) for f in frames]
    tracks = _merge_fragments(_link_tracks(per_sample, scale))
    return FaceAnalysis(width, height, SAMPLE_FPS, len(frames), tracks, frames, scale, detector.name)


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def _fill_series(track: Track, n: int, axis: str) -> np.ndarray:
    """Per-sample face centre (x or y); gaps are linearly interpolated."""
    idxs = np.array(sorted(track.boxes))
    vals = []
    for i in idxs:
        x, y, w, h = track.boxes[i]
        vals.append(x + w / 2 if axis == "x" else y + h / 2)
    vals = np.array(vals, dtype=float)
    return np.interp(np.arange(n), idxs, vals)


def _moving_average(v: np.ndarray, half_window: int) -> np.ndarray:
    if half_window <= 0 or len(v) < 2:
        return v.copy()
    k = 2 * half_window + 1
    padded = np.pad(v, half_window, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def _deadzone(v: np.ndarray, dead: float) -> np.ndarray:
    out = np.empty_like(v)
    anchor = v[0]
    for i, val in enumerate(v):
        if abs(val - anchor) > dead:
            anchor = val
        out[i] = anchor
    return out


def _smooth_position(centers: np.ndarray, size: float, limit: float) -> np.ndarray:
    """
    centers: per-sample face centre along one axis.
    size:    crop size along that axis; limit: frame size along that axis.
    Returns the per-sample crop origin (top-left) along that axis.
    """
    half = int(round(SMOOTH_SECONDS * SAMPLE_FPS))
    glide = int(round(GLIDE_SECONDS * SAMPLE_FPS))
    sm = _moving_average(centers, half)
    sm = _deadzone(sm, DEADZONE_FRAC * size)
    sm = _moving_average(sm, glide)
    origin = sm - size / 2
    return np.clip(origin, 0, max(limit - size, 0))


def _keyframes(values: np.ndarray) -> list[tuple[float, float]]:
    """Drop samples where the value does not change (keeps ffmpeg expressions short)."""
    keys = []
    n = len(values)
    for i in range(n):
        t = i / SAMPLE_FPS
        v = float(round(values[i], 1))
        if i == 0 or i == n - 1:
            keys.append((t, v))
            continue
        prev, nxt = values[i - 1], values[i + 1]
        if abs(v - prev) < 0.5 and abs(v - nxt) < 0.5:
            continue
        keys.append((t, v))
    return keys


# --------------------------------------------------------------------------- #
# Layout planning
# --------------------------------------------------------------------------- #
def _even(v: float) -> int:
    return int(round(v / 2)) * 2


def _handoff_candidates(a: FaceAnalysis, primary: Track, others: list[Track]) -> tuple[list[Track], list[dict], list[dict]]:
    """
    Other tracks that may stand in for the primary while it is absent.
    Returns (accepted tracks, accepted infos, rejected infos with reasons).
    """
    p_idx = set(primary.boxes)
    p_h = primary.median_box()[3]
    accepted, acc_info, rej_info = [], [], []
    for t in others:
        idx = set(t.boxes)
        h = t.median_box()[3]
        info = {"track": t.id, "samples": len(idx), "median_x": round(float(t.center_x()), 1),
                "face_h": round(float(h), 1), "size_ratio": round(float(h / p_h), 2) if p_h else None,
                "copresence": round(len(idx & p_idx) / len(idx), 3) if idx else 1.0}
        if len(idx) < HANDOFF_MIN_SAMPLES:
            info["reason"] = f"fewer than {HANDOFF_MIN_SAMPLES} samples"
        elif info["copresence"] > HANDOFF_MAX_COPRESENCE:
            info["reason"] = "co-present with the primary (another person)"
        elif not (HANDOFF_SIZE_RANGE[0] <= info["size_ratio"] <= HANDOFF_SIZE_RANGE[1]):
            info["reason"] = f"face size {info['size_ratio']}x the primary (outside {HANDOFF_SIZE_RANGE})"
        else:
            accepted.append(t)
            acc_info.append(info)
            continue
        rej_info.append(info)
    order = sorted(range(len(accepted)), key=lambda k: -accepted[k].median_box()[3])
    return [accepted[k] for k in order], [acc_info[k] for k in order], rej_info


def _centre_series_with_handoff(primary: Track, handoffs: list[Track], n: int, axis: str):
    """
    Per-sample face centre: the primary where it is detected, else the
    largest accepted handoff track detected at that sample, else linear
    interpolation. Also returns the origin of every sample.
    """
    known: dict[int, tuple[float, str]] = {}
    for i, (x, y, w, h) in primary.boxes.items():
        known[i] = (x + w / 2 if axis == "x" else y + h / 2, "primary")
    for t in handoffs:
        for i, (x, y, w, h) in t.boxes.items():
            if i not in known:
                known[i] = (x + w / 2 if axis == "x" else y + h / 2, f"handoff:{t.id}")
    idxs = np.array(sorted(known))
    vals = np.array([known[i][0] for i in idxs], dtype=float)
    series = np.interp(np.arange(n), idxs, vals)
    source = ["interpolated"] * n
    for i in known:
        source[i] = known[i][1]
    return series, source


def _origin_boundaries(source: list[str]) -> list[int]:
    """Sample indexes where the followed track changes (primary <-> handoff, handoff <-> handoff)."""
    boundaries = []
    last = None
    for i, s in enumerate(source):
        if s == "interpolated":
            continue
        if last is not None and s != last:
            boundaries.append(i)
        last = s
    return boundaries


def _smooth_segmented(centres: np.ndarray, size: float, limit: float, boundaries: list[int]) -> np.ndarray:
    """
    The usual smoothing, applied independently on each side of a handoff so
    the crop steps at a source cut instead of panning across it. With no
    boundaries this is exactly _smooth_position.
    """
    if not boundaries:
        return _smooth_position(centres, size, limit)
    out = np.empty_like(centres)
    edges = [0] + sorted(boundaries) + [len(centres)]
    for s, e in zip(edges[:-1], edges[1:]):
        if e > s:
            out[s:e] = _smooth_position(centres[s:e], size, limit)
    return out


def _keyframes_with_steps(values: np.ndarray, boundaries: list[int]) -> list[tuple[float, float]]:
    """_keyframes plus an instantaneous step at each handoff boundary."""
    keys = _keyframes(values)
    for b in boundaries:
        if 0 < b < len(values):
            t = b / SAMPLE_FPS
            keys.append((round(t - 0.02, 3), float(round(values[b - 1], 1))))
            keys.append((t, float(round(values[b], 1))))
    keys = sorted(set(keys))
    return keys


def _single_region(a: FaceAnalysis, tr: Track, others: list[Track] | None = None) -> tuple[Region, dict]:
    """
    9:16 crop that follows the primary face horizontally, handing off to a
    credible other track while the primary is absent (see _handoff_candidates).
    Returns the region and a diagnostics dict that makes the decision
    auditable in job.json: the raw per-sample target (face centre - crop_w/2,
    clamped), the final smoothed crop x per sample, where each sample's
    position came from, and the geometric-centre x it did NOT use.
    """
    W, H = a.width, a.height
    n = a.n_samples
    crop_h = H
    crop_w = _even(H * 9 / 16)
    if crop_w > W:
        crop_w = _even(W)
        crop_h = _even(W * 16 / 9)
    handoffs, handoff_info, rejected_info = _handoff_candidates(a, tr, others or [])
    centres, source = _centre_series_with_handoff(tr, handoffs, n, "x")
    boundaries = _origin_boundaries(source)
    target = np.clip(centres - crop_w / 2, 0, max(W - crop_w, 0))
    xs = _smooth_segmented(centres, crop_w, W, boundaries)
    # other faces visible while the primary is absent but NOT followed (rejected candidates)
    rejected_ids = {r["track"] for r in rejected_info}
    other_visible: dict[str, str] = {}
    for t in others or []:
        if t.id not in rejected_ids:
            continue
        reason = next(r["reason"] for r in rejected_info if r["track"] == t.id)
        for i, (x, y, w, h) in t.boxes.items():
            if i not in tr.boxes:
                other_visible[str(i)] = f"#{t.id} x={x + w / 2:.0f} h={h:.0f} not followed: {reason}"
    cy = float(np.median(_fill_series(tr, n, "y")))
    y = float(np.clip(cy - crop_h * 0.4, 0, max(H - crop_h, 0)))
    mx, _, mw, _ = tr.median_box()
    geometric_x = round((W - crop_w) / 2, 1)
    right_limit = max(W - crop_w, 0)
    handoff_samples = sum(1 for s in source if s.startswith("handoff:"))
    interpolated_samples = sum(1 for s in source if s == "interpolated")
    # Samples where the face is so close to a source edge that the crop
    # cannot centre it (nothing to the left/right of the frame to show).
    clamped_left = int(np.sum(centres - crop_w / 2 < 0))
    clamped_right = int(np.sum(centres - crop_w / 2 > right_limit))
    # Face position inside the rendered crop, as a fraction of crop width
    # (0.5 = centred). Uses the smoothed crop actually rendered.
    face_pos = (centres - xs) / crop_w
    diag = {
        "primary_track": tr.id,
        "source_width": W,
        "source_height": H,
        "coverage": round(tr.coverage(n), 3),
        "span": round(tr.span(n), 3),
        "median_face_x": round(float(mx + mw / 2), 1),
        "crop_w": crop_w,
        "geometric_centre_x": geometric_x,
        "geometric_centre_used": False,
        "detected_samples": len(tr.boxes),
        "fallback_samples": n - len(tr.boxes),          # samples without a primary detection
        "handoff_samples": handoff_samples,              # ... of which followed another credible track
        "interpolated_samples": interpolated_samples,    # ... of which held / interpolated
        "handoff_tracks": handoff_info,
        "handoff_rejected": rejected_info,
        "handoff_boundaries": boundaries,
        "position_source": source,
        "other_visible": other_visible,
        "clamped_left_samples": clamped_left,
        "clamped_right_samples": clamped_right,
        "face_in_crop_min": round(float(face_pos.min()), 3),
        "face_in_crop_median": round(float(np.median(face_pos)), 3),
        "face_in_crop_max": round(float(face_pos.max()), 3),
        "target_crop_x": [round(float(v), 1) for v in target],
        "smoothed_crop_x": [round(float(v), 1) for v in xs],
        "crop_x_min": round(float(xs.min()), 1),
        "crop_x_max": round(float(xs.max()), 1),
    }
    return Region(crop_w, crop_h, _keyframes_with_steps(xs, boundaries), [(0.0, round(y, 1))], tr.id), diag


def _primary_track(a: FaceAnalysis, candidates: list[Track]) -> Track:
    """
    The person to centre on. Among tracks that are present often enough,
    prefer the one that is both present and large: coverage x median face
    height. A small face that is always in frame (poster, picture-in-picture,
    logo) must not win over the actual speaker. With one track this is a no-op.
    """
    n = a.n_samples
    eligible = [t for t in candidates if t.coverage(n) >= SINGLE_MIN_COVERAGE] or candidates
    return max(eligible, key=lambda t: t.coverage(n) * t.median_box()[3])


def _half_region(a: FaceAnalysis, tr: Track) -> Region:
    W, H = a.width, a.height
    _, _, _, face_h = tr.median_box()
    crop_h = float(np.clip(face_h * 3.2, H * 0.45, H))
    crop_w = crop_h * 9 / 8
    if crop_w > W:
        crop_w = W
        crop_h = W * 8 / 9
    crop_w, crop_h = _even(crop_w), _even(crop_h)
    xs = _smooth_position(_fill_series(tr, a.n_samples, "x"), crop_w, W)
    # keep the face around 42 % from the top of its half
    ys = _smooth_position(_fill_series(tr, a.n_samples, "y") + crop_h * (0.5 - 0.42), crop_h, H)
    return Region(crop_w, crop_h, _keyframes(xs), _keyframes(ys), tr.id)


def _is_persistent(t: Track, n: int) -> bool:
    cov, span = t.coverage(n), t.span(n)
    return cov >= PERSISTENT_COVERAGE or (cov >= SPAN_COVERAGE and span >= SPAN_FRAC)


def classification_metrics(a: FaceAnalysis) -> dict:
    """Everything the layout decision looks at, for the debug output."""
    n = a.n_samples
    ranked = sorted(a.tracks, key=lambda t: t.coverage(n), reverse=True)
    m = {
        "samples": n,
        "sample_fps": a.sample_fps,
        "clip_seconds": round(n / a.sample_fps, 1) if a.sample_fps else 0,
        "tracks": [t.metrics(n, a.sample_fps) for t in ranked],
        "thresholds": {
            "persistent_coverage": PERSISTENT_COVERAGE,
            "span_coverage": SPAN_COVERAGE,
            "span_frac": SPAN_FRAC,
            "single_min_coverage": SINGLE_MIN_COVERAGE,
            "dominance_ratio": DOMINANCE_RATIO,
            "min_separation_frac": MIN_SEPARATION_FRAC,
        },
    }
    if len(ranked) >= 2:
        first, second = ranked[0], ranked[1]
        c1, c2 = first.coverage(n), second.coverage(n)
        m["separation"] = round(abs(first.center_x() - second.center_x()) / a.width, 3)
        m["strength_ratio"] = round(c2 / c1, 3) if c1 else 0.0
        m["second_span"] = round(second.span(n), 3)
        m["second_longest_gap_s"] = round(second.longest_gap(a.sample_fps), 1)
    return m


def plan_layout(a: FaceAnalysis) -> FramePlan:
    if a.n_samples == 0 or not a.tracks:
        return FramePlan(LAYOUT_CENTER, [], "no frames or no faces detected",
                         {"geometric_centre_used": True, "reason": "no faces detected"})

    n = a.n_samples
    ranked = sorted(a.tracks, key=lambda t: t.coverage(n), reverse=True)
    cov = [t.coverage(n) for t in ranked]
    persistent = [t for t in ranked if _is_persistent(t, n)]

    if len(persistent) >= 2:
        first, second = persistent[0], persistent[1]
        c1, c2 = first.coverage(n), second.coverage(n)
        sep = abs(first.center_x() - second.center_x()) / a.width
        span2 = second.span(n)
        # A second face that is in frame for most of the clip is a second
        # person even if the detector sees it less often than the first.
        dominated = c2 < DOMINANCE_RATIO * c1 and span2 < SPAN_FRAC
        stats = (f"coverage {c1:.0%} / {c2:.0%}, ratio {c2 / c1:.2f}, span {span2:.0%}, "
                 f"gap {second.longest_gap(a.sample_fps):.0f}s, separation {sep:.0%}, samples {n}")
        if sep >= MIN_SEPARATION_FRAC and not dominated:
            left, right = sorted((first, second), key=lambda t: t.center_x())
            return FramePlan(LAYOUT_SPLIT, [_half_region(a, left), _half_region(a, right)],
                             f"two persistent faces ({stats})")
        reason = "too close" if sep < MIN_SEPARATION_FRAC else "second face dominated"
        primary = _primary_track(a, persistent)
        region, diag = _single_region(a, primary, [t for t in a.tracks if t is not primary])
        return FramePlan(LAYOUT_SINGLE, [region], f"{reason} ({stats}); {_single_note(diag)}", diag)

    if cov[0] >= SINGLE_MIN_COVERAGE:
        extra = ""
        if len(ranked) >= 2:
            extra = (f", second face coverage {cov[1]:.0%} span {ranked[1].span(n):.0%} "
                     f"separation {abs(ranked[0].center_x() - ranked[1].center_x()) / a.width:.0%}")
        primary = _primary_track(a, ranked)
        region, diag = _single_region(a, primary, [t for t in a.tracks if t is not primary])
        return FramePlan(LAYOUT_SINGLE, [region],
                         f"one persistent face (coverage {cov[0]:.0%}, samples {n}{extra}); {_single_note(diag)}", diag)

    return FramePlan(LAYOUT_CENTER, [], f"faces too sporadic (best coverage {cov[0]:.0%}, samples {n})",
                     {"geometric_centre_used": True, "reason": "no reliable face track"})


def _single_note(d: dict) -> str:
    clamp = ""
    if d.get("clamped_left_samples") or d.get("clamped_right_samples"):
        clamp = (f", crop clamped at source edge for {d['clamped_left_samples']} left / "
                 f"{d['clamped_right_samples']} right samples")
    handoff = ""
    if d.get("handoff_tracks"):
        handoff = ", handoff to " + " / ".join(f"#{h['track']} ({h['samples']} samples, x {h['median_x']:.0f})"
                                               for h in d["handoff_tracks"])
    return (f"primary track #{d['primary_track']} coverage {d['coverage']:.0%} span {d['span']:.0%}, "
            f"median face x {d['median_face_x']:.0f}, crop x {d['crop_x_min']:.0f}-{d['crop_x_max']:.0f} "
            f"(geometric centre would be {d['geometric_centre_x']:.0f}), face at "
            f"{d['face_in_crop_min']:.0%}-{d['face_in_crop_max']:.0%} of crop width (median {d['face_in_crop_median']:.0%}), "
            f"{d['fallback_samples']} of {d['detected_samples'] + d['fallback_samples']} samples without primary "
            f"({d.get('handoff_samples', 0)} followed another track, {d.get('interpolated_samples', d['fallback_samples'])} "
            f"held/interpolated){handoff}{clamp}")


# --------------------------------------------------------------------------- #
# ffmpeg filtergraph
# --------------------------------------------------------------------------- #
def _expr(keys: list[tuple[float, float]]) -> str:
    """Piecewise-linear ffmpeg expression in t for a keyframe list."""
    if len(keys) == 1 or all(abs(v - keys[0][1]) < 0.5 for _, v in keys):
        return f"{keys[0][1]:.1f}"
    # Build from the end so the nesting is if(lt(t,t1), seg0, if(lt(t,t2), seg1, ... last))
    expr = f"{keys[-1][1]:.1f}"
    for (t0, v0), (t1, v1) in reversed(list(zip(keys[:-1], keys[1:]))):
        dt = max(t1 - t0, 1e-3)
        seg = f"({v0:.1f}+({v1 - v0:.1f})*(t-{t0:.3f})/{dt:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return expr


def _crop(r: Region) -> str:
    return f"crop={r.w}:{r.h}:x='{_expr(r.x_keys)}':y='{_expr(r.y_keys)}'"


def build_filtergraph(plan: FramePlan, ass_filter: str | None) -> str:
    """
    Return a filter_complex string producing [v]. `ass_filter` is the
    already-escaped "ass='...'" fragment, or None for no subtitles.
    """
    sub = f",{ass_filter}" if ass_filter else ""
    if plan.layout == LAYOUT_SINGLE:
        r = plan.regions[0]
        return f"[0:v]{_crop(r)},scale={OUT_W}:{OUT_H},setsar=1{sub}[v]"
    if plan.layout == LAYOUT_SPLIT:
        top, bot = plan.regions
        return (
            f"[0:v]split=2[a][b];"
            f"[a]{_crop(top)},scale={OUT_W}:{HALF_H},setsar=1[top];"
            f"[b]{_crop(bot)},scale={OUT_W}:{HALF_H},setsar=1[bot];"
            f"[top][bot]vstack=inputs=2{sub}[v]"
        )
    raise ValueError(f"no filtergraph for layout {plan.layout}")


# --------------------------------------------------------------------------- #
# Debug output
# --------------------------------------------------------------------------- #
def debug_enabled() -> bool:
    return os.environ.get("DEBUG_FACES", "").strip().lower() in ("1", "true", "yes", "on")


def _region_at(r: Region, t: float) -> tuple[int, int, int, int]:
    def interp(keys):
        ts = [k[0] for k in keys]
        vs = [k[1] for k in keys]
        return float(np.interp(t, ts, vs))
    return int(interp(r.x_keys)), int(interp(r.y_keys)), r.w, r.h


def save_debug_sheet(a: FaceAnalysis, plan: FramePlan, out_path: Path, max_tiles: int = 12) -> None:
    """Contact sheet: face boxes, track ids, crop regions, layout label."""
    if a.n_samples == 0:
        return
    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255), (255, 128, 128)]
    region_colors = [(0, 255, 255), (255, 200, 0)]
    pick = np.linspace(0, a.n_samples - 1, min(max_tiles, a.n_samples)).astype(int)
    inv = 1.0 / a.scale
    tiles = []
    for i in pick:
        img = a.frames[i].copy()
        t = i / a.sample_fps
        for ri, r in enumerate(plan.regions):
            x, y, w, h = _region_at(r, t)
            p1 = (int(x * inv), int(y * inv))
            p2 = (int((x + w) * inv), int((y + h) * inv))
            cv2.rectangle(img, p1, p2, region_colors[ri % 2], 2)
            cxp = int((x + w / 2) * inv)
            cv2.line(img, (cxp, p1[1]), (cxp, p2[1]), region_colors[ri % 2], 1)
        if plan.layout == LAYOUT_CENTER:
            gx = (a.width - _even(a.height * 9 / 16)) / 2
            cv2.rectangle(img, (int(gx * inv), 0), (int((gx + _even(a.height * 9 / 16)) * inv), img.shape[0] - 1), (128, 128, 128), 1)
            cv2.putText(img, "GEOMETRIC CENTRE", (int(gx * inv) + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        primary_ids = {r.track_id for r in plan.regions}
        handoff_ids = {h["track"] for h in plan.diagnostics.get("handoff_tracks", [])}
        for tr in a.tracks:
            if i not in tr.boxes:
                continue
            x, y, w, h = tr.boxes[i]
            col = colors[tr.id % len(colors)]
            p1 = (int(x * inv), int(y * inv))
            p2 = (int((x + w) * inv), int((y + h) * inv))
            cv2.rectangle(img, p1, p2, col, 3 if tr.id in primary_ids or tr.id in handoff_ids else 1)
            tag = f"#{tr.id} {tr.coverage(a.n_samples):.0%}" + (
                " PRIMARY" if tr.id in primary_ids else " HANDOFF" if tr.id in handoff_ids else "")
            cv2.putText(img, tag, (p1[0], max(p1[1] - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"t={t:.1f}s", (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    sheet = np.zeros((rows * th + 40, cols * tw, 3), dtype=np.uint8)
    title = f"{plan.layout} | {plan.note} | detector={a.detector_name} | samples={a.n_samples}"
    cv2.putText(sheet, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[40 + r * th: 40 + (r + 1) * th, c * tw: (c + 1) * tw] = tile
    cv2.imwrite(str(out_path), sheet)

    summary = {
        "layout": plan.layout,
        "note": plan.note,
        "detector": a.detector_name,
        "samples": a.n_samples,
        "classification": classification_metrics(a),
        "plan_diagnostics": plan.diagnostics,
        "regions": [
            {"track": r.track_id, "w": r.w, "h": r.h, "x_keys": r.x_keys, "y_keys": r.y_keys}
            for r in plan.regions
        ],
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
