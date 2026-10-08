"""
Print the face-centred framing diagnostics of one clip in a job, second by
second, so a bad composition can be explained from numbers instead of a sheet.

    python app/framing_report.py output/<job_id>/job.json c001

Columns: t, target crop x (face centre - crop_w/2, clamped), smoothed crop x
(what was rendered), face position inside the crop as a percentage of the
crop width, and a flag when the crop is clamped at a source edge or when the
sample had no detection (held / interpolated).
"""

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    manifest = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    clip = next((c for c in manifest.get("clips", []) if c["id"] == argv[2]), None)
    if clip is None:
        print(f"no clip {argv[2]} in {argv[1]}")
        return 1
    plan = clip.get("plan") or {}
    d = plan.get("diagnostics") or {}
    print(f"{clip['label']}  layout={clip.get('layout')}  {clip.get('start')}s-{clip.get('end')}s")
    print(f"note: {plan.get('note', '')}")
    if plan.get("layout") != "SINGLE_PERSON" or "smoothed_crop_x" not in d:
        print("no single-person diagnostics on this clip (generated before v0.3 framing diagnostics, or not face-centred)")
        return 0
    crop_w = d["crop_w"]
    region = plan["regions"][0]
    print(f"primary track #{d['primary_track']}  coverage {d['coverage']:.0%}  span {d['span']:.0%}  "
          f"median face x {d['median_face_x']}  crop_w {crop_w}  geometric centre x {d['geometric_centre_x']}  "
          f"held/interpolated {d['fallback_samples']}/{d['fallback_samples'] + d['detected_samples']}")
    fps = 2.0
    target, smooth = d["target_crop_x"], d["smoothed_crop_x"]
    right_limit = d.get("source_width", 0) - crop_w if d.get("source_width") else None
    print(f"{'t':>6} {'target_x':>9} {'smooth_x':>9} {'face@crop':>9}  flags")
    for i, (tx, sx) in enumerate(zip(target, smooth)):
        t = i / fps
        face_cx = tx + crop_w / 2          # target is the clamped face centre minus half width
        pos = (face_cx - sx) / crop_w
        flags = []
        if sx <= 0.5:
            flags.append("CLAMPED-LEFT")
        if right_limit is not None and sx >= right_limit - 0.5:
            flags.append("CLAMPED-RIGHT")
        if abs(pos - 0.5) > 0.15:
            flags.append("OFF-CENTRE")
        if i % 2 == 0 or flags:
            print(f"{t:6.1f} {tx:9.1f} {sx:9.1f} {pos:8.0%}   {' '.join(flags)}")
    print(f"keyframes rendered: {len(region['x_keys'])}  x range {d['crop_x_min']}-{d['crop_x_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
