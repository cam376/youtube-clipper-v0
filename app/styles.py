"""
Subtitle style presets rendered as ASS files.

Four presets share one template; CLEAN is byte-for-byte the v0.2.1 look.
KARAOKE highlights the spoken word with ASS \\k tags, driven by the word
timings stored in each cue (no re-transcription needed).

Colours are ASS &HAABBGGRR. Electric blue #2F6BFF -> &H00FF6B2F.
"""

from __future__ import annotations

from pathlib import Path

from fonts import resolve_font

OUT_W, OUT_H = 1080, 1920

WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
SHADOW_HALF = "&H80000000"
ELECTRIC_BLUE = "&H00FF6B2F"
NAVY = "&H00201208"

# fontsize, bold, outline, shadow, margin_v (bottom placement), primary, secondary, karaoke
PRESETS: dict[str, dict] = {
    "CLEAN": {
        "label": "Clean",
        "description": "Professional white captions with a black outline (the Kivro default).",
        "fontsize": 72, "bold": -1, "outline": 5, "shadow": 2, "margin_v": 420,
        "primary": WHITE, "secondary": WHITE, "outline_colour": BLACK, "back": SHADOW_HALF,
        "karaoke": False, "uppercase": False,
    },
    "BOLD": {
        "label": "Bold",
        "description": "Large creator-style captions, heavy weight, strong outline.",
        "fontsize": 92, "bold": -1, "outline": 8, "shadow": 0, "margin_v": 400,
        "primary": WHITE, "secondary": WHITE, "outline_colour": BLACK, "back": BLACK,
        "karaoke": False, "uppercase": True,
    },
    "KARAOKE": {
        "label": "Karaoke",
        "description": "The spoken word lights up in electric blue, TikTok style.",
        "fontsize": 80, "bold": -1, "outline": 6, "shadow": 0, "margin_v": 420,
        "primary": ELECTRIC_BLUE, "secondary": WHITE, "outline_colour": BLACK, "back": BLACK,
        "karaoke": True, "uppercase": False,
    },
    "MINIMAL": {
        "label": "Minimal",
        "description": "Smaller type, light outline, low visual footprint.",
        "fontsize": 54, "bold": 0, "outline": 2, "shadow": 0, "margin_v": 360,
        "primary": WHITE, "secondary": WHITE, "outline_colour": NAVY, "back": BLACK,
        "karaoke": False, "uppercase": False,
    },
}
DEFAULT_STYLE = "CLEAN"

ASS_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},{primary},{secondary},{outline_colour},{back},{bold},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def style_names() -> list[dict]:
    return [{"id": k, "label": v["label"], "description": v["description"]} for k, v in PRESETS.items()]


def normalize_style(name: str | None) -> str:
    name = (name or DEFAULT_STYLE).upper()
    return name if name in PRESETS else DEFAULT_STYLE


def ass_time(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def _karaoke_text(cue: dict, uppercase: bool) -> str:
    """\\k tags: each word is highlighted from its start until the next word starts."""
    words = cue.get("words") or []
    if not words:
        return escape_ass(cue["text"])
    parts = []
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else cue["end"]
        dur_cs = max(int(round((nxt - w["start"]) * 100)), 1)
        tok = escape_ass(w["word"])
        parts.append(f"{{\\k{dur_cs}}}{tok.upper() if uppercase else tok}")
    return " ".join(parts)


def render_ass(cues: list[dict], out_path: Path, style: str = DEFAULT_STYLE,
               font_choice: str | None = None, split_screen: bool = False) -> dict:
    """
    Write an ASS file for `cues` and return {"style", "font_family", "font_choice",
    "font_fallback": bool, "font_note"}.
    """
    style = normalize_style(style)
    preset = PRESETS[style]
    font = resolve_font(font_choice)
    header = ASS_TEMPLATE.format(
        w=OUT_W, h=OUT_H, font=font["family"],
        fontsize=preset["fontsize"], primary=preset["primary"], secondary=preset["secondary"],
        outline_colour=preset["outline_colour"], back=preset["back"], bold=preset["bold"],
        outline=preset["outline"], shadow=preset["shadow"],
        alignment=5 if split_screen else 2, margin_v=preset["margin_v"],
    )
    lines = []
    for cue in cues:
        if preset["karaoke"]:
            text = _karaoke_text(cue, preset["uppercase"])
        else:
            text = escape_ass(cue["text"])
            if preset["uppercase"]:
                text = text.upper()
        if not text.strip():
            continue
        lines.append(f"Dialogue: 0,{ass_time(cue['start'])},{ass_time(cue['end'])},Default,,0,0,0,,{text}")
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return {"style": style, "font_choice": font["choice"], "font_family": font["family"],
            "font_fallback": font["fallback"], "font_note": font["note"]}
