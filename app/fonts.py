"""
Font choices for burned subtitles: four named options, each a chain of
real font families. The first family installed on this machine is used;
if none is, the last one is written anyway (libass then picks its own
substitute) and the result is flagged as a fallback so the UI can say so.

Detection: Windows registry (winreg) on Windows, `fc-list` elsewhere.
No font files are shipped with the repo.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys

FONT_CHOICES: dict[str, dict] = {
    "clean_sans": {"label": "Clean Sans", "families": ["Arial", "Liberation Sans", "DejaVu Sans", "Helvetica"]},
    "heavy_sans": {"label": "Heavy Sans", "families": ["Arial Black", "Impact", "Liberation Sans", "DejaVu Sans"]},
    "condensed": {"label": "Condensed", "families": ["Franklin Gothic Medium", "Arial Narrow", "Liberation Sans Narrow", "DejaVu Sans Condensed", "Arial"]},
    "classic": {"label": "Classic", "families": ["Georgia", "Times New Roman", "Liberation Serif", "DejaVu Serif"]},
}
DEFAULT_FONT = "clean_sans"


@functools.lru_cache(maxsize=1)
def installed_families() -> frozenset[str]:
    """Lower-cased family names available to libass on this machine."""
    names: set[str] = set()
    try:
        if sys.platform.startswith("win"):
            import winreg  # type: ignore[import-not-found]
            for hive, key in (
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"),
            ):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        i = 0
                        while True:
                            try:
                                name, _value, _type = winreg.EnumValue(k, i)
                            except OSError:
                                break
                            i += 1
                            # "Arial Black (TrueType)" -> "arial black"; "Arial Bold" stays
                            base = name.split(" (")[0]
                            names.add(base.lower())
                            for suffix in (" bold", " italic", " bold italic", " regular"):
                                if base.lower().endswith(suffix):
                                    names.add(base.lower()[: -len(suffix)])
                except OSError:
                    continue
        else:
            out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                for fam in line.split(","):
                    if fam.strip():
                        names.add(fam.strip().lower())
    except Exception:  # noqa: BLE001 - detection is best effort
        pass
    return frozenset(names)


def font_choices() -> list[dict]:
    """Choices with their resolved family, for the UI."""
    out = []
    for key, spec in FONT_CHOICES.items():
        r = resolve_font(key)
        out.append({"id": key, "label": spec["label"], "family": r["family"], "fallback": r["fallback"],
                    "note": r["note"]})
    return out


def resolve_font(choice: str | None) -> dict:
    """{"choice", "family", "fallback": bool, "note": str}"""
    choice = choice if choice in FONT_CHOICES else DEFAULT_FONT
    families = FONT_CHOICES[choice]["families"]
    installed = installed_families()
    for fam in families:
        if fam.lower() in installed:
            note = "" if fam == families[0] else f"{families[0]} not installed, using {fam}"
            return {"choice": choice, "family": fam, "fallback": fam != families[0], "note": note}
    last = families[-1]
    if not installed:
        note = "font detection unavailable on this machine; requested family written as is"
        return {"choice": choice, "family": families[0], "fallback": False, "note": note}
    return {"choice": choice, "family": last,
            "fallback": True,
            "note": f"none of {', '.join(families)} installed; libass will substitute for {last}"}


def describe() -> str:
    return os.linesep.join(f"{c['label']}: {c['family']}" + (f" ({c['note']})" if c["note"] else "")
                           for c in font_choices())
