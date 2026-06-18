import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class SubtitleStyle:
    font_path: str = ""
    font_name: str = "Arial"
    font_size: int = 24
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    outline_width: int = 2
    shadow: int = 0
    back_color: str = "&H00000000"
    border_style: int = 1
    alignment: int = 2
    margin_v: int = 40
    margin_l: int = 20
    margin_r: int = 20
    spacing: int = 0
    alpha_level: float = 0.0
    rounded: bool = False
    background_bar: bool = False
    bg_alpha: float = 0.75
    second_font_size_scale: float = 0.8


def _find_font_files(search_dirs: list, font_names: list) -> Optional[str]:
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                lower = f.lower()
                for name in font_names:
                    if name.lower() in lower:
                        return os.path.join(root, f)
    return None


def resolve_font_path() -> str:
    if sys.platform.startswith("win"):
        search_dirs = [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            r"C:\Program Files\Adobe\Acrobat DC\Resource\Font",
            r"C:\Program Files (x86)\Common Files\Adobe\Fonts",
            os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts"),
        ]
    elif sys.platform == "darwin":
        search_dirs = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    else:
        search_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]

    found = _find_font_files(
        search_dirs,
        ["SourceHanSans", "Source Han Sans", "NotoSansCJK", "Noto Sans CJK", "思源黑体"],
    )
    if found:
        return found

    found = _find_font_files(
        search_dirs,
        ["NotoSans", "Noto Sans"],
    )
    if found:
        return found

    found = _find_font_files(
        search_dirs,
        ["msyh", "Microsoft YaHei", "微软雅黑"],
    )
    if found:
        return found

    found = _find_font_files(
        search_dirs,
        ["Arial", "DejaVuSans", "DejaVu Sans", "PingFang"],
    )
    if found:
        return found

    return ""


def _to_ass_color(hex_rgb: str, alpha: float = 0.0) -> str:
    return hex_to_ass_color(hex_rgb, alpha)


def hex_to_ass_color(hex_rgb: str, alpha: float = 0.0) -> str:
    hex_rgb = hex_rgb.lstrip("#")
    if len(hex_rgb) == 3:
        hex_rgb = "".join(c * 2 for c in hex_rgb)
    if len(hex_rgb) != 6:
        return "&H00FFFFFF"
    r, g, b = int(hex_rgb[0:2], 16), int(hex_rgb[2:4], 16), int(hex_rgb[4:6], 16)
    a = int(alpha * 255)
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


PRESETS: Dict[str, SubtitleStyle] = {
    "white": SubtitleStyle(
        font_name="Source Han Sans / Arial",
        font_size=28,
        primary_color=_to_ass_color("#FFFFFF"),
        outline_color=_to_ass_color("#000000"),
        outline_width=3,
        shadow=1,
        border_style=1,
        alignment=2,
        margin_v=50,
        background_bar=False,
    ),
    "dark": SubtitleStyle(
        font_name="Source Han Sans / Arial",
        font_size=28,
        primary_color=_to_ass_color("#FFD700"),
        outline_color=_to_ass_color("#000000"),
        outline_width=3,
        shadow=2,
        border_style=1,
        alignment=2,
        margin_v=50,
        background_bar=False,
    ),
    "minimal": SubtitleStyle(
        font_name="Source Han Sans / Arial",
        font_size=24,
        primary_color=_to_ass_color("#FFFFFF"),
        outline_color=_to_ass_color("#000000"),
        outline_width=1,
        shadow=0,
        border_style=1,
        alignment=2,
        margin_v=40,
        background_bar=False,
    ),
    "rounded": SubtitleStyle(
        font_name="Source Han Sans / Arial",
        font_size=26,
        primary_color=_to_ass_color("#FFFFFF"),
        outline_color=_to_ass_color("#000000"),
        outline_width=1,
        shadow=0,
        border_style=3,
        back_color=_to_ass_color("#000000", alpha=0.7),
        alignment=2,
        margin_v=45,
        rounded=True,
        background_bar=True,
        bg_alpha=0.75,
    ),
}


def get_preset(name: str) -> SubtitleStyle:
    key = name.lower()
    if key in PRESETS:
        style = PRESETS[key]
        resolved = resolve_font_path()
        if resolved:
            style.font_path = resolved
            style.font_name = os.path.splitext(os.path.basename(resolved))[0]
        return style
    raise KeyError(f"Unknown preset: {name}. Available: {', '.join(PRESETS.keys())}")


def load_style_from_json(json_path: str) -> SubtitleStyle:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    style = SubtitleStyle()
    for k, v in data.items():
        if hasattr(style, k):
            setattr(style, k, v)

    if not style.font_path:
        resolved = resolve_font_path()
        if resolved:
            style.font_path = resolved
            style.font_name = os.path.splitext(os.path.basename(resolved))[0]

    if "primary_color_hex" in data:
        style.primary_color = _to_ass_color(data["primary_color_hex"], data.get("primary_alpha", 0.0))
    if "outline_color_hex" in data:
        style.outline_color = _to_ass_color(data["outline_color_hex"], data.get("outline_alpha", 0.0))
    if "back_color_hex" in data:
        style.back_color = _to_ass_color(data["back_color_hex"], data.get("back_alpha", style.bg_alpha))

    return style


def style_to_dict(style: SubtitleStyle) -> Dict[str, Any]:
    return asdict(style)


def list_presets() -> Dict[str, Dict[str, Any]]:
    return {k: style_to_dict(v) for k, v in PRESETS.items()}
