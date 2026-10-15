"""
Token cards: nine frames to choose from, drawn over full-bleed or windowed art.

Every frame is a Card Conjurer image measured at its 1500x2100 template size (the seventh
edition frames are 744x1039 and are scaled up first) and drawn at the regular card size. The
art is cover-fitted to the frame's art box, the frame goes on top, then the centred name,
the type line ("Token Creature — Goblin"), any abilities, the italic flavor text and P/T.

Text colour follows the paint underneath: each name, type line and text box is measured
after the art and frame are composited, and gets black ink on light paint and white ink
(with a soft shadow) on dark or see-through paint. That keeps one layout for the light
white token frame, the dark promo bars and whatever art shows through Expedition's box.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

from card_renderer import card_renderer


def _log(message: str) -> None:
    """Best-effort console logging: an unprintable message (emoji on a cp1252 console) must
    never break a token."""
    try:
        print(message)
    except Exception:
        pass

TEMPLATE_W, TEMPLATE_H = 1500, 2100
WUBRG = "WUBRG"
FULL_ART = (60, 60, 1440, 1940)          # inside the black border: art for full-bleed frames
PT_BOX_TOPLEFT = (1136, 1858)            # the M15 P/T box, same spot as regular cards
PT_BOX_SIZE = (282, 154)
LOGO_SIZE = 76                            # set symbol at the right end of the type line
NAME_MAX_SIZE, NAME_MIN_SIZE = 24, 14
TYPE_MAX_SIZE, TYPE_MIN_SIZE = 19, 12
TEXT_SIZES = [21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
FLAVOR_SLANT = 0.2                        # italic shear for the flavor text (no italic font ships)
LIGHT_INK_BELOW = 150                     # mean luminance under the text; darker gets white ink


@dataclass(frozen=True)
class TokenFrame:
    """A frame's measurements on the 1500x2100 template."""
    id: str
    label: str
    blurb: str
    path: str                              # under assets/, {code} = W U B R G M A L
    codes: str                             # codes that have their own image ("" = one image)
    art_box: Tuple[int, int, int, int]
    name_y: int                            # name centre line
    name_span: Tuple[int, int]             # left/right limits for the centred name
    type_y: int
    type_span: Tuple[int, int]             # type line from left, logo at the right end
    text_box: Optional[Tuple[int, int, int, int]]   # abilities + flavor; None = textless
    pt: str = "box"                        # "box" (M15 P/T box) or "corner" (old frames: ink in the border)
    pt_corner: Tuple[int, int] = (1330, 1965)
    land_code: str = "L"                   # the code used for colourless lands

    def asset(self, code: str) -> str:
        if not self.codes:
            return self.path
        if code not in self.codes:
            code = "M" if "M" in self.codes else self.codes[0]
        if code == "L":
            code = self.land_code
        return self.path.format(code=code)

    @property
    def has_text(self) -> bool:
        return self.text_box is not None


FRAMES: List[TokenFrame] = [
    TokenFrame("classic", "Classic Token", "The real thing: full art under a see-through text box.",
               "token/tokenFrame{code}Regular.png", "WUBRGMAL", FULL_ART,
               name_y=161, name_span=(150, 1350), type_y=1416, type_span=(130, 1370),
               text_box=(135, 1500, 1365, 1925)),
    TokenFrame("tall", "Tall Text", "Room for a wordy token, with the art still full bleed.",
               "token/tokenFrame{code}Tall.png", "WUBRGMAL", FULL_ART,
               name_y=161, name_span=(150, 1350), type_y=1241, type_span=(130, 1370),
               text_box=(135, 1325, 1365, 1925)),
    TokenFrame("fullart", "Full Art", "All painting, no text box. Flavor shows below the card.",
               "token/tokenFrame{code}Textless.png", "WUBRGMAL", FULL_ART,
               name_y=161, name_span=(150, 1350), type_y=1772, type_span=(130, 1370),
               text_box=None),
    TokenFrame("promo", "Borderless Promo", "Edge-to-edge art with smoky glass bars.",
               "m15Promo/m15PromoFrame{code}.png", "WUBRGMAL", (0, 0, 1500, 2100),
               name_y=161, name_span=(150, 1350), type_y=1416, type_span=(130, 1370),
               text_box=(135, 1500, 1365, 1925)),
    TokenFrame("expedition", "Expedition", "Zendikar's carved stone around a glassy text box.",
               "expedition/expeditionFrame{code}.png", "WUBRGMAL", (110, 225, 1390, 1710),
               name_y=157, name_span=(170, 1330), type_y=1772, type_span=(150, 1350),
               text_box=(140, 1268, 1360, 1690)),
    TokenFrame("map", "Treasure Map", "Ixalan's explorer parchment, scrolls and all.",
               "ixalan/ixalanFrame{code}.png", "WUBRGML", (60, 240, 1440, 1200),
               name_y=150, name_span=(170, 1330), type_y=1238, type_span=(170, 1330),
               text_box=(150, 1330, 1350, 1860)),
    TokenFrame("invention", "Masterpiece", "Kaladesh Invention filigree in copper and white.",
               "invention/inventionFrame.png", "", (60, 60, 1440, 1320),
               name_y=161, name_span=(150, 1350), type_y=1241, type_span=(130, 1370),
               text_box=(135, 1328, 1365, 1910)),
    TokenFrame("oldschool", "Old School", "Seventh Edition borders, straight out of a 2001 binder.",
               "seventh/seventhFrame{code}.png", "WUBRGMAL", (170, 200, 1330, 1146),
               name_y=132, name_span=(190, 1310), type_y=1203, type_span=(180, 1320),
               text_box=(185, 1282, 1315, 1845), pt="corner", land_code="CL"),
    TokenFrame("legends", "Legends Gold", "The 1994 gold frame, whatever the token's colors.",
               "legends/legendsFrameM.png", "", (160, 194, 1340, 1146),
               name_y=128, name_span=(190, 1310), type_y=1203, type_span=(180, 1320),
               text_box=(185, 1278, 1315, 1866), pt="corner"),
]
FRAMES_BY_ID = {f.id: f for f in FRAMES}
DEFAULT_FRAME = "classic"

# Placeholder paint for previews and thumbnails, per frame code.
TINTS = {"W": (222, 210, 168), "U": (64, 112, 178), "B": (74, 62, 86), "R": (184, 78, 48),
         "G": (58, 124, 70), "M": (196, 164, 78), "A": (140, 142, 152), "L": (146, 116, 86)}


def frame_code(colors, card_type: str = "") -> str:
    """W/U/B/R/G for one colour, M for two or more, L for a colourless land, else A."""
    found = [c for c in WUBRG if c in set(colors or [])]
    if len(found) > 1:
        return "M"
    if found:
        return found[0]
    return "L" if "land" in (card_type or "").lower() else "A"


def art_aspect(frame_id: str) -> float:
    """Width / height of the frame's art box (so the art can be painted to fit)."""
    x0, y0, x1, y1 = FRAMES_BY_ID.get(frame_id, FRAMES_BY_ID[DEFAULT_FRAME]).art_box
    return (x1 - x0) / (y1 - y0)


def type_line(card: dict) -> str:
    """"Token Creature — Goblin": tokens print the word Token before their card types."""
    main = " ".join(p for p in ("Token", card.get("supertype") or "", card.get("type") or "") if p)
    subtype = (card.get("subtype") or "").strip()
    return f"{main} — {subtype}" if subtype else main


# ----- drawing helpers -----

def _scale(x: float, y: float) -> Tuple[int, int]:
    return (round(x * card_renderer.card_width / TEMPLATE_W),
            round(y * card_renderer.card_height / TEMPLATE_H))


def _box(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (*_scale(box[0], box[1]), *_scale(box[2], box[3]))


def _font(size: int) -> ImageFont.FreeTypeFont:
    return card_renderer._title_font_at(size)


def _ink(card: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[str, bool]:
    """(fill colour, needs a shadow) for text drawn on this area of the composited card."""
    region = card.crop(box).convert("L")
    light = ImageStat.Stat(region).mean[0] >= LIGHT_INK_BELOW
    return ("black", False) if light else ("white", True)


def _shadowed_text(card: Image.Image, xy, text: str, font, fill: str, shadow: bool, anchor: str):
    if shadow:
        layer = Image.new("RGBA", card.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((xy[0] + 1, xy[1] + 1), text, font=font,
                                   fill=(0, 0, 0, 200), anchor=anchor)
        card.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.2)))
    ImageDraw.Draw(card).text(xy, text, font=font, fill=fill, anchor=anchor)


def _fit_size(text: str, width: int, largest: int, smallest: int) -> int:
    size = largest
    while size > smallest and _font(size).getlength(text) > width:
        size -= 1
    return size


def placeholder_art(code: str, size: Tuple[int, int]) -> Image.Image:
    """A soft painted-looking gradient in the frame's colour, where the art will go."""
    w, h = size
    r, g, b = TINTS.get(code, TINTS["A"])
    top = tuple(min(255, int(c * 1.25 + 30)) for c in (r, g, b))
    bottom = tuple(int(c * 0.45) for c in (r, g, b))
    grad = Image.new("RGB", (1, 256))
    for i in range(256):
        t = i / 255
        grad.putpixel((0, i), tuple(int(top[k] * (1 - t) + bottom[k] * t) for k in range(3)))
    art = grad.resize((w, h))
    glow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glow).ellipse([w * 0.15, h * 0.18, w * 0.85, h * 0.62], fill=110)
    art.paste(Image.new("RGB", (w, h), (255, 248, 225)), (0, 0),
              glow.filter(ImageFilter.GaussianBlur(min(w, h) / 6)))
    return art


def _wrap(text: str, font, width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words, line = paragraph.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if line and font.getlength(trial) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def _flavor_height(text: str, size: int, width: int) -> int:
    font = card_renderer.get_text_font(size)
    return len(_wrap(text, font, width - int(size * FLAVOR_SLANT) - 2)) * int(size * 1.1)


def _draw_flavor(card: Image.Image, text: str, x: int, y: int, width: int, size: int, fill: str):
    """Flavor text in a sheared (italic) Garamond, one line at a time."""
    font = card_renderer.get_text_font(size)
    line_h = int(size * 1.1)
    slant_pad = int(size * 1.3 * FLAVOR_SLANT) + 2
    for i, line in enumerate(_wrap(text, font, width - slant_pad)):
        if not line:
            continue
        w = int(font.getlength(line)) + slant_pad + 4
        h = int(size * 1.35)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((2, 0), line, font=font, fill=fill)
        k = FLAVOR_SLANT
        layer = layer.transform((w, h), Image.Transform.AFFINE, (1, k, -k * h, 0, 1, 0),
                                resample=Image.Resampling.BICUBIC)
        card.alpha_composite(layer, (x, y + i * line_h))


def _layout_text(rules: str, flavor: str, width: int, height: int) -> Tuple[int, int, int]:
    """(font size, rules height, flavor height) for the largest size where both fit."""
    scratch = Image.new("RGBA", (1, 1))
    gap_for = lambda size: int(size * 0.8) if rules and flavor else 0
    for size in TEXT_SIZES:
        rh = card_renderer.draw_text_with_mana_symbols(
            scratch, rules, (0, 0), width, font_size=size, measure_only=True) if rules else 0
        fh = _flavor_height(flavor, size, width) if flavor else 0
        if rh + gap_for(size) + fh <= height:
            return size, rh, fh
    size = TEXT_SIZES[-1]
    rh = card_renderer.draw_text_with_mana_symbols(
        scratch, rules, (0, 0), width, font_size=size, measure_only=True) if rules else 0
    return size, rh, _flavor_height(flavor, size, width) if flavor else 0


# ----- the token -----

def _art_layer(frame: TokenFrame, code: str, artwork_base64: Optional[str]) -> Image.Image:
    card = Image.new("RGBA", (card_renderer.card_width, card_renderer.card_height), (12, 12, 14, 255))
    x0, y0, x1, y1 = _box(frame.art_box)
    art = None
    if artwork_base64:
        try:
            data = artwork_base64.split(",", 1)[1] if artwork_base64.startswith("data:") else artwork_base64
            art = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        except Exception as e:
            _log(f"⚠️ Token art unreadable, using a placeholder: {e}")
    if art is None:
        art = placeholder_art(code, (x1 - x0, y1 - y0))
    # Bias the crop upward a little: heads and faces sit in the top half of most paintings
    art = ImageOps.fit(art, (x1 - x0, y1 - y0), Image.Resampling.LANCZOS, centering=(0.5, 0.4))
    card.paste(art, (x0, y0))
    return card


def _frame_image(frame: TokenFrame, code: str) -> Optional[Image.Image]:
    path = os.path.join(card_renderer.assets_dir, frame.asset(code))
    if not os.path.exists(path):
        _log(f"⚠️ Token frame missing: {path}")
        return None
    image = card_renderer._load_image_cached(path).convert("RGBA")
    return image.resize((card_renderer.card_width, card_renderer.card_height), Image.Resampling.LANCZOS)


def _pt_box(code: str) -> Optional[Image.Image]:
    pt_code = {"L": "C"}.get(code, code)
    path = os.path.join(card_renderer.m15_dir, f"m15PT{pt_code}.png")
    if not os.path.exists(path):
        path = os.path.join(card_renderer.m15_dir, "m15PTM.png")
        if not os.path.exists(path):
            return None
    w, h = _scale(*PT_BOX_SIZE)
    return card_renderer._load_image_cached(path).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)


def render_token_image(card: dict, artwork_base64: Optional[str] = None) -> Image.Image:
    """The token as a PIL image at the regular card size (rounded corners)."""
    frame = FRAMES_BY_ID.get(card.get("frame") or "", FRAMES_BY_ID[DEFAULT_FRAME])
    code = frame_code(card.get("colors"), card.get("type") or "")
    image = _art_layer(frame, code, artwork_base64)
    frame_img = _frame_image(frame, code)
    if frame_img is not None:
        image.alpha_composite(frame_img)

    is_creature = "creature" in (card.get("type") or "").lower()
    power, toughness = card.get("power"), card.get("toughness")
    has_pt = is_creature and power not in (None, "") and toughness not in (None, "")
    pt_box = _pt_box(code) if has_pt and frame.pt == "box" else None
    pt_top = _scale(0, PT_BOX_TOPLEFT[1])[1]
    if pt_box is not None:
        image.alpha_composite(pt_box, _scale(*PT_BOX_TOPLEFT))

    # Name, centred in its bar
    name = (card.get("name") or "").strip() or "Token"
    nx0, _ = _scale(frame.name_span[0], 0)
    nx1, ny = _scale(frame.name_span[1], frame.name_y)
    size = _fit_size(name, nx1 - nx0, NAME_MAX_SIZE, NAME_MIN_SIZE)
    fill, shadow = _ink(image, (nx0, ny - 10, nx1, ny + 10))
    _shadowed_text(image, ((nx0 + nx1) // 2, ny), name, _font(size), fill, shadow, "mm")

    # Type line, with the site logo as the set symbol at its right end
    tx0, ty = _scale(frame.type_span[0], frame.type_y)
    tx1, _ = _scale(frame.type_span[1], 0)
    logo_w = _scale(LOGO_SIZE, 0)[0]
    logo = card_renderer.load_site_logo("common")
    if logo is not None:
        logo = logo.resize((logo_w, logo_w), Image.Resampling.LANCZOS)
        image.alpha_composite(logo, (tx1 - logo_w, ty - logo_w // 2))
    line = type_line(card)
    type_width = tx1 - tx0 - logo_w - 6
    size = _fit_size(line, type_width, TYPE_MAX_SIZE, TYPE_MIN_SIZE)
    fill, shadow = _ink(image, (tx0, ty - 9, tx0 + type_width, ty + 9))
    _shadowed_text(image, (tx0, ty), line, _font(size), fill, shadow, "lm")

    # Abilities and flavor, centred as a block in the text box
    rules = (card.get("description") or "").strip()
    flavor = (card.get("flavorText") or "").strip()
    if frame.text_box and (rules or flavor):
        bx0, by0, bx1, by1 = _box(frame.text_box)
        bottom = min(by1, pt_top + 6) if pt_box is not None else by1
        width, height = bx1 - bx0, bottom - by0
        size, rh, fh = _layout_text(rules, flavor, width, height)
        gap = int(size * 0.8) if rules and flavor else 0
        fill, shadow = _ink(image, (bx0, by0, bx1, bottom))
        top = by0 + max(0, (height - rh - gap - fh) // 2)
        if shadow:  # white ink on see-through paint: darken behind it so it stays legible
            veil = Image.new("RGBA", image.size, (0, 0, 0, 0))
            ImageDraw.Draw(veil).rectangle([bx0, by0, bx1, bottom], fill=(0, 0, 0, 70))
            image.alpha_composite(veil)
        if rules:
            card_renderer.draw_text_with_mana_symbols(image, rules, (bx0, top), width,
                                                     font_size=size, fill=fill)
        if flavor:
            if rules:  # the thin rule real cards print between rules and flavor text
                y = top + rh + gap // 2 - 2
                ImageDraw.Draw(image).line([(bx0 + width * 0.2, y), (bx1 - width * 0.2, y)],
                                           fill=fill, width=1)
            _draw_flavor(image, flavor, bx0, top + rh + gap, width, size, fill)

    # Power / toughness
    if has_pt:
        pt_text = f"{power}/{toughness}"
        pt_font = card_renderer.pt_font
        if pt_box is not None:
            px, py = _scale(*PT_BOX_TOPLEFT)
            ImageDraw.Draw(image).text((px + pt_box.width // 2 + 4, py + pt_box.height // 2),
                                       pt_text, fill="black", font=pt_font, anchor="mm")
        else:
            cx, cy = _scale(*frame.pt_corner)
            fill, shadow = _ink(image, (cx - 50, cy - 10, cx, cy + 10))
            _shadowed_text(image, (cx, cy), pt_text, pt_font, fill, shadow, "rm")

    return card_renderer.add_rounded_corners(image, radius=18)


def render_token(card: dict, artwork_base64: Optional[str] = None) -> str:
    """The token as a base64 PNG (no data: prefix)."""
    buffer = io.BytesIO()
    render_token_image(card, artwork_base64).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def sample_token(frame_id: str, code: str) -> dict:
    """A sample token in this frame and colour code, for the frame picker's thumbnails."""
    sample = {"frame": frame_id, "name": "Token Name", "type": "Creature", "subtype": "Spirit",
              "colors": [] if code in ("A", "L") else (["W", "U"] if code == "M" else [code]),
              "power": "1", "toughness": "1", "description": "Flying",
              "flavorText": "It knows what you did."}
    if code == "L":
        sample.update(type="Land", subtype="", power="", toughness="", description="")
    return sample


def thumbnail(png_base64: str, width: int = 248) -> bytes:
    """A rendered token (base64 PNG) shrunk to the frame picker's size, as PNG bytes."""
    image = Image.open(io.BytesIO(base64.b64decode(png_base64)))
    image = image.resize((width, round(width * image.height / image.width)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
