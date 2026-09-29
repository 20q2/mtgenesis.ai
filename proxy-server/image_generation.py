"""
Card art generation (SDXL Lightning fine-tune) and art prompt construction.

Spec: docs/superpowers/specs/2026-09-28-ai-night-design.md §6.
torch/diffusers are imported lazily so placeholder mode and tests never load them.
"""
from __future__ import annotations

import re

# ===== PROMPT CONSTRUCTION =====

MAX_PROMPT_TOKENS = 75       # CLIP's 77 minus the start/end tokens
SUBJECT_MIN_TOKENS = 30      # the subject always keeps at least this much room

STYLE_SUFFIX = "painterly Magic: The Gathering fantasy illustration, dramatic lighting, highly detailed"
NEGATIVE_PROMPT = ("text, letters, watermark, signature, border, frame, card, UI, "
                   "blurry, lowres, deformed, extra limbs")
GENERIC_CONTEXT = "magical fantasy scene"

WUBRG = "WUBRG"
_COLOR_NAMES = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G", "colorless": "C"}

# Mood hints per color (spec §6). Mono-colored cards get both words; multicolor cards
# get the first word of each color so the hint stays short.
COLOR_MOODS = {
    "W": "radiant, ordered",
    "U": "arcane, ocean",
    "B": "shadow, decay",
    "R": "fiery, embers",
    "G": "verdant, primal",
}

# Color palettes keyed by the set of WUBRG colors (moved from app.createCardImage).
COLOR_PALETTES = {
    frozenset(): "metallic silver, steel gray",
    # mono
    frozenset("W"): "pure white, warm gold",
    frozenset("U"): "sapphire blue, silver",
    frozenset("B"): "void black, dark purple",
    frozenset("R"): "burning red, molten orange",
    frozenset("G"): "forest green, earth brown",
    # guilds
    frozenset("WU"): "pristine white, sapphire blue",
    frozenset("WB"): "pure white, deep black",
    frozenset("WR"): "ivory white, burning red",
    frozenset("WG"): "marble white, forest green",
    frozenset("UB"): "midnight blue, void black",
    frozenset("UR"): "electric blue, molten red",
    frozenset("UG"): "ocean blue, living green",
    frozenset("BR"): "shadow black, blood red",
    frozenset("BG"): "decay black, wild green",
    frozenset("RG"): "flame red, primal green",
    # shards and wedges
    frozenset("WUG"): "white marble, blue sapphire, green emerald",   # Bant
    frozenset("UBR"): "dark blues, void black, burning red",          # Grixis
    frozenset("BRG"): "shadow black, flame red, wild green",          # Jund
    frozenset("RGW"): "burning red, emerald green, pure white",       # Naya
    frozenset("WBG"): "ivory white, deep black, forest green",        # Abzan
    frozenset("URW"): "sapphire blue, flame red, pure white",         # Jeskai
    frozenset("BGU"): "shadow black, wild green, deep blue",          # Sultai
    frozenset("RWB"): "burning red, bone white, void black",          # Mardu
    frozenset("GUR"): "emerald green, ocean blue, molten red",        # Temur
    # all five
    frozenset(WUBRG): "rainbow prismatic, all five mana colors",
}
_PALETTE_FALLBACK_BY_COUNT = {
    3: "three-color blend, rich jewel tones",
    4: "four-color convergence, rich jewel tones",
}

# Type contexts (moved from app.createCardImage). Checked in order, so an
# "Artifact Creature" gets the creature context.
TYPE_CONTEXTS = {
    "creature": "detailed creature portrait, living being, character focus",
    "instant": "magical effect in progress, spell energy, dynamic action",
    "sorcery": "grand magical ritual, powerful spell effect",
    "artifact": "detailed artifact object, ancient relic, object focus",
    "enchantment": "magical aura, enchanted environment, mystical atmosphere",
    "land": "sweeping landscape view, terrain, natural environment",
    "planeswalker": "powerful planeswalker character, magical being, character focus",
    "battle": "epic battle scene, warfare, dramatic confrontation",
}


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of CLIP tokens - CLIP tokenizer splits on spaces and punctuation.
    This is a conservative estimate to stay under the 77 token limit.
    """
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return len(tokens)


def truncate_prompt_smartly(prompt: str, max_tokens: int = 75) -> str:
    """
    Intelligently truncate prompt while preserving the most important elements.
    Priority: subject > style > color palette > lighting
    """
    estimated_tokens = estimate_tokens(prompt)

    if estimated_tokens <= max_tokens:
        return prompt

    print(f"⚠️  Prompt too long ({estimated_tokens} tokens), truncating...")

    parts = prompt.split(', ')

    # Prioritize parts: Subject > Color > Style > Magic context > Lighting
    subject_parts = []
    color_parts = []
    style_parts = []
    magic_parts = []
    lighting_parts = []

    for part in parts:
        part_lower = part.lower()
        if 'color palette' in part_lower:
            color_parts.append(part)
        elif any(keyword in part_lower for keyword in ['magic: the gathering', 'card art']):
            magic_parts.append(part)
        elif any(keyword in part_lower for keyword in ['fantasy art', 'style', 'detailed', 'illustration', 'artwork']):
            style_parts.append(part)
        elif any(keyword in part_lower for keyword in ['lighting', 'contrast', 'dramatic']):
            lighting_parts.append(part)
        else:
            subject_parts.append(part)

    final_parts = subject_parts
    test_prompt = ', '.join(final_parts)
    for group in (color_parts, style_parts, magic_parts, lighting_parts):
        for part in group:
            if estimate_tokens(test_prompt + ', ' + part) <= max_tokens:
                final_parts.append(part)
                test_prompt = ', '.join(final_parts)
                break

    final_prompt = ', '.join(final_parts)
    print(f"✂️  Truncated to {estimate_tokens(final_prompt)} tokens: {final_prompt[:100]}...")
    return final_prompt


def _normalize_colors(card_data: dict | None) -> tuple[list[str], bool]:
    """(WUBRG colors in canonical order, explicitly colorless?)."""
    raw = (card_data or {}).get("colors") or []
    letters = set()
    for color in raw:
        if not isinstance(color, str):
            continue
        c = color.strip()
        letters.add(_COLOR_NAMES.get(c.lower(), c.upper()))
    return [c for c in WUBRG if c in letters], "C" in letters


def _type_context(card_data: dict | None) -> str:
    card_type = str((card_data or {}).get("type") or "").lower()
    for keyword, context in TYPE_CONTEXTS.items():
        if keyword in card_type:
            return context
    return GENERIC_CONTEXT


def _color_part(card_data: dict | None) -> str:
    """Mood hints plus palette, or '' when the card has no color information."""
    colors, colorless = _normalize_colors(card_data)
    if not colors:
        card_type = str((card_data or {}).get("type") or "").lower()
        if colorless or "artifact" in card_type:
            return f"{COLOR_PALETTES[frozenset()]} palette"
        return ""
    if len(colors) == 1:
        mood = COLOR_MOODS[colors[0]]
    else:
        mood = ", ".join(COLOR_MOODS[c].split(",")[0] for c in colors)
    palette = COLOR_PALETTES.get(frozenset(colors)) or _PALETTE_FALLBACK_BY_COUNT.get(len(colors), "")
    return f"{mood}, {palette} palette" if palette else mood


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    """Keep the leading words of text that fit in max_tokens."""
    if estimate_tokens(text) <= max_tokens:
        return text
    kept: list[str] = []
    for word in text.split():
        if estimate_tokens(" ".join(kept + [word])) > max_tokens:
            break
        kept.append(word)
    return " ".join(kept).rstrip(" ,;:.-")


def build_art_prompt(prompt: str, card_data: dict | None) -> tuple[str, str]:
    """(positive prompt of at most 75 CLIP tokens, subject first; negative prompt).

    The positive prompt is ordered: subject, type context, color mood and palette,
    style suffix. When the budget is tight the subject is trimmed (keeping its first
    words, never below SUBJECT_MIN_TOKENS), then the palette, type context and style
    are dropped in that order.
    """
    subject = " ".join(str(prompt or "").split()).strip(" ,;")
    if not subject:
        subject = str((card_data or {}).get("name") or "").strip() or "a fantasy scene"

    type_context = _type_context(card_data)
    color_part = _color_part(card_data)
    tail = [p for p in (type_context, color_part, STYLE_SUFFIX) if p]

    def tail_tokens(parts: list[str]) -> int:
        return sum(estimate_tokens(p) + 1 for p in parts)  # +1 for the joining comma

    subject = _trim_to_tokens(subject, max(MAX_PROMPT_TOKENS - tail_tokens(tail), SUBJECT_MIN_TOKENS))

    # Drop lowest-priority tail parts until everything fits.
    for part in [p for p in (color_part, type_context, STYLE_SUFFIX) if p]:
        if estimate_tokens(subject) + tail_tokens(tail) <= MAX_PROMPT_TOKENS:
            break
        tail.remove(part)

    positive = ", ".join([subject] + tail)
    # Final guard; a no-op because the budget above already holds.
    positive = truncate_prompt_smartly(positive, max_tokens=MAX_PROMPT_TOKENS)
    return positive, NEGATIVE_PROMPT


def generate_art(prompt: str, card_data: dict | None) -> str:
    """Raw base64 PNG (no data: prefix) at config.ART_BOX_SIZE."""
    raise NotImplementedError
