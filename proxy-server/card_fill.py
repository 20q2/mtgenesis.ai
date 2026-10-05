"""
Fill in the blanks: on the create page, any of name, mana cost, type, subtype and power /
toughness left empty is chosen by the card director in its existing call (director.py), so
the player can leave fields blank and still get a complete card. Supertype is never filled.

Everything the model returns is checked here before it is used; a value that fails is simply
dropped, and the card falls back to what it did before (an "Untitled" name, the stat curve).
"""
from __future__ import annotations

import re

import power_level
from commander_rules import NOT_CREATURE_TYPES

# The create form's type choices (src/app/models/card.model.ts CardTypeOptions).
CARD_TYPES = ("Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Artifact Creature",
              "Enchantment Creature", "Land", "Planeswalker", "Battle")
FILLABLE = ("name", "manaCost", "type", "subtype", "power", "toughness")
NAME_MAX = 30          # the form's limit
SUBTYPE_MAX = 30
MAX_MANA_VALUE = 10

_SYMBOL = r"\{(?:\d{1,2}|[WUBRGCX]|[WUBRG]/[WUBRGP]|2/[WUBRG])\}"
_COST_RE = re.compile(rf"(?:{_SYMBOL})+")
_SUBTYPE_RE = re.compile(r"[A-Za-z][A-Za-z' -]*")
_WHOLE_RE = re.compile(r"\d{1,2}")


def _blank(value) -> bool:
    return value is None or not str(value).strip()


def _has_body(card_type: str, subtype: str) -> bool:
    # A blank type is generated as a Creature (rules_text.build_messages), so treat it as one.
    card_type = (card_type or "").strip().lower()
    return not card_type or "creature" in card_type or "vehicle" in (subtype or "").lower()


def blank_fields(params: dict) -> list[str]:
    """The fields the player left empty that the AI should fill, in FILLABLE order. A Land
    needs no cost; only creatures (and Vehicles) need a body; an unknown type may need both."""
    params = params or {}
    card_type = (params.get("type") or "").strip()
    fields = [f for f in ("name", "type", "subtype") if _blank(params.get(f))]
    if _blank(params.get("manaCost")) and "land" not in card_type.lower():
        fields.append("manaCost")
    if (not card_type or _has_body(card_type, params.get("subtype") or "")) and \
            (_blank(params.get("power")) or _blank(params.get("toughness"))):
        fields += ["power", "toughness"]
    return [f for f in FILLABLE if f in fields]


def fill_schema(fields: list[str]) -> dict:
    """The JSON schema for the director reply's "fill" object: only the blank fields."""
    props = {f: {"type": "string"} for f in fields}
    if "type" in props:
        props["type"] = {"type": "string", "enum": list(CARD_TYPES)}
    return {"type": "object", "properties": props, "required": list(fields)}


def fill_instructions(fields: list[str]) -> str:
    """What the director is told about the blanks."""
    wanted = {
        "name": f"a fitting card name (at most {NAME_MAX} characters)",
        "manaCost": "a mana cost in braces like {2}{W}{U} that suits the rarity and type "
                    f"(mana value at most {MAX_MANA_VALUE}; colors from it are the card's colors)",
        "type": "the card type, one of: " + ", ".join(CARD_TYPES),
        "subtype": "its subtype(s), e.g. Human Wizard for a creature, or an empty string for a "
                   "spell that has none",
        "power": "its power, a whole number", "toughness": "its toughness, a whole number",
    }
    lines = ["Some of this card's fields are blank. In \"fill\", choose them so the whole card "
             "fits together, then write the brief for the finished card:"]
    lines += [f"- {f}: {wanted[f]}" for f in fields]
    if "power" in fields:
        lines.append("- keep the body modest for its cost: power plus toughness close to the "
                     "mana value plus 1 or 2")
    return "\n".join(lines)


def mana_value(cost: str) -> int:
    total = 0
    for sym in re.findall(r"\{([^}]+)\}", cost or ""):
        if sym.isdigit():
            total += int(sym)
        elif sym.upper() in ("X", "Y", "Z"):
            continue
        elif sym.startswith("2/"):
            total += 2
        else:
            total += 1
    return total


def clean_fill(raw: dict, params: dict, fields: list[str]) -> dict:
    """The model's fill, kept only where valid and only for fields that were blank."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    text = {f: " ".join(str(raw.get(f) or "").split()) for f in fields}

    if "name" in fields and 0 < len(text["name"]) <= NAME_MAX and not re.search(r"[{}<>]", text["name"]):
        out["name"] = text["name"]
    if "type" in fields and text["type"] in CARD_TYPES:
        out["type"] = text["type"]
    card_type = out.get("type") or (params.get("type") or "")

    if "manaCost" in fields and _COST_RE.fullmatch(text["manaCost"]) \
            and mana_value(text["manaCost"]) <= MAX_MANA_VALUE and "land" not in card_type.lower():
        out["manaCost"] = text["manaCost"]

    if "subtype" in fields and text["subtype"]:
        sub = text["subtype"]
        creature = not card_type.strip() or "creature" in card_type.lower()
        if len(sub) <= SUBTYPE_MAX and _SUBTYPE_RE.fullmatch(sub) and \
                not (creature and any(w.lower() in NOT_CREATURE_TYPES for w in sub.split())):
            out["subtype"] = sub

    if "power" in fields and _has_body(card_type, out.get("subtype") or params.get("subtype") or ""):
        p, t = text.get("power", ""), text.get("toughness", "")
        if _WHOLE_RE.fullmatch(p) and _WHOLE_RE.fullmatch(t) and int(t) >= 1:
            cost = out.get("manaCost") or params.get("manaCost") or ""
            judged = {**params, **out, "cmc": mana_value(cost) if cost else params.get("cmc", 0)}
            limit = sum(power_level.creature_stats(judged))
            if int(p) + int(t) <= limit:
                out["power"], out["toughness"] = p, t
    return out


def apply_fill(params: dict, fill: dict) -> dict:
    """The card params with the fill merged in; a filled cost also sets colors and mana value."""
    out = {**params, **fill}
    if "manaCost" in fill:
        colors = []
        for sym in re.findall(r"\{([^}]+)\}", fill["manaCost"]):
            for c in sym:
                if c in "WUBRG" and c not in colors:
                    colors.append(c)
        out["colors"] = colors
        out["cmc"] = mana_value(fill["manaCost"])
    return out
