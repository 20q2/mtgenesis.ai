"""
AI Night commander rules (docs/superpowers/specs/2026-10-03-commander-rules-design.md §2).

A player makes one commander at each of 3, 4 and 5 mana: a Legendary Creature or Vehicle,
Uncommon, Rare or Mythic, whose P/T is a point buy of mana value + 1 (a Vehicle gets 2 more).
commander_params turns a set request into the card params all three versions share.
"""
from __future__ import annotations

import re

import power_level
from storage import StorageError

COMMANDER_CMCS = (3, 4, 5)
COMMANDER_RARITIES = ("uncommon", "rare", "mythic")
VEHICLE_BONUS = 2
# Words that can't be in a creature commander's type line: other cards' subtypes and type words.
NOT_CREATURE_TYPES = frozenset("""
    vehicle equipment fortification aura saga curse shrine class room cartouche background role
    treasure food clue blood gold map powerstone incubator attraction contraption
    plains island swamp mountain forest desert gate lair locus mine tower cave sphere
    creature artifact enchantment instant sorcery land planeswalker battle kindred tribal
    legendary basic snow world ongoing elite host
""".split())

_MANA_SYMBOL_RE = re.compile(r"\{([^}]+)\}")
_WHOLE_NUMBER_RE = re.compile(r"\d+")


def stat_points(cmc: int, is_vehicle: bool) -> int:
    """Total power + toughness a commander may have."""
    return cmc + 1 + (VEHICLE_BONUS if is_vehicle else 0)


def auto_stats(cmc: int, is_vehicle: bool, subtype: str) -> tuple[int, int]:
    """A split that spends every point, leaning by creature type like the stat curve."""
    points = stat_points(cmc, is_vehicle)
    return power_level.lean(points // 2, points - points // 2, subtype)


def _cost(mana_cost: str, cmc: int) -> str:
    """Keeps colored/hybrid/Phyrexian pips, drops generic and X, pads generic up to cmc.
    Each commander is its own design, so its pips may fill its whole mana value."""
    pips = [s for s in _MANA_SYMBOL_RE.findall(mana_cost or "")
            if not s.isdigit() and s.upper() not in ("X", "Y", "Z")]
    pip_value = sum(2 if s.startswith("2/") else 1 for s in pips)
    if pip_value > cmc:
        raise StorageError(400, f"A {cmc}-mana commander's colored pips can add up to at most "
                                f"{cmc} mana")
    generic = cmc - pip_value
    return (f"{{{generic}}}" if generic else "") + "".join(f"{{{s}}}" for s in pips)


def _stats(card_data: dict, cmc: int, is_vehicle: bool, subtype: str) -> tuple[int, int]:
    power, toughness = (str(card_data.get(k) if card_data.get(k) is not None else "").strip()
                        for k in ("power", "toughness"))
    if not power and not toughness:
        return auto_stats(cmc, is_vehicle, subtype)
    if not power or not toughness:
        raise StorageError(400, "Give both power and toughness, or leave both blank")
    if not _WHOLE_NUMBER_RE.fullmatch(power) or not _WHOLE_NUMBER_RE.fullmatch(toughness):
        raise StorageError(400, "P/T must be whole numbers — X and * aren't allowed")
    p, t = int(power), int(toughness)
    if t < 1:
        raise StorageError(400, "Toughness must be at least 1")
    points = stat_points(cmc, is_vehicle)
    if p + t > points:
        raise StorageError(400, f"A {cmc}-mana commander has {points} points; {p}/{t} uses {p + t}")
    return p, t


def _creature_type(subtype) -> str:
    """The creature type line, whitespace tidied; 400 when a word isn't a creature type."""
    subtype = " ".join(str(subtype or "").split())
    for word in subtype.split():
        if word.lower() in NOT_CREATURE_TYPES:
            raise StorageError(400, f"{word} isn't a creature type")
    return subtype


def commander_params(card_data: dict, cmc: int) -> dict:
    """The card params shared by all three versions of a commander at this mana value.
    Raises StorageError 400 with a player-facing message when a rule is broken."""
    if cmc not in COMMANDER_CMCS:
        raise StorageError(400, "A commander costs 3, 4 or 5 mana")
    kind = card_data.get("commanderKind") or "creature"
    if kind not in ("creature", "vehicle"):
        raise StorageError(400, "A commander is a Creature or a Vehicle")
    rarity = str(card_data.get("rarity") or "").strip().lower()
    if rarity not in COMMANDER_RARITIES:
        raise StorageError(400, "Commanders are Uncommon, Rare or Mythic")

    is_vehicle = kind == "vehicle"
    # A Legendary Creature of creature types, or exactly a Legendary Artifact — Vehicle.
    subtype = "Vehicle" if is_vehicle else _creature_type(card_data.get("subtype"))
    power, toughness = _stats(card_data, cmc, is_vehicle, subtype)

    out = {k: v for k, v in card_data.items() if k != "commanderKind"}
    out.update(manaCost=_cost(card_data.get("manaCost") or "", cmc), cmc=cmc,
               supertype="Legendary", type="Artifact" if is_vehicle else "Creature",
               subtype=subtype, rarity=rarity, power=str(power), toughness=str(toughness))
    return out
