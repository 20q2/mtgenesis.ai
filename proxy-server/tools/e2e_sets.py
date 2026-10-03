"""
Commander sets shared by the e2e tools' --sets mode (tools/e2e_rules_text.py, tools/e2e_art.py).

Each set is one commander request at its own mana value, expanded to its three versions
(identical params) exactly as POST /generations does, with the director's three briefs
attached unless the run is a --no-director baseline. Specs:
docs/superpowers/specs/2026-10-02-card-director-design.md §7 and
docs/superpowers/specs/2026-10-03-commander-rules-design.md.
"""
from __future__ import annotations

import time

SET_SPECS = [
    {"name": "Zur'ka, Élan of Ash", "manaCost": "{B}", "colors": ["B"], "type": "Creature",
     "subtype": "Human Cleric", "rarity": "mythic", "cmc": 3},
    {"name": "Mother Thornwild", "manaCost": "{G}{G}", "colors": ["G"], "type": "Creature",
     "subtype": "Elf Druid", "rarity": "rare", "cmc": 4},
    {"name": "Ixen of the Split Sky", "manaCost": "{U}{R}", "colors": ["U", "R"], "type": "Creature",
     "subtype": "Human Wizard", "rarity": "mythic", "cmc": 5},
    {"name": "The Bronze Warden", "manaCost": "", "colors": [], "type": "Creature",
     "subtype": "Golem", "rarity": "uncommon", "cmc": 4},
    # A Vehicle commander: crew text and the Vehicle's +2 stat points (spec 2026-10-03 §7).
    {"name": "The Iron Pilgrim", "manaCost": "{R}", "colors": ["R"], "type": "Artifact",
     "subtype": "Vehicle", "rarity": "rare", "cmc": 5, "commanderKind": "vehicle"},
]


def art_prompt(params: dict) -> str:
    """The art prompt the set builder sends (card-form generateArtPromptText in commander mode)."""
    noun = (params.get("subtype") or "creature").lower()
    return f"{params['name']}, a legendary {noun}, medium scale"


def build_set(spec: dict, use_director: bool, client, model: str) -> tuple[list[dict], list | None, float]:
    """(slot params for versions 1-3, the briefs or None, seconds the director took)."""
    import director
    from commander_rules import commander_params

    params = commander_params(spec, spec["cmc"])
    slots = [dict(params) for _ in range(3)]
    if not use_director:
        return slots, None, 0.0
    t0 = time.time()
    briefs = director.write_briefs(slots[0], 3, [], client, model)
    seconds = time.time() - t0
    if briefs:
        slots = [{**params, "brief": b} for params, b in zip(slots, briefs)]
    return slots, briefs, seconds
