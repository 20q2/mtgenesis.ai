"""
Commander sets shared by the e2e tools' --sets mode (tools/e2e_rules_text.py, tools/e2e_art.py).

Each set is one commander request, expanded to its 3-, 4- and 5-mana versions exactly as
POST /generations does, with the director's three briefs attached unless the run is a
--no-director baseline. Spec: docs/superpowers/specs/2026-10-02-card-director-design.md §7.
"""
from __future__ import annotations

import time

SET_SPECS = [
    {"name": "Zur'ka, Élan of Ash", "manaCost": "{B}", "colors": ["B"], "type": "Creature",
     "subtype": "Human Cleric", "rarity": "mythic"},
    {"name": "Mother Thornwild", "manaCost": "{G}{G}", "colors": ["G"], "type": "Creature",
     "subtype": "Elf Druid", "rarity": "rare"},
    {"name": "Ixen of the Split Sky", "manaCost": "{U}{R}", "colors": ["U", "R"], "type": "Creature",
     "subtype": "Human Wizard", "rarity": "mythic"},
    {"name": "The Bronze Warden", "manaCost": "", "colors": [], "type": "Creature",
     "subtype": "Golem", "rarity": "rare"},
]


def art_prompt(params: dict) -> str:
    """The art prompt the set builder sends (card-form generateArtPromptText in commander mode)."""
    noun = (params.get("subtype") or "creature").lower()
    return f"{params['name']}, a legendary {noun}, medium scale"


def build_set(spec: dict, use_director: bool, client, model: str) -> tuple[list[dict], list | None, float]:
    """(slot params for versions 1-3, the briefs or None, seconds the director took)."""
    import director
    from storage import commander_slot_params

    slots = [commander_slot_params(spec, slot) for slot in (1, 2, 3)]
    if not use_director:
        return slots, None, 0.0
    t0 = time.time()
    briefs = director.write_briefs(slots[0], 3, [], client, model)
    seconds = time.time() - t0
    if briefs:
        slots = [{**params, "brief": b} for params, b in zip(slots, briefs)]
    return slots, briefs, seconds
