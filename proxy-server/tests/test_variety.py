"""Variety across cards: wider color hooks, the director's memory of recent mechanics, and
clichés that count against a rules-text candidate.

The red cards kept coming out as "Sacrifice this creature: it deals 3 damage to any target"
and a quarter of all cards opened with menace.
"""
import json
import random
import re

import director
import rules_text as rt
from rules_text import CLICHES, COLOR_HOOKS, cliche_count

HUMAN = {"name": "Brakka, Pit Boss", "type": "Creature", "supertype": "Legendary",
         "subtype": "Human Warrior", "colors": ["R"], "manaCost": "{2}{R}", "cmc": 3,
         "rarity": "rare"}


# ----- A. wider hooks, rotated keyword suggestions -----
def test_every_color_has_a_wide_hook_list():
    for color, hooks in COLOR_HOOKS.items():
        assert len(hooks) >= 14, color
        assert len(set(hooks)) == len(hooks), color


def test_damage_is_a_small_share_of_red():
    damage = [h for h in COLOR_HOOKS["R"] if re.search(r"damage|burst", h)]
    assert len(damage) <= 2


def test_grounded_keywords_rotate_and_menace_is_not_always_offered():
    lines = set()
    menace = 0
    for seed in range(40):
        user = rt.build_messages("", HUMAN, random.Random(seed))[1]["content"]
        (line,) = [l for l in user.splitlines() if "does not fly" in l]
        lines.add(line)
        menace += "menace" in line
    assert len(lines) > 5
    assert menace < 30


def test_activated_shape_does_not_ask_for_a_sacrifice():
    assert not any("real cost" in s for s in director.SHAPES)


# ----- C. clichés -----
def test_cliches_are_counted():
    sac = "Sacrifice Brakka: It deals 3 damage to any target."
    sac_other = "Sacrifice a creature: This creature deals 2 damage to any target."
    menace = "Vigilance, menace\nWhenever Brakka attacks, create a Treasure token."
    fresh = "Haste\nWhenever Brakka attacks, goad target creature an opponent controls."
    assert cliche_count(sac) == 1
    assert cliche_count(sac_other) == 1
    assert cliche_count(menace) == 1
    assert cliche_count(sac + "\n" + menace) == 2
    assert cliche_count(fresh) == 0
    assert cliche_count("") == 0
    assert all(isinstance(name, str) for name, _ in CLICHES)


class ReplyClient:
    """Answers each rules-text call with the next list of abilities."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        return {"message": {"content": json.dumps({"abilities": self.replies.pop(0)})}}


def test_a_cliche_triggers_another_attempt_and_loses_to_a_fresh_one():
    cliche = ["Sacrifice Brakka: Brakka deals 2 damage to any target."]
    fresh = ["Whenever Brakka attacks, goad target creature an opponent controls."]
    client = ReplyClient(cliche, fresh)
    text = rt.generate_rules_text("", HUMAN, client, "m", attempts=2, rng=random.Random(0))
    assert client.calls == 2
    assert "goad" in text


def test_a_cliche_still_wins_over_a_lint_error():
    cliche = ["Sacrifice Brakka: Brakka deals 2 damage to any target."]
    broken = ["Draw a card."]  # a bare instruction on a permanent is a lint error
    text = rt.generate_rules_text("", HUMAN, ReplyClient(broken, cliche), "m", attempts=2,
                                  rng=random.Random(0))
    assert "damage" in text


# ----- B. the director remembers recent mechanics -----
def _brief(mechanic):
    return {"identity": "A pit fighter who never stays down", "mechanic": mechanic,
            "art": {"subject": "a human warrior in scarred leather", "action": "roaring",
                    "setting": f"a fighting pit of {mechanic}", "framing": "low angle",
                    "light": "torchlight"}}


class StubClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": json.dumps({"briefs": self.replies.pop(0)})}}


RECENT = ["sacrifice creatures to deal direct damage", "treasure tokens fuel big attacks"]


def test_recent_mechanics_are_not_shown_to_the_model():
    # Listed in the prompt, they anchored the 8B model: 9 red cards in a row copied
    # "Steal fire from creatures to fuel your attacks". They steer the ingredients instead.
    client = StubClient([_brief("goad enemies into bad attacks")])
    director.write_briefs(HUMAN, 1, [], client, "m", rng=random.Random(1), recent=RECENT)
    user = client.calls[0]["messages"][-1]["content"]
    for mechanic in RECENT:
        assert mechanic not in user


def test_hooks_that_echo_recent_mechanics_are_not_drawn():
    recent = ["steal fire from creatures to fuel your attacks", "treasure tokens fuel big attacks"]
    for seed in range(40):
        (ing,) = director.draw_ingredients(HUMAN, 1, random.Random(seed), recent=recent)
        for hook in ing["hooks"]:
            assert not director.content_words(hook) & {"steal", "treasure", "tokens", "attacks", "fuel", "fire"}, hook


def test_recent_never_empties_the_hook_pool():
    every_word = [" ".join(COLOR_HOOKS["R"])]
    (ing,) = director.draw_ingredients(HUMAN, 1, random.Random(0), recent=every_word)
    assert len(ing["hooks"]) == 2


def test_repeating_a_recent_mechanic_is_retried():
    client = StubClient([_brief("sacrifice creatures to deal direct damage")],
                        [_brief("goad enemies into bad attacks")])
    briefs = director.write_briefs(HUMAN, 1, [], client, "m", rng=random.Random(1), recent=RECENT)
    assert len(client.calls) == 2
    assert briefs[0]["mechanic"] == "goad enemies into bad attacks"


def test_last_attempt_accepts_a_recent_repeat_rather_than_no_brief():
    same = [_brief("sacrifice creatures to deal direct damage")]
    briefs = director.write_briefs(HUMAN, 1, [], StubClient(same, same), "m",
                                   rng=random.Random(1), recent=RECENT)
    assert briefs is not None


def test_recent_mechanics_do_not_touch_the_art():
    # `avoid` (set siblings) shares its subject and settings; `recent` must not.
    client = StubClient([_brief("goad enemies into bad attacks")])
    briefs = director.write_briefs(HUMAN, 1, [], client, "m", rng=random.Random(1), recent=RECENT)
    assert briefs[0]["art"]["subject"] == "a human warrior in scarred leather"
    assert "setting different" not in client.calls[0]["messages"][-1]["content"]


def _add_card(storage, user_id, colors, mechanic):
    card = storage.create_card(user_id, "p", {"name": "x", "colors": colors, "type": "Creature"})
    if mechanic:
        storage.set_card_brief(card["id"], _brief(mechanic))
    return card


def test_storage_recent_mechanics_by_color(tmp_storage):
    user = tmp_storage.login("Andrew")
    _add_card(tmp_storage, user["id"], ["R"], "red one")
    _add_card(tmp_storage, user["id"], ["G"], "green one")
    _add_card(tmp_storage, user["id"], ["R", "G"], "gruul one")
    _add_card(tmp_storage, user["id"], [], "colorless one")
    _add_card(tmp_storage, user["id"], ["R"], None)  # no brief
    _add_card(tmp_storage, user["id"], ["R"], "red two")
    _add_card(tmp_storage, user["id"], ["R"], "red two")  # duplicates collapse
    assert tmp_storage.recent_mechanics(["R"], 10) == ["red two", "gruul one", "red one"]
    assert tmp_storage.recent_mechanics(["G"], 1) == ["gruul one"]
    assert tmp_storage.recent_mechanics([], 10) == ["colorless one"]
