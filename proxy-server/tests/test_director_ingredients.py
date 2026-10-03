"""Director: random on-color ingredients per brief, and the design guidance it shares with
the rules-text prompt.

Spec: docs/superpowers/specs/2026-10-02-card-director-design.md (Ingredients).
"""
import random

import director
import rules_text
from director import SHAPES, TWISTS, draw_ingredients
from rules_text import COLOR_HOOKS, COLORLESS_HOOKS, DESIGN_GUIDE, POWER_GUIDE

MONO = {"name": "Gravecaller of Vey", "type": "Creature", "subtype": "Zombie Cleric",
        "colors": ["B"], "manaCost": "{2}{B}", "cmc": 3, "rarity": "rare"}
DUO = {**MONO, "colors": ["W", "U"], "manaCost": "{1}{W}{U}"}
TRIO = {**MONO, "colors": ["B", "R", "G"], "manaCost": "{B}{R}{G}"}
COLORLESS = {**MONO, "colors": [], "type": "Artifact", "subtype": "", "manaCost": "{3}"}
INSTANT = {**MONO, "type": "Instant", "subtype": ""}


def test_mono_color_hooks_are_on_color():
    for seed in range(20):
        (ing,) = draw_ingredients(MONO, 1, random.Random(seed))
        assert len(ing["hooks"]) == 2
        assert all(h in COLOR_HOOKS["B"] for h in ing["hooks"])
        assert ing["shape"] in SHAPES
        assert ing["twist"] is None or ing["twist"] in TWISTS


def test_two_colors_draw_one_hook_from_each():
    for seed in range(20):
        (ing,) = draw_ingredients(DUO, 1, random.Random(seed))
        assert sum(h in COLOR_HOOKS["W"] for h in ing["hooks"]) >= 1
        assert sum(h in COLOR_HOOKS["U"] for h in ing["hooks"]) >= 1


def test_three_colors_stay_on_color():
    on = {h for c in "BRG" for h in COLOR_HOOKS[c]}
    for seed in range(20):
        for ing in draw_ingredients(TRIO, 3, random.Random(seed)):
            assert set(ing["hooks"]) <= on


def test_set_versions_never_share_a_hook_or_shape():
    for seed in range(50):
        ings = draw_ingredients(MONO, 3, random.Random(seed))
        hooks = [h for i in ings for h in i["hooks"]]
        assert len(hooks) == len(set(hooks))
        assert len({i["shape"] for i in ings}) == 3
        twists = [i["twist"] for i in ings if i["twist"]]
        assert len(twists) == len(set(twists))


def test_colorless_set_has_enough_hooks():
    # Only five colorless hooks for six slots: repeats are allowed then, but never within a brief.
    for seed in range(20):
        for ing in draw_ingredients(COLORLESS, 3, random.Random(seed)):
            assert len(set(ing["hooks"])) == 2
            assert set(ing["hooks"]) <= set(COLORLESS_HOOKS)


def test_spells_and_planeswalkers_have_no_shape():
    walker = {**MONO, "type": "Planeswalker", "subtype": "Vey"}
    for card in (INSTANT, {**MONO, "type": "Sorcery"}, walker):
        (ing,) = draw_ingredients(card, 1, random.Random(0))
        assert ing["shape"] is None


def test_twist_comes_up_sometimes():
    twists = [draw_ingredients(MONO, 1, random.Random(s))[0]["twist"] for s in range(200)]
    assert 40 < sum(t is not None for t in twists) < 160


def test_seeded_messages_are_deterministic():
    a = director._messages(MONO, 3, [], random.Random(7))
    b = director._messages(MONO, 3, [], random.Random(7))
    c = director._messages(MONO, 3, [], random.Random(8))
    assert a == b
    assert a != c


def test_prompt_carries_guides_and_each_briefs_ingredients():
    rng = random.Random(3)
    ings = draw_ingredients(MONO, 3, random.Random(3))
    system, user = (m["content"] for m in director._messages(MONO, 3, [], rng))
    assert DESIGN_GUIDE in system and POWER_GUIDE in system
    for n, ing in enumerate(ings, 1):
        assert f"Brief {n}" in user
        for hook in ing["hooks"]:
            assert hook in user
        if ing["twist"]:
            assert ing["twist"] in user
    # The shape goes to the rules-text writer, not the director (it made mechanics read like rules text).
    assert not any(shape in user for shape in SHAPES)
    # The full hook list is no longer dumped into the prompt.
    unused = set(COLOR_HOOKS["B"]) - {h for i in ings for h in i["hooks"]}
    assert not any(h in user for h in unused)


def test_rules_text_prompt_still_carries_both_guides():
    assert DESIGN_GUIDE in rules_text.SYSTEM_PROMPT
    assert POWER_GUIDE in rules_text.SYSTEM_PROMPT


def test_prompt_fits_the_context_window():
    # num_ctx is 2560 and the reply needs num_predict; ~3 characters per token is pessimistic.
    avoid = [{"mechanic": "sacrifice tokens to drain each opponent",
              "art": {"setting": "a burning cathedral of black glass"}}] * 2
    for card in (MONO, TRIO, INSTANT):
        messages = director._messages(card, 3, avoid, random.Random(0))
        tokens = sum(len(m["content"]) for m in messages) / 3
        assert tokens + (200 * 3 + 100) < 2560


def test_write_briefs_uses_the_given_rng():
    from test_director import StubClient, brief
    client = StubClient([brief()])
    director.write_briefs(MONO, 1, [], client, "qwen3:8b", rng=random.Random(5))
    expected = director._messages(MONO, 1, [], random.Random(5))
    assert client.calls[0]["messages"] == expected


def test_brief_carries_its_shape_to_the_rules_text():
    from test_director import StubClient, brief
    briefs = director.write_briefs(MONO, 1, [], StubClient([brief()]), "qwen3:8b", rng=random.Random(5))
    (ing,) = draw_ingredients(MONO, 1, random.Random(5))
    assert briefs[0]["shape"] == ing["shape"]
    messages = rules_text.build_messages("", {**MONO, "brief": briefs[0]}, random.Random(0))
    assert f"Carry the mechanic on {ing['shape']}." in messages[-1]["content"]


def test_model_cannot_set_the_shape():
    from test_director import StubClient, brief
    reply = {**brief(), "shape": "a planeswalker ultimate"}
    briefs = director.write_briefs(INSTANT, 1, [], StubClient([reply]), "qwen3:8b", rng=random.Random(1))
    assert "shape" not in briefs[0]


def test_director_stays_within_real_magic():
    assert "never invent" in director.SYSTEM_PROMPT.lower()
    assert "use a named counter type" not in TWISTS
