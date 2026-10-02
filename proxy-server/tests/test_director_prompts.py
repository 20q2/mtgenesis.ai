"""The rules-text and art prompts follow the director's brief, and are unchanged without one."""
import random

import rules_text as rt
from image_generation import ART_STYLE, NEGATIVE_PROMPT, build_art_prompt

CARD = {"name": "Zur'ka, Élan of Ash", "type": "Creature", "supertype": "Legendary",
        "subtype": "Human Cleric", "colors": ["B"], "manaCost": "{2}{B}", "cmc": 3,
        "rarity": "mythic"}
PROMPT = "Zur'ka, a legendary human cleric, medium scale"
BRIEF = {"identity": "An ash-priest who keeps dead fires burning",
         "mechanic": "sacrifice tokens to drain each opponent",
         "art": {"subject": "a human cleric in ash-grey robes", "action": "raising a smoking censer",
                 "setting": "a ruined fire temple", "framing": "low angle, close",
                 "light": "embers glowing from below"}}

# Snapshots of the prompts from before the director existed (commit 78a733e).
RULES_USER_BEFORE = (
    "Name: Zur'ka, Élan of Ash\nType line: Legendary Creature — Human Cleric\nRarity: mythic\n"
    "Mana cost: {2}{B} (mana value 3)\nColors: black\n"
    "Power/toughness: 3/3 (fixed; design the abilities around this body)\n"
    "Power budget: a 3-mana mythic card, so its abilities together are worth one strong effect or "
    "two medium ones, about \"destroy target creature with power 3 or less\" or a small effect that "
    "repeats every turn.\nConcept: Zur'ka, a legendary human cleric, medium scale\n\n"
    "- Write 4 abilities (a keyword item of up to two keywords counts as one).\n"
    "- Refer to the card as \"Zur'ka\".\n"
    "- A Human does not fly: no flying; prefer grounded keywords like vigilance, first strike, "
    "reach or menace.\n"
    "- Design hook to consider, scaled to the power budget: drain each opponent.")
ART_BEFORE = [
    (PROMPT, CARD,
     "Zur'ka, a legendary human cleric, medium scale, traditional oil painting on canvas, Magic: The "
     "Gathering card art, visible impasto brushstrokes, rich earthy colors, soft atmospheric light, "
     "human character, fully clothed, full figure, ominous, decay, dark violet, charcoal, bone palette"),
    ("Cinder Snap, an instant", {"type": "Instant", "colors": ["R"]},
     "Cinder Snap, an instant, traditional oil painting on canvas, Magic: The Gathering card art, "
     "visible impasto brushstrokes, rich earthy colors, soft atmospheric light, spell in motion, "
     "dynamic action, fiery, wild, crimson, ember orange, smoky brown palette"),
    ("Iron Idol, an artifact", {"type": "Artifact", "colors": []},
     "Iron Idol, an artifact, traditional oil painting on canvas, Magic: The Gathering card art, "
     "visible impasto brushstrokes, rich earthy colors, soft atmospheric light, ornate artifact, "
     "object focus, weathered bronze, stone gray palette"),
]


def user_message(card):
    return rt.build_messages(PROMPT, card, random.Random(7))[1]["content"]


# ----- rules text -----
def test_rules_prompt_without_a_brief_is_unchanged():
    assert user_message(CARD) == RULES_USER_BEFORE
    assert user_message({**CARD, "brief": None}) == RULES_USER_BEFORE


def test_rules_prompt_follows_the_brief():
    text = user_message({**CARD, "brief": BRIEF})
    assert ("Card idea: An ash-priest who keeps dead fires burning. Build the abilities around this "
            "mechanic, scaled to the power budget: sacrifice tokens to drain each opponent.") in text
    assert "Design hook" not in text
    assert "Concept:" not in text
    assert "Power budget:" in text and 'Refer to the card as "Zur\'ka"' in text


# ----- art -----
def test_art_prompt_without_a_brief_is_unchanged():
    for prompt, card, expected in ART_BEFORE:
        assert build_art_prompt(prompt, card)[0] == expected
        assert build_art_prompt(prompt, {**card, "brief": None})[0] == expected


def test_art_prompt_follows_the_brief():
    positive, negative = build_art_prompt(PROMPT, {**CARD, "brief": BRIEF})
    assert positive.startswith("a human cleric in ash-grey robes, raising a smoking censer, "
                               "a ruined fire temple, low angle, close")
    assert "Zur'ka" not in positive
    assert ART_STYLE.split(",")[0] in positive
    assert "fully clothed" in positive
    assert negative == NEGATIVE_PROMPT


def test_art_brief_trims_light_then_framing_then_setting():
    def words(text):  # one token per word keeps the budget arithmetic readable
        return len(text.replace(",", " ").split())

    long_art = {f: f"{f} " + " ".join(f"{f}{i}" for i in range(9)) for f in BRIEF["art"]}
    positive, _ = build_art_prompt(PROMPT, {**CARD, "brief": {**BRIEF, "art": long_art}},
                                   count_tokens=words)
    assert positive.startswith("subject subject0")
    assert "light0" not in positive
    assert "subject8" in positive
    # Whatever survived of the brief keeps its order.
    kept = [f for f in ("subject", "action", "setting", "framing") if f"{f}0" in positive]
    assert kept == ["subject", "action", "setting", "framing"][:len(kept)]
    assert words(positive) <= 75
