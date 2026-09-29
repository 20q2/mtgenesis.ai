"""rules_text: prompt building, the cleanup that makes model output legal, and the linter.

Every cleanup case below is a real model output seen in the e2e runs
(tools/e2e_rules_text.py). Fast: no app import, no Ollama.
"""
import json
import random

import pytest

import rules_text as rt

DRAGON = {"name": "Vyraxa, Ember Sovereign", "manaCost": "{4}{R}{R}", "colors": ["R"], "type": "Creature",
          "supertype": "Legendary", "subtype": "Dragon", "rarity": "mythic", "cmc": 6}
SOLDIER = {"name": "Dawnwatch Sentry", "manaCost": "{1}{W}", "colors": ["W"], "type": "Creature",
           "subtype": "Human Soldier", "rarity": "common", "cmc": 2}
INSTANT = {"name": "Cinder Snap", "manaCost": "{R}", "colors": ["R"], "type": "Instant", "rarity": "common", "cmc": 1}
AURA = {"name": "Oath", "manaCost": "{1}{W}", "colors": ["W"], "type": "Enchantment", "subtype": "Aura",
        "rarity": "uncommon", "cmc": 2}
EQUIPMENT = {"name": "Blade", "manaCost": "{2}", "colors": [], "type": "Artifact", "subtype": "Equipment",
             "rarity": "rare", "cmc": 2}
VEHICLE = {"name": "Glider", "manaCost": "{3}", "colors": [], "type": "Artifact", "subtype": "Vehicle",
           "rarity": "uncommon", "cmc": 3}
ENCHANTMENT = {"name": "Banner", "manaCost": "{2}{R}{W}", "colors": ["R", "W"], "type": "Enchantment",
               "rarity": "uncommon", "cmc": 4}
LAND = {"name": "Observatory", "manaCost": "", "colors": [], "type": "Land", "rarity": "rare", "cmc": 0}
WALKER = {"name": "Kaelis, Stormweaver", "manaCost": "{2}{U}{R}", "colors": ["U", "R"], "type": "Planeswalker",
          "supertype": "Legendary", "rarity": "mythic", "cmc": 4}
COLOSSUS = {"name": "Rootmass Colossus", "manaCost": "{2}{G}{G}", "colors": ["G"], "type": "Creature",
            "subtype": "Elemental", "rarity": "rare", "cmc": 4, "power": "*", "toughness": "*"}


def clean(abilities, card):
    return rt.clean_abilities(abilities, card)


# ===== Cleanup =====

@pytest.mark.parametrize("raw, card, expected", [
    # symbols and wording
    (["Tap: Add {G}."], SOLDIER, ["{T}: Add {G}."]),
    (["When this creature enters the battlefield, draw a card."], SOLDIER, ["When this creature enters, draw a card."]),
    (["Whenever you gain life, creatures you control gain +1/+1 until end of turn."], ENCHANTMENT,
     ["Whenever you gain life, creatures you control get +1/+1 until end of turn."]),
    (["Target creature gets +2 power until end of turn."], INSTANT, ["Target creature gets +2/+0 until end of turn."]),
    (["It cannot be blocked."], SOLDIER, ["It can't be blocked."]),
    (["Whenever this creature attacks, put a treasure token onto the battlefield."], SOLDIER,
     ["Whenever this creature attacks, create a Treasure token."]),
    # self references
    (["When Vyraxa, Ember Sovereign enters, draw a card."], DRAGON, ["When Vyraxa enters, draw a card."]),
    (["At the beginning of your end step, Vyrax, create a 1/1 red Dragon creature token with flying."], DRAGON,
     ["At the beginning of your end step, create a 1/1 red Dragon creature token with flying."]),
    (["Vyraxa, the dragon queen, can't be blocked by creatures with power 4 or less."], DRAGON,
     ["Vyraxa can't be blocked by creatures with power 4 or less."]),
    (["Enters with three +1/+1 counters."], SOLDIER,
     ["This creature enters with three +1/+1 counters on it."]),
    # triggers
    (["Whenever this enchantment enters, you gain 2 life."], ENCHANTMENT, ["When this enchantment enters, you gain 2 life."]),
    (["When you draw a card, scry 1."], SOLDIER, ["Whenever you draw a card, scry 1."]),
    (["When this creature attacks, you gain 1 life."], SOLDIER, ["Whenever this creature attacks, you gain 1 life."]),
    (["When Vyraxa attacks, you may pay {R}{R}: Vyraxa deals 2 damage to any target."], DRAGON,
     ["Whenever Vyraxa attacks, you may pay {R}{R}. If you do, Vyraxa deals 2 damage to any target."]),
    # costs
    (["Roots, {T}: Add {G}."], SOLDIER, ["{T}: Add {G}."]),
    (["{Q}, {T}: Sacrifice this land: Add {C}."], LAND, ["{Q}, {T}, Sacrifice this land: Add {C}."]),
    (["{T}, Pay 1 charge counter: Add {R} and {U}."], LAND, ["{T}, Remove a charge counter from this land: Add {R}{U}."]),
    (["{T}: Add {C} or {U} or {R} or {G} or {B}."], LAND, ["{T}: Add one mana of any color."]),
    (["{T}: Add {1}."], LAND, ["{T}: Add {C}."]),
])
def test_cleanup_fixes(raw, card, expected):
    assert clean(raw, card) == expected


def test_keywords_merge_onto_one_line_and_invented_ones_are_dropped():
    out = clean(["Flying, trample, primal growth", "Reach", "Miststep."], DRAGON)
    assert out == ["Flying, trample"]  # reach is redundant with flying; invented words go


def test_keywords_print_in_rules_order():
    assert clean(["Trample, flying"], DRAGON) == ["Flying, trample"]
    assert clean(["Lifelink, deathtouch"], DRAGON) == ["Deathtouch, lifelink"]


def test_a_during_clause_becomes_a_trigger():
    assert clean(["Vyraxa deals 2 damage to each opponent during your end step."], DRAGON) == \
        ["At the beginning of your end step, Vyraxa deals 2 damage to each opponent."]


def test_mana_symbol_used_as_an_amount_is_an_error():
    issues = rt.lint_rules_text("+1: Kaelis deals 1 damage to any target. You may pay {1} to add {C} damage.\n"
                                "−2: Draw a card.", WALKER)
    assert any("used as a number" in m for _, m in issues)


def test_protection_color_is_lowercase():
    assert clean(["Flash, Protection from Black"], SOLDIER) == ["Flash, protection from black"]


def test_run_on_abilities_are_split():
    out = clean(["Trample, menace. {T}, Sacrifice a Goblin: Draw a card. Whenever this creature attacks, you gain 1 life."],
                {**DRAGON, "rarity": "mythic"})
    assert out == ["Trample, menace", "{T}, Sacrifice a Goblin: Draw a card.",
                   "Whenever Vyraxa attacks, you gain 1 life."]


def test_keyword_then_trigger_joined_by_comma_is_split():
    assert clean(["Protection from blue, when this enchantment enters, deal 2 damage to any target."], ENCHANTMENT) == \
        ["Protection from blue", "When this enchantment enters, this enchantment deals 2 damage to any target."]


def test_spell_loses_costs_triggers_and_creature_keywords():
    assert clean(["Flying", "{R}: Deal 2 damage to any target."], INSTANT) == ["This spell deals 2 damage to any target."]
    assert clean(["When you cast this spell, draw a card."], INSTANT) == ["Draw a card."]


def test_type_specific_lines_are_added_or_removed():
    assert clean(["Enchanted creature gets +2/+2."], AURA)[0] == "Enchant creature"
    assert clean(["Equipped creature gets +1/+1."], EQUIPMENT)[-1] == "Equip {2}"
    assert clean(["Flying", "Crew 1", "Crew 2", "Crew N"], VEHICLE) == ["Flying", "Crew 1"]
    assert clean(["Whenever a creature you control attacks, it gets +1/+0 until end of turn.", "Equip {1}"],
                 ENCHANTMENT) == ["Whenever a creature you control attacks, it gets +1/+0 until end of turn."]


def test_equipment_wording_on_a_plain_enchantment_becomes_the_team():
    assert clean(["Equipped creature gets +1/+1 for each +1/+1 counter on another creature you control."], ENCHANTMENT) == \
        ["Creatures you control get +1/+1 for each +1/+1 counter on another creature you control."]


def test_planeswalker_loyalty_is_normalized_and_deduped():
    out = clean(["Starting loyalty: 4", "+1: Draw a card.", "-2: Destroy target artifact.",
                 "+1: Draw two cards.", "- 7: You get an emblem with \"Creatures you control have flying.\""], WALKER)
    assert out == ["+1: Draw a card.", "−2: Destroy target artifact.",
                   "−7: You get an emblem with \"Creatures you control have flying.\""]


def test_variable_power_toughness_gets_defined_first():
    assert clean(["Trample"], COLOSSUS) == \
        ["Trample", "This creature's power and toughness are each equal to the number of lands you control."]
    partial = clean(["Trample", "This creature's power is equal to the number of Forests you control."], COLOSSUS)
    assert partial[1] == "This creature's power and toughness are each equal to the number of Forests you control."


def test_reminder_text_and_labels_are_stripped():
    assert clean(['"Keywords: Flying"', "**Triggered ability:** When this creature dies, draw a card.",
                  "Cycling {2} (You may pay {2} and discard this card. Draw a card.)"], {**DRAGON, "rarity": "mythic"}) == \
        ["Flying, cycling {2}", "When Vyraxa dies, draw a card."]


def test_rarity_cap_drops_trailing_abilities():
    out = clean(["Flying", "When this creature enters, draw a card.", "Whenever this creature attacks, scry 1.",
                 "{2}: This creature gets +1/+0 until end of turn."], SOLDIER)
    assert len(out) == 2  # common: two abilities


# ===== Lint =====

@pytest.mark.parametrize("text, card, fragment", [
    ("10101011100010, legendary construct.", {**DRAGON, "name": "10101011100010", "type": "Artifact Creature",
                                               "subtype": "Construct"}, "header"),
    ("Flying, shadow, primal growth.", DRAGON, "unknown keyword"),
    ("{T}: Add {B}, {G}, Sacrifice a creature: Draw a card.", DRAGON, "two activated abilities"),
    ("Quick Strike: Target creature gains double strike until end of turn.", INSTANT, "label used as an activation cost"),
    ("Scry 1, then draw a card.", SOLDIER, "no trigger or cost"),
    ("Gain control of that creature until end of turn.", INSTANT, "dangling"),
    ("Whenever Vyr, the player gains 2 life.", DRAGON, "no event"),
    ("Create three lightning bolts.", INSTANT, "isn't a token"),
    ("Create a storm token.", INSTANT, "undefined token"),
    ("Remove any target from the game.", INSTANT, "any target"),
    ("Whenever this creature attacks, you get an emblem with \"You gain 1 life.\"", SOLDIER, "emblem"),
    ("Equipped creature has flying.\nCrew 1", VEHICLE, "equipped creature"),
    ("Bligma gets -1/-1 until end of turn.", DRAGON, "duration"),
    ("Enchant creature\nEnchanted creature has lifetap.", AURA, "unknown keyword"),
    ("Destroy target creature.", {**INSTANT, "manaCost": "{X}{R}"}, "X in mana cost"),
    ("Trample", COLOSSUS, "never defined"),
    ("+1: Draw a card.\n+1: Scry 2.", WALKER, "no minus"),
])
def test_lint_catches(text, card, fragment):
    issues = rt.lint_rules_text(text, card)
    assert any(s == "error" and fragment in m for s, m in issues), issues


@pytest.mark.parametrize("text, card", [
    ("Flying, trample\nWhenever Vyraxa deals combat damage to a player, create a Treasure token.", DRAGON),
    ("This spell deals 3 damage to any target. Scry 1.", INSTANT),
    ("Enchant creature\nEnchanted creature gets +2/+1 and has first strike.", AURA),
    ("Equipped creature gets +1/+0 and has \"Whenever this creature deals combat damage to a player, create a Treasure token.\"\nEquip {2}", EQUIPMENT),
    ("+1: Scry 2, then draw a card.\n−2: Return target creature to its owner's hand.", WALKER),
    ("This land enters tapped.\n{T}: Add {U} or {R}.", LAND),
    ("Creatures you control have protection from red and white.", ENCHANTMENT),
    ("This creature's power and toughness are each equal to the number of lands you control.", COLOSSUS),
])
def test_lint_accepts_real_templating(text, card):
    assert rt.error_count(rt.lint_rules_text(text, card)) == 0, rt.lint_rules_text(text, card)


# ===== Parsing and prompt =====

def test_parse_reply_json_and_fallbacks():
    assert rt.parse_reply('{"abilities": ["Flying", "Draw a card."]}') == ["Flying", "Draw a card."]
    assert rt.parse_reply('```json\n{"abilities": ["Flying"]}\n```') == ["Flying"]
    assert rt.parse_reply('"Flying" "When ~ enters, draw a card."') == ["Flying", "When ~ enters, draw a card."]
    assert rt.parse_reply("Flying\nWhen ~ enters, draw a card.") == ["Flying", "When ~ enters, draw a card."]


def test_prompt_carries_card_facts_and_requirements():
    messages = rt.build_messages("a dragon queen on a volcano", {**COLOSSUS, "subtype": "Human Warrior"}, random.Random(0))
    system, user = messages[0]["content"], messages[1]["content"]
    assert "Oracle templating" in system and '"abilities"' in system
    assert "Name: Rootmass Colossus" in user
    assert "power and toughness are variable" in user
    assert "does not fly" in user
    assert "Design hook" in user


def test_prompt_type_requirements():
    assert "Equip" in rt.build_messages("", EQUIPMENT, random.Random(0))[1]["content"]
    assert "exactly one Crew" in rt.build_messages("", VEHICLE, random.Random(0))[1]["content"]
    assert "Enchant creature" in rt.build_messages("", AURA, random.Random(0))[1]["content"]
    assert "loyalty abilities" in rt.build_messages("", WALKER, random.Random(0))[1]["content"]
    assert "Use X" in rt.build_messages("", {**INSTANT, "manaCost": "{X}{R}"}, random.Random(0))[1]["content"]


# ===== Generation loop =====

class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": json.dumps({"abilities": reply})}}


def test_generation_stops_at_the_first_clean_reply():
    client = FakeClient(["Flying", "Whenever Vyraxa attacks, create a Treasure token."], ["Haste"])
    text = rt.generate_rules_text("dragon", DRAGON, client, "m", attempts=3)
    assert text == "Flying\nWhenever Vyraxa attacks, create a Treasure token."
    assert len(client.calls) == 1
    assert client.calls[0]["format"] == rt.RESPONSE_SCHEMA


def test_generation_retries_and_keeps_the_best_attempt():
    bad = ["Flying", "Create a storm token."]  # lint error: undefined token
    good = ["Flying", "Whenever Vyraxa attacks, create a Treasure token."]
    client = FakeClient(bad, good)
    assert rt.generate_rules_text("dragon", DRAGON, client, "m", attempts=3).endswith("Treasure token.")
    assert len(client.calls) == 2


def test_generation_raises_when_the_model_is_unreachable():
    client = FakeClient(ConnectionError("ollama down"))
    with pytest.raises(ConnectionError):
        rt.generate_rules_text("dragon", DRAGON, client, "m")


def test_later_failure_keeps_the_earlier_attempt():
    client = FakeClient(["Flying", "Create a storm token."], TimeoutError("slow"))
    assert rt.generate_rules_text("dragon", DRAGON, client, "m", attempts=2) == "Flying\nCreate a storm token."
