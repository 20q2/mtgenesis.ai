from card_fill import apply_fill, blank_fields, clean_fill, fill_schema, mana_value


def test_blank_fields_lists_what_the_ai_should_fill():
    assert blank_fields({"name": "", "manaCost": "", "type": "", "subtype": ""}) == \
        ["name", "manaCost", "type", "subtype", "power", "toughness"]
    # a full creature: nothing to fill
    full = {"name": "Zur", "manaCost": "{2}{B}", "type": "Creature", "subtype": "Zombie",
            "power": "2", "toughness": "2"}
    assert blank_fields(full) == []
    # supertype is never filled
    assert "supertype" not in blank_fields({"name": ""})


def test_blank_fields_skips_what_the_type_does_not_use():
    # A land has no mana cost and no P/T; an instant has no P/T.
    assert blank_fields({"name": "X", "type": "Land", "manaCost": "", "subtype": "Desert"}) == []
    assert blank_fields({"name": "X", "type": "Instant", "manaCost": "{R}", "subtype": ""}) == ["subtype"]
    # A creature with a cost but no body gets its P/T filled
    assert blank_fields({"name": "X", "type": "Creature", "manaCost": "{1}{G}", "subtype": "Elf"}) == \
        ["power", "toughness"]
    # Whitespace counts as blank
    assert blank_fields({"name": "  ", "type": "Instant", "manaCost": "{R}", "subtype": "x"}) == ["name"]


def test_fill_schema_only_asks_for_the_blanks():
    schema = fill_schema(["name", "type"])
    assert set(schema["properties"]) == {"name", "type"}
    assert schema["required"] == ["name", "type"]
    assert "Creature" in schema["properties"]["type"]["enum"]


def test_mana_value():
    assert mana_value("{2}{W}{U}") == 4
    assert mana_value("{X}{R}") == 1
    assert mana_value("{2/W}{G}") == 3
    assert mana_value("") == 0


def test_clean_fill_keeps_valid_values():
    params = {"name": "", "manaCost": "", "type": "", "subtype": "", "rarity": "rare"}
    raw = {"name": "  Grimbold the Unbowed ", "manaCost": "{2}{R}{R}", "type": "Creature",
           "subtype": "Dwarf Warrior", "power": "4", "toughness": "3"}
    assert clean_fill(raw, params, blank_fields(params)) == {
        "name": "Grimbold the Unbowed", "manaCost": "{2}{R}{R}", "type": "Creature",
        "subtype": "Dwarf Warrior", "power": "4", "toughness": "3"}


def test_clean_fill_drops_bad_values():
    params = {"name": "", "manaCost": "", "type": "", "subtype": "", "rarity": "common"}
    fields = blank_fields(params)
    bad = clean_fill({"name": "x" * 31, "manaCost": "{2}{Q}", "type": "Tribal Wizard",
                      "subtype": "Equipment", "power": "*", "toughness": "3"}, params, fields)
    assert bad == {}
    # mana value above 10 is dropped
    assert "manaCost" not in clean_fill({"manaCost": "{11}"}, params, ["manaCost"])
    # an empty cost for a non-land is not a fill
    assert "manaCost" not in clean_fill({"manaCost": ""}, {**params, "type": "Instant"}, ["manaCost"])
    # fields that weren't blank are never overwritten
    assert clean_fill({"name": "Other"}, {**params, "name": "Mine"}, ["type"]) == {}


def test_clean_fill_keeps_the_body_within_the_stat_curve():
    params = {"name": "X", "manaCost": "{1}{G}", "type": "Creature", "subtype": "Elf", "rarity": "common"}
    # A 2-mana common creature's curve total is 4 (power_level.STAT_TOTAL)
    assert clean_fill({"power": "1", "toughness": "3"}, params, ["power", "toughness"]) == \
        {"power": "1", "toughness": "3"}
    assert clean_fill({"power": "5", "toughness": "5"}, params, ["power", "toughness"]) == {}
    assert clean_fill({"power": "2", "toughness": "0"}, params, ["power", "toughness"]) == {}


def test_clean_fill_judges_the_body_by_the_filled_cost_and_type():
    params = {"name": "", "manaCost": "", "type": "", "subtype": "", "rarity": "common"}
    raw = {"name": "Big", "manaCost": "{4}{G}", "type": "Creature", "subtype": "Beast",
           "power": "4", "toughness": "4"}  # 5-mana common: total 9, so 4/4 is fine
    assert clean_fill(raw, params, blank_fields(params))["power"] == "4"


def test_clean_fill_on_a_land_or_spell_never_returns_a_body():
    params = {"name": "", "manaCost": "", "type": "", "subtype": "", "rarity": "common"}
    out = clean_fill({"name": "Bolt", "manaCost": "{R}", "type": "Instant", "subtype": "",
                      "power": "3", "toughness": "3"}, params, blank_fields(params))
    assert out == {"name": "Bolt", "manaCost": "{R}", "type": "Instant"}


def test_apply_fill_sets_colors_and_mana_value_from_a_filled_cost():
    params = {"name": "", "manaCost": "", "colors": [], "cmc": 0, "type": "", "rarity": "rare"}
    out = apply_fill(params, {"name": "Ixen", "manaCost": "{3}{U}{R}", "type": "Creature"})
    assert out["name"] == "Ixen" and out["manaCost"] == "{3}{U}{R}"
    assert out["colors"] == ["U", "R"] and out["cmc"] == 5
    assert params["name"] == ""  # input untouched
    # a cost the player typed keeps its colors
    kept = apply_fill({"manaCost": "{W}", "colors": ["W"], "cmc": 1}, {"name": "A"})
    assert kept["colors"] == ["W"] and kept["cmc"] == 1
