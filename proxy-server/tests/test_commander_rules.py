import pytest

import power_level
from commander_rules import auto_stats, commander_params, stat_points
from storage import StorageError

BASE = {"name": "Zur", "manaCost": "{W}{U}", "colors": ["W", "U"], "subtype": "Human Wizard",
        "rarity": "Rare"}

WHOLE_NUMBERS = "P/T must be whole numbers — X and * aren't allowed"


def rejects(card_data, cmc=3):
    with pytest.raises(StorageError) as err:
        commander_params(card_data, cmc)
    assert err.value.status == 400
    return err.value.message


def test_cost_padded_to_each_cmc():
    costs = [commander_params(BASE, cmc) for cmc in (3, 4, 5)]
    assert [p["manaCost"] for p in costs] == ["{1}{W}{U}", "{2}{W}{U}", "{3}{W}{U}"]
    assert [p["cmc"] for p in costs] == [3, 4, 5]
    assert commander_params({**BASE, "manaCost": "{X}{4}{B}{B}"}, 5)["manaCost"] == "{3}{B}{B}"
    assert commander_params({**BASE, "manaCost": ""}, 3)["manaCost"] == "{3}"
    assert commander_params({**BASE, "manaCost": "{W/U}{B/P}"}, 3)["manaCost"] == "{1}{W/U}{B/P}"
    # {2/W} is worth 2 mana on its own.
    assert commander_params({**BASE, "manaCost": "{2/W}{G}"}, 3)["manaCost"] == "{2/W}{G}"
    assert BASE["manaCost"] == "{W}{U}" and "cmc" not in BASE  # input untouched


def test_pips_may_fill_the_commanders_own_mana_value():
    # Each commander is its own design, so a 5-drop can be {W}{W}{U}{U}{B}.
    assert commander_params({**BASE, "manaCost": "{W}{W}{U}{U}"}, 4)["manaCost"] == "{W}{W}{U}{U}"
    assert commander_params({**BASE, "manaCost": "{W}{W}{U}{U}{B}"}, 5)["manaCost"] == "{W}{W}{U}{U}{B}"
    assert commander_params({**BASE, "manaCost": "{W}{W}{U}{U}"}, 5)["manaCost"] == "{1}{W}{W}{U}{U}"
    assert commander_params({**BASE, "manaCost": "{2/W}{2/U}"}, 4)["manaCost"] == "{2/W}{2/U}"


def test_too_many_pips_is_400():
    assert rejects({**BASE, "manaCost": "{W}{W}{U}{U}"}, 3) ==         "A 3-mana commander's colored pips can add up to at most 3 mana"
    rejects({**BASE, "manaCost": "{W}{W}{U}{U}{B}{B}"}, 5)


def test_bad_cmc_is_400():
    rejects(BASE, 2)
    rejects(BASE, 6)


def test_creature_kind():
    p = commander_params(BASE, 3)
    assert p["supertype"] == "Legendary" and p["type"] == "Creature"
    assert p["subtype"] == "Human Wizard" and p["name"] == "Zur" and p["colors"] == ["W", "U"]
    assert "commanderKind" not in p
    assert commander_params({**BASE, "commanderKind": "creature"}, 3)["type"] == "Creature"


def test_vehicle_kind():
    vehicle = {**BASE, "commanderKind": "vehicle"}
    p = commander_params({**vehicle, "subtype": ""}, 3)
    assert p["supertype"] == "Legendary" and p["type"] == "Artifact" and p["subtype"] == "Vehicle"
    assert "commanderKind" not in p
    # A Vehicle commander is exactly "Legendary Artifact — Vehicle", whatever else is typed.
    assert commander_params({**vehicle, "subtype": "Construct"}, 3)["subtype"] == "Vehicle"
    assert commander_params({**vehicle, "subtype": "Vehicle"}, 3)["subtype"] == "Vehicle"
    rejects({**BASE, "commanderKind": "planeswalker"})


def test_rarity():
    for given, stored in (("Uncommon", "uncommon"), ("rare", "rare"), ("MYTHIC", "mythic")):
        assert commander_params({**BASE, "rarity": given}, 3)["rarity"] == stored
    for bad in ("common", ""):
        assert rejects({**BASE, "rarity": bad}) == "Commanders are Uncommon, Rare or Mythic"
    no_rarity = {k: v for k, v in BASE.items() if k != "rarity"}
    assert rejects(no_rarity) == "Commanders are Uncommon, Rare or Mythic"


def test_stat_points():
    assert stat_points(3, False) == 4
    assert stat_points(5, False) == 6
    assert stat_points(3, True) == 6


def test_pt_within_budget():
    for p, t in (("3", "1"), ("1", "3"), ("0", "1"), ("1", "1")):
        params = commander_params({**BASE, "power": p, "toughness": t}, 3)
        assert (params["power"], params["toughness"]) == (p, t)
    vehicle = commander_params({**BASE, "commanderKind": "vehicle", "power": "4", "toughness": "2"}, 3)
    assert (vehicle["power"], vehicle["toughness"]) == ("4", "2")


def test_pt_rejections():
    assert rejects({**BASE, "power": "3", "toughness": "2"}) == \
        "A 3-mana commander has 4 points; 3/2 uses 5"
    for p, t in (("*", "*"), ("X", "2"), ("1+*", "2"), ("1.5", "1")):
        assert rejects({**BASE, "power": p, "toughness": t}) == WHOLE_NUMBERS
    rejects({**BASE, "power": "-1", "toughness": "3"})
    rejects({**BASE, "power": "2", "toughness": "0"})


def test_pt_whitespace_and_leading_zero():
    params = commander_params({**BASE, "power": " 2 ", "toughness": "02"}, 3)
    assert (params["power"], params["toughness"]) == ("2", "2")
    rejects({**BASE, "power": "", "toughness": "3"})
    blank = commander_params({**BASE, "power": "", "toughness": ""}, 3)
    assert (blank["power"], blank["toughness"]) == ("2", "2")
    missing = commander_params(BASE, 3)
    assert (missing["power"], missing["toughness"]) == ("2", "2")


def test_auto_stats():
    assert auto_stats(3, False, "Human") == (2, 2)
    assert auto_stats(4, False, "Human") == (2, 3)
    assert auto_stats(5, False, "Goblin") == (4, 2)
    assert auto_stats(5, True, "Vehicle") == (4, 4)
    assert auto_stats(4, False, "Wall") == (1, 4)
    for cmc, vehicle, subtype in ((3, False, "Human"), (5, True, "Vehicle"), (4, False, "Wall")):
        assert sum(auto_stats(cmc, vehicle, subtype)) == stat_points(cmc, vehicle)


def test_power_level_lean_unchanged():
    assert power_level.creature_stats({"cmc": 4, "rarity": "rare", "subtype": "Goblin"}) == (4, 3)
    assert power_level.creature_stats({"cmc": 3, "subtype": "Wall"}) == (1, 4)
    assert power_level.lean(2, 2, "Goblin Warrior") == (3, 1)
    assert power_level.lean(2, 2, "Human") == (2, 2)


def test_creature_type_must_be_a_creature_type():
    for ok in ("Human Wizard", "Elf", "", "Dragon Spirit"):
        assert commander_params({**BASE, "subtype": ok}, 3)["subtype"] == ok
    assert rejects({**BASE, "subtype": "Equipment"}) == "Equipment isn't a creature type"
    assert rejects({**BASE, "subtype": "Human Vehicle"}) == "Vehicle isn't a creature type"
    for bad in ("Aura", "Instant", "Artifact", "Legendary", "Forest", "Saga"):
        rejects({**BASE, "subtype": bad})
