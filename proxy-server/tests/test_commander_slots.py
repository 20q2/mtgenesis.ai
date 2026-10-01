import pytest

from storage import COMMANDER_SLOT_CMC, StorageError, commander_slot_params

BASE = {"name": "Zur", "manaCost": "{W}{U}", "colors": ["W", "U"], "type": "Instant",
        "subtype": "Human Wizard", "rarity": "mythic", "cmc": 2}


def test_slots_are_3_4_5_mana_legendary_creatures():
    costs = [commander_slot_params(BASE, slot) for slot in (1, 2, 3)]
    assert [p["manaCost"] for p in costs] == ["{1}{W}{U}", "{2}{W}{U}", "{3}{W}{U}"]
    assert [p["cmc"] for p in costs] == [3, 4, 5] == list(COMMANDER_SLOT_CMC.values())
    for p in costs:
        assert p["type"] == "Creature" and p["supertype"] == "Legendary"
        assert p["subtype"] == "Human Wizard" and p["colors"] == ["W", "U"]
        assert p["name"] == "Zur" and p["rarity"] == "mythic"
    assert BASE["manaCost"] == "{W}{U}"  # input untouched


def test_generic_and_x_are_replaced():
    p = commander_slot_params({**BASE, "manaCost": "{X}{4}{B}{B}"}, 3)
    assert p["manaCost"] == "{3}{B}{B}" and p["cmc"] == 5


def test_colorless_commander_is_all_generic():
    assert commander_slot_params({**BASE, "manaCost": ""}, 1)["manaCost"] == "{3}"
    assert commander_slot_params({**BASE, "manaCost": "{C}"}, 2)["manaCost"] == "{3}{C}"


def test_hybrid_phyrexian_and_twobrid_pips():
    assert commander_slot_params({**BASE, "manaCost": "{W/U}{B/P}"}, 1)["manaCost"] == "{1}{W/U}{B/P}"
    # {2/W} is worth 2 mana on its own.
    assert commander_slot_params({**BASE, "manaCost": "{2/W}{G}"}, 1)["manaCost"] == "{2/W}{G}"


def test_pips_filling_the_slot_need_no_generic():
    assert commander_slot_params({**BASE, "manaCost": "{W}{U}{B}"}, 1)["manaCost"] == "{W}{U}{B}"


def test_too_many_pips_is_400():
    with pytest.raises(StorageError) as err:
        commander_slot_params({**BASE, "manaCost": "{W}{W}{U}{U}"}, 1)
    assert err.value.status == 400 and "3" in str(err.value)


def test_typed_stats_are_dropped_so_the_curve_sets_them():
    p = commander_slot_params({**BASE, "power": "3", "toughness": "4"}, 2)
    assert "power" not in p and "toughness" not in p
