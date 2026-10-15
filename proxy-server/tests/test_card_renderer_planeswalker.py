"""Planeswalkers render on the planeswalker frame: each loyalty cost in a +/−/0 badge left of
its ability, the starting loyalty in the frame's shield. Fast: imports card_renderer only."""
import base64
import io

import pytest
from PIL import Image, ImageDraw

from card_renderer import MagicCardRenderer

WALKER = {
    "name": "Nissa, Grove Warden", "manaCost": "{2}{G}{G}", "colors": ["G"], "cmc": 4,
    "supertype": "Legendary", "type": "Planeswalker", "subtype": "Nissa", "rarity": "rare",
    "loyalty": "4",
    "description": "+1: Untap up to two target lands.\n−2: Create a 3/3 green Elephant creature token.\n"
                   "−6: Lands you control become 5/5 Elemental creatures with trample.",
}


@pytest.fixture(scope="module")
def renderer():
    return MagicCardRenderer()


def test_parse_loyalty_abilities_normalizes_costs(renderer):
    rows = renderer.parse_loyalty_abilities(
        "Spells cost {1} more.\n+1: Scry 1.\n-2: Draw a card.\n0: Gain 2 life.\n−X: Tap X creatures.\n\n+0: Hm.")
    assert rows == [(None, "Spells cost {1} more."), ("+1", "Scry 1."), ("−2", "Draw a card."),
                    ("0", "Gain 2 life."), ("−X", "Tap X creatures."), ("0", "Hm.")]


@pytest.mark.parametrize("card_type, expected", [
    ("Planeswalker", True), ("Legendary Planeswalker", True), ("Creature", False), ("", False)])
def test_is_planeswalker(card_type, expected):
    assert MagicCardRenderer.is_planeswalker(card_type) is expected


def spy_text(monkeypatch):
    drawn = []
    original = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        drawn.append((xy, text, kwargs.get("anchor")))
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    return drawn


def test_costs_sit_in_badges_left_of_their_text_and_loyalty_in_the_shield(renderer, monkeypatch):
    drawn = spy_text(monkeypatch)
    out = renderer.generate_card_image(dict(WALKER))
    assert out
    card = Image.open(io.BytesIO(base64.b64decode(out)))
    assert card.size == (renderer.card_width, renderer.card_height)

    centred = {text: xy for xy, text, anchor in drawn if anchor == "mm"}
    badge_x = renderer._pw(renderer.PW_BADGE_BODY_X, 0)[0]
    text_x = renderer._pw(renderer.PW_TEXT_X, 0)[0]
    box_top = renderer._pw(0, renderer.PW_TEXT_BOX[1])[1]
    costs = ["+1", "−2", "−6"]
    assert all(c in centred for c in costs)
    for cost in costs:
        x, y = centred[cost]
        assert x == badge_x < text_x
        assert y > box_top
    # One row per ability, top to bottom in rules order
    assert [centred[c][1] for c in costs] == sorted(centred[c][1] for c in costs)
    # The ability text starts right of the badge, never with its cost
    words = [(xy, text) for xy, text, anchor in drawn
             if text.strip() in ("Untap", "Create", "Lands") and xy[1] > box_top]  # not the fitting pass
    assert len(words) == 3 and all(xy[0] >= text_x for xy, _ in words)
    assert not any(text.startswith(("+1:", "−2:")) for _, text, _ in drawn)

    assert centred["4"] == renderer._pw(*renderer.PW_SHIELD_CENTER)
    # The P/T box is never drawn on a planeswalker
    assert not any("/" in text and anchor == "mm" for _, text, anchor in drawn)


def test_uses_the_planeswalker_frame_for_its_colors(renderer, monkeypatch):
    loaded = []
    original = renderer._load_image_cached
    monkeypatch.setattr(renderer, "_load_image_cached", lambda path: loaded.append(path) or original(path))
    for colors, frame in ((["G"], "planeswalkerFrameG.png"), (["W", "U"], "planeswalkerFrameM.png"),
                          ([], "planeswalkerFrameA.png")):
        loaded.clear()
        assert renderer.generate_card_image({**WALKER, "colors": colors})
        assert any(p.endswith(frame) for p in loaded), (colors, loaded)
        assert not any("m15Frame" in p for p in loaded)


def test_without_loyalty_or_art_still_renders(renderer, monkeypatch):
    drawn = spy_text(monkeypatch)
    card = {k: v for k, v in WALKER.items() if k != "loyalty"}
    assert renderer.generate_card_image(card)
    shield = renderer._pw(*renderer.PW_SHIELD_CENTER)
    assert not any(xy == shield for xy, _, _ in drawn)


def test_long_abilities_shrink_to_fit_the_box(renderer):
    long = "Create two 2/2 green Wolf creature tokens with trample, then put a +1/+1 counter on each creature you control."
    rows = renderer.parse_loyalty_abilities("\n".join(f"{c}: {long}" for c in ("+1", "−2", "−4", "−9")))
    box = renderer._pw(0, renderer.PW_TEXT_BOX[3])[1] - renderer._pw(0, renderer.PW_TEXT_BOX[1])[1]
    size, heights = renderer._pw_fit(rows, box, 4)
    assert size < renderer.PW_FONT_SIZES[0]
    assert sum(heights) + 8 * len(rows) <= box
