"""card_renderer title fitting (final review M-2): long commander names must not run
into the mana cost. Fast: imports card_renderer only (no app.py, no models)."""
from PIL import ImageDraw

from card_renderer import MagicCardRenderer

LONG_NAME = "Zur'ka, Élan of Ash, the Undying Flame!!"  # 40 = storage.COMMANDER_NAME_MAX
SIX_SYMBOLS = "{4}{B}{B}{R}{R}{G}"


def test_long_name_shrinks_to_fit_left_of_the_mana_cost():
    assert len(LONG_NAME) == 40
    r = MagicCardRenderer()
    size, width, available = r.fit_title_font(LONG_NAME, SIX_SYMBOLS)
    mana_left = r.mana_cost_pos[0] - r.mana_cost_width(SIX_SYMBOLS)
    assert available <= mana_left - r.name_pos[0]
    assert width <= available
    assert r.TITLE_MIN_FONT_SIZE <= size < r.TITLE_FONT_SIZE


def test_short_name_keeps_the_full_title_size():
    r = MagicCardRenderer()
    size, width, available = r.fit_title_font("Zur the Ashen", "{2}{B}{R}")
    assert size == r.TITLE_FONT_SIZE
    assert width <= available


def test_rendered_card_draws_the_name_with_the_fitted_font(monkeypatch):
    r = MagicCardRenderer()
    drawn = []
    original = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        if text == LONG_NAME:
            drawn.append((xy, kwargs.get("font")))
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    out = r.generate_card_image({
        "name": LONG_NAME, "manaCost": SIX_SYMBOLS, "supertype": "Legendary",
        "type": "Creature", "subtype": "Dragon", "colors": ["B", "R", "G"],
        "description": "Flying", "power": "5", "toughness": "5", "rarity": "mythic"})
    assert out
    assert len(drawn) == 1
    (x, _), font = drawn[0]
    size, _, available = r.fit_title_font(LONG_NAME, SIX_SYMBOLS)
    assert font.size == size
    assert x + font.getlength(LONG_NAME) <= r.name_pos[0] + available
