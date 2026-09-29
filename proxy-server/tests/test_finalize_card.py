"""finalize_card: the post-processing + render step shared by the legacy route and the AI Night queue.

Imports app.py (torch/diffusers/ollama), so it is marked slow.
"""
import base64

import pytest

pytestmark = pytest.mark.slow

from app import finalize_card  # noqa: E402


def _params(**overrides):
    params = {
        "name": "Zur",
        "manaCost": "{2}{U}",
        "colors": ["U"],
        "type": "Instant",
        "rarity": "rare",
        "cmc": 3,
    }
    params.update(overrides)
    return params


def _is_png_b64(data):
    return isinstance(data, str) and base64.b64decode(data)[:8] == b"\x89PNG\r\n\x1a\n"


def test_finalize_replaces_tilde_and_periods():
    card, rendered = finalize_card(_params(), "When ~ enters, draw a card", None)
    assert card["description"] == "When Zur enters, draw a card."
    assert rendered
    assert _is_png_b64(rendered)


def test_finalize_generates_creature_pt():
    params = _params(type="Creature", subtype="Human Wizard")
    card, _ = finalize_card(params, "Flying", None)
    assert card.get("power")
    assert card.get("toughness")


def test_finalize_none_text_defaults():
    card, _ = finalize_card(_params(), None, None)
    assert card["description"] == "Generated card rules text"


def test_finalize_force_name_overrides_llm_name():
    card, _ = finalize_card(_params(name="Anything"),
                         '{"name": "Other", "description": "Flying"}', None,
                         force_name="Zur")
    assert card["name"] == "Zur"


def test_finalize_force_name_used_for_tilde():
    card, _ = finalize_card(_params(name="Anything"),
                            '{"name": "Other", "description": "When ~ attacks, draw a card"}', None,
                            force_name="Zur")
    assert card["description"] == "When Zur attacks, draw a card."


def test_finalize_does_not_mutate_input():
    params = _params()
    finalize_card(params, '{"name": "Other", "description": "Flying"}', None)
    assert params == _params()
