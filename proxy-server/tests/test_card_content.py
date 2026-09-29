"""createCardContent: rules-text generation must not crash on missing/empty/None subtypes.

Imports app.py (torch/diffusers/ollama), so it is marked slow. Ollama is monkeypatched.
"""
import pytest

pytestmark = pytest.mark.slow

import app  # noqa: E402

_MISSING = object()


@pytest.fixture
def fake_ollama(monkeypatch):
    # Unquoted on purpose: a multi-quoted response ('"A." "B."') is currently mangled to ''
    # by createCardContent's outer-quote strip + reorder_abilities_properly (separate issue).
    def fake_generate(*args, **kwargs):
        return {"response": "Trample. When ~ enters, deal 2 damage to any target."}

    monkeypatch.setattr(app.ollama, "generate", fake_generate)


def _card(supertype, subtype):
    card = {
        "name": "Zur'ka, Élan of Ash",
        "manaCost": "{3}{R}{G}",
        "type": "Creature",
        "colors": ["R", "G"],
        "cmc": 5,
        "rarity": "common",
    }
    if supertype is not None:
        card["supertype"] = supertype
    if subtype is not _MISSING:
        card["subtype"] = subtype
    return card


@pytest.mark.parametrize("supertype", ["Legendary", None])
@pytest.mark.parametrize("subtype", ["", _MISSING, None, "Dragon"], ids=["empty", "missing", "none", "dragon"])
def test_creature_content_generated_for_any_subtype(fake_ollama, supertype, subtype):
    text = app.createCardContent("A fiery legend", _card(supertype, subtype))
    assert isinstance(text, str)
    assert text.strip()


def test_content_generated_without_card_data(fake_ollama):
    text = app.createCardContent("A fiery legend", None)
    assert isinstance(text, str)
    assert text.strip()
