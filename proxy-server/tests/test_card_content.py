"""createCardContent: rules-text generation must not crash on missing/empty/None subtypes.

Imports app.py (torch/diffusers/ollama), so it is marked slow. Only the Ollama boundary
(app.ollama_client.generate) is monkeypatched.
"""
import pytest

pytestmark = pytest.mark.slow

import app  # noqa: E402

_MISSING = object()


@pytest.fixture
def fake_ollama(monkeypatch):
    def fake_generate(*args, **kwargs):
        return {"response": "Trample. When ~ enters, deal 2 damage to any target."}

    monkeypatch.setattr(app.ollama_client, "generate", fake_generate)


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


# Mistral is asked to wrap each ability in its own quoted section. Only a reply that is a
# single quoted section may lose its outer quotes; multi-quoted replies keep every ability.
@pytest.mark.parametrize("reply", [
    '"Flying." "When ~ enters, draw a card."',
    '"Flying."\n"When ~ enters, draw a card."',
    '"Flying"\n\n"When ~ enters, draw a card."',
], ids=["same-line", "newline", "blank-line"])
def test_multi_quoted_reply_keeps_every_ability(monkeypatch, reply):
    monkeypatch.setattr(app.ollama_client, "generate", lambda *a, **k: {"response": reply})
    text = app.createCardContent("A storm dragon", _card("Legendary", "Dragon"))
    assert "Flying" in text
    assert "draw a card" in text


def test_single_quoted_reply_loses_outer_quotes(monkeypatch):
    monkeypatch.setattr(app.ollama_client, "generate",
                        lambda *a, **k: {"response": '"When ~ enters, draw a card."'})
    text = app.createCardContent("A storm dragon", _card("Legendary", "Dragon"))
    assert "draw a card" in text
    assert not text.startswith('"')


def test_ollama_client_has_a_timeout():
    # A stalled Ollama must fail the card instead of blocking the text worker forever.
    assert app.OLLAMA_TIMEOUT_SECONDS == 120
    assert app.ollama_client._client.timeout.read == app.OLLAMA_TIMEOUT_SECONDS


def test_generate_keeps_mistral_resident(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        return {"response": "Flying."}

    monkeypatch.setattr(app.ollama_client, "generate", fake_generate)
    app.createCardContent("A storm dragon", _card("Legendary", "Dragon"))
    assert calls and all(c.get("keep_alive") == "30m" for c in calls)


def test_ollama_timeout_fails_the_card(monkeypatch):
    import httpx

    def stalled(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(app.ollama_client, "generate", stalled)
    assert not app.createCardContent("A storm dragon", _card("Legendary", "Dragon"))
