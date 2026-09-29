"""createCardContent: rules-text generation must not crash on missing/empty/None subtypes,
keeps every ability the model returns, and fails the card when Ollama fails.

Imports app.py (torch/diffusers/ollama), so it is marked slow. Only the Ollama boundary
(app.ollama_client.chat) is monkeypatched.
"""
import json

import pytest

pytestmark = pytest.mark.slow

import app  # noqa: E402

_MISSING = object()


def _reply(*abilities):
    return {"message": {"content": json.dumps({"abilities": list(abilities)})}}


@pytest.fixture
def fake_ollama(monkeypatch):
    monkeypatch.setattr(app.ollama_client, "chat",
                        lambda *a, **k: _reply("Trample", "When ~ enters, deal 2 damage to any target."))


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


def test_every_ability_is_kept_one_per_line(monkeypatch):
    monkeypatch.setattr(app.ollama_client, "chat",
                        lambda *a, **k: _reply("Flying", "When ~ enters, draw a card."))
    text = app.createCardContent("A storm dragon", _card("Legendary", "Dragon"))
    assert text.split("\n") == ["Flying", "When Zur'ka enters, draw a card."]


def test_plain_text_reply_still_parsed(monkeypatch):
    # A model that ignores the JSON format: one quoted ability per line still works
    monkeypatch.setattr(app.ollama_client, "chat",
                        lambda *a, **k: {"message": {"content": '"Flying."\n"When ~ enters, draw a card."'}})
    text = app.createCardContent("A storm dragon", _card("Legendary", "Dragon"))
    assert "Flying" in text
    assert "draw a card" in text
    assert '"' not in text


def test_ollama_client_has_a_timeout():
    # A stalled Ollama must fail the card instead of blocking the text worker forever.
    assert app.OLLAMA_TIMEOUT_SECONDS == 120
    assert app.ollama_client._client.timeout.read == app.OLLAMA_TIMEOUT_SECONDS


def test_chat_keeps_the_model_resident_and_small(monkeypatch):
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append(kwargs)
        return _reply("Flying")

    monkeypatch.setattr(app.ollama_client, "chat", fake_chat)
    app.createCardContent("A storm dragon", _card("Legendary", "Dragon"))
    assert calls
    for c in calls:
        assert c.get("keep_alive") == "30m"
        assert c.get("model") == app.TEXT_MODEL
        assert c.get("think") is False
        # a small context keeps the LLM beside SDXL on a 12 GB GPU
        assert c["options"]["num_ctx"] <= 2560


def test_ollama_timeout_fails_the_card(monkeypatch):
    import httpx

    def stalled(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(app.ollama_client, "chat", stalled)
    assert not app.createCardContent("A storm dragon", _card("Legendary", "Dragon"))
