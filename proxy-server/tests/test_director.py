"""Card director: briefs that steer rules text and art.

Spec: docs/superpowers/specs/2026-10-02-card-director-design.md.
"""
import json

import pytest

import director
from director import ART_FIELDS, content_words, jaccard, write_briefs

CARD = {"name": "Zur'ka, Élan of Ash", "type": "Creature", "supertype": "Legendary",
        "subtype": "Human Cleric", "colors": ["B"], "manaCost": "{2}{B}", "cmc": 3,
        "rarity": "mythic"}


def brief(mechanic="sacrifice tokens to drain each opponent", subject="a human cleric in ash robes",
          identity="An ash-priest who keeps dead fires burning", **art):
    fields = {"subject": subject, "action": "raising a smoking censer",
              "setting": "a ruined temple", "framing": "low angle, close",
              "light": "embers glowing from below"}
    fields.update(art)
    return {"identity": identity, "mechanic": mechanic, "art": fields}


class StubClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps({"briefs": reply})
        return {"message": {"content": content}}


def run(client, count=1, avoid=None, card=CARD):
    return write_briefs(card, count, avoid, client, "qwen3:8b")


def test_one_brief_parsed_and_trimmed():
    long_identity = " ".join(f"word{i}" for i in range(30))
    long_subject = "a human cleric " + " ".join(f"detail{i}" for i in range(20))
    result = run(StubClient([brief(identity=long_identity, subject=long_subject)]))
    assert len(result) == 1
    assert len(result[0]["identity"].split()) == 20
    assert len(result[0]["art"]["subject"].split()) == 12
    assert set(result[0]["art"]) == set(ART_FIELDS)


def test_three_briefs_with_distinct_mechanics():
    replies = [brief("sacrifice tokens to drain each opponent"),
               brief("return creature cards from your graveyard"),
               brief("attacking makes opponents discard")]
    result = run(StubClient(replies), count=3)
    assert [b["mechanic"] for b in result] == [r["mechanic"] for r in replies]


def test_overlapping_mechanics_retry_once_then_none():
    same = [brief("sacrifice tokens to drain each opponent"),
            brief("sacrifice tokens to drain each opponent quickly"),
            brief("attacking makes opponents discard")]
    client = StubClient(same, same)
    assert run(client, count=3) is None
    assert len(client.calls) == 2


def test_retry_can_succeed():
    bad = [brief("sacrifice tokens to drain"), brief("sacrifice tokens to drain")]
    good = [brief("sacrifice tokens to drain"), brief("return creatures from the graveyard")]
    client = StubClient(bad, good)
    assert [b["mechanic"] for b in run(client, count=2)] == [g["mechanic"] for g in good]


def test_avoid_is_sent_and_enforced():
    avoid = [brief("sacrifice tokens to drain each opponent")]
    client = StubClient([brief("sacrifice tokens to drain each opponent twice")],
                        [brief("discard cards to grow stronger")])
    result = run(client, avoid=avoid)
    assert result[0]["mechanic"] == "discard cards to grow stronger"
    assert "sacrifice tokens to drain each opponent" in client.calls[0]["messages"][-1]["content"]


def test_subject_gets_the_subtype_when_missing():
    result = run(StubClient([brief(subject="a robed priest at an altar")]))
    assert result[0]["art"]["subject"].startswith("a human cleric, a robed priest")


def test_subject_with_a_subtype_word_is_kept():
    result = run(StubClient([brief(subject="a gaunt cleric with ash on his hands")]))
    assert result[0]["art"]["subject"] == "a gaunt cleric with ash on his hands"


def test_art_word_filter():
    result = run(StubClient([brief(subject="a shirtless nude human cleric",
                                   action="baring cleavage and a bare chest")]))
    art = " ".join(result[0]["art"].values()).lower()
    for word in ("shirtless", "nude", "cleavage", "bare chest"):
        assert word not in art
    assert "  " not in art


@pytest.mark.parametrize("reply", ["not json", '{"briefs": []}', '{"briefs": [{"identity": "x"}]}',
                                   '{"briefs": [{"identity": "x", "mechanic": "", "art": {}}]}'])
def test_garbage_returns_none(reply):
    assert run(StubClient(reply, reply)) is None


def test_client_error_returns_none():
    client = StubClient(TimeoutError("ollama timed out"), ConnectionError("down"))
    assert run(client) is None


def test_call_options():
    client = StubClient([brief()])
    run(client)
    call = client.calls[0]
    assert call["model"] == "qwen3:8b" and call["think"] is False
    assert call["format"]["properties"]["briefs"]["minItems"] == 1
    assert call["options"]["num_ctx"] == 2560
    assert "Zur'ka" in call["messages"][-1]["content"]


def test_content_words_and_jaccard():
    a = content_words("Sacrifice tokens to drain each opponent")
    assert "sacrifice" in a and "each" not in a and "to" not in a
    assert jaccard(a, content_words("drain each opponent by sacrificing tokens")) > 0
    assert jaccard(set(), set()) == 0.0


def test_config_switches():
    import config
    assert isinstance(config.DIRECTOR_ENABLED, bool)
    assert config.DIRECTOR_MODEL
    assert director.MECHANIC_MAX_OVERLAP == 0.5


def test_set_overlap():
    same = director.set_overlap(["Flying. Draw a card.", "Flying. Draw a card.", "Trample"])
    different = director.set_overlap(["Flying", "Trample", "Deathtouch"])
    assert 0 < same < 1
    assert same > different == 0.0
    assert director.set_overlap(["Flying"]) == 0.0


def test_logging_never_breaks_a_good_brief(monkeypatch):
    import io
    import sys
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    assert run(StubClient([brief()])) is not None
    assert run(StubClient("not json", "not json")) is None


def test_a_transport_error_is_not_retried():
    client = StubClient(TimeoutError("ollama timed out"), [brief()])
    assert run(client) is None
    assert len(client.calls) == 1


def test_director_has_its_own_short_timeout():
    import config
    assert 0 < config.DIRECTOR_TIMEOUT_SECONDS <= 30


def test_colors_strengths_are_in_the_prompt():
    from rules_text import COLOR_HOOKS, COLORLESS_HOOKS
    client = StubClient([brief()])
    run(client)
    user = client.calls[0]["messages"][-1]["content"]
    assert all(hook in user for hook in COLOR_HOOKS["B"])
    assert "must fit" in user
    client = StubClient([brief(subject="a bronze golem")])
    run(client, card={**CARD, "colors": [], "manaCost": "{3}", "subtype": "Golem"})
    assert all(hook in client.calls[0]["messages"][-1]["content"] for hook in COLORLESS_HOOKS)


def test_mechanic_is_asked_for_as_a_short_theme():
    assert "not rules text" in director.SYSTEM_PROMPT


def test_a_set_shows_the_same_character():
    replies = [brief("sacrifice tokens to drain", subject="a gaunt human cleric in black robes"),
               brief("return creatures from the graveyard", subject="a young human cleric woman"),
               brief("attacking makes opponents discard", subject="a hooded human cleric")]
    result = run(StubClient(replies), count=3)
    assert {b["art"]["subject"] for b in result} == {"a gaunt human cleric in black robes"}


def test_a_reroll_keeps_the_siblings_character():
    avoid = [brief("sacrifice tokens to drain", subject="a gaunt human cleric in black robes")]
    result = run(StubClient([brief("discard cards to grow", subject="a cheerful human cleric")]),
                 avoid=avoid)
    assert result[0]["art"]["subject"] == "a gaunt human cleric in black robes"


def test_set_prompt_asks_for_one_character():
    client = StubClient([brief("a"), brief("b"), brief("c")])
    run(client, count=3)
    assert "same character" in client.calls[0]["messages"][-1]["content"]
