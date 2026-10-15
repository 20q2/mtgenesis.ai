"""Token Maker: params, flavor text, the token renderer and the /tokens routes (tokens.py)."""
import base64
import io
import json
import random

import pytest
from flask import Flask
from PIL import Image

import token_renderer
import tokens
from api_routes import create_api_blueprint
from generation_queue import GenerationQueue
from tests.test_api import PIN, FakeQueue, H, login

GOBLIN = {"frame": "classic", "name": "", "type": "Creature", "subtype": "Goblin",
          "colors": ["R"], "power": "1", "toughness": "1", "description": "Haste",
          "flavorText": "", "artPrompt": "a goblin counting coins"}


class FakeClient:
    """An Ollama client that answers with the given replies in turn."""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": json.dumps(reply)}}


def decode(b64: str) -> Image.Image:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ----- params -----

def test_token_params_keeps_the_form_and_marks_the_kind():
    params = tokens.token_params({**GOBLIN, "colors": ["G", "R"], "power": "x"})
    assert params["kind"] == "token"
    assert params["colors"] == ["R", "G"]          # WUBRG order
    assert params["power"] == "X" and params["toughness"] == "1"
    assert params["manaCost"] == "" and params["rarity"] == "common"


@pytest.mark.parametrize("change, message", [
    ({"frame": "nope"}, "Unknown frame"),
    ({"type": "Instant"}, "type must be"),
    ({"colors": ["P"]}, "colors"),
    ({"power": ""}, "go together"),
    ({"power": "big"}, "go together"),
    ({"name": "x" * 31}, "30 characters"),
])
def test_token_params_rejects_bad_input(change, message):
    with pytest.raises(tokens.TokenError, match=message):
        tokens.token_params({**GOBLIN, **change})


def test_noncreature_tokens_have_no_pt():
    params = tokens.token_params({**GOBLIN, "type": "Artifact", "subtype": "Treasure", "power": ""})
    assert "power" not in params and "toughness" not in params


def test_preview_params_tolerates_an_unfinished_form():
    params = tokens.preview_params({"frame": "bogus", "type": "Creature", "power": "?"})
    assert params["frame"] == token_renderer.DEFAULT_FRAME
    assert params["power"] == ""


def test_art_is_shaped_like_the_frame():
    tall, _ = tokens.art_sizes({"frame": "classic"})
    wide, stored = tokens.art_sizes({"frame": "oldschool"})
    assert tall[0] < tall[1] and wide[0] > wide[1]
    assert stored[0] > stored[1]


# ----- flavor -----

def test_flavor_gets_a_speaker_line_and_a_name_when_blank():
    client = FakeClient({"name": "Coin Goblin", "quote": "I only ate the shiny ones.", "speaker": "Grib"})
    reply = json.loads(tokens.generate_flavor(tokens.token_params(GOBLIN), client, "m",
                                              rng=random.Random(1)))
    assert reply == {"name": "Coin Goblin", "flavorText": "“I only ate the shiny ones.”\n—Grib"}
    assert "Also give it a name" in client.calls[0]["messages"][1]["content"]


def test_flavor_is_skipped_when_the_player_wrote_it():
    client = FakeClient()
    params = tokens.token_params({**GOBLIN, "name": "Grib", "flavorText": "Mine."})
    assert tokens.generate_flavor(params, client, "m") == "{}"
    assert client.calls == []


def test_a_failed_flavor_call_never_fails_the_token():
    client = FakeClient(RuntimeError("ollama down"), RuntimeError("still down"))
    assert tokens.generate_flavor(tokens.token_params(GOBLIN), client, "m") == "{}"


def test_rules_text_is_not_flavor():
    assert tokens.clean_flavor("Add {R} to your mana pool.", "") == ""
    assert tokens.clean_flavor('"Ha!"', "—Krenko") == "“Ha!”\n—Krenko"


def test_finalize_names_a_token_after_its_type_when_the_ai_did_not():
    card, rendered = tokens.finalize_token(tokens.token_params(GOBLIN), "{}", None,
                                           render=lambda c, a: "PNG")
    assert card["name"] == "Goblin" and rendered == "PNG"


# ----- renderer -----

@pytest.mark.parametrize("frame", [f.id for f in token_renderer.FRAMES])
def test_every_frame_renders(frame):
    card = {**tokens.token_params({**GOBLIN, "frame": frame}), "name": "Grib",
            "flavorText": "“I only ate the shiny ones.”\n—Grib"}
    image = token_renderer.render_token_image(card)
    assert image.size == (496, 693) and image.mode == "RGBA"


@pytest.mark.parametrize("colors, card_type, code", [
    ([], "Artifact", "A"), ([], "Land", "L"), (["U"], "Creature", "U"), (["W", "B"], "Creature", "M")])
def test_frame_code(colors, card_type, code):
    assert token_renderer.frame_code(colors, card_type) == code


def test_every_frame_asset_exists():
    import os
    for frame in token_renderer.FRAMES:
        for code in "WUBRGMAL":
            path = os.path.join(token_renderer.card_renderer.assets_dir, frame.asset(code))
            assert os.path.exists(path), path


def test_type_line():
    assert token_renderer.type_line({"type": "Artifact", "subtype": "Treasure"}) == "Token Artifact — Treasure"
    assert token_renderer.type_line({"type": "Enchantment"}) == "Token Enchantment"


# ----- queue -----

def test_tokens_skip_the_director(tmp_storage, tmp_path):
    briefs = []
    q = GenerationQueue(tmp_storage, tmp_path, lambda p, c: "{}", lambda p, c: "", lambda *a: ({}, ""),
                        start_workers=False, brief_fn=lambda *a, **k: briefs.append(a) or [{}])
    user_id = tmp_storage.login("alice")["id"]
    card = tmp_storage.create_card(user_id, "goblin", tokens.token_params(GOBLIN))
    q.enqueue(card["id"])
    assert q.process_next_brief()
    assert briefs == []
    assert tmp_storage.get_card(card["id"])["brief"] is None


# ----- routes -----

@pytest.fixture
def queue():
    return FakeQueue()


@pytest.fixture
def client(tmp_storage, queue, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(tmp_storage, queue, tmp_path, PIN), url_prefix="/api/v1")
    return app.test_client()


def test_frames_route_lists_every_frame(client):
    data = client.get("/api/v1/tokens/frames").get_json()
    assert [f["id"] for f in data["frames"]] == [f.id for f in token_renderer.FRAMES]
    assert "Creature" in data["types"]


def test_frame_thumbnail_is_a_png(client):
    res = client.get("/api/v1/tokens/frames/oldschool/G.png")
    assert res.status_code == 200 and res.mimetype == "image/png"
    assert decode(base64.b64encode(res.data).decode()).width == 248
    assert client.get("/api/v1/tokens/frames/nope/G.png").status_code == 404


def test_preview_needs_a_login_and_draws_the_token(client):
    assert client.post("/api/v1/tokens/preview", json={"token": GOBLIN}).status_code == 401
    uid = login(client, "alice")
    res = client.post("/api/v1/tokens/preview", headers=H(uid), json={"token": {"frame": "map"}})
    assert res.status_code == 200
    assert decode(res.get_json()["image"]).size == (496, 693)


def test_create_token_queues_a_card(client, queue, tmp_storage):
    uid = login(client, "alice")
    res = client.post("/api/v1/tokens", headers=H(uid), json={"token": GOBLIN})
    assert res.status_code == 200, res.get_json()
    view = res.get_json()
    assert queue.enqueued == [view["id"]]
    assert view["card"]["kind"] == "token" and view["status"] == "queued"
    assert tmp_storage.get_card(view["id"])["prompt"] == "a goblin counting coins"


def test_create_token_reports_bad_input(client):
    uid = login(client, "alice")
    res = client.post("/api/v1/tokens", headers=H(uid), json={"token": {**GOBLIN, "type": "Instant"}})
    assert res.status_code == 400 and "type must be" in res.get_json()["error"]


def test_tokens_count_toward_the_pending_cap(client):
    uid = login(client, "alice")
    for _ in range(3):
        assert client.post("/api/v1/tokens", headers=H(uid), json={"token": GOBLIN}).status_code == 200
    assert client.post("/api/v1/tokens", headers=H(uid), json={"token": GOBLIN}).status_code == 429


def test_tokens_cannot_enter_the_pool():
    from storage import pool_card_eligible
    assert not pool_card_eligible(tokens.token_params(GOBLIN))


def test_an_unprintable_log_line_never_fails_the_flavor(monkeypatch):
    """The server console can be cp1252: printing an emoji raises UnicodeEncodeError."""
    def broken_print(*args, **kwargs):
        raise UnicodeEncodeError("charmap", "🃏", 0, 1, "character maps to <undefined>")
    monkeypatch.setattr("builtins.print", broken_print)
    client = FakeClient({"name": "Grib", "quote": "Mine now.", "speaker": ""})
    assert json.loads(tokens.generate_flavor(tokens.token_params(GOBLIN), client, "m")) == \
        {"name": "Grib", "flavorText": "Mine now."}


def test_flavor_cleanup_drops_copies_markdown_and_invented_types():
    assert tokens.clean_flavor(tokens.TONE_EXAMPLES[0], "") == ""
    assert tokens.clean_flavor("I was *made* this way.", "") == "I was made this way."
    assert tokens.clean_flavor("The form said 'list all.'", "") == "The form said 'list all.'"
    assert tokens.clean_name("Goblin Gold Hoarder", "Treasure") == ""
    assert tokens.clean_name("Zombie Intern", "Zombie") == "Zombie Intern"


# ----- leaving fields blank -----

def test_every_field_can_be_left_blank():
    params = tokens.token_params({"frame": "classic", "type": "Creature", "aiName": False, "aiFlavor": False})
    assert params["name"] == "" and params["subtype"] == "" and "power" not in params
    assert tokens.art_prompt(params) == "mysterious creature"
    assert tokens.art_prompt({**params, "type": "Artifact"}) == "mysterious artifact"


def test_the_ai_fills_blanks_by_default():
    params = tokens.token_params(GOBLIN)
    assert params["aiName"] is True and params["aiFlavor"] is True


def test_blank_fields_the_ai_should_not_fill_skip_the_model():
    client = FakeClient()
    params = tokens.token_params({**GOBLIN, "aiName": False, "aiFlavor": False})
    assert tokens.generate_flavor(params, client, "m") == "{}"
    assert client.calls == []


def test_only_the_wanted_blank_is_filled():
    client = FakeClient({"name": "Coin Goblin", "quote": "Mine.", "speaker": ""})
    params = tokens.token_params({**GOBLIN, "aiName": False})
    assert json.loads(tokens.generate_flavor(params, client, "m")) == {"flavorText": "Mine."}
    assert "Leave name empty" in client.calls[0]["messages"][1]["content"]


def test_a_name_left_blank_stays_blank():
    params = tokens.token_params({**GOBLIN, "aiName": False, "aiFlavor": False})
    card, _ = tokens.finalize_token(params, "{}", None, render=lambda c, a: "PNG")
    assert card["name"] == "" and card["flavorText"] == ""
    assert tokens.preview_params({**GOBLIN, "aiName": False})["name"] == ""
    assert tokens.preview_params(GOBLIN)["name"] == "Goblin"
    token_renderer.render_token_image(card)  # an empty title bar draws fine


def test_a_creature_without_pt_prints_no_pt_box():
    params = tokens.token_params({**GOBLIN, "power": "", "toughness": ""})
    assert "power" not in params
    token_renderer.render_token_image(params)
