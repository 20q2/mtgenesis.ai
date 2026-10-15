"""Host errata (docs/superpowers/specs/2026-10-14-errata-design.md): the host rewrites a card's
rules text and P/T from /admin and it re-renders, for cards in the commander vote, the
legal commanders of the last closed event, and the Knowledge Pool."""
import base64
import io

import pytest
from flask import Flask
from PIL import Image

from api_routes import create_api_blueprint
from test_api import ADMIN, PIN, FakeQueue, H, generate_set, login

CREATURE = {"name": "Zur the Ashen", "manaCost": "{2}{B}{R}", "colors": ["B", "R"], "type": "Creature",
            "supertype": "Legendary", "subtype": "Zombie Wizard", "rarity": "rare", "cmc": 4,
            "description": "Flying\nWhenever Zur the Ashen attacks, draw a card.", "power": "3",
            "toughness": "4"}
POOL_CARD = {"name": "Ember Pup", "manaCost": "{1}{R}", "colors": ["R"], "type": "Creature",
             "rarity": "common", "cmc": 2, "power": "2", "toughness": "2", "description": "Haste"}


def png_b64(color=(10, 20, 30)) -> str:
    out = io.BytesIO()
    Image.new("RGB", (8, 11), color).save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


class FakeRenderer:
    """Records what it drew; the image's red channel is the rules text length, so tests can tell renders apart."""
    def __init__(self):
        self.calls = []

    def __call__(self, card, art_b64):
        self.calls.append((dict(card), art_b64))
        return png_b64((len(card.get("description", "")) % 256, 0, 0))


@pytest.fixture
def renderer():
    return FakeRenderer()


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(tmp_storage, data_dir, renderer):
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(tmp_storage, FakeQueue(), data_dir, PIN,
                                                render_card=renderer), url_prefix="/api/v1")
    return app.test_client()


def finish(storage, data_dir, card_id, card=CREATURE):
    """A finished card with art and a rendered image on disk."""
    (data_dir / "art").mkdir(parents=True, exist_ok=True)
    (data_dir / "cards").mkdir(parents=True, exist_ok=True)
    art = data_dir / "art" / f"{card_id}.png"
    art.write_bytes(base64.b64decode(png_b64((200, 100, 0))))
    rendered = data_dir / "cards" / f"{card_id}.png"
    rendered.write_bytes(base64.b64decode(png_b64()))
    storage.update_card(card_id, status="done", text_ready=True, art_ready=True, card=dict(card),
                        art_path=str(art), card_path=str(rendered))


def open_event(client):
    res = client.post("/api/v1/admin/events", json={"name": "AI Night"}, headers=ADMIN)
    assert res.status_code == 200, res.get_json()
    return res.get_json()["id"]


def locked_set(client, storage, data_dir, owner, name="Zur the Ashen"):
    body = generate_set(client, owner, name)
    for c in body["cards"]:
        finish(storage, data_dir, c["id"], {**CREATURE, "name": name})
    res = client.post(f"/api/v1/sets/{body['setId']}/lock", headers=H(owner), json={})
    assert res.status_code == 200, res.get_json()
    return body["setId"], [c["id"] for c in body["cards"]]


def errata(client, card_id, headers=ADMIN, **fields):
    return client.post(f"/api/v1/admin/cards/{card_id}/errata", headers=headers, json=fields)


def stored_red(storage, card_id):
    path = storage.get_card(card_id)["card_path"]
    return Image.open(path).convert("RGB").getpixel((0, 0))[0]


# ----- storage -----
def test_errata_keeps_the_first_original_and_revert_restores_it(tmp_storage):
    user = tmp_storage.login("Andrew")
    card_id = tmp_storage.create_card(user["id"], "p", CREATURE)["id"]
    tmp_storage.update_card(card_id, status="done", card=dict(CREATURE))
    assert tmp_storage.get_card(card_id)["original_card"] is None
    assert tmp_storage.get_card(card_id)["errata_at"] is None

    tmp_storage.errata_card(card_id, {**CREATURE, "description": "Flying"})
    tmp_storage.errata_card(card_id, {**CREATURE, "description": "Vigilance"})
    card = tmp_storage.get_card(card_id)
    assert card["card"]["description"] == "Vigilance"
    assert card["original_card"] == CREATURE
    assert card["errata_at"]

    tmp_storage.revert_errata(card_id)
    card = tmp_storage.get_card(card_id)
    assert card["card"] == CREATURE and card["original_card"] is None and card["errata_at"] is None


# ----- API: who and what -----
def test_errata_needs_the_host_pin(client, tmp_storage, data_dir):
    open_event(client)
    owner = login(client, "Andrew")
    _, cards = locked_set(client, tmp_storage, data_dir, owner)
    assert errata(client, cards[0], headers={}, description="x").status_code == 403
    assert errata(client, cards[0], headers={"X-Admin-Pin": "nope"}, description="x").status_code == 403
    assert client.get("/api/v1/admin/errata").status_code == 403


def test_cards_in_the_commander_vote_can_get_errata(client, tmp_storage, data_dir, renderer):
    open_event(client)
    owner = login(client, "Andrew")
    set_id, cards = locked_set(client, tmp_storage, data_dir, owner)
    res = errata(client, cards[1], description="Flying, lifelink\nWhen ~ enters, draw a card.",
                 power="2", toughness="5")
    assert res.status_code == 200, res.get_json()
    item = res.get_json()
    assert item["card"]["errata"] is True
    assert item["card"]["card"]["description"] == "Flying, lifelink\nWhen Zur the Ashen enters, draw a card."
    assert (item["card"]["card"]["power"], item["card"]["card"]["toughness"]) == ("2", "5")
    assert item["original"]["description"] == CREATURE["description"]

    # re-rendered from the stored art, the image file replaced, and its URL changed
    assert renderer.calls[-1][1] == png_b64((200, 100, 0))
    assert stored_red(tmp_storage, cards[1]) == len(item["card"]["card"]["description"])
    url = item["card"]["cardImageUrl"]
    assert url.startswith(f"/api/v1/media/cards/{cards[1]}.png?v=")
    assert client.get(url).status_code == 200

    # players see it in the vote
    voter = login(client, "Beth")
    view = client.get("/api/v1/events/current", headers=H(voter)).get_json()
    shown = next(c for s in view["sets"] if s["id"] == set_id for c in s["cards"] if c["id"] == cards[1])
    assert shown["errata"] is True and shown["cardImageUrl"] == url


def test_name_cost_type_and_rarity_stay_as_printed(client, tmp_storage, data_dir):
    open_event(client)
    owner = login(client, "Andrew")
    _, cards = locked_set(client, tmp_storage, data_dir, owner)
    tmp_storage.update_card(cards[0], card={**CREATURE, "flavorText": "Ash remembers."})
    res = errata(client, cards[0], description="Flying", name="Mega Zur", manaCost="{R}",
                 type="Instant", rarity="mythic", power="3", toughness="4", flavorText="New")
    card = res.get_json()["card"]["card"]
    assert (card["name"], card["manaCost"], card["type"], card["rarity"], card["flavorText"]) == (
        "Zur the Ashen", "{2}{B}{R}", "Creature", "rare", "Ash remembers.")


@pytest.mark.parametrize("fields, message", [
    ({"description": 5}, "Rules text"),
    ({"description": "x" * 1001}, "Rules text"),
    ({"description": "ok", "power": "99999", "toughness": "1"}, "Power"),
    ({"description": "ok", "power": "2", "toughness": "lots"}, "Toughness"),
])
def test_errata_validation(client, tmp_storage, data_dir, fields, message):
    open_event(client)
    owner = login(client, "Andrew")
    _, cards = locked_set(client, tmp_storage, data_dir, owner)
    res = errata(client, cards[0], **fields)
    assert res.status_code == 400 and message in res.get_json()["error"]
    assert tmp_storage.get_card(cards[0])["errata_at"] is None


def test_power_and_toughness_only_for_creatures_and_vehicles(client, tmp_storage, data_dir):
    owner = login(client, "Andrew")
    instant = {"name": "Bolt", "manaCost": "{R}", "colors": ["R"], "type": "Instant", "rarity": "common",
               "cmc": 1, "description": "Bolt deals 3 damage to any target."}
    card_id = tmp_storage.create_card(owner, "p", instant)["id"]
    finish(tmp_storage, data_dir, card_id, instant)
    client.post("/api/v1/admin/pools", headers=ADMIN, json={"name": "Pool", "maxEntriesPerUser": 3})
    client.post("/api/v1/pools/entries", headers=H(owner), json={"cardId": card_id})
    card = errata(client, card_id, description="Deal 2.", power="9", toughness="9").get_json()["card"]["card"]
    assert "power" not in card and "toughness" not in card


def test_only_cards_in_the_vote_a_closed_event_or_the_pool(client, tmp_storage, data_dir):
    owner = login(client, "Andrew")
    loose = tmp_storage.create_card(owner, "p", CREATURE)["id"]
    finish(tmp_storage, data_dir, loose)
    assert errata(client, loose, description="x").status_code == 409
    draft = generate_set(client, owner)["cards"][0]["id"]  # a draft set, not locked in
    finish(tmp_storage, data_dir, draft)
    assert errata(client, draft, description="x").status_code == 409
    assert errata(client, "8f0e2a5c-1d3b-4c7e-9a6f-2b4d8e1c3a5f", description="x").status_code == 404


def test_pool_entries_and_closed_event_winners_can_get_errata(client, tmp_storage, data_dir):
    event_id = open_event(client)
    owner = login(client, "Andrew")
    set_id, cards = locked_set(client, tmp_storage, data_dir, owner)
    client.post("/api/v1/votes", headers=H(owner), json={"setId": set_id, "cardId": cards[2]})
    client.post(f"/api/v1/admin/events/{event_id}/close", headers=ADMIN)
    assert errata(client, cards[2], description="Flying", power="3", toughness="4").status_code == 200

    pup = tmp_storage.create_card(owner, "p", POOL_CARD)["id"]
    finish(tmp_storage, data_dir, pup, POOL_CARD)
    client.post("/api/v1/admin/pools", headers=ADMIN, json={"name": "Pool", "maxEntriesPerUser": 3})
    client.post("/api/v1/pools/entries", headers=H(owner), json={"cardId": pup})
    res = errata(client, pup, description="Haste, menace", power="2", toughness="1")
    assert res.status_code == 200
    pool = client.get("/api/v1/pools/current").get_json()
    assert pool["entries"][0]["card"]["errata"] is True


def test_preview_renders_without_saving(client, tmp_storage, data_dir):
    open_event(client)
    owner = login(client, "Andrew")
    _, cards = locked_set(client, tmp_storage, data_dir, owner)
    before = tmp_storage.get_card(cards[0])
    res = errata(client, cards[0], description="Flying, deathtouch", power="3", toughness="4", preview=True)
    body = res.get_json()
    assert res.status_code == 200
    assert body["image"].startswith("data:image/png;base64,")
    assert body["power"]["verdict"] in ("fair", "pushed", "over")
    assert tmp_storage.get_card(cards[0]) == before
    assert stored_red(tmp_storage, cards[0]) == 10


def test_revert_restores_the_printed_card(client, tmp_storage, data_dir):
    open_event(client)
    owner = login(client, "Andrew")
    _, cards = locked_set(client, tmp_storage, data_dir, owner)
    errata(client, cards[0], description="Defender", power="0", toughness="9")
    errata(client, cards[0], description="Defender, reach", power="0", toughness="9")
    res = client.post(f"/api/v1/admin/cards/{cards[0]}/errata/revert", headers=ADMIN)
    item = res.get_json()
    assert res.status_code == 200
    assert item["card"]["card"]["description"] == CREATURE["description"]
    assert item["card"]["errata"] is False and item["original"] is None
    assert "?v=" not in item["card"]["cardImageUrl"]
    assert stored_red(tmp_storage, cards[0]) == len(CREATURE["description"])
    assert client.post(f"/api/v1/admin/cards/{cards[0]}/errata/revert").status_code == 403


def test_errata_list_groups_the_vote_the_legal_commanders_and_the_pool(client, tmp_storage, data_dir):
    first = open_event(client)
    alice, bob = login(client, "Alice"), login(client, "Bob")
    set_a, cards_a = locked_set(client, tmp_storage, data_dir, alice, "Alpha")
    set_b, cards_b = locked_set(client, tmp_storage, data_dir, bob, "Beta")
    dana = login(client, "Dana")
    client.post("/api/v1/votes", headers=H(alice), json={"setId": set_a, "cardId": cards_a[1]})
    # Beta ties 1-1 between versions 1 and 3 (its owner, Bob, didn't vote)
    client.post("/api/v1/votes", headers=H(alice), json={"setId": set_b, "cardId": cards_b[0]})
    client.post("/api/v1/votes", headers=H(dana), json={"setId": set_b, "cardId": cards_b[2]})
    client.post(f"/api/v1/admin/events/{first}/close", headers=ADMIN)

    # A new night is voting now
    open_event(client)
    carl = login(client, "Carl")
    _, cards_c = locked_set(client, tmp_storage, data_dir, carl, "Gamma")
    pup = tmp_storage.create_card(carl, "p", POOL_CARD)["id"]
    finish(tmp_storage, data_dir, pup, POOL_CARD)
    client.post("/api/v1/admin/pools", headers=ADMIN, json={"name": "Pool", "maxEntriesPerUser": 3})
    client.post("/api/v1/pools/entries", headers=H(carl), json={"cardId": pup})

    groups = client.get("/api/v1/admin/errata", headers=ADMIN).get_json()["groups"]
    assert [g["id"] for g in groups] == ["vote", "legal", "pool"]
    vote, legal, pool = groups
    assert [i["card"]["id"] for i in vote["items"]] == cards_c
    assert vote["items"][0]["label"] == "Gamma" and vote["items"][0]["username"] == "Carl"
    assert vote["items"][0]["detail"] == "Version 1"
    # The last closed event: Alpha's winner, and both of Beta's tied versions
    legal_cards = {i["card"]["id"]: i["detail"] for i in legal["items"]}
    assert legal_cards == {cards_a[1]: "Winner", cards_b[0]: "Tied", cards_b[2]: "Tied"}
    assert legal["items"][0]["label"] == "Alpha"
    assert [i["card"]["id"] for i in pool["items"]] == [pup]


def test_media_route_ignores_the_version_suffix(client, tmp_storage, data_dir):
    open_event(client)
    owner = login(client, "Andrew")
    _, cards = locked_set(client, tmp_storage, data_dir, owner)
    assert client.get(f"/api/v1/media/cards/{cards[0]}.png?v=123").status_code == 200
