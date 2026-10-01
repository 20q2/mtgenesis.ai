"""Sharing finished cards to the gallery's Community tab."""
import sqlite3
import uuid

import pytest

from storage import Storage, StorageError
from test_api import CARD_DATA, CARD_VIEW_KEYS, H, client, login, queue  # noqa: F401 (fixtures)

PARAMS = {"name": "Zur", "manaCost": "{2}{B}", "colors": ["B"], "type": "Creature",
          "rarity": "rare", "cmc": 3}


def done_card(storage, user_id, name="Zur"):
    card = storage.create_card(user_id, "a lich", {**PARAMS, "name": name})
    storage.update_card(card["id"], status="done", card={**PARAMS, "name": name})
    return card


@pytest.fixture
def user(tmp_storage):
    return tmp_storage.login("Andrew")


@pytest.fixture
def other(tmp_storage):
    return tmp_storage.login("Beth")


# ----- storage -----

def test_cards_start_private(tmp_storage, user):
    assert done_card(tmp_storage, user["id"])["shared_at"] is None
    assert tmp_storage.list_shared_cards() == []


def test_share_and_unshare(tmp_storage, user):
    card = done_card(tmp_storage, user["id"])
    shared = tmp_storage.set_card_shared(card["id"], user["id"], True)
    assert shared["shared_at"]
    listed = tmp_storage.list_shared_cards()
    assert [c["id"] for c in listed] == [card["id"]]
    assert listed[0]["username"] == "Andrew"

    # sharing again keeps the original time
    assert tmp_storage.set_card_shared(card["id"], user["id"], True)["shared_at"] == shared["shared_at"]

    assert tmp_storage.set_card_shared(card["id"], user["id"], False)["shared_at"] is None
    assert tmp_storage.list_shared_cards() == []


def test_shared_cards_newest_share_first(tmp_storage, user, other):
    a = done_card(tmp_storage, user["id"], "A")
    b = done_card(tmp_storage, other["id"], "B")
    tmp_storage.set_card_shared(b["id"], other["id"], True)
    tmp_storage.set_card_shared(a["id"], user["id"], True)
    listed = tmp_storage.list_shared_cards()
    assert [c["id"] for c in listed] == [a["id"], b["id"]]
    assert [c["username"] for c in listed] == ["Andrew", "Beth"]
    assert [c["id"] for c in tmp_storage.list_shared_cards(limit=1)] == [a["id"]]


def test_share_rules(tmp_storage, user, other):
    card = done_card(tmp_storage, user["id"])
    with pytest.raises(StorageError) as err:
        tmp_storage.set_card_shared(card["id"], other["id"], True)
    assert err.value.status == 403
    with pytest.raises(StorageError) as err:
        tmp_storage.set_card_shared(str(uuid.uuid4()), user["id"], True)
    assert err.value.status == 404
    for status in ("queued", "generating", "rendering", "failed"):
        pending = tmp_storage.create_card(user["id"], "x", PARAMS)
        tmp_storage.update_card(pending["id"], status=status)
        with pytest.raises(StorageError) as err:
            tmp_storage.set_card_shared(pending["id"], user["id"], True)
        assert err.value.status == 409, status


def test_old_database_gains_the_shared_column(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE cards (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, set_id TEXT, "
                 "slot INTEGER, replaced INTEGER NOT NULL DEFAULT 0, prompt TEXT NOT NULL, "
                 "card_params_json TEXT NOT NULL, card_json TEXT, art_path TEXT, card_path TEXT, "
                 "status TEXT NOT NULL, text_ready INTEGER NOT NULL DEFAULT 0, "
                 "art_ready INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL, "
                 "finished_at TEXT)")
    conn.execute("INSERT INTO cards (id, user_id, prompt, card_params_json, status, created_at) "
                 "VALUES ('c1', 'u1', 'p', '{}', 'done', '2026-01-01')")
    conn.commit()
    conn.close()

    storage = Storage(db)
    assert storage.get_card("c1")["shared_at"] is None
    Storage(db)  # reopening doesn't try to add it twice


# ----- API -----

def test_share_endpoint_and_community_list(client, tmp_storage):
    uid = login(client, "Andrew")
    card = done_card(tmp_storage, uid)

    mine = client.get("/api/v1/me/cards", headers=H(uid)).get_json()
    assert mine[0]["shared"] is False and set(mine[0]) == CARD_VIEW_KEYS

    res = client.post(f"/api/v1/cards/{card['id']}/share", headers=H(uid), json={"shared": True})
    assert res.status_code == 200, res.get_json()
    assert res.get_json()["shared"] is True and res.get_json()["id"] == card["id"]

    # anyone can see the Community list, with the maker's name
    listed = client.get("/api/v1/cards/shared").get_json()
    assert [c["id"] for c in listed] == [card["id"]]
    assert listed[0]["username"] == "Andrew" and listed[0]["shared"] is True
    assert set(listed[0]) == CARD_VIEW_KEYS | {"username"}

    res = client.post(f"/api/v1/cards/{card['id']}/share", headers=H(uid), json={"shared": False})
    assert res.get_json()["shared"] is False
    assert client.get("/api/v1/cards/shared").get_json() == []


def test_share_endpoint_errors(client, tmp_storage):
    uid = login(client, "Andrew")
    other = login(client, "Beth")
    card = done_card(tmp_storage, uid)
    url = f"/api/v1/cards/{card['id']}/share"
    assert client.post(url, json={"shared": True}).status_code == 401
    assert client.post(url, headers=H(other), json={"shared": True}).status_code == 403
    for payload in [{}, {"shared": "yes"}, {"shared": 1}]:
        assert client.post(url, headers=H(uid), json=payload).status_code == 400, payload
    queued = client.post("/api/v1/generations", headers=H(uid),
                         json={"prompt": "p", "cardData": CARD_DATA, "count": 1}).get_json()
    res = client.post(f"/api/v1/cards/{queued['cards'][0]['id']}/share", headers=H(uid),
                      json={"shared": True})
    assert res.status_code == 409
