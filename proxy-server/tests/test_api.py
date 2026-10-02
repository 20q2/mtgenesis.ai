import uuid

import pytest
from flask import Flask

from api_routes import create_api_blueprint

PIN = "9999"
CARD_DATA = {"name": "Placeholder", "manaCost": "{2}{B}{R}", "colors": ["B", "R"],
             "type": "Creature", "subtype": "Zombie Wizard", "rarity": "mythic", "cmc": 4,
             "power": "3", "toughness": "4"}
CARD_VIEW_KEYS = {"id", "userId", "setId", "slot", "replaced", "status", "error", "textReady",
                  "artReady", "queuePosition", "etaSeconds", "card", "cardImageUrl",
                  "artImageUrl", "createdAt", "shared", "poolEntryId"}
SET_VIEW_KEYS = {"id", "userId", "username", "eventId", "commanderName", "prompt", "status",
                 "lockedAt", "cards", "myVoteCardId"}
EVENT_SUMMARY_KEYS = {"id", "name", "status", "createdAt", "closedAt"}


class FakeQueue:
    def __init__(self):
        self.enqueued = []
        self.batches = []
        self.pos = (2, 25.0)
        self.position_calls = []
        self.stat = {"busy": True, "cardsAhead": 4, "generatingNow": 1,
                     "avgImageSeconds": 9.5, "etaSeconds": 47.5}

    def enqueue(self, card_id):
        self.enqueued.append(card_id)

    def enqueue_many(self, card_ids):
        self.batches.append(list(card_ids))
        self.enqueued.extend(card_ids)

    def position(self, card_id):
        self.position_calls.append(card_id)
        return self.pos

    def status(self):
        return dict(self.stat)


@pytest.fixture
def queue():
    return FakeQueue()


@pytest.fixture
def client(tmp_storage, queue, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(tmp_storage, queue, tmp_path, PIN),
                           url_prefix="/api/v1")
    return app.test_client()


def login(client, name):
    res = client.post("/api/v1/users/login", json={"username": name})
    assert res.status_code == 200, res.get_json()
    return res.get_json()["id"]


def H(user_id):
    return {"X-User-Id": user_id}


ADMIN = {"X-Admin-Pin": PIN}


def generate_set(client, user_id, name="Zur the Ashen"):
    res = client.post("/api/v1/generations", headers=H(user_id),
                      json={"prompt": "a fiery lich", "cardData": CARD_DATA, "count": 3,
                            "commanderName": name})
    assert res.status_code == 200, res.get_json()
    return res.get_json()


def finish_cards(storage, cards):
    for c in cards:
        storage.update_card(c["id"], status="done", text_ready=True, art_ready=True)


def done_set(client, storage, user_id, name="Zur the Ashen"):
    body = generate_set(client, user_id, name)
    finish_cards(storage, body["cards"])
    return body


def test_login_and_auth(client):
    res = client.post("/api/v1/users/login", json={"username": " Andrew "})
    assert res.status_code == 200
    body = res.get_json()
    assert set(body) == {"id", "username"} and body["username"] == "Andrew"
    assert client.post("/api/v1/users/login", json={"username": "andrew"}).get_json() == body

    bad = client.post("/api/v1/users/login", json={"username": "bad!name"})
    assert bad.status_code == 400 and "error" in bad.get_json()
    assert client.post("/api/v1/users/login", data="nope").status_code == 400

    res = client.get("/api/v1/me/cards")
    assert res.status_code == 401 and "error" in res.get_json()
    assert client.get("/api/v1/me/cards", headers=H(str(uuid.uuid4()))).status_code == 401
    ok = client.get("/api/v1/me/cards", headers=H(body["id"]))
    assert ok.status_code == 200 and ok.get_json() == []


def test_generations_free_play(client, queue):
    uid = login(client, "Andrew")
    res = client.post("/api/v1/generations", headers=H(uid),
                      json={"prompt": "a dragon", "cardData": CARD_DATA, "count": 1})
    assert res.status_code == 200
    body = res.get_json()
    assert body["setId"] is None
    assert len(body["cards"]) == 1
    card = body["cards"][0]
    assert set(card) == CARD_VIEW_KEYS
    assert card["status"] == "queued"
    assert card["userId"] == uid and card["setId"] is None and card["slot"] is None
    assert card["replaced"] is False and card["textReady"] is False and card["artReady"] is False
    assert card["queuePosition"] == 2 and card["etaSeconds"] == 25.0
    assert card["card"] == CARD_DATA  # pending cards show their requested params
    assert card["cardImageUrl"] is None and card["artImageUrl"] is None
    assert card["error"] is None and card["createdAt"]
    assert queue.enqueued == [card["id"]]

    mine = client.get("/api/v1/me/cards", headers=H(uid)).get_json()
    assert [c["id"] for c in mine] == [card["id"]]


def test_generations_validation(client, queue):
    uid = login(client, "Andrew")
    url = "/api/v1/generations"
    assert client.post(url, json={"prompt": "x", "cardData": CARD_DATA,
                                  "count": 1}).status_code == 401
    for payload in [
        {"cardData": CARD_DATA, "count": 1},
        {"prompt": "   ", "cardData": CARD_DATA, "count": 1},
        {"prompt": "x", "count": 1},
        {"prompt": "x", "cardData": "nope", "count": 1},
        {"prompt": "x", "cardData": CARD_DATA, "count": 2},
        {"prompt": "x", "cardData": CARD_DATA, "count": "3"},
        {"prompt": "x", "cardData": CARD_DATA, "count": True},
        {"prompt": "x", "cardData": CARD_DATA},
    ]:
        res = client.post(url, headers=H(uid), json=payload)
        assert res.status_code == 400, payload
        assert "error" in res.get_json()
    assert client.post(url, headers=H(uid), data="not json").status_code == 400
    assert queue.enqueued == []


def test_generations_set(client, queue, tmp_storage):
    uid = login(client, "Andrew")
    name = "Zur'ka, Élan of Ash"
    body = generate_set(client, uid, "  " + name + " ")
    assert body["setId"]
    assert [c["slot"] for c in body["cards"]] == [1, 2, 3]
    assert all(c["setId"] == body["setId"] for c in body["cards"])
    assert all(c["card"]["name"] == name for c in body["cards"])
    assert all(c["card"]["type"] == "Creature" for c in body["cards"])
    assert all(c["card"]["supertype"] == "Legendary" for c in body["cards"])
    assert [c["card"]["manaCost"] for c in body["cards"]] == ["{1}{B}{R}", "{2}{B}{R}", "{3}{B}{R}"]
    assert [c["card"]["cmc"] for c in body["cards"]] == [3, 4, 5]
    assert queue.enqueued == [c["id"] for c in body["cards"]]
    assert queue.batches == [[c["id"] for c in body["cards"]]]  # one batch: one director call
    assert tmp_storage.get_set(body["setId"])["commander_name"] == name

    other = login(client, "Beth")
    for extra in [{}, {"commanderName": ""}, {"commanderName": "   "},
                  {"commanderName": "x" * 41},
                  {"commanderName": "Ok", "cardData": {**CARD_DATA, "manaCost": "{B}{B}{R}{R}"}}]:
        res = client.post("/api/v1/generations", headers=H(other),
                          json={"prompt": "p", "cardData": CARD_DATA, "count": 3, **extra})
        assert res.status_code == 400, extra
    assert tmp_storage.current_set(other) is None


def test_pending_cap(client, queue, tmp_storage):
    uid = login(client, "Andrew")
    generate_set(client, uid)
    res = client.post("/api/v1/generations", headers=H(uid),
                      json={"prompt": "p", "cardData": CARD_DATA, "count": 3,
                            "commanderName": "Again"})
    assert res.status_code == 429 and "error" in res.get_json()
    res = client.post("/api/v1/generations", headers=H(uid),
                      json={"prompt": "p", "cardData": CARD_DATA, "count": 1})
    assert res.status_code == 429
    assert len(queue.enqueued) == 3

    # 2 free-play pending + a set of 3 would exceed the cap as well
    other = login(client, "Beth")
    for _ in range(2):
        assert client.post("/api/v1/generations", headers=H(other),
                           json={"prompt": "p", "cardData": CARD_DATA,
                                 "count": 1}).status_code == 200
    res = client.post("/api/v1/generations", headers=H(other),
                      json={"prompt": "p", "cardData": CARD_DATA, "count": 3,
                            "commanderName": "Zur"})
    assert res.status_code == 429
    assert tmp_storage.current_set(other) is None


def test_card_view_positions(client, queue, tmp_storage):
    uid = login(client, "Andrew")
    card = client.post("/api/v1/generations", headers=H(uid),
                       json={"prompt": "p", "cardData": CARD_DATA, "count": 1}
                       ).get_json()["cards"][0]
    cid = card["id"]
    queue.pos = (0, 3.5)
    view = client.get(f"/api/v1/cards/{cid}", headers=H(uid)).get_json()
    assert view["queuePosition"] == 0 and view["etaSeconds"] == 3.5

    tmp_storage.update_card(cid, status="generating", text_ready=True)
    view = client.get(f"/api/v1/cards/{cid}", headers=H(uid)).get_json()
    assert view["status"] == "generating" and view["textReady"] is True
    assert view["queuePosition"] == 0

    tmp_storage.update_card(cid, status="rendering", art_ready=True, art_path="x")
    view = client.get(f"/api/v1/cards/{cid}", headers=H(uid)).get_json()
    assert view["queuePosition"] is None and view["etaSeconds"] is None
    assert view["artImageUrl"] == f"/api/v1/media/art/{cid}.png"
    assert view["cardImageUrl"] is None

    final = dict(CARD_DATA, name="Final Name", flavorText="Ash remembers.")
    tmp_storage.update_card(cid, status="done", card=final, card_path="y")
    view = client.get(f"/api/v1/cards/{cid}", headers=H(uid)).get_json()
    assert view["status"] == "done"
    assert view["queuePosition"] is None and view["etaSeconds"] is None
    assert view["cardImageUrl"] == f"/api/v1/media/cards/{cid}.png"
    assert view["card"] == final

    failed = client.post("/api/v1/generations", headers=H(uid),
                         json={"prompt": "p", "cardData": CARD_DATA, "count": 1}
                         ).get_json()["cards"][0]
    tmp_storage.update_card(failed["id"], status="failed", error="Ollama is down")
    view = client.get(f"/api/v1/cards/{failed['id']}", headers=H(uid)).get_json()
    assert view["status"] == "failed" and view["error"] == "Ollama is down"
    assert view["queuePosition"] is None and view["etaSeconds"] is None

    assert client.get(f"/api/v1/cards/{uuid.uuid4()}", headers=H(uid)).status_code == 404


def test_reroll_endpoint(client, queue, tmp_storage):
    uid = login(client, "Andrew")
    body = done_set(client, tmp_storage, uid)
    old = body["cards"][1]
    res = client.post(f"/api/v1/cards/{old['id']}/reroll", headers=H(uid))
    assert res.status_code == 200, res.get_json()
    new = res.get_json()
    assert set(new) == CARD_VIEW_KEYS
    assert new["id"] != old["id"] and new["slot"] == 2 and new["status"] == "queued"
    assert new["card"]["manaCost"] == "{2}{B}{R}" and new["card"]["cmc"] == 4
    assert new["setId"] == body["setId"]
    assert queue.enqueued[-1] == new["id"]

    # the replaced card stays in the gallery, flagged
    gallery = {c["id"]: c for c in client.get("/api/v1/me/cards", headers=H(uid)).get_json()}
    assert gallery[old["id"]]["replaced"] is True

    # rerolling the still-queued new card is rejected and nothing new is queued
    count = len(queue.enqueued)
    res = client.post(f"/api/v1/cards/{new['id']}/reroll", headers=H(uid))
    assert res.status_code == 409 and "error" in res.get_json()
    assert len(queue.enqueued) == count

    other = login(client, "Beth")
    res = client.post(f"/api/v1/cards/{body['cards'][0]['id']}/reroll", headers=H(other))
    assert res.status_code == 403
    assert client.post(f"/api/v1/cards/{uuid.uuid4()}/reroll",
                       headers=H(uid)).status_code == 404
    assert client.post(f"/api/v1/cards/{old['id']}/reroll").status_code == 401


def test_reroll_respects_pending_cap(client, queue, tmp_storage):
    uid = login(client, "Andrew")
    body = done_set(client, tmp_storage, uid)
    for _ in range(3):
        client.post("/api/v1/generations", headers=H(uid),
                    json={"prompt": "p", "cardData": CARD_DATA, "count": 1})
    assert tmp_storage.count_pending(uid) == 3
    res = client.post(f"/api/v1/cards/{body['cards'][0]['id']}/reroll", headers=H(uid))
    assert res.status_code == 429


def test_lock_vote_flow(client, queue, tmp_storage):
    res = client.post("/api/v1/admin/events", json={"name": "AI Night 1"},
                      headers={"X-Admin-Pin": "0000"})
    assert res.status_code == 403 and "error" in res.get_json()
    assert client.post("/api/v1/admin/events", json={"name": "x"}).status_code == 403
    res = client.post("/api/v1/admin/events", json={"name": "AI Night 1"}, headers=ADMIN)
    assert res.status_code == 200
    event = res.get_json()
    assert set(event) == EVENT_SUMMARY_KEYS | {"sets"}
    assert event["status"] == "open" and event["sets"] == [] and event["closedAt"] is None
    assert client.post("/api/v1/admin/events", json={"name": "Two"},
                       headers=ADMIN).status_code == 409

    owner = login(client, "Andrew")
    voter = login(client, "Beth")
    body = done_set(client, tmp_storage, owner)
    set_id = body["setId"]

    assert client.post(f"/api/v1/sets/{set_id}/lock", headers=H(voter),
                       json={}).status_code == 403
    res = client.post(f"/api/v1/sets/{set_id}/lock", headers=H(owner),
                      json={"commanderName": "Zur the Ashen"})
    assert res.status_code == 200, res.get_json()
    locked = res.get_json()
    assert set(locked) == SET_VIEW_KEYS
    assert locked["status"] == "locked" and locked["eventId"] == event["id"]
    assert locked["username"] == "Andrew" and locked["lockedAt"]
    assert [c["slot"] for c in locked["cards"]] == [1, 2, 3]
    assert all(set(c) == CARD_VIEW_KEYS | {"votes", "leader", "tied"} for c in locked["cards"])

    target = body["cards"][2]["id"]
    res = client.post("/api/v1/votes", headers=H(voter), json={"setId": set_id, "cardId": target})
    assert res.status_code == 200, res.get_json()
    voted = res.get_json()
    assert voted["myVoteCardId"] == target
    assert [c["votes"] for c in voted["cards"]] == [0, 0, 1]
    assert assert_votes_rejected_without_user(client, set_id, target)
    bad = client.post("/api/v1/votes", headers=H(voter), json={"setId": set_id})
    assert bad.status_code == 400

    current = client.get("/api/v1/events/current", headers=H(voter)).get_json()
    assert current["id"] == event["id"] and current["name"] == "AI Night 1"
    assert len(current["sets"]) == 1
    view = current["sets"][0]
    assert view["myVoteCardId"] == target
    by_id = {c["id"]: c for c in view["cards"]}
    assert by_id[target] == {**by_id[target], "votes": 1, "leader": True, "tied": False}
    others = [c for c in view["cards"] if c["id"] != target]
    assert all(c["votes"] == 0 and not c["leader"] and not c["tied"] for c in others)

    # the owner has not voted; anonymous callers get myVoteCardId null
    owner_view = client.get("/api/v1/events/current", headers=H(owner)).get_json()
    assert owner_view["sets"][0]["myVoteCardId"] is None
    anon = client.get("/api/v1/events/current")
    assert anon.status_code == 200 and anon.get_json()["sets"][0]["myVoteCardId"] is None
    # a stale user id is rejected so the frontend can send the user back to /login
    assert client.get("/api/v1/events/current",
                      headers=H(str(uuid.uuid4()))).status_code == 401

    # the owner self-votes for another card -> tie
    client.post("/api/v1/votes", headers=H(owner),
                json={"setId": set_id, "cardId": body["cards"][0]["id"]})
    tied = client.get("/api/v1/events/current", headers=H(owner)).get_json()["sets"][0]
    assert [c["tied"] for c in tied["cards"]] == [True, False, True]
    assert not any(c["leader"] for c in tied["cards"])


def assert_votes_rejected_without_user(client, set_id, card_id):
    return client.post("/api/v1/votes", json={"setId": set_id,
                                              "cardId": card_id}).status_code == 401


def test_unlock_endpoint(client, queue, tmp_storage):
    client.post("/api/v1/admin/events", json={"name": "Night"}, headers=ADMIN)
    owner = login(client, "Andrew")
    voter = login(client, "Beth")
    body = done_set(client, tmp_storage, owner)
    set_id = body["setId"]
    client.post(f"/api/v1/sets/{set_id}/lock", headers=H(owner), json={})
    client.post("/api/v1/votes", headers=H(voter),
                json={"setId": set_id, "cardId": body["cards"][0]["id"]})
    assert client.post(f"/api/v1/sets/{set_id}/unlock", headers=H(voter)).status_code == 403
    res = client.post(f"/api/v1/sets/{set_id}/unlock", headers=H(owner))
    assert res.status_code == 200
    view = res.get_json()
    assert view["status"] == "draft" and view["eventId"] is None and view["lockedAt"] is None
    assert [c["votes"] for c in view["cards"]] == [0, 0, 0]
    assert client.get("/api/v1/events/current").get_json()["sets"] == []


def test_lock_without_event_is_409(client, tmp_storage):
    owner = login(client, "Andrew")
    body = done_set(client, tmp_storage, owner)
    res = client.post(f"/api/v1/sets/{body['setId']}/lock", headers=H(owner))
    assert res.status_code == 409 and "error" in res.get_json()
    assert client.post(f"/api/v1/sets/{uuid.uuid4()}/lock",
                       headers=H(owner), json={}).status_code == 404


def test_close_event_freezes(client, queue, tmp_storage):
    event = client.post("/api/v1/admin/events", json={"name": "Night"},
                        headers=ADMIN).get_json()
    owner = login(client, "Andrew")
    voter = login(client, "Beth")
    body = done_set(client, tmp_storage, owner)
    set_id = body["setId"]
    client.post(f"/api/v1/sets/{set_id}/lock", headers=H(owner), json={})
    client.post("/api/v1/votes", headers=H(voter),
                json={"setId": set_id, "cardId": body["cards"][1]["id"]})

    assert client.post(f"/api/v1/admin/events/{event['id']}/close").status_code == 403
    res = client.post(f"/api/v1/admin/events/{event['id']}/close", headers=ADMIN)
    assert res.status_code == 200
    closed = res.get_json()
    assert closed["status"] == "closed" and closed["closedAt"]
    winner = [c for c in closed["sets"][0]["cards"] if c["leader"]]
    assert [c["id"] for c in winner] == [body["cards"][1]["id"]]
    assert client.post(f"/api/v1/admin/events/{event['id']}/close",
                       headers=ADMIN).status_code == 409
    assert client.post(f"/api/v1/admin/events/{uuid.uuid4()}/close",
                       headers=ADMIN).status_code == 404

    res = client.post("/api/v1/votes", headers=H(voter),
                      json={"setId": set_id, "cardId": body["cards"][0]["id"]})
    assert res.status_code == 409
    assert res.get_json() == {"error": "Voting is closed"}
    assert client.post(f"/api/v1/sets/{set_id}/unlock", headers=H(owner)).status_code == 409

    history = client.get(f"/api/v1/events/{event['id']}", headers=H(voter)).get_json()
    assert history["status"] == "closed"
    assert [s["id"] for s in history["sets"]] == [set_id]
    assert history["sets"][0]["myVoteCardId"] == body["cards"][1]["id"]
    assert client.get(f"/api/v1/events/{uuid.uuid4()}").status_code == 404

    listed = client.get("/api/v1/events").get_json()
    assert len(listed) == 1
    assert set(listed[0]) == EVENT_SUMMARY_KEYS
    assert listed[0]["id"] == event["id"] and listed[0]["status"] == "closed"
    assert client.get("/api/v1/events/current").get_json() is None


def test_me_sets_current(client, queue, tmp_storage):
    uid = login(client, "Andrew")
    res = client.get("/api/v1/me/sets/current", headers=H(uid))
    assert res.status_code == 200 and res.get_json() is None
    body = generate_set(client, uid)
    view = client.get("/api/v1/me/sets/current", headers=H(uid)).get_json()
    assert set(view) == SET_VIEW_KEYS
    assert view["id"] == body["setId"] and view["status"] == "draft"
    assert view["commanderName"] == "Zur the Ashen" and view["prompt"] == "a fiery lich"
    assert view["userId"] == uid and view["username"] == "Andrew"
    assert view["eventId"] is None and view["lockedAt"] is None and view["myVoteCardId"] is None
    assert [c["slot"] for c in view["cards"]] == [1, 2, 3]
    assert all(c["queuePosition"] == 2 for c in view["cards"])
    assert client.get("/api/v1/me/sets/current").status_code == 401


def test_queue_status(client, queue):
    res = client.get("/api/v1/queue_status")
    assert res.status_code == 200
    assert res.get_json() == queue.stat


def test_media_rejects_bad_id(client, tmp_path):
    assert client.get("/api/v1/media/cards/..%2Fapp.py").status_code == 404
    assert client.get("/api/v1/media/cards/../app.py").status_code == 404
    assert client.get("/api/v1/media/cards/notauuid.png").status_code == 404
    assert client.get("/api/v1/media/cards/%2E%2E%2Ft.db").status_code == 404

    card_id = str(uuid.uuid4())
    (tmp_path / "cards").mkdir()
    (tmp_path / "cards" / f"{card_id}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    res = client.get(f"/api/v1/media/cards/{card_id}.png")
    assert res.status_code == 200
    assert res.mimetype == "image/png"
    assert res.data.startswith(b"\x89PNG")
    res.close()

    assert client.get(f"/api/v1/media/art/{card_id}.png").status_code == 404  # missing file
    assert client.get(f"/api/v1/media/other/{card_id}.png").status_code == 404
    assert client.get(f"/api/v1/media/cards/{card_id}.jpg").status_code == 404
    assert client.get(f"/api/v1/media/cards/{card_id.upper()}.png").status_code == 404
    assert client.get(f"/api/v1/media/cards/{card_id.replace('-', '')}.png").status_code == 404


def test_options_preflight(client):
    res = client.options("/api/v1/generations",
                         headers={"Origin": "http://localhost:4200",
                                  "Access-Control-Request-Method": "POST",
                                  "Access-Control-Request-Headers": "x-user-id"})
    assert res.status_code == 200
    assert client.options("/api/v1/votes").status_code == 200


# ----- final-review fixes: prompt cap (M-3), admin lockout (I-1), conditional GETs (I-2) -----

def test_generations_prompt_cap(client, queue):
    from api_routes import MAX_PROMPT_CHARS
    uid = login(client, "Andrew")
    url = "/api/v1/generations"
    too_long = "x" * (MAX_PROMPT_CHARS + 1)
    res = client.post(url, headers=H(uid),
                      json={"prompt": too_long, "cardData": CARD_DATA, "count": 1})
    assert res.status_code == 400
    assert str(MAX_PROMPT_CHARS) in res.get_json()["error"]
    res = client.post(url, headers=H(uid), json={"prompt": too_long, "cardData": CARD_DATA,
                                                 "count": 3, "commanderName": "Zur"})
    assert res.status_code == 400
    assert queue.enqueued == []
    # Exactly the cap (after trimming) is fine.
    res = client.post(url, headers=H(uid), json={
        "prompt": "  " + "z" * MAX_PROMPT_CHARS + "  ", "cardData": CARD_DATA, "count": 1})
    assert res.status_code == 200


def test_prompt_cap_leaves_room_for_the_forms_longest_art_prompt():
    from api_routes import MAX_PROMPT_CHARS
    # card-form's generateArtPromptText with a 30-char name, 50-char type, all five colours,
    # mythic, big creature and two description keywords is ~450 characters.
    assert MAX_PROMPT_CHARS >= 600


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def clocked_client(tmp_storage, queue, tmp_path, clock):
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(tmp_storage, queue, tmp_path, PIN, clock=clock),
                           url_prefix="/api/v1")
    return app.test_client()


def _admin_create(client, pin, name="Night"):
    return client.post("/api/v1/admin/events", json={"name": name}, headers={"X-Admin-Pin": pin})


def test_admin_lockout_after_five_wrong_pins(clocked_client, clock):
    for _ in range(5):
        assert _admin_create(clocked_client, "0000").status_code == 403
        clock.now += 10
    res = _admin_create(clocked_client, "0000")
    assert res.status_code == 429
    assert res.get_json() == {"error": "Too many wrong PINs - wait a few minutes"}
    # The right PIN is refused too while locked out, on every admin route.
    assert _admin_create(clocked_client, PIN).status_code == 429
    assert clocked_client.post(f"/api/v1/admin/events/{uuid.uuid4()}/close",
                               headers=ADMIN).status_code == 429
    # Non-admin routes are unaffected.
    assert clocked_client.get("/api/v1/events").status_code == 200


def test_admin_lockout_expires(clocked_client, clock):
    for _ in range(5):
        _admin_create(clocked_client, "0000")
    assert _admin_create(clocked_client, PIN).status_code == 429
    clock.now += 299
    assert _admin_create(clocked_client, PIN).status_code == 429
    clock.now += 2
    assert _admin_create(clocked_client, PIN).status_code == 200
    # The counter starts over after the lockout: one wrong PIN is a plain 403.
    assert _admin_create(clocked_client, "0000").status_code == 403


def test_wrong_pins_outside_the_window_do_not_lock_out(clocked_client, clock):
    for _ in range(4):
        assert _admin_create(clocked_client, "0000").status_code == 403
    clock.now += 301  # those four fall out of the 5-minute window
    for _ in range(4):
        assert _admin_create(clocked_client, "0000").status_code == 403
    assert _admin_create(clocked_client, PIN).status_code == 200


def test_admin_pin_warning():
    from api_routes import admin_pin_warning
    assert admin_pin_warning("") and admin_pin_warning("   ")
    assert admin_pin_warning("1234") and admin_pin_warning(None)
    assert admin_pin_warning("k7#Qm2vX9p") is None


@pytest.mark.parametrize("which", ["current", "by-id"])
def test_event_views_support_conditional_get(client, queue, tmp_storage, which):
    event = client.post("/api/v1/admin/events", json={"name": "Night"}, headers=ADMIN).get_json()
    owner, voter = login(client, "Andrew"), login(client, "Beth")
    body = done_set(client, tmp_storage, owner)
    assert client.post(f"/api/v1/sets/{body['setId']}/lock", headers=H(owner),
                       json={"commanderName": "Zur the Ashen"}).status_code == 200
    url = "/api/v1/events/current" if which == "current" else f"/api/v1/events/{event['id']}"

    first = client.get(url, headers=H(voter))
    assert first.status_code == 200
    etag = first.headers.get("ETag")
    assert etag
    assert "no-cache" in first.headers.get("Cache-Control", "")

    again = client.get(url, headers={**H(voter), "If-None-Match": etag})
    assert again.status_code == 304
    assert again.data == b""

    res = client.post("/api/v1/votes", headers=H(voter),
                      json={"setId": body["setId"], "cardId": body["cards"][1]["id"]})
    assert res.status_code == 200
    after = client.get(url, headers={**H(voter), "If-None-Match": etag})
    assert after.status_code == 200
    assert after.headers["ETag"] != etag
    assert after.get_json()["sets"][0]["myVoteCardId"] == body["cards"][1]["id"]
