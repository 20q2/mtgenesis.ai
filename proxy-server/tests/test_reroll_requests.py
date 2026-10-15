"""Reroll requests (docs/superpowers/specs/2026-10-14-reroll-requests-design.md): players ask the
owner to reroll a broken version on the vote page; at 3 requests the owner may reroll it there."""
import pytest

from storage import REROLL_REQUESTS_NEEDED, StorageError
from test_api import ADMIN, H, client, done_set, login, queue  # noqa: F401 (fixtures)
from test_storage_votes import locked_set


@pytest.fixture
def owner(tmp_storage):
    return tmp_storage.login("Owner")["id"]


@pytest.fixture
def players(tmp_storage):
    return [tmp_storage.login(name)["id"] for name in ("Ann", "Ben", "Cat", "Dan")]


def ask(storage, card_id, users):
    for u in users:
        storage.set_reroll_request(card_id, u, True)


def test_threshold_is_three():
    assert REROLL_REQUESTS_NEEDED == 3


def test_request_toggles_and_counts(tmp_storage, owner, players):
    s, (c1, c2, _) = locked_set(tmp_storage, owner)
    assert tmp_storage.set_reroll_request(c1, players[0], True) == s["id"]
    tmp_storage.set_reroll_request(c1, players[0], True)  # asking twice is still one request
    tmp_storage.set_reroll_request(c1, players[1], True)
    tmp_storage.set_reroll_request(c2, players[1], True)
    assert tmp_storage.reroll_request_counts(s["id"]) == {c1: 2, c2: 1}
    assert tmp_storage.my_reroll_requests(players[1], s["id"]) == {c1, c2}

    tmp_storage.set_reroll_request(c1, players[1], False)
    tmp_storage.set_reroll_request(c1, players[1], False)  # taking back twice is fine
    assert tmp_storage.reroll_request_counts(s["id"]) == {c1: 1, c2: 1}
    assert tmp_storage.my_reroll_requests(players[1], s["id"]) == {c2}


def test_request_rules(tmp_storage, owner, players):
    s, (c1, c2, c3) = locked_set(tmp_storage, owner)

    def status_of(card_id, user_id, requested=True):
        with pytest.raises(StorageError) as exc:
            tmp_storage.set_reroll_request(card_id, user_id, requested)
        return exc.value.status

    assert status_of("missing", players[0]) == 404
    assert status_of(c1, owner) == 403  # the owner rerolls; they don't ask themselves

    free = tmp_storage.create_card(players[0], "p", {"name": "x"})
    assert status_of(free["id"], players[1]) == 400

    draft = tmp_storage.create_set(owner, "Draft", "p", {"name": "x"}, cmc=5, rarity="mythic")
    draft_card = tmp_storage.create_card(owner, "p", {"name": "x"}, set_id=draft["id"], slot=1)
    tmp_storage.update_card(draft_card["id"], status="done")
    assert status_of(draft_card["id"], players[0]) == 409

    tmp_storage.update_card(c3, status="failed")
    assert status_of(c3, players[0]) == 409  # nothing finished to judge

    tmp_storage.close_event(tmp_storage.current_event()["id"])
    assert status_of(c1, players[0]) == 409
    assert tmp_storage.reroll_request_counts(s["id"]) == {}


def test_owner_rerolls_at_three_requests(tmp_storage, owner, players):
    s, (c1, c2, _) = locked_set(tmp_storage, owner)
    with pytest.raises(StorageError) as exc:
        tmp_storage.reroll_card(c1, owner)
    assert exc.value.status == 409

    ask(tmp_storage, c1, players[:2])
    assert not tmp_storage.can_reroll(c1, owner)
    with pytest.raises(StorageError):
        tmp_storage.reroll_card(c1, owner)

    ask(tmp_storage, c1, players[2:3])
    assert tmp_storage.can_reroll(c1, owner)
    assert not tmp_storage.can_reroll(c1, players[0])  # only the owner rerolls
    with pytest.raises(StorageError) as exc:
        tmp_storage.reroll_card(c1, players[0])
    assert exc.value.status == 403

    new = tmp_storage.reroll_card(c1, owner)
    assert new["status"] == "queued" and new["slot"] == 1 and new["set_id"] == s["id"]
    assert tmp_storage.get_set(s["id"])["status"] == "locked"
    assert [c["id"] for c in tmp_storage.set_cards(s["id"])][0] == new["id"]
    # the new version starts over
    assert tmp_storage.reroll_request_counts(s["id"]) == {}
    assert not tmp_storage.can_reroll(new["id"], owner)


def test_reroll_clears_only_that_versions_votes(tmp_storage, owner, players):
    s, (c1, c2, _) = locked_set(tmp_storage, owner)
    tmp_storage.cast_vote(players[0], s["id"], c1)
    tmp_storage.cast_vote(players[1], s["id"], c2)
    tmp_storage.cast_vote(owner, s["id"], c1)
    ask(tmp_storage, c1, players[:3])
    ask(tmp_storage, c2, players[3:])
    new = tmp_storage.reroll_card(c1, owner)

    assert tmp_storage.vote_tally(s["id"]) == {c2: 1}
    assert tmp_storage.user_vote(players[0], s["id"]) is None
    assert tmp_storage.owner_vote(s["id"]) is None
    assert tmp_storage.user_vote(players[1], s["id"]) == c2
    assert tmp_storage.reroll_request_counts(s["id"]) == {c2: 1}

    # the rerolled version can't be voted on until it's finished
    with pytest.raises(StorageError) as exc:
        tmp_storage.cast_vote(players[0], s["id"], new["id"])
    assert exc.value.status == 409
    tmp_storage.update_card(new["id"], status="done")
    tmp_storage.cast_vote(players[0], s["id"], new["id"])
    assert tmp_storage.vote_tally(s["id"]) == {c2: 1, new["id"]: 1}


def test_failed_version_can_be_rerolled_without_requests(tmp_storage, owner, players):
    s, (c1, _, _) = locked_set(tmp_storage, owner)
    ask(tmp_storage, c1, players[:3])
    new = tmp_storage.reroll_card(c1, owner)
    for status in ("queued", "generating", "rendering"):
        tmp_storage.update_card(new["id"], status=status)
        assert not tmp_storage.can_reroll(new["id"], owner)
        with pytest.raises(StorageError):
            tmp_storage.reroll_card(new["id"], owner)
    tmp_storage.update_card(new["id"], status="failed", error="Server restarted - please reroll")
    assert tmp_storage.can_reroll(new["id"], owner)
    assert tmp_storage.reroll_card(new["id"], owner)["status"] == "queued"


def test_no_reroll_after_voting_closes(tmp_storage, owner, players):
    s, (c1, _, _) = locked_set(tmp_storage, owner)
    ask(tmp_storage, c1, players[:3])
    tmp_storage.close_event(tmp_storage.current_event()["id"])
    assert not tmp_storage.can_reroll(c1, owner)
    with pytest.raises(StorageError) as exc:
        tmp_storage.reroll_card(c1, owner)
    assert exc.value.status == 409


def test_draft_reroll_unchanged(tmp_storage, owner):
    s = tmp_storage.create_set(owner, "Zur", "p", {"name": "x"}, cmc=3, rarity="rare")
    card = tmp_storage.create_card(owner, "p", {"name": "x"}, set_id=s["id"], slot=1)
    tmp_storage.update_card(card["id"], status="done")
    assert tmp_storage.can_reroll(card["id"], owner)
    assert tmp_storage.reroll_card(card["id"], owner)["status"] == "queued"


# ----- API -----
def locked_api_set(client, storage, owner_id):
    client.post("/api/v1/admin/events", json={"name": "AI Night"}, headers=ADMIN)
    body = done_set(client, storage, owner_id)
    res = client.post(f"/api/v1/sets/{body['setId']}/lock", headers=H(owner_id), json={})
    assert res.status_code == 200, res.get_json()
    return body["setId"], [c["id"] for c in body["cards"]]


def request(client, card_id, user_id, requested=True):
    return client.post(f"/api/v1/cards/{card_id}/reroll-request", headers=H(user_id),
                       json={"requested": requested})


def test_reroll_request_endpoint(client, queue, tmp_storage):
    owner_id = login(client, "Owner")
    ann, ben, cat = (login(client, n) for n in ("Ann", "Ben", "Cat"))
    set_id, (c1, c2, _) = locked_api_set(client, tmp_storage, owner_id)

    res = request(client, c1, ann)
    assert res.status_code == 200, res.get_json()
    view = res.get_json()
    assert view["id"] == set_id and view["rerollRequestsNeeded"] == 3
    first = view["cards"][0]
    assert first["rerollRequests"] == 1 and first["myRerollRequest"] is True
    assert first["canReroll"] is False
    assert view["cards"][1]["rerollRequests"] == 0 and view["cards"][1]["myRerollRequest"] is False

    request(client, c1, ben)
    request(client, c1, cat)
    event = client.get("/api/v1/events/current", headers=H(owner_id)).get_json()
    mine = event["sets"][0]["cards"]
    assert [c["canReroll"] for c in mine] == [True, False, False]
    assert [c["rerollRequests"] for c in mine] == [3, 0, 0]
    assert mine[0]["myRerollRequest"] is False
    as_ann = client.get("/api/v1/events/current", headers=H(ann)).get_json()["sets"][0]["cards"]
    assert as_ann[0]["canReroll"] is False and as_ann[0]["myRerollRequest"] is True

    # taking a request back
    view = request(client, c1, cat, False).get_json()
    assert view["cards"][0]["rerollRequests"] == 2

    assert request(client, c1, owner_id).status_code == 403
    assert client.post(f"/api/v1/cards/{c1}/reroll-request", headers=H(ann), json={}).status_code == 400
    assert client.post(f"/api/v1/cards/{c1}/reroll-request", json={"requested": True}).status_code == 401


def test_owner_rerolls_from_the_vote(client, queue, tmp_storage):
    owner_id = login(client, "Owner")
    voters = [login(client, n) for n in ("Ann", "Ben", "Cat")]
    set_id, (c1, c2, _) = locked_api_set(client, tmp_storage, owner_id)
    client.post("/api/v1/votes", headers=H(voters[0]), json={"setId": set_id, "cardId": c1})
    client.post("/api/v1/votes", headers=H(voters[1]), json={"setId": set_id, "cardId": c2})

    assert client.post(f"/api/v1/cards/{c1}/reroll", headers=H(owner_id)).status_code == 409
    for v in voters:
        request(client, c1, v)
    res = client.post(f"/api/v1/cards/{c1}/reroll", headers=H(owner_id))
    assert res.status_code == 200, res.get_json()
    new = res.get_json()
    assert new["status"] == "queued" and new["slot"] == 1 and queue.enqueued[-1] == new["id"]

    event = client.get("/api/v1/events/current", headers=H(voters[0])).get_json()
    s = event["sets"][0]
    assert s["status"] == "locked" and s["myVoteCardId"] is None
    assert [c["id"] for c in s["cards"]] == [new["id"], c2, s["cards"][2]["id"]]
    assert [c["votes"] for c in s["cards"]] == [0, 1, 0]
    assert s["cards"][0]["rerollRequests"] == 0

    res = client.post("/api/v1/votes", headers=H(voters[0]), json={"setId": set_id, "cardId": new["id"]})
    assert res.status_code == 409 and "ready" in res.get_json()["error"]
    assert request(client, new["id"], voters[0]).status_code == 409
