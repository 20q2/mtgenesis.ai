import pytest

from storage import StorageError

PARAMS = {"name": "Zur", "manaCost": "{2}{B}{R}", "colors": ["B", "R"], "type": "Creature",
          "rarity": "mythic", "cmc": 4}


def make_set(storage, user_id, name="Zur the Ashen", status="done"):
    s = storage.create_set(user_id, name, "a fiery lich", PARAMS)
    cards = [storage.create_card(user_id, "a fiery lich", PARAMS, set_id=s["id"], slot=slot)
             for slot in (1, 2, 3)]
    if status is not None:
        for c in cards:
            storage.update_card(c["id"], status=status)
    return s, cards


def vote_rows(storage, set_id):
    return storage._conn().execute(
        "SELECT COUNT(*) FROM votes WHERE set_id = ?", (set_id,)).fetchone()[0]


@pytest.fixture
def user(tmp_storage):
    return tmp_storage.login("Andrew")


@pytest.fixture
def other(tmp_storage):
    return tmp_storage.login("Beth")


def test_create_card_row_shape(tmp_storage, user):
    card = tmp_storage.create_card(user["id"], "a dragon", PARAMS)
    assert set(card) == {"id", "user_id", "set_id", "slot", "replaced", "prompt", "card_params",
                         "card", "art_path", "card_path", "status", "text_ready", "art_ready",
                         "error", "created_at", "finished_at", "shared_at"}
    assert card["status"] == "queued"
    assert card["set_id"] is None and card["slot"] is None
    assert card["replaced"] == 0 and card["text_ready"] == 0 and card["art_ready"] == 0
    assert card["card_params"] == PARAMS and card["card"] is None
    assert tmp_storage.get_card(card["id"]) == card


def test_update_card_fields(tmp_storage, user):
    card = tmp_storage.create_card(user["id"], "a dragon", PARAMS)
    tmp_storage.update_card(card["id"], status="done", text_ready=True, art_ready=True,
                            card={"name": "Final", "flavorText": "Élan"}, art_path="a.png",
                            card_path="c.png", error=None, finished_at="2026-09-28T00:00:00+00:00")
    got = tmp_storage.get_card(card["id"])
    assert got["status"] == "done"
    assert got["text_ready"] == 1 and got["art_ready"] == 1
    assert got["card"] == {"name": "Final", "flavorText": "Élan"}
    assert got["art_path"] == "a.png" and got["card_path"] == "c.png"
    assert got["finished_at"] == "2026-09-28T00:00:00+00:00"
    with pytest.raises(ValueError):
        tmp_storage.update_card(card["id"], replaced=1)


def test_get_missing(tmp_storage):
    assert tmp_storage.get_card("nope") is None
    assert tmp_storage.get_set("nope") is None


def test_create_set_abandons_previous_draft(tmp_storage, user):
    s1 = tmp_storage.create_set(user["id"], "First", "p", PARAMS)
    assert s1["status"] == "draft" and s1["event_id"] is None and s1["card_params"] == PARAMS
    s2 = tmp_storage.create_set(user["id"], "Second", "p", PARAMS)
    assert tmp_storage.get_set(s1["id"])["status"] == "abandoned"
    assert tmp_storage.current_set(user["id"]) == s2


def test_current_set_none_and_locked(tmp_storage, user):
    assert tmp_storage.current_set(user["id"]) is None
    tmp_storage.create_event("Night")
    s, _ = make_set(tmp_storage, user["id"])
    tmp_storage.lock_set(s["id"], user["id"])
    assert tmp_storage.current_set(user["id"])["id"] == s["id"]
    # a new draft takes precedence over the locked set
    d = tmp_storage.create_set(user["id"], "Next", "p", PARAMS)
    assert tmp_storage.current_set(user["id"])["id"] == d["id"]
    assert tmp_storage.get_set(s["id"])["status"] == "locked"


def test_current_set_ignores_locked_in_closed_event(tmp_storage, user):
    event = tmp_storage.create_event("Night")
    s, _ = make_set(tmp_storage, user["id"])
    tmp_storage.lock_set(s["id"], user["id"])
    tmp_storage.close_event(event["id"])
    assert tmp_storage.current_set(user["id"]) is None


def test_commander_name_validation(tmp_storage, user):
    for bad in ["", "   ", "x" * 41, None]:
        with pytest.raises(StorageError) as exc:
            tmp_storage.create_set(user["id"], bad, "p", PARAMS)
        assert exc.value.status == 400
    s = tmp_storage.create_set(user["id"], "Zur'ka, Élan of Ash", "p", PARAMS)
    assert tmp_storage.get_set(s["id"])["commander_name"] == "Zur'ka, Élan of Ash"
    assert tmp_storage.create_set(user["id"], "  " + "x" * 40 + " ", "p",
                                  PARAMS)["commander_name"] == "x" * 40


def test_reroll_replaces_slot(tmp_storage, user):
    s, cards = make_set(tmp_storage, user["id"], status=None)
    tmp_storage.update_card(cards[1]["id"], status="done")
    new = tmp_storage.reroll_card(cards[1]["id"], user["id"])
    old = tmp_storage.get_card(cards[1]["id"])
    assert old["replaced"] == 1
    assert new["id"] != old["id"]
    assert new["slot"] == 2 and new["set_id"] == s["id"]
    assert new["prompt"] == old["prompt"] and new["card_params"] == old["card_params"]
    assert new["status"] == "queued" and new["replaced"] == 0
    current = tmp_storage.set_cards(s["id"])
    assert [c["slot"] for c in current] == [1, 2, 3]
    assert current[1]["id"] == new["id"]
    assert old["id"] in {c["id"] for c in tmp_storage.list_user_cards(user["id"])}


def test_reroll_failed_card_allowed(tmp_storage, user):
    _, cards = make_set(tmp_storage, user["id"], status="failed")
    assert tmp_storage.reroll_card(cards[0]["id"], user["id"])["status"] == "queued"


def test_reroll_rejects_in_progress(tmp_storage, user):
    _, cards = make_set(tmp_storage, user["id"], status=None)
    for card, status in zip(cards, ["queued", "generating", "rendering"]):
        tmp_storage.update_card(card["id"], status=status)
        with pytest.raises(StorageError) as exc:
            tmp_storage.reroll_card(card["id"], user["id"])
        assert exc.value.status == 409
    assert len(tmp_storage.list_user_cards(user["id"])) == 3


def test_reroll_rules(tmp_storage, user, other):
    _, cards = make_set(tmp_storage, user["id"])
    with pytest.raises(StorageError) as exc:
        tmp_storage.reroll_card(cards[0]["id"], other["id"])
    assert exc.value.status == 403

    free = tmp_storage.create_card(user["id"], "p", PARAMS)
    tmp_storage.update_card(free["id"], status="done")
    with pytest.raises(StorageError) as exc:
        tmp_storage.reroll_card(free["id"], user["id"])
    assert exc.value.status == 400

    with pytest.raises(StorageError) as exc:
        tmp_storage.reroll_card("missing", user["id"])
    assert exc.value.status == 404

    tmp_storage.create_event("Night")
    locked, locked_cards = make_set(tmp_storage, user["id"])
    tmp_storage.lock_set(locked["id"], user["id"])
    with pytest.raises(StorageError) as exc:
        tmp_storage.reroll_card(locked_cards[0]["id"], user["id"])
    assert exc.value.status == 409


def test_reroll_twice_on_replaced_card_rejected(tmp_storage, user):
    _, cards = make_set(tmp_storage, user["id"])
    tmp_storage.reroll_card(cards[0]["id"], user["id"])
    with pytest.raises(StorageError) as exc:
        tmp_storage.reroll_card(cards[0]["id"], user["id"])
    assert exc.value.status == 409


def test_lock_preconditions(tmp_storage, user, other):
    s, cards = make_set(tmp_storage, user["id"])

    with pytest.raises(StorageError) as exc:  # no open event
        tmp_storage.lock_set(s["id"], user["id"])
    assert exc.value.status == 409

    event = tmp_storage.create_event("Night")
    tmp_storage.update_card(cards[2]["id"], status="generating")
    with pytest.raises(StorageError) as exc:  # a card not done
        tmp_storage.lock_set(s["id"], user["id"])
    assert exc.value.status == 409
    tmp_storage.update_card(cards[2]["id"], status="done")

    with pytest.raises(StorageError) as exc:  # wrong owner
        tmp_storage.lock_set(s["id"], other["id"])
    assert exc.value.status == 403

    locked = tmp_storage.lock_set(s["id"], user["id"])
    assert locked["status"] == "locked"
    assert locked["event_id"] == event["id"]
    assert locked["locked_at"] is not None

    s2, _ = make_set(tmp_storage, user["id"], name="Another")
    with pytest.raises(StorageError) as exc:  # second locked set in the same event
        tmp_storage.lock_set(s2["id"], user["id"])
    assert exc.value.status == 409

    with pytest.raises(StorageError) as exc:  # already locked
        tmp_storage.lock_set(s["id"], user["id"])
    assert exc.value.status == 409


def test_lock_requires_three_current_cards(tmp_storage, user):
    tmp_storage.create_event("Night")
    s = tmp_storage.create_set(user["id"], "Zur", "p", PARAMS)
    c = tmp_storage.create_card(user["id"], "p", PARAMS, set_id=s["id"], slot=1)
    tmp_storage.update_card(c["id"], status="done")
    with pytest.raises(StorageError) as exc:
        tmp_storage.lock_set(s["id"], user["id"])
    assert exc.value.status == 409


def test_lock_with_commander_name(tmp_storage, user):
    tmp_storage.create_event("Night")
    s, _ = make_set(tmp_storage, user["id"])
    with pytest.raises(StorageError) as exc:
        tmp_storage.lock_set(s["id"], user["id"], "   ")
    assert exc.value.status == 400
    assert tmp_storage.get_set(s["id"])["status"] == "draft"
    locked = tmp_storage.lock_set(s["id"], user["id"], "  Renamed  ")
    assert locked["commander_name"] == "Renamed"


def test_lock_unknown_set(tmp_storage, user):
    with pytest.raises(StorageError) as exc:
        tmp_storage.lock_set("missing", user["id"])
    assert exc.value.status == 404


def test_locked_sets_oldest_first(tmp_storage, user, other):
    event = tmp_storage.create_event("Night")
    s1, _ = make_set(tmp_storage, user["id"])
    s2, _ = make_set(tmp_storage, other["id"])
    tmp_storage.lock_set(s2["id"], other["id"])
    tmp_storage.lock_set(s1["id"], user["id"])
    assert [s["id"] for s in tmp_storage.locked_sets(event["id"])] == [s2["id"], s1["id"]]


def test_unlock_clears_votes_and_returns_to_draft(tmp_storage, user, other):
    event = tmp_storage.create_event("Night")
    s, cards = make_set(tmp_storage, user["id"])
    tmp_storage.lock_set(s["id"], user["id"])
    # seed a vote directly (voting itself is covered in test_storage_votes.py)
    tmp_storage._conn().execute(
        "INSERT INTO votes (voter_id, set_id, card_id, created_at) VALUES (?, ?, ?, 'now')",
        (other["id"], s["id"], cards[0]["id"]))
    assert vote_rows(tmp_storage, s["id"]) == 1

    with pytest.raises(StorageError) as exc:
        tmp_storage.unlock_set(s["id"], other["id"])
    assert exc.value.status == 403

    unlocked = tmp_storage.unlock_set(s["id"], user["id"])
    assert unlocked["status"] == "draft"
    assert unlocked["event_id"] is None
    assert unlocked["locked_at"] is None
    assert vote_rows(tmp_storage, s["id"]) == 0

    with pytest.raises(StorageError) as exc:  # not locked any more
        tmp_storage.unlock_set(s["id"], user["id"])
    assert exc.value.status == 409

    tmp_storage.lock_set(s["id"], user["id"])
    tmp_storage.close_event(event["id"])
    with pytest.raises(StorageError) as exc:
        tmp_storage.unlock_set(s["id"], user["id"])
    assert exc.value.status == 409
    assert tmp_storage.get_set(s["id"])["status"] == "locked"


def test_unlock_abandons_other_draft(tmp_storage, user):
    tmp_storage.create_event("Night")
    s, _ = make_set(tmp_storage, user["id"])
    tmp_storage.lock_set(s["id"], user["id"])
    newer = tmp_storage.create_set(user["id"], "Newer", "p", PARAMS)
    tmp_storage.unlock_set(s["id"], user["id"])
    assert tmp_storage.get_set(newer["id"])["status"] == "abandoned"
    assert tmp_storage.current_set(user["id"])["id"] == s["id"]


def test_list_user_cards_newest_first(tmp_storage, user, other):
    ids = [tmp_storage.create_card(user["id"], f"p{i}", PARAMS)["id"] for i in range(3)]
    tmp_storage.create_card(other["id"], "theirs", PARAMS)
    assert [c["id"] for c in tmp_storage.list_user_cards(user["id"])] == ids[::-1]


def test_count_pending_and_unfinished(tmp_storage, user, other):
    statuses = ["queued", "generating", "rendering", "done", "failed"]
    mine = {}
    for status in statuses:
        c = tmp_storage.create_card(user["id"], "p", PARAMS)
        tmp_storage.update_card(c["id"], status=status)
        mine[status] = c["id"]
    theirs = tmp_storage.create_card(other["id"], "p", PARAMS)
    assert tmp_storage.count_pending(user["id"]) == 3
    assert tmp_storage.count_pending(other["id"]) == 1
    assert tmp_storage.count_pending("nobody") == 0
    assert set(tmp_storage.unfinished_card_ids()) == {
        mine["queued"], mine["generating"], mine["rendering"], theirs["id"]}
