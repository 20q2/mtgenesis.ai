import threading

import pytest

from storage import Storage, StorageError, leader_flags

PARAMS = {"name": "Zur", "manaCost": "{2}{B}", "colors": ["B"], "type": "Creature",
          "rarity": "rare", "cmc": 3}


def locked_set(storage, user_id):
    s = storage.create_set(user_id, "Zur", "p", PARAMS)
    cards = [storage.create_card(user_id, "p", PARAMS, set_id=s["id"], slot=slot)
             for slot in (1, 2, 3)]
    for c in cards:
        storage.update_card(c["id"], status="done")
    if storage.current_event() is None:
        storage.create_event("Night")
    storage.lock_set(s["id"], user_id)
    return s, [c["id"] for c in cards]


@pytest.fixture
def owner(tmp_storage):
    return tmp_storage.login("Owner")["id"]


@pytest.fixture
def voter(tmp_storage):
    return tmp_storage.login("Voter")["id"]


def test_vote_overwrites(tmp_storage, owner, voter):
    s, (c1, c2, _) = locked_set(tmp_storage, owner)
    assert tmp_storage.user_vote(voter, s["id"]) is None
    tmp_storage.cast_vote(voter, s["id"], c1)
    assert tmp_storage.vote_tally(s["id"]) == {c1: 1}
    tmp_storage.cast_vote(voter, s["id"], c2)
    assert tmp_storage.vote_tally(s["id"]) == {c2: 1}
    assert tmp_storage.user_vote(voter, s["id"]) == c2


def test_self_vote_allowed(tmp_storage, owner, voter):
    s, (c1, c2, _) = locked_set(tmp_storage, owner)
    tmp_storage.cast_vote(owner, s["id"], c1)
    tmp_storage.cast_vote(voter, s["id"], c1)
    assert tmp_storage.vote_tally(s["id"]) == {c1: 2}
    assert tmp_storage.user_vote(owner, s["id"]) == c1


def test_vote_rejections(tmp_storage, owner, voter):
    event = tmp_storage.current_event() or tmp_storage.create_event("Night")
    # a draft set with a rerolled (replaced) card
    draft = tmp_storage.create_set(owner, "Zur", "p", PARAMS)
    cards = [tmp_storage.create_card(owner, "p", PARAMS, set_id=draft["id"], slot=slot)
             for slot in (1, 2, 3)]
    for c in cards:
        tmp_storage.update_card(c["id"], status="done")
    tmp_storage.reroll_card(cards[0]["id"], owner)

    with pytest.raises(StorageError) as exc:  # draft set
        tmp_storage.cast_vote(voter, draft["id"], cards[1]["id"])
    assert exc.value.status == 409

    new_card = tmp_storage.set_cards(draft["id"])[0]
    tmp_storage.update_card(new_card["id"], status="done")
    tmp_storage.lock_set(draft["id"], owner)

    other_owner = tmp_storage.login("Other")["id"]
    other_set = tmp_storage.create_set(other_owner, "Else", "p", PARAMS)
    foreign = tmp_storage.create_card(other_owner, "p", PARAMS, set_id=other_set["id"], slot=1)

    with pytest.raises(StorageError) as exc:  # card not in the set
        tmp_storage.cast_vote(voter, draft["id"], foreign["id"])
    assert exc.value.status == 400
    with pytest.raises(StorageError) as exc:  # replaced card
        tmp_storage.cast_vote(voter, draft["id"], cards[0]["id"])
    assert exc.value.status == 400
    with pytest.raises(StorageError) as exc:  # unknown set
        tmp_storage.cast_vote(voter, "missing", cards[1]["id"])
    assert exc.value.status == 404

    tmp_storage.cast_vote(voter, draft["id"], cards[1]["id"])
    tmp_storage.close_event(event["id"])
    with pytest.raises(StorageError) as exc:  # closed event
        tmp_storage.cast_vote(voter, draft["id"], cards[2]["id"])
    assert exc.value.status == 409
    assert exc.value.message == "Voting is closed"
    assert tmp_storage.user_vote(voter, draft["id"]) == cards[1]["id"]


def test_concurrent_votes_single_row(tmp_path):
    storage = Storage(tmp_path / "c.db")
    owner = storage.login("Owner")["id"]
    voter = storage.login("Voter")["id"]
    s, card_ids = locked_set(storage, owner)
    errors = []
    barrier = threading.Barrier(10)

    def vote(i):
        try:
            barrier.wait()
            # each thread gets its own Storage connection (threading.local)
            storage.cast_vote(voter, s["id"], card_ids[i % 3])
        except Exception as e:  # noqa: BLE001 - the test asserts there are none
            errors.append(e)

    threads = [threading.Thread(target=vote, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    rows = storage._conn().execute(
        "SELECT COUNT(*) FROM votes WHERE voter_id = ? AND set_id = ?", (voter, s["id"])
    ).fetchone()[0]
    assert rows == 1
    assert sum(storage.vote_tally(s["id"]).values()) == 1


def test_concurrent_votes_many_voters(tmp_path):
    storage = Storage(tmp_path / "m.db")
    owner = storage.login("Owner")["id"]
    voters = [storage.login(f"v{i}")["id"] for i in range(10)]
    s, card_ids = locked_set(storage, owner)
    errors = []
    barrier = threading.Barrier(len(voters))

    def vote(voter_id):
        try:
            barrier.wait()
            storage.cast_vote(voter_id, s["id"], card_ids[0])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=vote, args=(v,)) for v in voters]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert storage.vote_tally(s["id"]) == {card_ids[0]: 10}


def test_leader_flags():
    ids = ["a", "b", "c"]
    flags = leader_flags({"a": 2, "b": 1}, ids)
    assert flags == {
        "a": {"votes": 2, "leader": True, "tied": False},
        "b": {"votes": 1, "leader": False, "tied": False},
        "c": {"votes": 0, "leader": False, "tied": False},
    }
    flags = leader_flags({"a": 2, "b": 2}, ids)
    assert flags["a"] == {"votes": 2, "leader": False, "tied": True}
    assert flags["b"] == {"votes": 2, "leader": False, "tied": True}
    assert flags["c"] == {"votes": 0, "leader": False, "tied": False}
    flags = leader_flags({}, ids)
    assert all(f == {"votes": 0, "leader": False, "tied": False} for f in flags.values())
    assert set(flags) == set(ids)
    # tally entries for cards outside card_ids are ignored
    assert leader_flags({"zzz": 5, "a": 1}, ids)["a"] == {"votes": 1, "leader": True, "tied": False}
