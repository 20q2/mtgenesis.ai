"""Knowledge Pool: one medal-voted list; the top half of the submitters' count make the pool.

Spec: docs/superpowers/specs/2026-09-29-knowledge-pool-design.md.
"""
import sqlite3
import uuid

import pytest

from api_routes import power_check
from storage import (Storage, StorageError, pool_card_eligible, pool_cutoff, pool_ranking)
from test_api import ADMIN, H, client, login, queue  # noqa: F401  (fixtures)

RED_CREATURE = {"name": "Ember Pup", "manaCost": "{1}{R}", "colors": ["R"], "type": "Creature",
                "rarity": "common", "cmc": 2, "power": "2", "toughness": "2",
                "description": "Haste"}
POT_OF_GREEN = {"name": "Pot of Green", "manaCost": "{0}", "colors": [], "type": "Artifact",
                "rarity": "common", "cmc": 0, "description": "Draw three cards."}
GOLD_CARD = {"name": "Azorius Envoy", "manaCost": "{W}{U}", "colors": ["W", "U"],
             "type": "Creature", "rarity": "uncommon", "cmc": 2}
MULTICOLOR_ERROR = "Only colorless or mono-colored cards can enter the pool"


def done_card(storage, user_id, card=RED_CREATURE):
    row = storage.create_card(user_id, "prompt", card)
    storage.update_card(row["id"], status="done", text_ready=True, art_ready=True, card=card)
    return row["id"]


def user(storage, name):
    return storage.login(name)["id"]


@pytest.fixture
def pool(tmp_storage):
    return tmp_storage.create_pool("Knowledge Pool 2026", 3)


def status_of(call) -> int:
    with pytest.raises(StorageError) as err:
        call()
    return err.value.status


# ----- eligibility -----
@pytest.mark.parametrize("colors, cost, ok", [
    (["W"], "{1}{W}", True), ([], "{3}", True), (["C"], "{2}", True), ([], "{G}{G}", True),
    (["W", "U"], "{W}{U}", False), ([], "{W/U}", False), ([], "{1}{B}{R}", False)])
def test_pool_card_eligible(colors, cost, ok):
    assert pool_card_eligible({"colors": colors, "manaCost": cost}) is ok


# ----- lifecycle -----
def test_one_open_pool_and_close(tmp_storage, pool):
    assert pool["max_entries_per_user"] == 3 and pool["status"] == "open"
    assert status_of(lambda: tmp_storage.create_pool("Another", 3)) == 409
    closed = tmp_storage.close_pool(pool["id"])
    assert closed["status"] == "closed" and closed["closed_at"]
    assert tmp_storage.current_pool() is None
    assert status_of(lambda: tmp_storage.close_pool(pool["id"])) == 409


@pytest.mark.parametrize("name, cap", [("", 3), ("x" * 81, 3), ("ok", 0), ("ok", 11),
                                       ("ok", "3"), ("ok", True)])
def test_create_pool_validation(tmp_storage, name, cap):
    assert status_of(lambda: tmp_storage.create_pool(name, cap)) == 400


# ----- submissions -----
def test_submit_rules(tmp_storage):
    alice, bob = user(tmp_storage, "alice"), user(tmp_storage, "bob")
    pup = done_card(tmp_storage, alice)
    assert status_of(lambda: tmp_storage.submit_pool_entry(alice, pup)) == 404  # no pool open
    pool = tmp_storage.create_pool("Night", 3)

    entry = tmp_storage.submit_pool_entry(alice, pup)
    assert entry["card_id"] == pup and entry["user_id"] == alice and entry["pool_id"] == pool["id"]
    assert status_of(lambda: tmp_storage.submit_pool_entry(alice, pup)) == 409  # twice
    assert status_of(lambda: tmp_storage.submit_pool_entry(bob, pup)) == 403   # not theirs
    assert status_of(lambda: tmp_storage.submit_pool_entry(alice, str(uuid.uuid4()))) == 404

    queued = tmp_storage.create_card(alice, "p", RED_CREATURE)["id"]
    assert status_of(lambda: tmp_storage.submit_pool_entry(alice, queued)) == 400

    gold = done_card(tmp_storage, alice, GOLD_CARD)
    with pytest.raises(StorageError) as err:
        tmp_storage.submit_pool_entry(alice, gold)
    assert err.value.status == 400 and err.value.message == MULTICOLOR_ERROR

    tmp_storage.submit_pool_entry(alice, done_card(tmp_storage, alice, POT_OF_GREEN))
    tmp_storage.submit_pool_entry(alice, done_card(tmp_storage, alice))
    assert status_of(lambda: tmp_storage.submit_pool_entry(
        alice, done_card(tmp_storage, alice))) == 409  # cap of 3

    tmp_storage.close_pool(pool["id"])
    assert status_of(lambda: tmp_storage.submit_pool_entry(
        bob, done_card(tmp_storage, bob))) == 404  # closed: no pool open


def test_withdraw_frees_medals_and_moves_cutoff(tmp_storage, pool):
    names = ["alice", "bob", "carl", "dana"]
    ids = {n: user(tmp_storage, n) for n in names}
    entries = {n: tmp_storage.submit_pool_entry(ids[n], done_card(tmp_storage, ids[n]))
               for n in names}
    assert pool_cutoff(tmp_storage.pool_entries(pool["id"])) == 2

    tmp_storage.award_pool_medal(ids["bob"], entries["alice"]["id"], "gold")
    assert status_of(lambda: tmp_storage.withdraw_pool_entry(entries["alice"]["id"], ids["bob"])) == 403
    assert tmp_storage.withdraw_pool_entry(entries["alice"]["id"], ids["alice"]) == pool["id"]

    remaining = tmp_storage.pool_entries(pool["id"])
    assert pool_cutoff(remaining) == 1
    assert tmp_storage.pool_medal_counts(pool["id"]) == {}
    assert tmp_storage.my_pool_medals(ids["bob"], pool["id"]) == {}
    tmp_storage.award_pool_medal(ids["bob"], entries["carl"]["id"], "gold")
    assert tmp_storage.pool_medal_counts(pool["id"])[entries["carl"]["id"]]["gold"] == 1


def test_medals_one_each_per_pool_and_never_own(tmp_storage, pool):
    alice, bob, carl = (user(tmp_storage, n) for n in ("alice", "bob", "carl"))
    a = tmp_storage.submit_pool_entry(alice, done_card(tmp_storage, alice))["id"]
    b = tmp_storage.submit_pool_entry(bob, done_card(tmp_storage, bob))["id"]

    tmp_storage.award_pool_medal(carl, a, "gold")
    tmp_storage.award_pool_medal(carl, b, "gold")  # moves the gold
    assert tmp_storage.my_pool_medals(carl, pool["id"]) == {b: "gold"}
    tmp_storage.award_pool_medal(carl, b, "silver")  # replaces the gold on b
    tmp_storage.award_pool_medal(carl, a, "bronze")
    assert tmp_storage.my_pool_medals(carl, pool["id"]) == {b: "silver", a: "bronze"}
    counts = tmp_storage.pool_medal_counts(pool["id"])
    assert counts[b] == {"gold": 0, "silver": 1, "bronze": 0}

    assert status_of(lambda: tmp_storage.award_pool_medal(alice, a, "gold")) == 403
    assert status_of(lambda: tmp_storage.award_pool_medal(carl, a, "platinum")) == 400
    assert status_of(lambda: tmp_storage.award_pool_medal(carl, str(uuid.uuid4()), "gold")) == 404
    tmp_storage.clear_pool_medal(carl, a)
    assert tmp_storage.my_pool_medals(carl, pool["id"]) == {b: "silver"}


def test_closed_pool_is_frozen(tmp_storage, pool):
    alice, bob = user(tmp_storage, "alice"), user(tmp_storage, "bob")
    a = tmp_storage.submit_pool_entry(alice, done_card(tmp_storage, alice))
    tmp_storage.close_pool(pool["id"])
    for call in (lambda: tmp_storage.award_pool_medal(bob, a["id"], "gold"),
                 lambda: tmp_storage.clear_pool_medal(bob, a["id"]),
                 lambda: tmp_storage.withdraw_pool_entry(a["id"], alice)):
        assert status_of(call) == 409


def test_open_pool_entry_ids(tmp_storage, pool):
    alice = user(tmp_storage, "alice")
    card = done_card(tmp_storage, alice)
    assert tmp_storage.open_pool_entry_ids() == {}
    entry = tmp_storage.submit_pool_entry(alice, card)
    assert tmp_storage.open_pool_entry_ids() == {card: entry["id"]}
    tmp_storage.close_pool(pool["id"])
    assert tmp_storage.open_pool_entry_ids() == {}


# ----- ranking -----
def rank(spec: dict[str, tuple[int, int, int]], submitters: int) -> dict[str, dict]:
    """spec: entry id -> (gold, silver, bronze); entries are owned round-robin by `submitters` users."""
    entries = [{"id": eid, "user_id": f"u{i % submitters}", "created_at": f"2026-10-01T00:00:{i:02d}"}
               for i, eid in enumerate(spec)]
    medals = {eid: {"gold": g, "silver": s, "bronze": b} for eid, (g, s, b) in spec.items()}
    return pool_ranking(entries, medals)


def ins(table):
    return {eid for eid, row in table.items() if row["in"]}


def test_ranking_points_then_golds_then_silvers():
    # points 9, 7, 6 (1 gold), 6 (0 gold, 3 silver), 6 (0 gold, 2 silver), 2, 0, 0
    table = rank({"a": (3, 0, 0), "b": (1, 2, 0), "c": (1, 1, 1), "d": (0, 3, 0),
                  "e": (0, 2, 2), "f": (0, 1, 0), "g": (0, 0, 0), "h": (0, 0, 0)},
                 submitters=8)
    assert [table[e]["points"] for e in "abcdef"] == [9, 7, 6, 6, 6, 2]
    assert ins(table) == {"a", "b", "c", "d"}
    assert not any(row["tiedAtCutoff"] for row in table.values())
    assert [table[e]["rank"] for e in "abcdef"] == [1, 2, 3, 4, 5, 6]


def test_ranking_tie_at_the_line_all_get_in():
    table = rank({"a": (3, 0, 0), "b": (1, 2, 0), "c": (1, 1, 1), "d": (0, 3, 0),
                  "e": (0, 3, 0), "f": (0, 1, 0), "g": (0, 0, 0), "h": (0, 0, 0)},
                 submitters=8)
    assert ins(table) == {"a", "b", "c", "d", "e"}
    assert table["d"]["tiedAtCutoff"] and table["e"]["tiedAtCutoff"]
    assert not table["c"]["tiedAtCutoff"]
    assert table["d"]["rank"] == table["e"]["rank"] == 4 and table["f"]["rank"] == 6


def test_ranking_zero_points_never_in():
    table = rank({"a": (1, 0, 0), "b": (0, 0, 0), "c": (0, 0, 0)}, submitters=3)
    assert ins(table) == {"a"}
    table = rank({"a": (0, 0, 0), "b": (0, 0, 0)}, submitters=2)
    assert ins(table) == set()


def test_ranking_cutoff_from_submitters():
    assert pool_cutoff([]) == 0
    assert ins(rank({"a": (5, 0, 0), "b": (1, 0, 0)}, submitters=1)) == set()
    entries = [{"id": "a", "user_id": "u1"}, {"id": "b", "user_id": "u1"},
               {"id": "c", "user_id": "u2"}, {"id": "d", "user_id": "u3"}]
    assert pool_cutoff(entries) == 1


# ----- migration -----
OLD_POOL_SCHEMA = """
CREATE TABLE pools (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
    max_entries_per_user INTEGER NOT NULL, created_at TEXT NOT NULL, closed_at TEXT);
CREATE TABLE pool_slots (id TEXT PRIMARY KEY, pool_id TEXT NOT NULL, position INTEGER NOT NULL,
    label TEXT NOT NULL, color_rule TEXT NOT NULL, type_rule TEXT NOT NULL);
CREATE TABLE pool_entries (id TEXT PRIMARY KEY, pool_id TEXT NOT NULL, slot_id TEXT NOT NULL,
    card_id TEXT NOT NULL, user_id TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE pool_medals (voter_id TEXT NOT NULL, pool_id TEXT NOT NULL, slot_id TEXT NOT NULL,
    entry_id TEXT NOT NULL, medal TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE pool_bans (voter_id TEXT NOT NULL, pool_id TEXT NOT NULL, entry_id TEXT NOT NULL,
    created_at TEXT NOT NULL);
"""


def old_db(tmp_path, with_rows: bool):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(OLD_POOL_SCHEMA)
    if with_rows:
        conn.execute("INSERT INTO pools VALUES ('p', 'Old', 'closed', 2, '2026-09-29', NULL)")
    conn.commit()
    conn.close()
    return db


def table_names(db):
    conn = sqlite3.connect(db)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def test_migration_drops_empty_old_tables(tmp_path):
    db = old_db(tmp_path, with_rows=False)
    storage = Storage(db)
    names = table_names(db)
    assert "pool_slots" not in names and "pool_bans" not in names
    cols = {r[1] for r in storage._conn().execute("PRAGMA table_info(pool_entries)")}
    assert "slot_id" not in cols
    Storage(db)  # reopening is a no-op
    assert storage.create_pool("Night", 3)["status"] == "open"


def test_migration_refuses_old_tables_with_rows(tmp_path):
    db = old_db(tmp_path, with_rows=True)
    with pytest.raises(RuntimeError, match="pool_slots"):
        Storage(db)
    assert "pool_slots" in table_names(db)


# ----- power check -----
def test_power_check_flags_pot_of_green():
    assert power_check(POT_OF_GREEN)["verdict"] == "over"
    assert power_check(RED_CREATURE)["verdict"] == "fair"
    assert power_check(None) is None


# ----- API -----
def create_pool_api(client, cap=3):
    res = client.post("/api/v1/admin/pools", headers=ADMIN,
                      json={"name": "Knowledge Pool 2026", "maxEntriesPerUser": cap})
    assert res.status_code == 200, res.get_json()
    return res.get_json()


def test_api_admin_create_requires_pin(client):
    res = client.post("/api/v1/admin/pools", json={"name": "x", "maxEntriesPerUser": 2})
    assert res.status_code == 403
    assert client.get("/api/v1/pools/current").get_json() is None


ENTRY_KEYS = {"id", "cardId", "card", "username", "mine", "power", "gold", "silver", "bronze",
              "points", "rank", "in", "tiedAtCutoff", "myMedal", "createdAt"}


def submit_api(client, user_id, card_id):
    return client.post("/api/v1/pools/entries", headers=H(user_id), json={"cardId": card_id})


def test_api_full_flow(client, tmp_storage):
    pool = create_pool_api(client)
    assert pool["status"] == "open" and pool["maxEntriesPerUser"] == 3
    assert pool["entries"] == [] and pool["submitters"] == 0 and pool["cutoff"] == 0

    names = ["Alice", "Bob", "Carl", "Dana"]
    ids = {n: login(client, n) for n in names}
    cards = {n: done_card(tmp_storage, ids[n]) for n in names}
    cards["Bob"] = done_card(tmp_storage, ids["Bob"], POT_OF_GREEN)

    gold = done_card(tmp_storage, ids["Alice"], GOLD_CARD)
    res = submit_api(client, ids["Alice"], gold)
    assert res.status_code == 400 and res.get_json()["error"] == MULTICOLOR_ERROR

    entries = {}
    for n in names:
        res = submit_api(client, ids[n], cards[n])
        assert res.status_code == 200, res.get_json()
        view = res.get_json()
        entries[n] = next(e for e in view["entries"] if e["cardId"] == cards[n])
    assert view["submitters"] == 4 and view["cutoff"] == 2 and view["myEntryCount"] == 1
    assert set(entries["Dana"]) == ENTRY_KEYS
    assert entries["Dana"]["mine"] is True and entries["Dana"]["username"] == "Dana"
    assert entries["Bob"]["power"]["verdict"] == "over"

    # Everything is visible: Bob sees Alice's name.
    bob_view = client.get("/api/v1/pools/current", headers=H(ids["Bob"])).get_json()
    alice_seen = next(e for e in bob_view["entries"] if e["id"] == entries["Alice"]["id"])
    assert alice_seen["username"] == "Alice" and alice_seen["mine"] is False

    def medal(voter, target, kind):
        res = client.post("/api/v1/pools/medals", headers=H(ids[voter]),
                          json={"entryId": entries[target]["id"], "medal": kind})
        assert res.status_code == 200, res.get_json()
        return res.get_json()

    medal("Bob", "Carl", "gold")
    medal("Alice", "Carl", "gold")
    view = medal("Carl", "Dana", "gold")
    view = medal("Bob", "Alice", "silver")
    assert [e["id"] for e in view["entries"][:3]] == [entries[n]["id"] for n in ("Carl", "Dana", "Alice")]
    assert [e["in"] for e in view["entries"]] == [True, True, False, False]
    assert [e["rank"] for e in view["entries"][:3]] == [1, 2, 3]
    assert view["myMedals"] == {"gold": entries["Carl"]["id"], "silver": entries["Alice"]["id"],
                                "bronze": None}
    assert client.post("/api/v1/pools/medals", headers=H(ids["Bob"]),
                       json={"entryId": entries["Bob"]["id"], "medal": "gold"}).status_code == 403
    res = client.post("/api/v1/pools/medals/clear", headers=H(ids["Bob"]),
                      json={"entryId": entries["Alice"]["id"]})
    assert res.get_json()["myMedals"]["silver"] is None

    # The card views know which cards are in the open pool.
    mine = client.get("/api/v1/me/cards", headers=H(ids["Alice"])).get_json()
    by_id = {c["id"]: c for c in mine}
    assert by_id[cards["Alice"]]["poolEntryId"] == entries["Alice"]["id"]
    assert by_id[gold]["poolEntryId"] is None
    assert client.get(f"/api/v1/cards/{cards['Alice']}").get_json()["poolEntryId"] == entries["Alice"]["id"]

    assert client.post("/api/v1/pools/bans", headers=H(ids["Bob"]),
                       json={"entryId": entries["Alice"]["id"]}).status_code in (404, 405)

    closed = client.post(f"/api/v1/admin/pools/{pool['id']}/close", headers=ADMIN).get_json()
    assert closed["status"] == "closed"
    assert [e["in"] for e in closed["entries"]] == [True, True, False, False]
    assert client.get("/api/v1/pools/current").get_json() is None
    assert client.get(f"/api/v1/cards/{cards['Alice']}").get_json()["poolEntryId"] is None
    assert [p["id"] for p in client.get("/api/v1/pools").get_json()] == [pool["id"]]
    past = client.get(f"/api/v1/pools/{pool['id']}", headers=H(ids["Bob"])).get_json()
    assert len(past["entries"]) == 4
    assert client.post("/api/v1/pools/medals", headers=H(ids["Bob"]),
                       json={"entryId": entries["Dana"]["id"], "medal": "bronze"}).status_code == 409
    assert client.post(f"/api/v1/pools/entries/{entries['Dana']['id']}/withdraw",
                       headers=H(ids["Dana"])).status_code == 409


def test_api_withdraw_and_auth(client, tmp_storage):
    create_pool_api(client)
    alice, bob = login(client, "Alice"), login(client, "Bob")
    pup = done_card(tmp_storage, alice)
    assert client.post("/api/v1/pools/entries", json={"cardId": pup}).status_code == 401
    assert client.post("/api/v1/pools/entries", headers=H(alice), json={}).status_code == 400
    view = submit_api(client, alice, pup).get_json()
    entry_id = view["entries"][0]["id"]
    assert client.post(f"/api/v1/pools/entries/{entry_id}/withdraw",
                       headers=H(bob)).status_code == 403
    res = client.post(f"/api/v1/pools/entries/{entry_id}/withdraw", headers=H(alice))
    assert res.status_code == 200 and res.get_json()["entries"] == []
    assert client.get("/api/v1/pools/nope").status_code == 404
    assert client.post("/api/v1/pools/medals", headers=H(bob), json={}).status_code == 400


def test_api_admin_cap_default_and_range(client):
    res = client.post("/api/v1/admin/pools", headers=ADMIN, json={"name": "Night"})
    assert res.status_code == 200 and res.get_json()["maxEntriesPerUser"] == 3
    client.post(f"/api/v1/admin/pools/{res.get_json()['id']}/close", headers=ADMIN)
    for cap in (0, 11):
        res = client.post("/api/v1/admin/pools", headers=ADMIN,
                          json={"name": "Night", "maxEntriesPerUser": cap})
        assert res.status_code == 400, cap


def test_api_pool_current_etag(client):
    create_pool_api(client)
    first = client.get("/api/v1/pools/current")
    again = client.get("/api/v1/pools/current", headers={"If-None-Match": first.headers["ETag"]})
    assert again.status_code == 304
