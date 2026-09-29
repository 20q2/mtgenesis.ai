"""Knowledge Pool: slot rules, submissions, medals, bans and the API views.

Spec: docs/superpowers/specs/2026-09-29-knowledge-pool-design.md.
"""
import pytest

from api_routes import power_check
from storage import (POOL_BAN_THRESHOLD, StorageError, card_fits_slot, clean_pool_slots,
                     pool_standings, slot_rule_text)
from test_api import ADMIN, H, client, login, queue  # noqa: F401  (fixtures)

SLOTS = [
    {"label": "Red creature", "colorRule": "R", "typeRule": "creature"},
    {"label": "Colorless", "colorRule": "colorless", "typeRule": "any"},
    {"label": "Wild", "colorRule": "any", "typeRule": "any"},
]
RED_CREATURE = {"name": "Ember Pup", "manaCost": "{1}{R}", "colors": ["R"], "type": "Creature",
                "rarity": "common", "cmc": 2, "power": "2", "toughness": "2",
                "description": "Haste"}
POT_OF_GREEN = {"name": "Pot of Green", "manaCost": "{0}", "colors": [], "type": "Artifact",
                "rarity": "common", "cmc": 0, "description": "Draw three cards."}


def done_card(storage, user_id, card):
    row = storage.create_card(user_id, "prompt", card)
    storage.update_card(row["id"], status="done", text_ready=True, art_ready=True, card=card)
    return row["id"]


def user(storage, name):
    return storage.login(name)["id"]


@pytest.fixture
def pool(tmp_storage):
    return tmp_storage.create_pool("Knowledge Pool 2026", 2, SLOTS)


def slot_ids(storage, pool):
    return [s["id"] for s in storage.pool_slots(pool["id"])]


# ----- slot rules -----
@pytest.mark.parametrize("card, color, kind, fits", [
    (RED_CREATURE, "R", "creature", True),
    (RED_CREATURE, "R", "noncreature", False),
    (RED_CREATURE, "G", "any", False),
    (RED_CREATURE, "multicolor", "any", False),
    (RED_CREATURE, "any", "any", True),
    (POT_OF_GREEN, "colorless", "noncreature", True),
    (POT_OF_GREEN, "G", "any", False),
    ({"colors": ["R", "G"], "type": "Creature"}, "multicolor", "creature", True),
    ({"colors": ["R", "G"], "type": "Creature"}, "R", "creature", False),
    ({"colors": [], "manaCost": "{2}{U}", "type": "Instant"}, "U", "noncreature", True),
    ({"colors": ["C"], "type": "Land"}, "colorless", "land", True),
    ({"colors": ["C"], "type": "Land"}, "any", "noncreature", False),
    ({"colors": ["W"], "type": "Artifact Creature"}, "W", "creature", True),
])
def test_card_fits_slot(card, color, kind, fits):
    assert card_fits_slot(card, color, kind) is fits


def test_slot_rule_text():
    assert slot_rule_text("U", "creature") == "Blue creature"
    assert slot_rule_text("multicolor", "any") == "Multicolor card"
    assert slot_rule_text("any", "land") == "Any land"
    assert slot_rule_text("any", "any") == "Any card"


def test_clean_pool_slots_defaults_label_and_validates():
    assert clean_pool_slots([{"colorRule": "G", "typeRule": "noncreature"}]) == [
        {"label": "Green noncreature", "color_rule": "G", "type_rule": "noncreature"}]
    for bad in ([], None, [{"colorRule": "purple"}], [{"typeRule": "tribal"}], ["x"],
                [{"label": "x" * 41}], [{}] * 41):
        with pytest.raises(StorageError) as err:
            clean_pool_slots(bad)
        assert err.value.status == 400


# ----- lifecycle -----
def test_one_open_pool_and_close(tmp_storage, pool):
    with pytest.raises(StorageError) as err:
        tmp_storage.create_pool("Another", 2, SLOTS)
    assert err.value.status == 409
    assert tmp_storage.current_pool()["id"] == pool["id"]
    tmp_storage.close_pool(pool["id"])
    assert tmp_storage.current_pool() is None
    with pytest.raises(StorageError) as err:
        tmp_storage.close_pool(pool["id"])
    assert err.value.status == 409
    tmp_storage.create_pool("Next year", 2, SLOTS)
    assert [p["name"] for p in tmp_storage.list_pools()] == ["Next year", "Knowledge Pool 2026"]


@pytest.mark.parametrize("name, cap", [("", 2), ("x" * 81, 2), ("ok", 0), ("ok", 41), ("ok", "2"),
                                       ("ok", True)])
def test_create_pool_validation(tmp_storage, name, cap):
    with pytest.raises(StorageError) as err:
        tmp_storage.create_pool(name, cap, SLOTS)
    assert err.value.status == 400


def test_slots_keep_their_order(tmp_storage, pool):
    assert [(s["position"], s["label"]) for s in tmp_storage.pool_slots(pool["id"])] == [
        (1, "Red creature"), (2, "Colorless"), (3, "Wild")]


# ----- submissions -----
def test_submit_rules(tmp_storage, pool):
    red, colorless, wild = slot_ids(tmp_storage, pool)
    alice, bob = user(tmp_storage, "alice"), user(tmp_storage, "bob")
    pup = done_card(tmp_storage, alice, RED_CREATURE)
    pot = done_card(tmp_storage, alice, POT_OF_GREEN)

    def fails(status, *args):
        with pytest.raises(StorageError) as err:
            tmp_storage.submit_pool_entry(*args)
        assert err.value.status == status, err.value.message

    fails(403, bob, red, pup)                       # not the owner
    fails(400, alice, colorless, pup)               # doesn't fit
    pending = tmp_storage.create_card(alice, "p", RED_CREATURE)["id"]
    fails(400, alice, red, pending)                 # unfinished
    fails(404, alice, "nope", pup)
    entry = tmp_storage.submit_pool_entry(alice, red, pup)
    assert entry["slot_id"] == red and entry["user_id"] == alice
    fails(409, alice, wild, pup)                    # card already in the pool
    fails(409, alice, red, done_card(tmp_storage, alice, RED_CREATURE))  # one per slot
    tmp_storage.submit_pool_entry(alice, colorless, pot)
    fails(409, alice, wild, done_card(tmp_storage, alice, RED_CREATURE))  # cap of 2 reached


def test_withdraw_removes_medals_bans_and_frees_the_slot(tmp_storage, pool):
    red = slot_ids(tmp_storage, pool)[0]
    alice, bob = user(tmp_storage, "alice"), user(tmp_storage, "bob")
    entry = tmp_storage.submit_pool_entry(alice, red, done_card(tmp_storage, alice, RED_CREATURE))
    tmp_storage.award_pool_medal(bob, entry["id"], "gold")
    tmp_storage.ban_pool_entry(bob, entry["id"])
    with pytest.raises(StorageError) as err:
        tmp_storage.withdraw_pool_entry(entry["id"], bob)
    assert err.value.status == 403
    assert tmp_storage.withdraw_pool_entry(entry["id"], alice) == pool["id"]
    assert tmp_storage.pool_entries(pool["id"]) == []
    assert tmp_storage.pool_medal_counts(pool["id"]) == {}
    assert tmp_storage.pool_ban_counts(pool["id"]) == {}          # Bob's ban is refunded
    tmp_storage.submit_pool_entry(alice, red, done_card(tmp_storage, alice, RED_CREATURE))


def slot_with_entries(storage, pool, n):
    """n players each submit a red creature to the first slot; returns (players, entries)."""
    red = slot_ids(storage, pool)[0]
    players = [user(storage, f"p{i}") for i in range(n)]
    entries = [storage.submit_pool_entry(pid, red, done_card(storage, pid, RED_CREATURE))["id"]
               for pid in players]
    return players, entries


# ----- medals -----
def test_medals_move_and_never_go_to_your_own_card(tmp_storage, pool):
    (a, b, c), (ea, eb, ec) = slot_with_entries(tmp_storage, pool, 3)
    with pytest.raises(StorageError) as err:
        tmp_storage.award_pool_medal(a, ea, "gold")
    assert err.value.status == 403
    with pytest.raises(StorageError) as err:
        tmp_storage.award_pool_medal(a, eb, "platinum")
    assert err.value.status == 400

    tmp_storage.award_pool_medal(a, eb, "gold")
    tmp_storage.award_pool_medal(a, ec, "silver")
    assert tmp_storage.my_pool_medals(a, pool["id"]) == {eb: "gold", ec: "silver"}
    tmp_storage.award_pool_medal(a, ec, "gold")      # gold moves to C, C's silver is replaced
    assert tmp_storage.my_pool_medals(a, pool["id"]) == {ec: "gold"}
    tmp_storage.award_pool_medal(a, eb, "bronze")
    tmp_storage.clear_pool_medal(a, ec)
    assert tmp_storage.my_pool_medals(a, pool["id"]) == {eb: "bronze"}
    assert tmp_storage.pool_medal_counts(pool["id"]) == {
        eb: {"gold": 0, "silver": 0, "bronze": 1}}


def test_standings_points_gold_tiebreak_and_ties():
    medals = {"x": {"gold": 1, "silver": 0, "bronze": 1},   # 4 points, 1 gold
              "y": {"gold": 0, "silver": 2, "bronze": 0},   # 4 points, 0 golds
              "z": {"gold": 0, "silver": 0, "bronze": 1}}
    table = pool_standings(["x", "y", "z", "w"], medals, {}, apply_bans=False)
    assert table["x"]["points"] == 4 and table["x"]["leader"] and not table["y"]["leader"]
    assert table["w"] == {"gold": 0, "silver": 0, "bronze": 0, "points": 0,
                          "disqualified": False, "leader": False, "tied": False}

    medals["y"] = {"gold": 1, "silver": 0, "bronze": 1}
    table = pool_standings(["x", "y", "z"], medals, {}, apply_bans=False)
    assert table["x"]["tied"] and table["y"]["tied"] and not table["x"]["leader"]

    assert not any(r["leader"] or r["tied"]
                   for r in pool_standings(["x"], {}, {}, apply_bans=False).values())


def test_standings_bans_only_count_once_closed():
    medals = {"pot": {"gold": 3, "silver": 0, "bronze": 0},
              "fair": {"gold": 0, "silver": 1, "bronze": 0}}
    bans = {"pot": POOL_BAN_THRESHOLD, "fair": POOL_BAN_THRESHOLD - 1}
    open_ = pool_standings(["pot", "fair"], medals, bans, apply_bans=False)
    assert open_["pot"]["leader"] and not open_["pot"]["disqualified"]
    closed = pool_standings(["pot", "fair"], medals, bans, apply_bans=True)
    assert closed["pot"]["disqualified"] and not closed["pot"]["leader"]
    assert closed["fair"]["leader"] and not closed["fair"]["disqualified"]


# ----- bans -----
def test_bans_two_per_player(tmp_storage, pool):
    (a, *_), (ea, eb, ec, ed) = slot_with_entries(tmp_storage, pool, 4)
    with pytest.raises(StorageError) as err:
        tmp_storage.ban_pool_entry(a, ea)
    assert err.value.status == 403
    tmp_storage.ban_pool_entry(a, eb)
    tmp_storage.ban_pool_entry(a, eb)                # idempotent: still one ban used
    tmp_storage.ban_pool_entry(a, ec)
    with pytest.raises(StorageError) as err:
        tmp_storage.ban_pool_entry(a, ed)
    assert err.value.status == 409
    tmp_storage.unban_pool_entry(a, eb)
    tmp_storage.ban_pool_entry(a, ed)
    assert tmp_storage.my_pool_bans(a, pool["id"]) == {ec, ed}
    assert tmp_storage.pool_ban_counts(pool["id"]) == {ec: 1, ed: 1}


def test_closed_pool_is_frozen(tmp_storage, pool):
    red = slot_ids(tmp_storage, pool)[0]
    alice, bob = user(tmp_storage, "alice"), user(tmp_storage, "bob")
    a = tmp_storage.submit_pool_entry(alice, red, done_card(tmp_storage, alice, RED_CREATURE))
    tmp_storage.close_pool(pool["id"])
    for call in (lambda: tmp_storage.award_pool_medal(bob, a["id"], "gold"),
                 lambda: tmp_storage.clear_pool_medal(bob, a["id"]),
                 lambda: tmp_storage.ban_pool_entry(bob, a["id"]),
                 lambda: tmp_storage.unban_pool_entry(bob, a["id"]),
                 lambda: tmp_storage.withdraw_pool_entry(a["id"], alice),
                 lambda: tmp_storage.submit_pool_entry(
                     bob, red, done_card(tmp_storage, bob, RED_CREATURE))):
        with pytest.raises(StorageError) as err:
            call()
        assert err.value.status == 409


# ----- power check -----
def test_power_check_flags_pot_of_green():
    assert power_check(POT_OF_GREEN)["verdict"] == "over"
    assert power_check(RED_CREATURE)["verdict"] == "fair"
    assert power_check(None) is None


# ----- API -----
def create_pool_api(client, slots=SLOTS, cap=2):
    res = client.post("/api/v1/admin/pools", headers=ADMIN,
                      json={"name": "Knowledge Pool 2026", "maxEntriesPerUser": cap,
                            "slots": slots})
    assert res.status_code == 200, res.get_json()
    return res.get_json()


def test_api_admin_create_requires_pin(client):
    res = client.post("/api/v1/admin/pools", json={"name": "x", "maxEntriesPerUser": 2,
                                                   "slots": SLOTS})
    assert res.status_code == 403
    assert client.get("/api/v1/pools/current").get_json() is None


def test_api_full_flow(client, tmp_storage):
    pool = create_pool_api(client)
    assert pool["status"] == "open" and pool["maxEntriesPerUser"] == 2
    assert [s["ruleText"] for s in pool["slots"]] == ["Red creature", "Colorless card", "Any card"]
    red_slot, colorless_slot = pool["slots"][0]["id"], pool["slots"][1]["id"]

    alice, bob = login(client, "Alice"), login(client, "Bob")
    pup = done_card(tmp_storage, alice, RED_CREATURE)
    pot = done_card(tmp_storage, bob, POT_OF_GREEN)

    res = client.post("/api/v1/pools/entries", headers=H(alice),
                      json={"slotId": red_slot, "cardId": pup})
    assert res.status_code == 200, res.get_json()
    view = res.get_json()
    entry = view["slots"][0]["entries"][0]
    assert view["myEntryCount"] == 1 and view["slots"][0]["myEntryId"] == entry["id"]
    assert entry["mine"] is True and entry["username"] == "Alice"
    assert entry["card"]["status"] == "done" and entry["power"]["verdict"] == "fair"

    res = client.post("/api/v1/pools/entries", headers=H(bob),
                      json={"slotId": red_slot, "cardId": pot})
    assert res.status_code == 400 and "Red creature" in res.get_json()["error"]
    res = client.post("/api/v1/pools/entries", headers=H(bob),
                      json={"slotId": colorless_slot, "cardId": pot})
    assert res.status_code == 200
    pot_entry = res.get_json()["slots"][1]["entries"][0]
    assert pot_entry["power"]["verdict"] == "over"

    # Anonymous while open: Bob sees Alice's entry without her name.
    bob_view = client.get("/api/v1/pools/current", headers=H(bob)).get_json()
    seen = bob_view["slots"][0]["entries"][0]
    assert seen["mine"] is False and seen["username"] is None

    res = client.post("/api/v1/pools/medals", headers=H(alice),
                      json={"entryId": pot_entry["id"], "medal": "gold"})
    assert res.status_code == 200
    view = res.get_json()
    assert view["slots"][1]["myMedals"] == {"gold": pot_entry["id"], "silver": None, "bronze": None}
    assert view["slots"][1]["entries"][0]["myMedal"] == "gold"
    res = client.post("/api/v1/pools/medals", headers=H(bob),
                      json={"entryId": pot_entry["id"], "medal": "gold"})
    assert res.status_code == 403
    res = client.post("/api/v1/pools/medals", headers=H(bob),
                      json={"entryId": entry["id"], "medal": "silver"})
    red_view = res.get_json()["slots"][0]["entries"][0]
    assert red_view["leader"] is True and red_view["points"] == 2 and red_view["silver"] == 1

    # Bans: only your own are visible while open, and they don't change the leader yet.
    res = client.post("/api/v1/pools/bans", headers=H(alice), json={"entryId": pot_entry["id"]})
    view = res.get_json()
    assert view["myBansLeft"] == 1 and view["bansPerPlayer"] == 2 and view["banThreshold"] == 3
    banned = view["slots"][1]["entries"][0]
    assert banned["bannedByMe"] is True and banned["bans"] is None and banned["leader"] is True
    carl, dana = login(client, "Carl"), login(client, "Dana")
    for voter in (carl, dana):
        assert client.post("/api/v1/pools/bans", headers=H(voter),
                           json={"entryId": pot_entry["id"]}).status_code == 200
    bob_view = client.get("/api/v1/pools/current", headers=H(bob)).get_json()
    assert bob_view["slots"][1]["entries"][0]["bannedByMe"] is False
    assert bob_view["slots"][1]["entries"][0]["bans"] is None
    res = client.post("/api/v1/pools/medals/clear", headers=H(bob), json={"entryId": entry["id"]})
    assert res.get_json()["slots"][0]["entries"][0]["points"] == 0
    client.post("/api/v1/pools/medals", headers=H(bob),
                json={"entryId": entry["id"], "medal": "bronze"})
    res = client.post("/api/v1/pools/bans/clear", headers=H(carl), json={"entryId": pot_entry["id"]})
    assert res.get_json()["myBansLeft"] == 2
    client.post("/api/v1/pools/bans", headers=H(carl), json={"entryId": pot_entry["id"]})

    closed = client.post(f"/api/v1/admin/pools/{pool['id']}/close", headers=ADMIN).get_json()
    assert closed["status"] == "closed"
    red = closed["slots"][0]["entries"][0]
    assert red["leader"] is True and red["points"] == 1 and red["username"] == "Alice"
    pot_final = closed["slots"][1]["entries"][0]
    # Pot of Green had the only gold in its slot, but 3 bans knock it out at close.
    assert pot_final["bans"] == 3 and pot_final["disqualified"] is True
    assert pot_final["leader"] is False and pot_final["username"] == "Bob"

    assert client.get("/api/v1/pools/current").get_json() is None
    listed = client.get("/api/v1/pools").get_json()
    assert [p["id"] for p in listed] == [pool["id"]]
    past = client.get(f"/api/v1/pools/{pool['id']}", headers=H(bob)).get_json()
    assert past["slots"][0]["entries"][0]["username"] == "Alice"
    res = client.post("/api/v1/pools/medals", headers=H(bob),
                      json={"entryId": entry["id"], "medal": "gold"})
    assert res.status_code == 409


def test_api_withdraw_and_auth(client, tmp_storage):
    pool = create_pool_api(client)
    red_slot = pool["slots"][0]["id"]
    alice, bob = login(client, "Alice"), login(client, "Bob")
    pup = done_card(tmp_storage, alice, RED_CREATURE)
    assert client.post("/api/v1/pools/entries",
                       json={"slotId": red_slot, "cardId": pup}).status_code == 401
    view = client.post("/api/v1/pools/entries", headers=H(alice),
                       json={"slotId": red_slot, "cardId": pup}).get_json()
    entry_id = view["slots"][0]["entries"][0]["id"]
    assert client.post(f"/api/v1/pools/entries/{entry_id}/withdraw",
                       headers=H(bob)).status_code == 403
    res = client.post(f"/api/v1/pools/entries/{entry_id}/withdraw", headers=H(alice))
    assert res.status_code == 200 and res.get_json()["slots"][0]["entries"] == []
    assert client.get("/api/v1/pools/nope").status_code == 404
    assert client.post("/api/v1/pools/medals", headers=H(bob), json={}).status_code == 400
    assert client.post("/api/v1/pools/bans", headers=H(bob), json={}).status_code == 400


def test_api_pool_current_etag(client):
    create_pool_api(client)
    first = client.get("/api/v1/pools/current")
    again = client.get("/api/v1/pools/current", headers={"If-None-Match": first.headers["ETag"]})
    assert again.status_code == 304
