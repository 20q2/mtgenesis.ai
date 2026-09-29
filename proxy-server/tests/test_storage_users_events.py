import pytest

from storage import Storage, StorageError


def test_login_creates_then_reuses(tmp_storage):
    first = tmp_storage.login("Andrew")
    again = tmp_storage.login(" andrew ")
    assert again["id"] == first["id"]
    assert again["username"] == "Andrew"
    assert tmp_storage.get_user(first["id"]) == {"id": first["id"], "username": "Andrew"}


def test_login_distinct_users(tmp_storage):
    a = tmp_storage.login("Andrew")
    b = tmp_storage.login("Beth_2 x-y")
    assert a["id"] != b["id"]
    assert b["username"] == "Beth_2 x-y"


def test_get_user_unknown(tmp_storage):
    assert tmp_storage.get_user("nope") is None


@pytest.mark.parametrize("bad", ["", " ", "x" * 25, "bad!name", None, 42])
def test_login_validation(tmp_storage, bad):
    with pytest.raises(StorageError) as exc:
        tmp_storage.login(bad)
    assert exc.value.status == 400


def test_login_max_length_ok(tmp_storage):
    assert tmp_storage.login("x" * 24)["username"] == "x" * 24


def test_login_persists_across_instances(tmp_path):
    user = Storage(tmp_path / "p.db").login("Andrew")
    assert Storage(tmp_path / "p.db").get_user(user["id"])["username"] == "Andrew"


def test_single_open_event(tmp_storage):
    a = tmp_storage.create_event("A")
    assert a["status"] == "open" and a["name"] == "A" and a["closed_at"] is None
    with pytest.raises(StorageError) as exc:
        tmp_storage.create_event("B")
    assert exc.value.status == 409
    tmp_storage.close_event(a["id"])
    b = tmp_storage.create_event("B")
    assert tmp_storage.current_event() == b


def test_create_event_requires_name(tmp_storage):
    with pytest.raises(StorageError) as exc:
        tmp_storage.create_event("   ")
    assert exc.value.status == 400


def test_close_event_sets_closed_at(tmp_storage):
    a = tmp_storage.create_event("A")
    closed = tmp_storage.close_event(a["id"])
    assert closed["status"] == "closed"
    assert closed["closed_at"] is not None
    assert tmp_storage.current_event() is None
    assert tmp_storage.get_event(a["id"]) == closed
    with pytest.raises(StorageError) as exc:
        tmp_storage.close_event(a["id"])
    assert exc.value.status == 409


def test_close_unknown_event(tmp_storage):
    with pytest.raises(StorageError) as exc:
        tmp_storage.close_event("missing")
    assert exc.value.status == 404
    assert tmp_storage.get_event("missing") is None


def test_list_events_newest_first(tmp_storage):
    names = ["one", "two", "three"]
    for name in names:
        event = tmp_storage.create_event(name)
        tmp_storage.close_event(event["id"])
    listed = tmp_storage.list_events()
    assert [e["name"] for e in listed] == ["three", "two", "one"]
    assert set(listed[0]) == {"id", "name", "status", "created_at", "closed_at"}
