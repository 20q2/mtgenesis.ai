"""Profile pictures (docs/superpowers/specs/2026-10-14-profile-pictures-design.md)."""
import io
import sqlite3

import pytest
from flask import Flask
from PIL import Image

from api_routes import create_api_blueprint
from avatars import AVATAR_SIZE, process_avatar
from storage import Storage, StorageError
from test_api import FakeQueue, H, PIN, generate_set, login


def image_bytes(fmt="JPEG", size=(640, 480), color=(200, 40, 40), exif=None) -> bytes:
    out = io.BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    Image.new("RGB", size, color).save(out, fmt, **kwargs)
    return out.getvalue()


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(tmp_storage, data_dir):
    app = Flask(__name__)
    app.register_blueprint(create_api_blueprint(tmp_storage, FakeQueue(), data_dir, PIN),
                           url_prefix="/api/v1")
    return app.test_client()


def upload(client, user_id, data, content_type="image/jpeg"):
    return client.put("/api/v1/me/avatar", headers=H(user_id), data=data, content_type=content_type)


# ----- image processing -----
def test_process_avatar_crops_square_and_reencodes_as_jpeg():
    out = Image.open(io.BytesIO(process_avatar(image_bytes("PNG", (640, 480)))))
    assert out.format == "JPEG"
    assert out.size == (AVATAR_SIZE, AVATAR_SIZE) == (256, 256)


def test_process_avatar_drops_metadata():
    exif = Image.Exif()
    exif[0x010F] = "SpyCam"  # Make
    exif[0x8825] = {2: (51, 30, 0)}  # GPS IFD
    out = Image.open(io.BytesIO(process_avatar(image_bytes(exif=exif.tobytes()))))
    assert not out.getexif()
    assert "exif" not in out.info


def test_process_avatar_applies_exif_rotation():
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° clockwise on display
    # A wide red-left / blue-right image; after the rotation red is on top.
    img = Image.new("RGB", (400, 200), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 200, 200))
    raw = io.BytesIO()
    img.save(raw, "JPEG", exif=exif.tobytes())
    out = Image.open(io.BytesIO(process_avatar(raw.getvalue()))).convert("RGB")
    top, bottom = out.getpixel((128, 20)), out.getpixel((128, 236))
    assert top[0] > 200 and top[2] < 60
    assert bottom[2] > 200 and bottom[0] < 60


@pytest.mark.parametrize("data", [b"", b"not an image", b"GIF89a" + b"\x00" * 20])
def test_process_avatar_rejects_non_images(data):
    with pytest.raises(StorageError) as err:
        process_avatar(data)
    assert err.value.status == 400


def test_process_avatar_rejects_huge_dimensions():
    with pytest.raises(StorageError) as err:
        process_avatar(image_bytes("PNG", (5000, 10), (0, 0, 0)))
    assert err.value.status == 400


# ----- storage -----
def test_users_have_no_avatar_until_set(tmp_storage):
    user = tmp_storage.login("Andrew")
    assert user == {"id": user["id"], "username": "Andrew", "avatar_id": None}
    assert tmp_storage.get_user(user["id"])["avatar_id"] is None


def test_set_avatar_returns_the_previous_one(tmp_storage):
    user = tmp_storage.login("Andrew")
    assert tmp_storage.set_avatar(user["id"], "a1") is None
    assert tmp_storage.set_avatar(user["id"], "a2") == "a1"
    assert tmp_storage.get_user(user["id"])["avatar_id"] == "a2"
    assert tmp_storage.login("andrew")["avatar_id"] == "a2"
    assert tmp_storage.set_avatar(user["id"], None) == "a2"
    assert tmp_storage.get_user(user["id"])["avatar_id"] is None


def test_set_avatar_unknown_user(tmp_storage):
    with pytest.raises(StorageError) as err:
        tmp_storage.set_avatar("nope", "a1")
    assert err.value.status == 404


def test_migration_adds_avatar_column_to_an_old_database(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, username TEXT NOT NULL, created_at TEXT NOT NULL)")
    conn.execute("INSERT INTO users VALUES ('u1', 'Old', '2026-09-01T00:00:00+00:00')")
    conn.commit()
    conn.close()
    assert Storage(db).get_user("u1") == {"id": "u1", "username": "Old", "avatar_id": None}


# ----- API -----
def test_login_and_me_carry_avatar_url(client):
    res = client.post("/api/v1/users/login", json={"username": "Andrew"})
    body = res.get_json()
    assert body == {"id": body["id"], "username": "Andrew", "avatarUrl": None}
    assert client.get("/api/v1/me", headers=H(body["id"])).get_json() == body
    assert client.get("/api/v1/me").status_code == 401


def test_upload_saves_a_square_jpeg_and_serves_it(client, data_dir):
    uid = login(client, "Andrew")
    res = upload(client, uid, image_bytes("PNG"), "image/png")
    assert res.status_code == 200, res.get_json()
    url = res.get_json()["avatarUrl"]
    assert url.startswith("/api/v1/media/avatars/") and url.endswith(".jpg")
    assert client.get("/api/v1/me", headers=H(uid)).get_json()["avatarUrl"] == url

    served = client.get(url)
    assert served.status_code == 200 and served.mimetype == "image/jpeg"
    assert Image.open(io.BytesIO(served.data)).size == (256, 256)
    assert len(list((data_dir / "avatars").iterdir())) == 1


def test_a_new_picture_gets_a_new_url_and_replaces_the_old_file(client, data_dir):
    uid = login(client, "Andrew")
    first = upload(client, uid, image_bytes()).get_json()["avatarUrl"]
    second = upload(client, uid, image_bytes(color=(0, 90, 200))).get_json()["avatarUrl"]
    assert first != second
    assert client.get(first).status_code == 404
    assert client.get(second).status_code == 200
    assert len(list((data_dir / "avatars").iterdir())) == 1


def test_delete_removes_the_picture(client, data_dir):
    uid = login(client, "Andrew")
    url = upload(client, uid, image_bytes()).get_json()["avatarUrl"]
    res = client.delete("/api/v1/me/avatar", headers=H(uid))
    assert res.status_code == 200 and res.get_json()["avatarUrl"] is None
    assert client.get(url).status_code == 404
    assert list((data_dir / "avatars").iterdir()) == []
    # Removing again is fine
    assert client.delete("/api/v1/me/avatar", headers=H(uid)).status_code == 200


def test_upload_rejects_non_images_and_needs_a_user(client):
    uid = login(client, "Andrew")
    bad = upload(client, uid, b"<svg/>", "image/svg+xml")
    assert bad.status_code == 400 and "image" in bad.get_json()["error"].lower()
    assert upload(client, uid, b"").status_code == 400
    assert client.put("/api/v1/me/avatar", data=image_bytes(), content_type="image/jpeg").status_code == 401
    assert client.get("/api/v1/me", headers=H(uid)).get_json()["avatarUrl"] is None


@pytest.mark.parametrize("name", ["../t.db.jpg", "not-a-uuid.jpg",
                                  "8F0E2A5C-1D3B-4C7E-9A6F-2B4D8E1C3A5F.jpg"])
def test_media_route_only_serves_canonical_avatar_names(client, name):
    assert client.get(f"/api/v1/media/avatars/{name}").status_code == 404


def test_media_route_avatars_are_jpg_only(client):
    uid = login(client, "Andrew")
    url = upload(client, uid, image_bytes()).get_json()["avatarUrl"]
    assert client.get(url[:-len(".jpg")] + ".png").status_code == 404


def test_views_carry_the_owner_avatar(client, tmp_storage):
    uid = login(client, "Andrew")
    url = upload(client, uid, image_bytes()).get_json()["avatarUrl"]
    set_id = generate_set(client, uid)["setId"]
    view = client.get("/api/v1/me/sets/current", headers=H(uid)).get_json()
    mine = next(s for s in view if s["id"] == set_id)
    assert mine["avatarUrl"] == url

    other = login(client, "Beth")
    assert client.get("/api/v1/me", headers=H(other)).get_json()["avatarUrl"] is None


def test_gallery_and_pool_carry_the_maker_avatar(client, tmp_storage):
    from test_pool import create_pool_api, done_card, submit_api
    uid = login(client, "Andrew")
    url = upload(client, uid, image_bytes()).get_json()["avatarUrl"]
    card_id = done_card(tmp_storage, uid)

    client.post(f"/api/v1/cards/{card_id}/share", headers=H(uid), json={"shared": True})
    shared = client.get("/api/v1/cards/shared").get_json()
    assert shared[0]["avatarUrl"] == url

    create_pool_api(client)
    view = submit_api(client, uid, card_id).get_json()
    assert view["entries"][0]["avatarUrl"] == url
