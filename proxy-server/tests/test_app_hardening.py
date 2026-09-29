"""app.py hardening from the final review: body size cap (M-3), the default/empty admin PIN
startup warning (I-1) and CORS headers on conditional-GET 304s (I-2).

Imports app.py (torch/diffusers/ollama), so it is marked slow.
"""
import pytest
from flask import jsonify, request

pytestmark = pytest.mark.slow

import app as app_module  # noqa: E402
from app import app  # noqa: E402

ORIGIN = {"Origin": "http://localhost:4200"}


@app.route("/__test_conditional")
def _test_conditional():
    resp = jsonify({"sets": [1, 2, 3]})
    resp.add_etag()
    return resp.make_conditional(request)


@app.route("/__test_upload", methods=["POST"])
def _test_upload():
    return jsonify({"size": len(request.get_data())})


@pytest.fixture
def client():
    return app.test_client()


def test_request_bodies_are_capped(client):
    assert app.config["MAX_CONTENT_LENGTH"] == 64 * 1024
    ok = client.post("/__test_upload", data=b"x" * 1000, headers=ORIGIN)
    assert ok.status_code == 200
    big = client.post("/__test_upload", data=b"x" * (65 * 1024), headers=ORIGIN)
    assert big.status_code == 413
    assert len(big.headers.getlist("Access-Control-Allow-Origin")) == 1


def test_304_keeps_a_single_allow_origin(client):
    first = client.get("/__test_conditional", headers=ORIGIN)
    etag = first.headers["ETag"]
    again = client.get("/__test_conditional", headers={**ORIGIN, "If-None-Match": etag})
    assert again.status_code == 304
    assert again.headers.getlist("Access-Control-Allow-Origin") == ["*"]
    assert len(again.headers.getlist("ngrok-skip-browser-warning")) == 1


@pytest.mark.parametrize("pin", ["", "1234", None])
def test_default_or_empty_admin_pin_warns_loudly(capsys, pin):
    app_module.warn_about_admin_pin(pin)
    out = capsys.readouterr().out
    assert "ADMIN_PIN" in out and "WARNING" in out


def test_strong_admin_pin_does_not_warn(capsys):
    app_module.warn_about_admin_pin("k7#Qm2vX9p")
    assert "WARNING" not in capsys.readouterr().out
