"""CORS headers must be single-valued on every response, success and error alike.

Flask-CORS and app.py's after_request both touch Access-Control-Allow-Origin; on
responses produced by an exception handler both used to add one, and browsers
reject a response with two values (the frontend then sees status 0).

Imports app.py (torch/diffusers/ollama), so it is marked slow.
"""
import pytest
from flask import abort, jsonify

pytestmark = pytest.mark.slow

from app import app  # noqa: E402

ORIGIN = {"Origin": "http://localhost:4200"}


# Throwaway routes registered before the app serves any request.
@app.route("/__test_cors_ok")
def _test_cors_ok():
    return jsonify({"ok": True})


@app.route("/__test_cors_forbidden", methods=["POST"])
def _test_cors_forbidden():
    abort(403)


@pytest.fixture
def client():
    return app.test_client()


def _assert_single_cors_headers(resp):
    assert len(resp.headers.getlist("Access-Control-Allow-Origin")) == 1
    assert len(resp.headers.getlist("ngrok-skip-browser-warning")) == 1


def test_success_response_has_one_allow_origin(client):
    resp = client.get("/__test_cors_ok", headers=ORIGIN)
    assert resp.status_code == 200
    _assert_single_cors_headers(resp)


def test_exception_handled_error_has_one_allow_origin(client):
    resp = client.post("/__test_cors_forbidden", headers=ORIGIN)
    assert resp.status_code == 403
    _assert_single_cors_headers(resp)


def test_preflight_still_allows_ai_night_headers(client):
    resp = client.options("/__test_cors_forbidden", headers={
        **ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-User-Id, X-Admin-Pin, ngrok-skip-browser-warning",
    })
    assert len(resp.headers.getlist("Access-Control-Allow-Origin")) == 1
    allowed = resp.headers.get("Access-Control-Allow-Headers", "").lower()
    for header in ("x-user-id", "x-admin-pin", "ngrok-skip-browser-warning"):
        assert header in allowed
