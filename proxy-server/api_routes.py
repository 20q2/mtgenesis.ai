"""
AI Night HTTP API (users, generations, sets, events, votes, queue status, media).

Spec: docs/superpowers/specs/2026-09-28-ai-night-design.md §4.
The blueprint receives its dependencies so it can be tested without importing app.py.

CardView.card is the final card dict when set, otherwise the card's card_params,
so pending cards already show their name and type.

Auth: user-scoped routes require X-User-Id (401 if missing or unknown). Read-only
event/card routes accept it optionally (for myVoteCardId) but still reject an unknown
id with 401 so a stale browser login is detected. Admin routes require X-Admin-Pin (403);
after ADMIN_MAX_WRONG_PINS wrong PINs within ADMIN_LOCKOUT_SECONDS every admin request gets
429 for ADMIN_LOCKOUT_SECONDS (right PIN included), so the PIN can't be brute-forced.
GET /events/current and /events/<id> carry an ETag and answer If-None-Match with 304.
Media is public because <img> tags cannot send headers.
CORS/ngrok headers and OPTIONS answers come from app.py's global hooks and Flask.
"""
from __future__ import annotations

import collections
import hmac
import threading
import time
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from generation_queue import GenerationQueue
from storage import (PENDING_STATUSES, Storage, StorageError, clean_commander_name,
                     leader_flags)

MAX_PENDING_PER_USER = 3
MAX_PROMPT_CHARS = 1000  # the form's auto art prompt can reach ~450 chars (5 colours, long types)
ADMIN_MAX_WRONG_PINS = 5
ADMIN_LOCKOUT_SECONDS = 300  # also the window the wrong PINs are counted in
ADMIN_LOCKOUT_MESSAGE = "Too many wrong PINs - wait a few minutes"
DEFAULT_ADMIN_PIN = "1234"
MEDIA_KINDS = ("cards", "art")
API_PREFIX = "/api/v1"


def admin_pin_warning(admin_pin) -> str | None:
    """A startup warning when ADMIN_PIN is empty or still the default, else None."""
    pin = str(admin_pin or "").strip()
    if not pin:
        return "ADMIN_PIN is empty - set a strong ADMIN_PIN in config.py before the event!"
    if pin == DEFAULT_ADMIN_PIN:
        return (f"ADMIN_PIN is still the default '{DEFAULT_ADMIN_PIN}' - anyone can close the "
                "event. Set a random ADMIN_PIN (8+ characters) in config.py before the event!")
    return None


def create_api_blueprint(storage: Storage, gen_queue: GenerationQueue, data_dir: Path,
                         admin_pin: str, clock=time.monotonic) -> Blueprint:
    """All spec §4 endpoints; app.py registers the result with url_prefix="/api/v1".

    `clock` (seconds, monotonic) is injectable so tests can expire the admin lockout.
    """
    bp = Blueprint("ai_night_api", __name__)
    data_dir = Path(data_dir)
    # Serializes "check the pending cap, then create cards" so a double-submit cannot
    # slip two batches past the per-user cap (single-process server).
    create_lock = threading.Lock()
    # Wrong admin PIN bookkeeping (global, not per client: ngrok hides client IPs).
    admin_lock = threading.Lock()
    wrong_pin_times: collections.deque = collections.deque()
    locked_until = [0.0]

    # ----- errors and request helpers -----
    @bp.errorhandler(StorageError)
    def _storage_error(err: StorageError):
        return jsonify({"error": err.message}), err.status

    def body() -> dict:
        data = request.get_json(silent=True)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise StorageError(400, "Request body must be a JSON object")
        return data

    def required_body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise StorageError(400, "Request body must be a JSON object")
        return data

    def optional_user() -> dict | None:
        user_id = (request.headers.get("X-User-Id") or "").strip()
        if not user_id:
            return None
        user = storage.get_user(user_id)
        if user is None:
            raise StorageError(401, "Unknown user - please log in again")
        return user

    def require_user() -> dict:
        user = optional_user()
        if user is None:
            raise StorageError(401, "Please log in first")
        return user

    def require_admin() -> None:
        pin = request.headers.get("X-Admin-Pin") or ""
        with admin_lock:
            now = clock()
            if now < locked_until[0]:
                raise StorageError(429, ADMIN_LOCKOUT_MESSAGE)
            if hmac.compare_digest(pin.encode("utf-8"), str(admin_pin).encode("utf-8")):
                return
            while wrong_pin_times and wrong_pin_times[0] <= now - ADMIN_LOCKOUT_SECONDS:
                wrong_pin_times.popleft()
            wrong_pin_times.append(now)
            if len(wrong_pin_times) >= ADMIN_MAX_WRONG_PINS:
                wrong_pin_times.clear()
                locked_until[0] = now + ADMIN_LOCKOUT_SECONDS
        raise StorageError(403, "Wrong admin PIN")

    def check_pending_cap(user_id: str, adding: int) -> None:
        pending = storage.count_pending(user_id)
        if pending + adding > MAX_PENDING_PER_USER:
            raise StorageError(
                429, f"You already have {pending} card(s) in progress - wait for them to "
                     f"finish (max {MAX_PENDING_PER_USER} at a time)")

    # ----- views -----
    def card_view(row: dict) -> dict:
        position, eta = None, None
        if row["status"] in PENDING_STATUSES and not row["art_ready"]:
            position, eta = gen_queue.position(row["id"])
        cid = row["id"]
        return {
            "id": cid,
            "userId": row["user_id"],
            "setId": row["set_id"],
            "slot": row["slot"],
            "replaced": bool(row["replaced"]),
            "status": row["status"],
            "error": row["error"],
            "textReady": bool(row["text_ready"]),
            "artReady": bool(row["art_ready"]),
            "queuePosition": position,
            "etaSeconds": eta,
            "card": row["card"] if row["card"] is not None else row["card_params"],
            "cardImageUrl": f"{API_PREFIX}/media/cards/{cid}.png" if row["card_path"] else None,
            "artImageUrl": f"{API_PREFIX}/media/art/{cid}.png" if row["art_path"] else None,
            "createdAt": row["created_at"],
        }

    def set_view(row: dict, voter_id: str | None) -> dict:
        cards = storage.set_cards(row["id"])
        flags = leader_flags(storage.vote_tally(row["id"]), [c["id"] for c in cards])
        owner = storage.get_user(row["user_id"])
        return {
            "id": row["id"],
            "userId": row["user_id"],
            "username": owner["username"] if owner else "",
            "eventId": row["event_id"],
            "commanderName": row["commander_name"],
            "prompt": row["prompt"],
            "status": row["status"],
            "lockedAt": row["locked_at"],
            "cards": [{**card_view(c), **flags[c["id"]]} for c in cards],
            "myVoteCardId": storage.user_vote(voter_id, row["id"]) if voter_id else None,
        }

    def event_summary(row: dict) -> dict:
        return {"id": row["id"], "name": row["name"], "status": row["status"],
                "createdAt": row["created_at"], "closedAt": row["closed_at"]}

    def event_view(row: dict, voter_id: str | None) -> dict:
        return {**event_summary(row),
                "sets": [set_view(s, voter_id) for s in storage.locked_sets(row["id"])]}

    def voter_id() -> str | None:
        user = optional_user()
        return user["id"] if user else None

    def conditional_json(payload):
        """JSON with an ETag over the body; If-None-Match on it gets an empty 304.

        no-cache makes the browser store the body but revalidate on every poll, so
        the unchanged event views most polls return cost a few hundred bytes.
        """
        resp = jsonify(payload)
        resp.add_etag()
        resp.cache_control.no_cache = True
        return resp.make_conditional(request)

    # ----- users -----
    @bp.post("/users/login")
    def login():
        return jsonify(storage.login(required_body().get("username")))

    @bp.get("/me/cards")
    def my_cards():
        user = require_user()
        return jsonify([card_view(c) for c in storage.list_user_cards(user["id"])])

    @bp.get("/me/sets/current")
    def my_current_set():
        user = require_user()
        row = storage.current_set(user["id"])
        return jsonify(set_view(row, user["id"]) if row else None)

    # ----- generation -----
    @bp.post("/generations")
    def create_generations():
        user = require_user()
        data = required_body()
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise StorageError(400, "No prompt provided")
        prompt = prompt.strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            raise StorageError(400, f"Prompt is too long (max {MAX_PROMPT_CHARS} characters)")
        card_data = data.get("cardData")
        if not isinstance(card_data, dict):
            raise StorageError(400, "cardData must be an object")
        count = data.get("count")
        if type(count) is not int or count not in (1, 3):
            raise StorageError(400, "count must be 1 or 3")
        commander_name = clean_commander_name(data.get("commanderName")) if count == 3 else None

        with create_lock:
            check_pending_cap(user["id"], count)
            if count == 1:
                set_id = None
                cards = [storage.create_card(user["id"], prompt, card_data)]
            else:
                params = {**card_data, "name": commander_name}
                set_id = storage.create_set(user["id"], commander_name, prompt, params)["id"]
                cards = [storage.create_card(user["id"], prompt, params, set_id=set_id, slot=slot)
                         for slot in (1, 2, 3)]
        for card in cards:
            gen_queue.enqueue(card["id"])
        return jsonify({"setId": set_id,
                        "cards": [card_view(storage.get_card(c["id"])) for c in cards]})

    @bp.post("/cards/<card_id>/reroll")
    def reroll(card_id):
        user = require_user()
        with create_lock:
            card = storage.get_card(card_id)
            if card is None:
                raise StorageError(404, "Card not found")
            # Ownership and in-progress problems are reported by reroll_card (403/409)
            # before the cap, which only applies when a new card would really be queued.
            if card["user_id"] == user["id"] and card["status"] not in PENDING_STATUSES:
                check_pending_cap(user["id"], 1)
            new = storage.reroll_card(card_id, user["id"])
        gen_queue.enqueue(new["id"])
        return jsonify(card_view(storage.get_card(new["id"])))

    @bp.get("/cards/<card_id>")
    def get_card(card_id):
        optional_user()
        card = storage.get_card(card_id)
        if card is None:
            raise StorageError(404, "Card not found")
        return jsonify(card_view(card))

    # ----- sets -----
    @bp.post("/sets/<set_id>/lock")
    def lock(set_id):
        user = require_user()
        row = storage.lock_set(set_id, user["id"], body().get("commanderName"))
        return jsonify(set_view(row, user["id"]))

    @bp.post("/sets/<set_id>/unlock")
    def unlock(set_id):
        user = require_user()
        row = storage.unlock_set(set_id, user["id"])
        return jsonify(set_view(row, user["id"]))

    # ----- events and votes -----
    @bp.get("/events/current")
    def current_event():
        voter = voter_id()
        row = storage.current_event()
        return conditional_json(event_view(row, voter) if row else None)

    @bp.get("/events")
    def list_events():
        return jsonify([event_summary(e) for e in storage.list_events()])

    @bp.get("/events/<event_id>")
    def get_event(event_id):
        voter = voter_id()
        row = storage.get_event(event_id)
        if row is None:
            raise StorageError(404, "Event not found")
        return conditional_json(event_view(row, voter))

    @bp.post("/votes")
    def vote():
        user = require_user()
        data = required_body()
        set_id, card_id = data.get("setId"), data.get("cardId")
        if not isinstance(set_id, str) or not isinstance(card_id, str):
            raise StorageError(400, "setId and cardId are required")
        storage.cast_vote(user["id"], set_id, card_id)
        return jsonify(set_view(storage.get_set(set_id), user["id"]))

    @bp.get("/queue_status")
    def queue_status():
        return jsonify(gen_queue.status())

    # ----- media -----
    @bp.get("/media/<kind>/<filename>")
    def media(kind, filename):
        not_found = StorageError(404, "Not found")
        if kind not in MEDIA_KINDS or not filename.endswith(".png"):
            raise not_found
        stem = filename[:-len(".png")]
        try:
            canonical = str(uuid.UUID(stem))
        except ValueError:
            raise not_found from None
        if canonical != stem:  # rejects braces, urn: prefixes, hex-only and upper-case forms
            raise not_found
        path = data_dir / kind / f"{canonical}.png"
        if not path.is_file():
            raise not_found
        return send_file(path, mimetype="image/png", max_age=3600)

    # ----- admin -----
    @bp.post("/admin/events")
    def admin_create_event():
        require_admin()
        return jsonify(event_view(storage.create_event(required_body().get("name")), None))

    @bp.post("/admin/events/<event_id>/close")
    def admin_close_event(event_id):
        require_admin()
        return jsonify(event_view(storage.close_event(event_id), None))

    return bp
