"""
AI Night HTTP API (users, generations, sets, events, votes, queue status, media)
and the Knowledge Pool (pools, entries, medals).

Specs: docs/superpowers/specs/2026-09-28-ai-night-design.md §4 and
docs/superpowers/specs/2026-09-29-knowledge-pool-design.md §4.
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
from storage import (MEDAL_POINTS, PENDING_STATUSES, POOL_DEFAULT_ENTRIES, Storage,
                     StorageError, clean_commander_name, commander_slot_params, leader_flags,
                     pool_cutoff, pool_ranking)

try:  # the power heuristic is advisory: pools still work without it
    import power_level
except ImportError:  # pragma: no cover
    power_level = None

MAX_PENDING_PER_USER = 3
MAX_PROMPT_CHARS = 1000  # the form's auto art prompt can reach ~450 chars (5 colours, long types)
ADMIN_MAX_WRONG_PINS = 5
ADMIN_LOCKOUT_SECONDS = 300  # also the window the wrong PINs are counted in
ADMIN_LOCKOUT_MESSAGE = "Too many wrong PINs - wait a few minutes"
DEFAULT_ADMIN_PIN = "1234"
MEDIA_KINDS = ("cards", "art")
API_PREFIX = "/api/v1"


def power_check(card: dict | None) -> dict | None:
    """{estimate, budget, verdict: fair|pushed|over} for a finished card's rules text, or None."""
    if power_level is None or not isinstance(card, dict):
        return None
    try:
        estimate, budget = power_level.assess(card.get("description") or "", card)
    except Exception:  # a heuristic must never break the pool view
        return None
    over = estimate - budget
    verdict = ("over" if over >= power_level.ERROR_OVER
               else "pushed" if over >= power_level.WARN_OVER else "fair")
    return {"estimate": estimate, "budget": budget, "verdict": verdict}


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
    def card_view(row: dict, pool_entry_ids: dict[str, str] | None = None) -> dict:
        """pool_entry_ids: storage.open_pool_entry_ids(), fetched once by list routes."""
        if pool_entry_ids is None:
            pool_entry_ids = storage.open_pool_entry_ids()
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
            "shared": row.get("shared_at") is not None,
            "poolEntryId": pool_entry_ids.get(cid),
        }

    def set_view(row: dict, voter_id: str | None) -> dict:
        pool_entry_ids = storage.open_pool_entry_ids()
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
            "cards": [{**card_view(c, pool_entry_ids), **flags[c["id"]]} for c in cards],
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

    # ----- knowledge pool views -----
    def pool_summary(row: dict) -> dict:
        return {"id": row["id"], "name": row["name"], "status": row["status"],
                "maxEntriesPerUser": row["max_entries_per_user"],
                "createdAt": row["created_at"], "closedAt": row["closed_at"]}

    def pool_view(row: dict, viewer_id: str | None) -> dict:
        """The whole pool as `viewer_id` sees it, entries in rank order (spec §4)."""
        entries = storage.pool_entries(row["id"])
        ranking = pool_ranking(entries, storage.pool_medal_counts(row["id"]))
        my_medals = storage.my_pool_medals(viewer_id, row["id"]) if viewer_id else {}
        pool_entry_ids = storage.open_pool_entry_ids()
        usernames: dict[str, str] = {}

        def username(user_id: str) -> str:
            if user_id not in usernames:
                user = storage.get_user(user_id)
                usernames[user_id] = user["username"] if user else ""
            return usernames[user_id]

        entry_views = []
        for e in sorted(entries, key=lambda e: ranking[e["id"]]["rank"]):  # stable: oldest first
            card = storage.get_card(e["card_id"])
            if card is None:
                continue
            view = card_view(card, pool_entry_ids)
            entry_views.append({
                "id": e["id"],
                "cardId": e["card_id"],
                "card": view,
                "username": username(e["user_id"]),
                "mine": e["user_id"] == viewer_id,
                "power": power_check(view["card"]),
                **ranking[e["id"]],
                "myMedal": my_medals.get(e["id"]),
                "createdAt": e["created_at"],
            })
        return {**pool_summary(row),
                "submitters": len({e["user_id"] for e in entries}),
                "cutoff": pool_cutoff(entries),
                "myEntryCount": sum(1 for e in entries if e["user_id"] == viewer_id),
                "myMedals": {m: next((eid for eid, given in my_medals.items() if given == m), None)
                             for m in MEDAL_POINTS},
                "entries": entry_views}

    def pool_or_404(pool_id: str) -> dict:
        row = storage.get_pool(pool_id)
        if row is None:
            raise StorageError(404, "Pool not found")
        return row

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
        pool_entry_ids = storage.open_pool_entry_ids()
        return jsonify([card_view(c, pool_entry_ids) for c in storage.list_user_cards(user["id"])])

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
        if count == 3:
            params = {**card_data, "name": commander_name}
            # One 3-, 4- and 5-mana version; checked before the old draft is abandoned.
            slot_params = {slot: commander_slot_params(params, slot) for slot in (1, 2, 3)}

        with create_lock:
            check_pending_cap(user["id"], count)
            if count == 1:
                set_id = None
                cards = [storage.create_card(user["id"], prompt, card_data)]
            else:
                set_id = storage.create_set(user["id"], commander_name, prompt, params)["id"]
                cards = [storage.create_card(user["id"], prompt, slot_params[slot], set_id=set_id,
                                             slot=slot)
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

    @bp.post("/cards/<card_id>/share")
    def share_card(card_id):
        user = require_user()
        shared = required_body().get("shared")
        if not isinstance(shared, bool):
            raise StorageError(400, "shared must be true or false")
        return jsonify(card_view(storage.set_card_shared(card_id, user["id"], shared)))

    @bp.get("/cards/shared")
    def shared_cards():
        """The gallery's Community tab: every shared card, with its maker."""
        optional_user()
        pool_entry_ids = storage.open_pool_entry_ids()
        return jsonify([{**card_view(c, pool_entry_ids), "username": c["username"]}
                        for c in storage.list_shared_cards()])

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

    # ----- knowledge pool -----
    @bp.get("/pools/current")
    def current_pool():
        viewer = voter_id()
        row = storage.current_pool()
        return conditional_json(pool_view(row, viewer) if row else None)

    @bp.get("/pools")
    def list_pools():
        return jsonify([pool_summary(p) for p in storage.list_pools()])

    @bp.get("/pools/<pool_id>")
    def get_pool(pool_id):
        viewer = voter_id()
        return conditional_json(pool_view(pool_or_404(pool_id), viewer))

    @bp.post("/pools/entries")
    def submit_pool_entry():
        user = require_user()
        card_id = required_body().get("cardId")
        if not isinstance(card_id, str):
            raise StorageError(400, "cardId is required")
        entry = storage.submit_pool_entry(user["id"], card_id)
        return jsonify(pool_view(pool_or_404(entry["pool_id"]), user["id"]))

    @bp.post("/pools/entries/<entry_id>/withdraw")
    def withdraw_pool_entry(entry_id):
        user = require_user()
        pool_id = storage.withdraw_pool_entry(entry_id, user["id"])
        return jsonify(pool_view(pool_or_404(pool_id), user["id"]))

    def entry_id_from_body() -> str:
        entry_id = required_body().get("entryId")
        if not isinstance(entry_id, str):
            raise StorageError(400, "entryId is required")
        return entry_id

    @bp.post("/pools/medals")
    def award_pool_medal():
        user = require_user()
        data = required_body()
        entry_id, medal = data.get("entryId"), data.get("medal")
        if not isinstance(entry_id, str) or not isinstance(medal, str):
            raise StorageError(400, "entryId and medal are required")
        pool_id = storage.award_pool_medal(user["id"], entry_id, medal)
        return jsonify(pool_view(pool_or_404(pool_id), user["id"]))

    @bp.post("/pools/medals/clear")
    def clear_pool_medal():
        user = require_user()
        pool_id = storage.clear_pool_medal(user["id"], entry_id_from_body())
        return jsonify(pool_view(pool_or_404(pool_id), user["id"]))

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

    @bp.post("/admin/pools")
    def admin_create_pool():
        require_admin()
        data = required_body()
        row = storage.create_pool(data.get("name"),
                                  data.get("maxEntriesPerUser", POOL_DEFAULT_ENTRIES))
        return jsonify(pool_view(row, None))

    @bp.post("/admin/pools/<pool_id>/close")
    def admin_close_pool(pool_id):
        require_admin()
        return jsonify(pool_view(storage.close_pool(pool_id), None))

    return bp
