"""
AI Night HTTP API (users, generations, sets, events, votes, queue status, media)
and the Knowledge Pool (pools, entries, medals).

Specs: docs/superpowers/specs/2026-09-28-ai-night-design.md §4 and
docs/superpowers/specs/2026-09-29-knowledge-pool-design.md §4.
The blueprint receives its dependencies so it can be tested without importing app.py.

CardView.card is the final card dict when set, otherwise the card's card_params,
so pending cards already show their name and type.

Host errata: GET /admin/errata, POST /admin/cards/<id>/errata[/revert] (docs/superpowers/specs/2026-10-14-errata-design.md).
Profile pictures: GET /me, PUT/DELETE /me/avatar (docs/superpowers/specs/2026-10-14-profile-pictures-design.md).
Reroll requests: POST /cards/<id>/reroll-request (docs/superpowers/specs/2026-10-14-reroll-requests-design.md).

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

import base64
import collections
import hmac
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from flask import Blueprint, jsonify, request, send_file

from avatars import process_avatar
from commander_rules import COMMANDER_CMCS, commander_params
from generation_queue import GenerationQueue
from storage import (MEDAL_POINTS, PENDING_STATUSES, POOL_DEFAULT_ENTRIES, REROLL_REQUESTS_NEEDED,
                     Storage, StorageError, clean_commander_name, leader_flags, pool_cutoff,
                     pool_ranking)

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
# media kind -> (file extension, mimetype); every file is named by a canonical UUID
MEDIA_KINDS = {"cards": (".png", "image/png"), "art": (".png", "image/png"),
               "avatars": (".jpg", "image/jpeg")}
API_PREFIX = "/api/v1"


# Host errata (docs/superpowers/specs/2026-10-14-errata-design.md)
ERRATA_TEXT_MAX = 1000
# "3", "12", "*", "X", "1+*", "*+1"
PT_RE = re.compile(r"^(\d{1,2}|\*|X|\d[+-]\*|\*[+-]\d)$")
NOT_ERRATABLE = ("Errata is for cards in the commander vote, a closed event's results "
                 "or the Knowledge Pool")
# (card dict, art as raw base64 PNG or None) -> rendered card as base64 PNG (app.py: the card renderer)
RenderCard = Callable[[dict, "str | None"], "str | None"]


def has_body(card: dict) -> bool:
    """Creatures and Vehicles print power/toughness."""
    return ("creature" in str(card.get("type") or "").lower()
            or "vehicle" in str(card.get("subtype") or "").lower())


def errata_card_data(card: dict, data: dict) -> dict:
    """The card with the host's rules text and P/T; everything else stays as printed (the
    renderer prints no flavor text, so there is none to errata).
    ~ and CARDNAME in the rules text become the card's name. StorageError(400) on bad input."""
    new = dict(card)
    text = data.get("description")
    if not isinstance(text, str):
        raise StorageError(400, "Rules text must be text")
    text = text.replace("\r\n", "\n").strip()
    if len(text) > ERRATA_TEXT_MAX:
        raise StorageError(400, f"Rules text is too long (at most {ERRATA_TEXT_MAX} characters)")
    name = str(card.get("name") or "")
    new["description"] = text.replace("CARDNAME", name).replace("~", name)

    if has_body(card):
        for key, label in (("power", "Power"), ("toughness", "Toughness")):
            value = data.get(key, card.get(key))
            value = "" if value is None else str(value).strip().upper()
            if not PT_RE.fullmatch(value):
                raise StorageError(400, f"{label} must be a number, * or X")
            new[key] = value
    else:
        new.pop("power", None)
        new.pop("toughness", None)
    return new


def avatar_url(avatar_id: str | None) -> str | None:
    return f"{API_PREFIX}/media/avatars/{avatar_id}.jpg" if avatar_id else None


def user_json(user: dict) -> dict:
    """The User the frontend stores: {id, username, avatarUrl}."""
    return {"id": user["id"], "username": user["username"],
            "avatarUrl": avatar_url(user.get("avatar_id"))}


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
                         admin_pin: str, clock=time.monotonic,
                         render_card: RenderCard | None = None) -> Blueprint:
    """All spec §4 endpoints; app.py registers the result with url_prefix="/api/v1".

    `clock` (seconds, monotonic) is injectable so tests can expire the admin lockout.
    `render_card` re-draws a card for host errata (None: errata answer 503).
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
        # An erratum re-renders the same file: the version makes it a new URL for every cache.
        version = f"?v={re.sub(r'[^0-9]', '', row['errata_at'])}" if row.get("errata_at") else ""
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
            "cardImageUrl": f"{API_PREFIX}/media/cards/{cid}.png{version}" if row["card_path"] else None,
            "artImageUrl": f"{API_PREFIX}/media/art/{cid}.png" if row["art_path"] else None,
            "createdAt": row["created_at"],
            "shared": row.get("shared_at") is not None,
            "poolEntryId": pool_entry_ids.get(cid),
            "errata": row.get("errata_at") is not None,
        }

    def set_view(row: dict, voter_id: str | None) -> dict:
        pool_entry_ids = storage.open_pool_entry_ids()
        cards = storage.set_cards(row["id"])
        flags = leader_flags(storage.vote_tally(row["id"]), [c["id"] for c in cards])
        owner_pick = storage.owner_vote(row["id"])
        owner = storage.get_user(row["user_id"])
        requests = storage.reroll_request_counts(row["id"])
        my_requests = storage.my_reroll_requests(voter_id, row["id"]) if voter_id else set()
        is_owner = voter_id == row["user_id"]
        return {
            "id": row["id"],
            "userId": row["user_id"],
            "username": owner["username"] if owner else "",
            "avatarUrl": avatar_url(owner["avatar_id"]) if owner else None,
            "eventId": row["event_id"],
            "commanderName": row["commander_name"],
            "prompt": row["prompt"],
            "status": row["status"],
            "lockedAt": row["locked_at"],
            "cmc": row["cmc"],
            "rarity": row["rarity"],
            # votes are weighted: the owner's own vote counts 2 (ownerVote marks that card).
            # Reroll requests: players asking the owner to reroll a broken version; canReroll is
            # only ever true for the owner.
            "cards": [{**card_view(c, pool_entry_ids), **flags[c["id"]],
                       "ownerVote": c["id"] == owner_pick,
                       "rerollRequests": requests.get(c["id"], 0),
                       "myRerollRequest": c["id"] in my_requests,
                       "canReroll": is_owner and storage.can_reroll(c["id"], voter_id)} for c in cards],
            "myVoteCardId": storage.user_vote(voter_id, row["id"]) if voter_id else None,
            "rerollRequestsNeeded": REROLL_REQUESTS_NEEDED,
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
        wtf_voters = storage.pool_wtf_voters(row["id"])
        pool_entry_ids = storage.open_pool_entry_ids()
        users: dict[str, dict] = {}

        def maker(user_id: str) -> dict:
            if user_id not in users:
                users[user_id] = storage.get_user(user_id) or {"username": "", "avatar_id": None}
            return users[user_id]

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
                "username": maker(e["user_id"])["username"],
                "avatarUrl": avatar_url(maker(e["user_id"])["avatar_id"]),
                "mine": e["user_id"] == viewer_id,
                "power": power_check(view["card"]),
                **ranking[e["id"]],
                "myMedal": my_medals.get(e["id"]),
                "wtfBy": [maker(v)["username"] for v in wtf_voters.get(e["id"], [])],
                "myWtf": viewer_id in wtf_voters.get(e["id"], []),
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
        return jsonify(user_json(storage.login(required_body().get("username"))))

    @bp.get("/me")
    def me():
        return jsonify(user_json(require_user()))

    # Profile pictures (docs/superpowers/specs/2026-10-14-profile-pictures-design.md)
    def avatar_path(avatar_id: str) -> Path:
        return data_dir / "avatars" / f"{avatar_id}.jpg"

    def replace_avatar(user_id: str, avatar_id: str | None) -> None:
        """Points the user at avatar_id and deletes the file it replaced."""
        previous = storage.set_avatar(user_id, avatar_id)
        if previous and previous != avatar_id:
            avatar_path(previous).unlink(missing_ok=True)

    @bp.put("/me/avatar")
    def set_my_avatar():
        """The raw image as the body; stored as a fresh 256x256 JPEG under a new id."""
        user = require_user()
        jpeg = process_avatar(request.get_data(cache=False))
        avatar_id = str(uuid.uuid4())
        path = avatar_path(avatar_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(jpeg)
        tmp.replace(path)
        try:
            replace_avatar(user["id"], avatar_id)
        except StorageError:
            path.unlink(missing_ok=True)
            raise
        return jsonify(user_json(storage.get_user(user["id"])))

    @bp.delete("/me/avatar")
    def delete_my_avatar():
        user = require_user()
        replace_avatar(user["id"], None)
        return jsonify(user_json(storage.get_user(user["id"])))

    @bp.get("/me/cards")
    def my_cards():
        user = require_user()
        pool_entry_ids = storage.open_pool_entry_ids()
        return jsonify([card_view(c, pool_entry_ids) for c in storage.list_user_cards(user["id"])])

    @bp.get("/me/sets/current")
    def my_current_set():
        user = require_user()
        # One per mana value: the player's three commanders (spec 2026-10-03 §4).
        return jsonify([set_view(row, user["id"]) for row in storage.current_sets(user["id"])])

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
            cmc = data.get("cmc")
            if type(cmc) is not int or cmc not in COMMANDER_CMCS:
                raise StorageError(400, "cmc must be 3, 4 or 5")
            # Three versions of one commander share these params; a broken rule is a 400
            # before the old draft at this mana value is abandoned.
            params = commander_params({**card_data, "name": commander_name}, cmc)

        with create_lock:
            check_pending_cap(user["id"], count)
            if count == 1:
                set_id = None
                cards = [storage.create_card(user["id"], prompt, card_data)]
            else:
                set_id = storage.create_set(user["id"], commander_name, prompt, params,
                                            cmc=cmc, rarity=params["rarity"])["id"]
                cards = [storage.create_card(user["id"], prompt, params, set_id=set_id, slot=slot)
                         for slot in (1, 2, 3)]
        # All at once, so a set's versions share one director call.
        gen_queue.enqueue_many([card["id"] for card in cards])
        return jsonify({"setId": set_id,
                        "cards": [card_view(storage.get_card(c["id"])) for c in cards]})

    @bp.post("/cards/<card_id>/reroll")
    def reroll(card_id):
        user = require_user()
        with create_lock:
            # Ownership, in-progress and locked-set problems are reported by reroll_card
            # (403/409) before the cap, which only applies when a new card would really be queued.
            if storage.can_reroll(card_id, user["id"]):
                check_pending_cap(user["id"], 1)
            new = storage.reroll_card(card_id, user["id"])
        gen_queue.enqueue(new["id"])
        return jsonify(card_view(storage.get_card(new["id"])))

    @bp.post("/cards/<card_id>/reroll-request")
    def reroll_request(card_id):
        """Ask the owner to reroll a broken version in the vote, or take the request back."""
        user = require_user()
        requested = required_body().get("requested")
        if not isinstance(requested, bool):
            raise StorageError(400, "requested must be true or false")
        set_id = storage.set_reroll_request(card_id, user["id"], requested)
        return jsonify(set_view(storage.get_set(set_id), user["id"]))

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
        return jsonify([{**card_view(c, pool_entry_ids), "username": c["username"],
                         "avatarUrl": avatar_url(c["avatar_id"])}
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

    @bp.post("/pools/wtf")
    def flag_pool_wtf():
        user = require_user()
        pool_id = storage.flag_pool_wtf(user["id"], entry_id_from_body())
        return jsonify(pool_view(pool_or_404(pool_id), user["id"]))

    @bp.post("/pools/wtf/clear")
    def clear_pool_wtf():
        user = require_user()
        pool_id = storage.clear_pool_wtf(user["id"], entry_id_from_body())
        return jsonify(pool_view(pool_or_404(pool_id), user["id"]))

    @bp.get("/queue_status")
    def queue_status():
        return jsonify(gen_queue.status())

    # ----- media -----
    @bp.get("/media/<kind>/<filename>")
    def media(kind, filename):
        not_found = StorageError(404, "Not found")
        if kind not in MEDIA_KINDS:
            raise not_found
        ext, mimetype = MEDIA_KINDS[kind]
        if not filename.endswith(ext):
            raise not_found
        stem = filename[:-len(ext)]
        try:
            canonical = str(uuid.UUID(stem))
        except ValueError:
            raise not_found from None
        if canonical != stem:  # rejects braces, urn: prefixes, hex-only and upper-case forms
            raise not_found
        path = data_dir / kind / f"{canonical}{ext}"
        if not path.is_file():
            raise not_found
        return send_file(path, mimetype=mimetype, max_age=3600)

    # ----- host errata (docs/superpowers/specs/2026-10-14-errata-design.md) -----
    def errata_item(row: dict, label: str = "", username: str = "", detail: str = "") -> dict:
        return {"card": card_view(row), "original": row.get("original_card"),
                "label": label or str((row.get("card") or {}).get("name") or ""),
                "username": username, "detail": detail}

    def owner_name(user_id: str) -> str:
        user = storage.get_user(user_id)
        return user["username"] if user else ""

    def errata_groups() -> list[dict]:
        """What the host may errata now: the cards being voted on, the last closed event's
        legal commanders (winners, or every tied version), and the Knowledge Pool."""
        groups = []
        event = storage.current_event()
        if event:
            items = [errata_item(c, s["commander_name"], owner_name(s["user_id"]), f"Version {c['slot']}")
                     for s in storage.locked_sets(event["id"]) for c in storage.set_cards(s["id"])]
            groups.append({"id": "vote", "title": f"Commander vote · {event['name']}", "items": items})
        closed = next((e for e in storage.list_events() if e["status"] == "closed"), None)
        if closed:
            items = []
            for s in storage.locked_sets(closed["id"]):
                cards = storage.set_cards(s["id"])
                flags = leader_flags(storage.vote_tally(s["id"]), [c["id"] for c in cards])
                items += [errata_item(c, s["commander_name"], owner_name(s["user_id"]),
                                      "Winner" if flags[c["id"]]["leader"] else "Tied")
                          for c in cards if flags[c["id"]]["leader"] or flags[c["id"]]["tied"]]
            groups.append({"id": "legal", "title": f"Legal commanders · {closed['name']}", "items": items})
        pool = storage.current_pool() or next(iter(storage.list_pools()), None)
        if pool:
            entries = storage.pool_entries(pool["id"])
            ranking = pool_ranking(entries, storage.pool_medal_counts(pool["id"]))
            items = []
            for e in sorted(entries, key=lambda e: ranking[e["id"]]["rank"]):
                rank = ranking[e["id"]]
                card = storage.get_card(e["card_id"])
                if card is None or (pool["status"] == "closed" and not rank["in"]):
                    continue
                points = f"{rank['points']} {'point' if rank['points'] == 1 else 'points'}"
                items.append(errata_item(card, "", owner_name(e["user_id"]),
                                         f"In · {points}" if rank["in"] else points))
            groups.append({"id": "pool", "title": f"Knowledge Pool · {pool['name']}", "items": items})
        return groups

    def errata_target(card_id: str) -> dict:
        card = storage.get_card(card_id)
        if card is None:
            raise StorageError(404, "Card not found")
        if card["status"] != "done" or not storage.errata_eligible(card_id):
            raise StorageError(409, NOT_ERRATABLE)
        return card

    def render_errata(row: dict, card: dict) -> str:
        """The card drawn on its stored art, as raw base64 PNG."""
        if render_card is None:
            raise StorageError(503, "Card rendering is not available")
        art = Path(row["art_path"]) if row.get("art_path") else None
        art_b64 = base64.b64encode(art.read_bytes()).decode() if art and art.is_file() else None
        rendered = render_card(card, art_b64)
        if not rendered:
            raise StorageError(500, "The card did not render")
        return rendered.split(",", 1)[1] if rendered.startswith("data:") else rendered

    def save_errata(row: dict, rendered: str, update: Callable[[], dict]) -> dict:
        """Writes the new image beside the old one, records the change, then swaps the image in."""
        path = Path(row["card_path"]) if row.get("card_path") else data_dir / "cards" / f"{row['id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".errata.tmp")
        tmp.write_bytes(base64.b64decode(rendered))
        try:
            updated = update()
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)
        return updated

    @bp.get("/admin/errata")
    def admin_errata():
        require_admin()
        return jsonify({"groups": errata_groups()})

    @bp.post("/admin/cards/<card_id>/errata")
    def admin_errata_card(card_id):
        """{description, power?, toughness?, preview?}: preview renders without saving."""
        require_admin()
        row = errata_target(card_id)
        data = required_body()
        card = errata_card_data(row["card"] or {}, data)
        rendered = render_errata(row, card)
        if data.get("preview") is True:
            return jsonify({"image": f"data:image/png;base64,{rendered}", "power": power_check(card)})
        updated = save_errata(row, rendered, lambda: storage.errata_card(card_id, card))
        return jsonify(errata_item(updated))

    @bp.post("/admin/cards/<card_id>/errata/revert")
    def admin_revert_errata(card_id):
        require_admin()
        row = errata_target(card_id)
        if row["original_card"] is None:
            return jsonify(errata_item(row))
        rendered = render_errata(row, row["original_card"])
        return jsonify(errata_item(save_errata(row, rendered, lambda: storage.revert_errata(card_id))))

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
