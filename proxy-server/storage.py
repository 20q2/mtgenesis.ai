"""
SQLite persistence for AI Night: users, events, commander sets, cards and votes.

Spec: docs/superpowers/specs/2026-09-28-ai-night-design.md §3.
Rows are returned as plain dicts with snake_case keys; JSON columns
(`card_params`, `card`) are decoded to dicts. IDs are uuid4 strings and
timestamps are UTC ISO-8601 strings.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

USERNAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,24}$")
COMMANDER_NAME_MAX = 40
EVENT_NAME_MAX = 80
PENDING_STATUSES = ("queued", "generating", "rendering")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    username    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS users_username_ci ON users(lower(username));

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    created_at  TEXT NOT NULL,
    closed_at   TEXT
);
-- at most one open event
CREATE UNIQUE INDEX IF NOT EXISTS events_one_open ON events(status) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS sets (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    event_id          TEXT,
    commander_name    TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    card_params_json  TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('draft', 'locked', 'abandoned')),
    created_at        TEXT NOT NULL,
    locked_at         TEXT
);
CREATE INDEX IF NOT EXISTS sets_user ON sets(user_id, status);
CREATE INDEX IF NOT EXISTS sets_event ON sets(event_id, status);

CREATE TABLE IF NOT EXISTS cards (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    set_id            TEXT,
    slot              INTEGER,
    replaced          INTEGER NOT NULL DEFAULT 0,
    prompt            TEXT NOT NULL,
    card_params_json  TEXT NOT NULL,
    card_json         TEXT,
    art_path          TEXT,
    card_path         TEXT,
    status            TEXT NOT NULL
                      CHECK (status IN ('queued', 'generating', 'rendering', 'done', 'failed')),
    text_ready        INTEGER NOT NULL DEFAULT 0,
    art_ready         INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    created_at        TEXT NOT NULL,
    finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS cards_user ON cards(user_id);
CREATE INDEX IF NOT EXISTS cards_set ON cards(set_id, replaced);
CREATE INDEX IF NOT EXISTS cards_status ON cards(status);

CREATE TABLE IF NOT EXISTS votes (
    voter_id    TEXT NOT NULL,
    set_id      TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (voter_id, set_id)
);
CREATE INDEX IF NOT EXISTS votes_set ON votes(set_id);
"""


_clock_lock = threading.Lock()
_last_now = datetime.min.replace(tzinfo=timezone.utc)


def _now() -> str:
    """UTC ISO-8601 timestamp, strictly increasing within this process.

    The Windows wall clock can return the same value for calls made in quick
    succession, which would make "newest first" / "oldest lock first" orderings
    ambiguous; bump by a microsecond on collisions."""
    global _last_now
    with _clock_lock:
        now = datetime.now(timezone.utc)
        if now <= _last_now:
            now = _last_now + timedelta(microseconds=1)
        _last_now = now
        return now.isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def clean_commander_name(name) -> str:
    """Trimmed commander name, 1-40 characters, else StorageError 400."""
    if not isinstance(name, str) or not name.strip():
        raise StorageError(400, "Commander name is required")
    name = name.strip()
    if len(name) > COMMANDER_NAME_MAX:
        raise StorageError(400, f"Commander name must be at most {COMMANDER_NAME_MAX} characters")
    return name


class StorageError(Exception):
    """A rule violation that maps directly to an HTTP status (400/403/404/409)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Storage:
    def __init__(self, db_path: str | Path):
        """Open (creating if needed) the database; WAL mode; one connection per thread."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)

    # ----- connection helpers -----
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # isolation_level=None: autocommit; write transactions are opened explicitly
            # with BEGIN IMMEDIATE so concurrent writers wait on the busy timeout
            # instead of failing with "database is locked" on a read->write lock upgrade.
            conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self):
        """A serialized write transaction (BEGIN IMMEDIATE ... COMMIT, ROLLBACK on error)."""
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _one(self, sql: str, params: tuple = ()) -> dict | None:
        row = self._conn().execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def _all(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self._conn().execute(sql, params).fetchall()]

    # ----- users -----
    def login(self, username: str) -> dict:
        """Create-or-get by case-insensitive username. Trimmed, 1-24 chars of [A-Za-z0-9 _-], else 400.
        Returns {id, username} with the originally stored casing."""
        if not isinstance(username, str):
            raise StorageError(400, "Username is required")
        name = username.strip()
        if not USERNAME_RE.fullmatch(name):
            raise StorageError(
                400, "Username must be 1-24 characters: letters, digits, spaces, _ or -")
        with self._tx() as conn:
            row = conn.execute(
                "SELECT id, username FROM users WHERE lower(username) = lower(?)", (name,)
            ).fetchone()
            if row is not None:
                return dict(row)
            user = {"id": _new_id(), "username": name}
            conn.execute("INSERT INTO users (id, username, created_at) VALUES (?, ?, ?)",
                         (user["id"], name, _now()))
        return user

    def get_user(self, user_id: str) -> dict | None:
        return self._one("SELECT id, username FROM users WHERE id = ?", (user_id,))

    # ----- cards -----
    # Row keys: id, user_id, set_id, slot, replaced, prompt, card_params, card, art_path,
    # card_path, status, text_ready, art_ready, error, created_at, finished_at
    _CARD_COLS = ("id, user_id, set_id, slot, replaced, prompt, card_params_json, card_json, "
                  "art_path, card_path, status, text_ready, art_ready, error, created_at, "
                  "finished_at")
    _CARD_UPDATABLE = frozenset({"status", "text_ready", "art_ready", "card", "art_path",
                                 "card_path", "error", "finished_at"})

    @staticmethod
    def _decode_card(row: dict | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        params, card = d.pop("card_params_json"), d.pop("card_json")
        d["card_params"] = json.loads(params) if params is not None else None
        d["card"] = json.loads(card) if card is not None else None
        return d

    @staticmethod
    def _insert_card(conn: sqlite3.Connection, card_id: str, user_id: str, prompt: str,
                     params_json: str, set_id: str | None, slot: int | None) -> None:
        conn.execute(
            "INSERT INTO cards (id, user_id, set_id, slot, replaced, prompt, card_params_json, "
            "status, text_ready, art_ready, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?, ?, 'queued', 0, 0, ?)",
            (card_id, user_id, set_id, slot, prompt, params_json, _now()))

    def create_card(self, user_id: str, prompt: str, card_params: dict,
                    set_id: str | None = None, slot: int | None = None) -> dict:
        """New card with status 'queued'."""
        card_id = _new_id()
        with self._tx() as conn:
            self._insert_card(conn, card_id, user_id, prompt, _dumps(card_params), set_id, slot)
        return self.get_card(card_id)

    def get_card(self, card_id: str) -> dict | None:
        return self._decode_card(
            self._one(f"SELECT {self._CARD_COLS} FROM cards WHERE id = ?", (card_id,)))

    def update_card(self, card_id: str, **fields) -> None:
        """Accepts status, text_ready, art_ready, card (dict), art_path, card_path, error, finished_at."""
        unknown = set(fields) - self._CARD_UPDATABLE
        if unknown:
            raise ValueError(f"update_card: unsupported fields {sorted(unknown)}")
        if not fields:
            return
        cols, values = [], []
        for key, value in fields.items():
            if key == "card":
                key, value = "card_json", (_dumps(value) if value is not None else None)
            elif key in ("text_ready", "art_ready"):
                value = int(bool(value))
            elif isinstance(value, Path):
                value = str(value)
            cols.append(f"{key} = ?")
            values.append(value)
        with self._tx() as conn:
            conn.execute(f"UPDATE cards SET {', '.join(cols)} WHERE id = ?", (*values, card_id))

    def list_user_cards(self, user_id: str) -> list[dict]:
        """Newest first, replaced cards included."""
        rows = self._all(f"SELECT {self._CARD_COLS} FROM cards WHERE user_id = ? "
                         "ORDER BY created_at DESC, rowid DESC", (user_id,))
        return [self._decode_card(r) for r in rows]

    def count_pending(self, user_id: str) -> int:
        """Cards in queued|generating|rendering for this user (spec §5 per-user cap)."""
        row = self._conn().execute(
            "SELECT COUNT(*) FROM cards WHERE user_id = ? AND status IN (?, ?, ?)",
            (user_id, *PENDING_STATUSES)).fetchone()
        return row[0]

    def unfinished_card_ids(self) -> list[str]:
        """Cards in queued|generating|rendering for all users (startup recovery)."""
        rows = self._conn().execute(
            "SELECT id FROM cards WHERE status IN (?, ?, ?) ORDER BY created_at, rowid",
            PENDING_STATUSES).fetchall()
        return [r["id"] for r in rows]

    # ----- sets -----
    # Row keys: id, user_id, event_id, commander_name, prompt, card_params, status, created_at, locked_at
    _SET_COLS = ("id, user_id, event_id, commander_name, prompt, card_params_json, status, "
                 "created_at, locked_at")

    @staticmethod
    def _decode_set(row: dict | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        params = d.pop("card_params_json")
        d["card_params"] = json.loads(params) if params is not None else None
        return d

    def create_set(self, user_id: str, commander_name: str, prompt: str, card_params: dict) -> dict:
        """New draft set; any existing draft of this user becomes 'abandoned'.
        commander_name trimmed, 1-40 chars, else 400."""
        name = clean_commander_name(commander_name)
        set_id = _new_id()
        with self._tx() as conn:
            conn.execute("UPDATE sets SET status = 'abandoned' WHERE user_id = ? AND status = 'draft'",
                         (user_id,))
            conn.execute(
                "INSERT INTO sets (id, user_id, event_id, commander_name, prompt, card_params_json, "
                "status, created_at) VALUES (?, ?, NULL, ?, ?, ?, 'draft', ?)",
                (set_id, user_id, name, prompt, _dumps(card_params), _now()))
        return self.get_set(set_id)

    def get_set(self, set_id: str) -> dict | None:
        return self._decode_set(
            self._one(f"SELECT {self._SET_COLS} FROM sets WHERE id = ?", (set_id,)))

    def current_set(self, user_id: str) -> dict | None:
        """The user's draft, else their set locked in the open event, else None."""
        row = self._one(
            f"SELECT {self._SET_COLS} FROM sets WHERE user_id = ? AND status = 'draft' "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1", (user_id,))
        if row is None:
            row = self._one(
                f"SELECT {self._SET_COLS} FROM sets WHERE user_id = ? AND status = 'locked' "
                "AND event_id IN (SELECT id FROM events WHERE status = 'open') "
                "ORDER BY locked_at DESC LIMIT 1", (user_id,))
        return self._decode_set(row)

    def set_cards(self, set_id: str) -> list[dict]:
        """Current (replaced = 0) cards ordered by slot."""
        rows = self._all(f"SELECT {self._CARD_COLS} FROM cards WHERE set_id = ? AND replaced = 0 "
                         "ORDER BY slot, rowid", (set_id,))
        return [self._decode_card(r) for r in rows]

    def reroll_card(self, card_id: str, user_id: str) -> dict:
        """Mark the card replaced and create a new queued card in the same slot with the same
        prompt/card_params. 403 not owner, 400 free-play card, 409 set not draft or card still in progress."""
        new_id = _new_id()
        with self._tx() as conn:
            card = conn.execute(
                "SELECT user_id, set_id, slot, replaced, prompt, card_params_json, status "
                "FROM cards WHERE id = ?", (card_id,)).fetchone()
            if card is None:
                raise StorageError(404, "Card not found")
            if card["user_id"] != user_id:
                raise StorageError(403, "You can only reroll your own cards")
            if card["set_id"] is None:
                raise StorageError(400, "Only commander set cards can be rerolled")
            set_row = conn.execute("SELECT status FROM sets WHERE id = ?",
                                   (card["set_id"],)).fetchone()
            if set_row is None or set_row["status"] != "draft":
                raise StorageError(409, "Rerolls are only allowed on draft sets")
            if card["replaced"]:
                raise StorageError(409, "This card has already been rerolled")
            if card["status"] in PENDING_STATUSES:
                raise StorageError(409, "This card is still being generated")
            conn.execute("UPDATE cards SET replaced = 1 WHERE id = ?", (card_id,))
            self._insert_card(conn, new_id, user_id, card["prompt"], card["card_params_json"],
                              card["set_id"], card["slot"])
        return self.get_card(new_id)

    def lock_set(self, set_id: str, user_id: str, commander_name: str | None = None) -> dict:
        """Requires open event, all 3 current cards done, owner, non-empty commander name,
        and no other locked set by this user in the open event. Sets event_id and locked_at."""
        with self._tx() as conn:
            s = conn.execute("SELECT user_id, status, commander_name FROM sets WHERE id = ?",
                             (set_id,)).fetchone()
            if s is None:
                raise StorageError(404, "Set not found")
            if s["user_id"] != user_id:
                raise StorageError(403, "You can only lock your own set")
            if s["status"] != "draft":
                raise StorageError(409, "Only a draft set can be locked")
            event = conn.execute("SELECT id FROM events WHERE status = 'open'").fetchone()
            if event is None:
                raise StorageError(409, "No event open — ask the host")
            statuses = [r["status"] for r in conn.execute(
                "SELECT status FROM cards WHERE set_id = ? AND replaced = 0", (set_id,))]
            if len(statuses) != 3 or any(st != "done" for st in statuses):
                raise StorageError(409, "All 3 cards must be finished before locking")
            if conn.execute(
                    "SELECT 1 FROM sets WHERE user_id = ? AND event_id = ? AND status = 'locked'",
                    (user_id, event["id"])).fetchone():
                raise StorageError(409, "You already have a set locked in this event — unlock it first")
            name = clean_commander_name(
                s["commander_name"] if commander_name is None else commander_name)
            conn.execute(
                "UPDATE sets SET status = 'locked', event_id = ?, locked_at = ?, commander_name = ? "
                "WHERE id = ?", (event["id"], _now(), name, set_id))
        return self.get_set(set_id)

    def unlock_set(self, set_id: str, user_id: str) -> dict:
        """Only while its event is open. Deletes the set's votes; back to draft with event_id NULL.
        Any other draft of the owner is abandoned so they keep at most one draft."""
        with self._tx() as conn:
            s = conn.execute("SELECT user_id, status, event_id FROM sets WHERE id = ?",
                             (set_id,)).fetchone()
            if s is None:
                raise StorageError(404, "Set not found")
            if s["user_id"] != user_id:
                raise StorageError(403, "You can only unlock your own set")
            if s["status"] != "locked":
                raise StorageError(409, "Set is not locked")
            event = conn.execute("SELECT status FROM events WHERE id = ?",
                                 (s["event_id"],)).fetchone()
            if event is None or event["status"] != "open":
                raise StorageError(409, "The event is closed")
            conn.execute("DELETE FROM votes WHERE set_id = ?", (set_id,))
            conn.execute("UPDATE sets SET status = 'abandoned' "
                         "WHERE user_id = ? AND status = 'draft' AND id != ?", (user_id, set_id))
            conn.execute("UPDATE sets SET status = 'draft', event_id = NULL, locked_at = NULL "
                         "WHERE id = ?", (set_id,))
        return self.get_set(set_id)

    # ----- events -----
    # Row keys: id, name, status, created_at, closed_at
    def create_event(self, name: str) -> dict:
        """409 if an event is already open."""
        if not isinstance(name, str) or not name.strip():
            raise StorageError(400, "Event name is required")
        name = name.strip()
        if len(name) > EVENT_NAME_MAX:
            raise StorageError(400, f"Event name must be at most {EVENT_NAME_MAX} characters")
        event_id = _new_id()
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM events WHERE status = 'open'").fetchone():
                raise StorageError(409, "An event is already open")
            conn.execute(
                "INSERT INTO events (id, name, status, created_at) VALUES (?, ?, 'open', ?)",
                (event_id, name, _now()))
        return self.get_event(event_id)

    def close_event(self, event_id: str) -> dict:
        """409 if already closed, 404 if unknown."""
        with self._tx() as conn:
            row = conn.execute("SELECT status FROM events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                raise StorageError(404, "Event not found")
            if row["status"] != "open":
                raise StorageError(409, "Event is already closed")
            conn.execute("UPDATE events SET status = 'closed', closed_at = ? WHERE id = ?",
                         (_now(), event_id))
        return self.get_event(event_id)

    _EVENT_COLS = "id, name, status, created_at, closed_at"

    def current_event(self) -> dict | None:
        return self._one(f"SELECT {self._EVENT_COLS} FROM events WHERE status = 'open'")

    def get_event(self, event_id: str) -> dict | None:
        return self._one(f"SELECT {self._EVENT_COLS} FROM events WHERE id = ?", (event_id,))

    def list_events(self) -> list[dict]:
        """Newest first."""
        return self._all(
            f"SELECT {self._EVENT_COLS} FROM events ORDER BY created_at DESC, rowid DESC")

    def locked_sets(self, event_id: str) -> list[dict]:
        """Sets locked in this event, oldest lock first."""
        rows = self._all(f"SELECT {self._SET_COLS} FROM sets WHERE event_id = ? "
                         "AND status = 'locked' ORDER BY locked_at, rowid", (event_id,))
        return [self._decode_set(r) for r in rows]

    # ----- votes -----
    def cast_vote(self, voter_id: str, set_id: str, card_id: str) -> None:
        """One vote per (voter, set); a new vote overwrites. 400 card not a current card of the set,
        409 set not locked, 409 "Voting is closed" when the event is closed."""
        with self._tx() as conn:
            s = conn.execute("SELECT status, event_id FROM sets WHERE id = ?", (set_id,)).fetchone()
            if s is None:
                raise StorageError(404, "Set not found")
            if s["status"] != "locked":
                raise StorageError(409, "This set is not locked in")
            event = conn.execute("SELECT status FROM events WHERE id = ?",
                                 (s["event_id"],)).fetchone()
            if event is None or event["status"] != "open":
                raise StorageError(409, "Voting is closed")
            card = conn.execute("SELECT set_id, replaced FROM cards WHERE id = ?",
                                (card_id,)).fetchone()
            if card is None or card["set_id"] != set_id or card["replaced"]:
                raise StorageError(400, "That card is not one of this set's versions")
            # Upsert: a double-click or two tabs racing still leave exactly one row.
            conn.execute(
                "INSERT INTO votes (voter_id, set_id, card_id, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(voter_id, set_id) DO UPDATE SET "
                "card_id = excluded.card_id, created_at = excluded.created_at",
                (voter_id, set_id, card_id, _now()))

    def vote_tally(self, set_id: str) -> dict[str, int]:
        """card_id -> count, only cards that have votes."""
        rows = self._conn().execute(
            "SELECT card_id, COUNT(*) AS n FROM votes WHERE set_id = ? GROUP BY card_id",
            (set_id,)).fetchall()
        return {r["card_id"]: r["n"] for r in rows}

    def user_vote(self, voter_id: str, set_id: str) -> str | None:
        row = self._conn().execute(
            "SELECT card_id FROM votes WHERE voter_id = ? AND set_id = ?",
            (voter_id, set_id)).fetchone()
        return row["card_id"] if row is not None else None


def leader_flags(tally: dict[str, int], card_ids: list[str]) -> dict[str, dict]:
    """{card_id: {"votes": int, "leader": bool, "tied": bool}} for every id in card_ids.
    A top count > 0 held by one card -> leader; held by several -> all tied; no votes -> neither."""
    votes = {cid: int(tally.get(cid, 0)) for cid in card_ids}
    top = max(votes.values(), default=0)
    top_ids = [cid for cid, n in votes.items() if n == top] if top > 0 else []
    return {
        cid: {"votes": n,
              "leader": len(top_ids) == 1 and cid in top_ids,
              "tied": len(top_ids) > 1 and cid in top_ids}
        for cid, n in votes.items()
    }
