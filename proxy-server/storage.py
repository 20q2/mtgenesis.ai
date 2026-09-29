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
from datetime import datetime, timezone
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


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
    def create_card(self, user_id: str, prompt: str, card_params: dict,
                    set_id: str | None = None, slot: int | None = None) -> dict:
        """New card with status 'queued'."""
        raise NotImplementedError

    def get_card(self, card_id: str) -> dict | None:
        raise NotImplementedError

    def update_card(self, card_id: str, **fields) -> None:
        """Accepts status, text_ready, art_ready, card (dict), art_path, card_path, error, finished_at."""
        raise NotImplementedError

    def list_user_cards(self, user_id: str) -> list[dict]:
        """Newest first, replaced cards included."""
        raise NotImplementedError

    def count_pending(self, user_id: str) -> int:
        """Cards in queued|generating|rendering for this user (spec §5 per-user cap)."""
        raise NotImplementedError

    def unfinished_card_ids(self) -> list[str]:
        """Cards in queued|generating|rendering for all users (startup recovery)."""
        raise NotImplementedError

    # ----- sets -----
    # Row keys: id, user_id, event_id, commander_name, prompt, card_params, status, created_at, locked_at
    def create_set(self, user_id: str, commander_name: str, prompt: str, card_params: dict) -> dict:
        """New draft set; any existing draft of this user becomes 'abandoned'.
        commander_name trimmed, 1-40 chars, else 400."""
        raise NotImplementedError

    def get_set(self, set_id: str) -> dict | None:
        raise NotImplementedError

    def current_set(self, user_id: str) -> dict | None:
        """The user's draft, else their set locked in the open event, else None."""
        raise NotImplementedError

    def set_cards(self, set_id: str) -> list[dict]:
        """Current (replaced = 0) cards ordered by slot."""
        raise NotImplementedError

    def reroll_card(self, card_id: str, user_id: str) -> dict:
        """Mark the card replaced and create a new queued card in the same slot with the same
        prompt/card_params. 403 not owner, 400 free-play card, 409 set not draft or card still in progress."""
        raise NotImplementedError

    def lock_set(self, set_id: str, user_id: str, commander_name: str | None = None) -> dict:
        """Requires open event, all 3 current cards done, owner, non-empty commander name,
        and no other locked set by this user in the open event. Sets event_id and locked_at."""
        raise NotImplementedError

    def unlock_set(self, set_id: str, user_id: str) -> dict:
        """Only while its event is open. Deletes the set's votes; back to draft with event_id NULL."""
        raise NotImplementedError

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
        raise NotImplementedError

    # ----- votes -----
    def cast_vote(self, voter_id: str, set_id: str, card_id: str) -> None:
        """One vote per (voter, set); a new vote overwrites. 400 card not a current card of the set,
        409 set not locked, 409 "Voting is closed" when the event is closed."""
        raise NotImplementedError

    def vote_tally(self, set_id: str) -> dict[str, int]:
        """card_id -> count, only cards that have votes."""
        raise NotImplementedError

    def user_vote(self, voter_id: str, set_id: str) -> str | None:
        raise NotImplementedError


def leader_flags(tally: dict[str, int], card_ids: list[str]) -> dict[str, dict]:
    """{card_id: {"votes": int, "leader": bool, "tied": bool}} for every id in card_ids.
    A top count > 0 held by one card -> leader; held by several -> all tied; no votes -> neither."""
    raise NotImplementedError
