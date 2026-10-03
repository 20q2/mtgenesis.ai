"""
SQLite persistence for AI Night: users, events, commander sets, cards and votes,
plus the Knowledge Pool (docs/superpowers/specs/2026-09-29-knowledge-pool-design.md).

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
    finished_at       TEXT,
    shared_at         TEXT,  -- set while the card is shared to the gallery's Community tab
    brief_json        TEXT   -- the director's brief (director.py); never shown to players
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

-- Knowledge Pool (docs/superpowers/specs/2026-09-29-knowledge-pool-design.md)
CREATE TABLE IF NOT EXISTS pools (
    id                    TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    max_entries_per_user  INTEGER NOT NULL,
    created_at            TEXT NOT NULL,
    closed_at             TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS pools_one_open ON pools(status) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS pool_entries (
    id          TEXT PRIMARY KEY,
    pool_id     TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (pool_id, card_id)
);
CREATE INDEX IF NOT EXISTS pool_entries_user ON pool_entries(pool_id, user_id);

-- Each voter gives at most one gold, silver and bronze per pool, and one medal per card.
CREATE TABLE IF NOT EXISTS pool_medals (
    voter_id    TEXT NOT NULL,
    pool_id     TEXT NOT NULL,
    entry_id    TEXT NOT NULL,
    medal       TEXT NOT NULL CHECK (medal IN ('gold', 'silver', 'bronze')),
    created_at  TEXT NOT NULL,
    UNIQUE (voter_id, pool_id, medal),
    UNIQUE (voter_id, entry_id)
);
CREATE INDEX IF NOT EXISTS pool_medals_pool ON pool_medals(pool_id);
"""

POOL_NAME_MAX = 80
POOL_ENTRY_CAP_MAX = 10
POOL_DEFAULT_ENTRIES = 3
MEDAL_POINTS = {"gold": 3, "silver": 2, "bronze": 1}
MONO_COLORS = ("W", "U", "B", "R", "G")
# The slot/ban pool's tables (replaced 2026-10-01); dropped on startup only while empty.
_OLD_POOL_TABLES = ("pools", "pool_slots", "pool_entries", "pool_medals", "pool_bans")


def card_colors(card: dict) -> set[str]:
    """The card's colors among WUBRG: its `colors` list, else the symbols in its mana cost."""
    card = card or {}
    colors = {c for c in (card.get("colors") or []) if c in MONO_COLORS}
    if not colors:
        colors = set(re.findall(r"[WUBRG]", (card.get("manaCost") or "").upper()))
    return colors


def pool_card_eligible(card: dict) -> bool:
    """Only colorless or mono-colored cards can enter the Knowledge Pool."""
    return len(card_colors(card)) <= 1


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


def _rarity_taken(rarity: str, cmc: int) -> str:
    return f"You already have a {rarity.capitalize()} commander ({cmc} CMC)"


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
        self._drop_old_pool_tables(conn)
        conn.executescript(_SCHEMA)
        self._migrate(conn)

    @staticmethod
    def _drop_old_pool_tables(conn: sqlite3.Connection) -> None:
        """The slot/ban Knowledge Pool's tables give way to the voted list's, but only while
        they are empty: a pool with data is never dropped silently."""
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        entry_cols = {r[1] for r in conn.execute("PRAGMA table_info(pool_entries)")}
        # Any trace of the old layout counts, so a drop interrupted halfway is finished next start.
        if not ({"pool_slots", "pool_bans"} & tables or "slot_id" in entry_cols):
            return
        for table in _OLD_POOL_TABLES:
            if table in tables and conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                raise RuntimeError(
                    f"Old Knowledge Pool tables (pool_slots, pool_bans) hold data ({table} has "
                    "rows); migrate them by hand")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table in _OLD_POOL_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Columns added after a table first shipped (CREATE TABLE IF NOT EXISTS skips them)."""
        card_cols = {r[1] for r in conn.execute("PRAGMA table_info(cards)")}
        if "shared_at" not in card_cols:
            conn.execute("ALTER TABLE cards ADD COLUMN shared_at TEXT")
        if "brief_json" not in card_cols:
            conn.execute("ALTER TABLE cards ADD COLUMN brief_json TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS cards_shared ON cards(shared_at) "
                     "WHERE shared_at IS NOT NULL")
        # A set is one commander (docs/superpowers/specs/2026-10-03-commander-rules-design.md);
        # sets made before that have NULL cmc/rarity and are "legacy".
        set_cols = {r[1] for r in conn.execute("PRAGMA table_info(sets)")}
        if "cmc" not in set_cols:
            conn.execute("ALTER TABLE sets ADD COLUMN cmc INTEGER")
        if "rarity" not in set_cols:
            conn.execute("ALTER TABLE sets ADD COLUMN rarity TEXT")

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
    # card_path, status, text_ready, art_ready, error, created_at, finished_at, shared_at, brief
    _CARD_COLS = ("id, user_id, set_id, slot, replaced, prompt, card_params_json, card_json, "
                  "art_path, card_path, status, text_ready, art_ready, error, created_at, "
                  "finished_at, shared_at, brief_json")
    _CARD_UPDATABLE = frozenset({"status", "text_ready", "art_ready", "card", "art_path",
                                 "card_path", "error", "finished_at"})

    @staticmethod
    def _decode_card(row: dict | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        params, card, brief = d.pop("card_params_json"), d.pop("card_json"), d.pop("brief_json")
        d["card_params"] = json.loads(params) if params is not None else None
        d["card"] = json.loads(card) if card is not None else None
        d["brief"] = json.loads(brief) if brief is not None else None
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

    def set_card_brief(self, card_id: str, brief: dict | None) -> None:
        """Store the director's brief for a card (None clears it)."""
        with self._tx() as conn:
            conn.execute("UPDATE cards SET brief_json = ? WHERE id = ?",
                         (_dumps(brief) if brief is not None else None, card_id))

    def set_card_shared(self, card_id: str, user_id: str, shared: bool) -> dict:
        """Share a finished card to the Community tab, or take it back. Sharing an already
        shared card keeps its original time. 404 unknown, 403 not owner, 409 not finished."""
        with self._tx() as conn:
            card = conn.execute("SELECT user_id, status, shared_at FROM cards WHERE id = ?",
                                (card_id,)).fetchone()
            if card is None:
                raise StorageError(404, "Card not found")
            if card["user_id"] != user_id:
                raise StorageError(403, "You can only share your own cards")
            if card["status"] != "done":
                raise StorageError(409, "Only finished cards can be shared")
            shared_at = (card["shared_at"] or _now()) if shared else None
            conn.execute("UPDATE cards SET shared_at = ? WHERE id = ?", (shared_at, card_id))
        return self.get_card(card_id)

    def list_shared_cards(self, limit: int = 200) -> list[dict]:
        """Shared cards with their maker's `username`, most recently shared first."""
        cols = ", ".join(f"c.{c.strip()}" for c in self._CARD_COLS.split(","))
        rows = self._all(f"SELECT {cols}, u.username FROM cards c JOIN users u ON u.id = c.user_id "
                         "WHERE c.shared_at IS NOT NULL ORDER BY c.shared_at DESC, c.rowid DESC "
                         "LIMIT ?", (limit,))
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
    # Row keys: id, user_id, event_id, commander_name, prompt, card_params, status, created_at,
    # locked_at, cmc, rarity. A set is one commander: three versions at one mana value. Sets with
    # cmc NULL predate that (legacy): they are listed but never count toward the rules below.
    _SET_COLS = ("id, user_id, event_id, commander_name, prompt, card_params_json, status, "
                 "created_at, locked_at, cmc, rarity")
    # A user's live commanders: drafts, and sets locked in the open event.
    _LIVE = ("(status = 'draft' OR (status = 'locked' AND "
             "event_id IN (SELECT id FROM events WHERE status = 'open')))")

    @staticmethod
    def _decode_set(row: dict | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        params = d.pop("card_params_json")
        d["card_params"] = json.loads(params) if params is not None else None
        return d

    def create_set(self, user_id: str, commander_name: str, prompt: str, card_params: dict, *,
                   cmc: int, rarity: str) -> dict:
        """New draft commander at this mana value; the user's draft at the same mana value (and
        any legacy draft) becomes 'abandoned'. commander_name trimmed, 1-40 chars, else 400.
        409 when the user's commander at this mana value is locked in the open event, or another
        of their commanders locked in the open event already has this rarity."""
        name = clean_commander_name(commander_name)
        set_id = _new_id()
        with self._tx() as conn:
            if conn.execute(
                    "SELECT 1 FROM sets WHERE user_id = ? AND cmc = ? AND status = 'locked' "
                    "AND event_id IN (SELECT id FROM events WHERE status = 'open')",
                    (user_id, cmc)).fetchone():
                raise StorageError(409, f"Your {cmc} CMC commander is locked in — unlock it to start over")
            # Only a locked commander holds its rarity: drafts may share one while the player
            # reassigns them, and lock_set enforces one of each.
            taken = conn.execute(
                "SELECT cmc FROM sets WHERE user_id = ? AND rarity = ? AND cmc IS NOT NULL "
                "AND cmc != ? AND status = 'locked' "
                "AND event_id IN (SELECT id FROM events WHERE status = 'open') LIMIT 1",
                (user_id, rarity, cmc)).fetchone()
            if taken:
                raise StorageError(409, _rarity_taken(rarity, taken["cmc"]))
            conn.execute("UPDATE sets SET status = 'abandoned' WHERE user_id = ? AND status = 'draft' "
                         "AND (cmc = ? OR cmc IS NULL)", (user_id, cmc))
            conn.execute(
                "INSERT INTO sets (id, user_id, event_id, commander_name, prompt, card_params_json, "
                "status, created_at, cmc, rarity) VALUES (?, ?, NULL, ?, ?, ?, 'draft', ?, ?, ?)",
                (set_id, user_id, name, prompt, _dumps(card_params), _now(), cmc, rarity))
        return self.get_set(set_id)

    def get_set(self, set_id: str) -> dict | None:
        return self._decode_set(
            self._one(f"SELECT {self._SET_COLS} FROM sets WHERE id = ?", (set_id,)))

    def current_sets(self, user_id: str) -> list[dict]:
        """Per mana value, the user's newest draft, else their commander locked in the open
        event; in mana value order. Legacy sets are left out."""
        rows = self._all(
            f"SELECT {self._SET_COLS} FROM sets WHERE user_id = ? AND cmc IS NOT NULL "
            f"AND {self._LIVE} ORDER BY status = 'locked', created_at DESC, rowid DESC", (user_id,))
        newest: dict[int, dict] = {}
        for row in rows:  # drafts come first, newest first
            newest.setdefault(row["cmc"], row)
        return [self._decode_set(newest[cmc]) for cmc in sorted(newest)]

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
        """Requires open event, all 3 current cards done, owner, non-empty commander name, a
        non-legacy set, and no other commander of this user locked in the open event at the same
        mana value or with the same rarity. Sets event_id and locked_at."""
        with self._tx() as conn:
            s = conn.execute("SELECT user_id, status, commander_name, cmc, rarity FROM sets "
                             "WHERE id = ?", (set_id,)).fetchone()
            if s is None:
                raise StorageError(404, "Set not found")
            if s["user_id"] != user_id:
                raise StorageError(403, "You can only lock your own set")
            if s["status"] != "draft":
                raise StorageError(409, "Only a draft set can be locked")
            if s["cmc"] is None:
                raise StorageError(409, "This set was made under the old rules — start a new commander")
            event = conn.execute("SELECT id FROM events WHERE status = 'open'").fetchone()
            if event is None:
                raise StorageError(409, "No event open — ask the host")
            statuses = [r["status"] for r in conn.execute(
                "SELECT status FROM cards WHERE set_id = ? AND replaced = 0", (set_id,))]
            if len(statuses) != 3 or any(st != "done" for st in statuses):
                raise StorageError(409, "All 3 cards must be finished before locking")
            for other in conn.execute(
                    "SELECT cmc, rarity FROM sets WHERE user_id = ? AND event_id = ? "
                    "AND status = 'locked' AND cmc IS NOT NULL", (user_id, event["id"])):
                if other["cmc"] == s["cmc"]:
                    raise StorageError(409, f"Your {s['cmc']} CMC commander is already locked in "
                                            "— unlock it first")
                if other["rarity"] == s["rarity"]:
                    raise StorageError(409, _rarity_taken(s["rarity"], other["cmc"]))
            name = clean_commander_name(
                s["commander_name"] if commander_name is None else commander_name)
            conn.execute(
                "UPDATE sets SET status = 'locked', event_id = ?, locked_at = ?, commander_name = ? "
                "WHERE id = ?", (event["id"], _now(), name, set_id))
        return self.get_set(set_id)

    def unlock_set(self, set_id: str, user_id: str) -> dict:
        """Only while its event is open. Deletes the set's votes; back to draft with event_id NULL.
        The owner's other draft at the same mana value is abandoned (one draft per mana value)."""
        with self._tx() as conn:
            s = conn.execute("SELECT user_id, status, event_id, cmc FROM sets WHERE id = ?",
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
                         "WHERE user_id = ? AND status = 'draft' AND cmc IS ? AND id != ?",
                         (user_id, s["cmc"], set_id))
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
        """Sets locked in this event by mana value (legacy sets last), then oldest lock first."""
        rows = self._all(f"SELECT {self._SET_COLS} FROM sets WHERE event_id = ? "
                         "AND status = 'locked' ORDER BY cmc IS NULL, cmc, locked_at, rowid",
                         (event_id,))
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
        """card_id -> votes, only cards that have votes. The owner's own vote counts 2, except on
        legacy sets (made before the commander rules), which keep the results they had."""
        rows = self._conn().execute(
            "SELECT v.card_id, SUM(CASE WHEN v.voter_id = s.user_id AND s.cmc IS NOT NULL "
            "THEN 2 ELSE 1 END) AS n "
            "FROM votes v JOIN sets s ON s.id = v.set_id WHERE v.set_id = ? GROUP BY v.card_id",
            (set_id,)).fetchall()
        return {r["card_id"]: r["n"] for r in rows}

    def owner_vote(self, set_id: str) -> str | None:
        """The card the set's owner voted for (the vote that counts 2), if any; None on legacy sets."""
        row = self._conn().execute(
            "SELECT v.card_id FROM votes v JOIN sets s ON s.id = v.set_id "
            "WHERE v.set_id = ? AND v.voter_id = s.user_id AND s.cmc IS NOT NULL",
            (set_id,)).fetchone()
        return row["card_id"] if row is not None else None

    def user_vote(self, voter_id: str, set_id: str) -> str | None:
        row = self._conn().execute(
            "SELECT card_id FROM votes WHERE voter_id = ? AND set_id = ?",
            (voter_id, set_id)).fetchone()
        return row["card_id"] if row is not None else None

    # ----- knowledge pool -----
    # Pool row keys: id, name, status, max_entries_per_user, created_at, closed_at
    # Entry row keys: id, pool_id, card_id, user_id, created_at
    _POOL_COLS = "id, name, status, max_entries_per_user, created_at, closed_at"
    _ENTRY_COLS = "id, pool_id, card_id, user_id, created_at"

    def create_pool(self, name: str, max_entries_per_user: int) -> dict:
        """New open pool. 400 bad name or cap, 409 if a pool is already open."""
        if not isinstance(name, str) or not name.strip():
            raise StorageError(400, "Pool name is required")
        name = name.strip()
        if len(name) > POOL_NAME_MAX:
            raise StorageError(400, f"Pool name must be at most {POOL_NAME_MAX} characters")
        if (type(max_entries_per_user) is not int
                or not 1 <= max_entries_per_user <= POOL_ENTRY_CAP_MAX):
            raise StorageError(
                400, f"Entries per player must be a whole number from 1 to {POOL_ENTRY_CAP_MAX}")
        pool_id = _new_id()
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM pools WHERE status = 'open'").fetchone():
                raise StorageError(409, "A Knowledge Pool is already open")
            conn.execute(
                "INSERT INTO pools (id, name, status, max_entries_per_user, created_at) "
                "VALUES (?, ?, 'open', ?, ?)", (pool_id, name, max_entries_per_user, _now()))
        return self.get_pool(pool_id)

    def close_pool(self, pool_id: str) -> dict:
        """409 if already closed, 404 if unknown."""
        with self._tx() as conn:
            row = conn.execute("SELECT status FROM pools WHERE id = ?", (pool_id,)).fetchone()
            if row is None:
                raise StorageError(404, "Pool not found")
            if row["status"] != "open":
                raise StorageError(409, "This Knowledge Pool is already closed")
            conn.execute("UPDATE pools SET status = 'closed', closed_at = ? WHERE id = ?",
                         (_now(), pool_id))
        return self.get_pool(pool_id)

    def current_pool(self) -> dict | None:
        return self._one(f"SELECT {self._POOL_COLS} FROM pools WHERE status = 'open'")

    def get_pool(self, pool_id: str) -> dict | None:
        return self._one(f"SELECT {self._POOL_COLS} FROM pools WHERE id = ?", (pool_id,))

    def list_pools(self) -> list[dict]:
        """Newest first."""
        return self._all(
            f"SELECT {self._POOL_COLS} FROM pools ORDER BY created_at DESC, rowid DESC")

    def pool_entries(self, pool_id: str) -> list[dict]:
        """All entries of the pool, oldest first."""
        return self._all(f"SELECT {self._ENTRY_COLS} FROM pool_entries WHERE pool_id = ? "
                         "ORDER BY created_at, rowid", (pool_id,))

    def open_pool_entry_ids(self) -> dict[str, str]:
        """card id -> entry id for the open pool ({} when no pool is open)."""
        rows = self._all("SELECT e.card_id, e.id FROM pool_entries e JOIN pools p "
                         "ON p.id = e.pool_id WHERE p.status = 'open'")
        return {r["card_id"]: r["id"] for r in rows}

    def pool_medal_counts(self, pool_id: str) -> dict[str, dict[str, int]]:
        """entry_id -> {"gold": n, "silver": n, "bronze": n}, only entries with medals."""
        counts: dict[str, dict[str, int]] = {}
        for r in self._conn().execute(
                "SELECT entry_id, medal, COUNT(*) AS n FROM pool_medals WHERE pool_id = ? "
                "GROUP BY entry_id, medal", (pool_id,)):
            counts.setdefault(r["entry_id"], dict.fromkeys(MEDAL_POINTS, 0))[r["medal"]] = r["n"]
        return counts

    def my_pool_medals(self, voter_id: str, pool_id: str) -> dict[str, str]:
        """entry_id -> medal for this voter's medals in the pool."""
        rows = self._conn().execute(
            "SELECT entry_id, medal FROM pool_medals WHERE pool_id = ? AND voter_id = ?",
            (pool_id, voter_id)).fetchall()
        return {r["entry_id"]: r["medal"] for r in rows}

    def submit_pool_entry(self, user_id: str, card_id: str) -> dict:
        """Enter one of the user's finished, colorless or mono-colored cards in the open pool.
        404 no open pool or unknown card, 403 not the owner, 400 unfinished or multicolor,
        409 card already entered or the cap reached."""
        entry_id = _new_id()
        with self._tx() as conn:
            pool = conn.execute("SELECT id, max_entries_per_user FROM pools "
                                "WHERE status = 'open'").fetchone()
            if pool is None:
                raise StorageError(404, "No Knowledge Pool is open")
            card = self._decode_card(conn.execute(
                f"SELECT {self._CARD_COLS} FROM cards WHERE id = ?", (card_id,)).fetchone())
            if card is None:
                raise StorageError(404, "Card not found")
            if card["user_id"] != user_id:
                raise StorageError(403, "You can only submit your own cards")
            if card["status"] != "done":
                raise StorageError(400, "Only finished cards can be submitted")
            data = card["card"] if card["card"] is not None else card["card_params"]
            if not pool_card_eligible(data):
                raise StorageError(400, "Only colorless or mono-colored cards can enter the pool")
            if conn.execute("SELECT 1 FROM pool_entries WHERE pool_id = ? AND card_id = ?",
                            (pool["id"], card_id)).fetchone():
                raise StorageError(409, "That card is already in the pool")
            count = conn.execute(
                "SELECT COUNT(*) FROM pool_entries WHERE pool_id = ? AND user_id = ?",
                (pool["id"], user_id)).fetchone()[0]
            if count >= pool["max_entries_per_user"]:
                raise StorageError(
                    409, f"You've used all {pool['max_entries_per_user']} of your submissions - "
                         "withdraw one to submit another")
            conn.execute(
                "INSERT INTO pool_entries (id, pool_id, card_id, user_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (entry_id, pool["id"], card_id, user_id, _now()))
        return self._one(f"SELECT {self._ENTRY_COLS} FROM pool_entries WHERE id = ?", (entry_id,))

    @staticmethod
    def _open_entry(conn: sqlite3.Connection, entry_id: str) -> sqlite3.Row:
        """The entry joined with its pool's status; 404 unknown entry, 409 pool closed."""
        entry = conn.execute(
            "SELECT e.id, e.pool_id, e.user_id, p.status FROM pool_entries e "
            "JOIN pools p ON p.id = e.pool_id WHERE e.id = ?", (entry_id,)).fetchone()
        if entry is None:
            raise StorageError(404, "Submission not found")
        if entry["status"] != "open":
            raise StorageError(409, "This Knowledge Pool is closed")
        return entry

    def withdraw_pool_entry(self, entry_id: str, user_id: str) -> str:
        """Remove the user's entry and its medals while the pool is open. Returns the pool id."""
        with self._tx() as conn:
            entry = self._open_entry(conn, entry_id)
            if entry["user_id"] != user_id:
                raise StorageError(403, "You can only withdraw your own submissions")
            conn.execute("DELETE FROM pool_medals WHERE entry_id = ?", (entry_id,))
            conn.execute("DELETE FROM pool_entries WHERE id = ?", (entry_id,))
        return entry["pool_id"]

    def _votable_entry(self, conn: sqlite3.Connection, entry_id: str, voter_id: str) -> sqlite3.Row:
        """An open pool's entry that `voter_id` may medal (not their own card)."""
        entry = self._open_entry(conn, entry_id)
        if entry["user_id"] == voter_id:
            raise StorageError(403, "You can't give a medal to your own card")
        return entry

    def award_pool_medal(self, voter_id: str, entry_id: str, medal: str) -> str:
        """Give a card gold, silver or bronze. The medal moves off any other card it was on,
        and the card's previous medal from this voter is replaced.
        400 unknown medal, 403 own card, 404 unknown entry, 409 pool closed. Returns the pool id."""
        if medal not in MEDAL_POINTS:
            raise StorageError(400, "Medal must be gold, silver or bronze")
        with self._tx() as conn:
            entry = self._votable_entry(conn, entry_id, voter_id)
            conn.execute("DELETE FROM pool_medals WHERE voter_id = ? AND "
                         "((pool_id = ? AND medal = ?) OR entry_id = ?)",
                         (voter_id, entry["pool_id"], medal, entry_id))
            conn.execute(
                "INSERT INTO pool_medals (voter_id, pool_id, entry_id, medal, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (voter_id, entry["pool_id"], entry_id, medal, _now()))
        return entry["pool_id"]

    def clear_pool_medal(self, voter_id: str, entry_id: str) -> str:
        """Take back this voter's medal on the card (no-op if none). Returns the pool id."""
        with self._tx() as conn:
            entry = self._votable_entry(conn, entry_id, voter_id)
            conn.execute("DELETE FROM pool_medals WHERE voter_id = ? AND entry_id = ?",
                         (voter_id, entry_id))
        return entry["pool_id"]


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


def pool_cutoff(entries: list[dict]) -> int:
    """How many cards make the pool: half the players with at least one entry, rounded down."""
    return len({e["user_id"] for e in entries}) // 2


def pool_ranking(entries: list[dict], medals: dict[str, dict[str, int]]) -> dict[str, dict]:
    """Per entry id: {gold, silver, bronze, points, rank, in, tiedAtCutoff}.

    Entries rank by points (gold 3, silver 2, bronze 1), then golds, then silvers; equal keys
    share a rank. The top pool_cutoff(entries) are in, plus every card level with the last
    of them on all three; a card with no points is never in.
    """
    table: dict[str, dict] = {}
    for e in entries:
        counts = {m: int((medals.get(e["id"]) or {}).get(m, 0)) for m in MEDAL_POINTS}
        table[e["id"]] = {**counts, "points": sum(MEDAL_POINTS[m] * n for m, n in counts.items())}
    key = {eid: (row["points"], row["gold"], row["silver"]) for eid, row in table.items()}
    ordered = sorted(key.values(), reverse=True)
    cutoff = pool_cutoff(entries)
    line = ordered[cutoff - 1] if 0 < cutoff <= len(ordered) else None
    for eid, row in table.items():
        row["rank"] = 1 + sum(1 for k in key.values() if k > key[eid])
        row["in"] = line is not None and row["points"] > 0 and key[eid] >= line
        row["tiedAtCutoff"] = row["in"] and key[eid] == line and ordered.count(line) > 1
    return table
