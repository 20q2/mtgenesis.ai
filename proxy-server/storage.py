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
    shared_at         TEXT   -- set while the card is shared to the gallery's Community tab
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

CREATE TABLE IF NOT EXISTS pool_slots (
    id          TEXT PRIMARY KEY,
    pool_id     TEXT NOT NULL,
    position    INTEGER NOT NULL,
    label       TEXT NOT NULL,
    color_rule  TEXT NOT NULL,
    type_rule   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pool_slots_pool ON pool_slots(pool_id, position);

CREATE TABLE IF NOT EXISTS pool_entries (
    id          TEXT PRIMARY KEY,
    pool_id     TEXT NOT NULL,
    slot_id     TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (pool_id, card_id),
    UNIQUE (slot_id, user_id)
);
CREATE INDEX IF NOT EXISTS pool_entries_slot ON pool_entries(slot_id);

-- Each voter gives at most one gold, silver and bronze per slot, and one medal per card.
CREATE TABLE IF NOT EXISTS pool_medals (
    voter_id    TEXT NOT NULL,
    pool_id     TEXT NOT NULL,
    slot_id     TEXT NOT NULL,
    entry_id    TEXT NOT NULL,
    medal       TEXT NOT NULL CHECK (medal IN ('gold', 'silver', 'bronze')),
    created_at  TEXT NOT NULL,
    UNIQUE (voter_id, slot_id, medal),
    UNIQUE (voter_id, entry_id)
);
CREATE INDEX IF NOT EXISTS pool_medals_pool ON pool_medals(pool_id);

-- Hidden until the pool closes; POOL_BAN_THRESHOLD bans disqualify a card.
CREATE TABLE IF NOT EXISTS pool_bans (
    voter_id    TEXT NOT NULL,
    pool_id     TEXT NOT NULL,
    entry_id    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (voter_id, entry_id)
);
CREATE INDEX IF NOT EXISTS pool_bans_pool ON pool_bans(pool_id);
"""

POOL_NAME_MAX = 80
POOL_SLOT_LABEL_MAX = 40
POOL_MAX_SLOTS = 40
POOL_MAX_ENTRIES_CAP = 40
MEDAL_POINTS = {"gold": 3, "silver": 2, "bronze": 1}
POOL_BANS_PER_PLAYER = 2
POOL_BAN_THRESHOLD = 3
MONO_COLORS = ("W", "U", "B", "R", "G")
COLOR_RULES = ("any", *MONO_COLORS, "multicolor", "colorless")
TYPE_RULES = ("any", "creature", "noncreature", "land")
_COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green",
                "multicolor": "Multicolor", "colorless": "Colorless"}
_TYPE_NAMES = {"creature": "creature", "noncreature": "noncreature", "land": "land"}


def card_colors(card: dict) -> set[str]:
    """The card's colors among WUBRG: its `colors` list, else the symbols in its mana cost."""
    card = card or {}
    colors = {c for c in (card.get("colors") or []) if c in MONO_COLORS}
    if not colors:
        colors = set(re.findall(r"[WUBRG]", (card.get("manaCost") or "").upper()))
    return colors


def card_fits_slot(card: dict, color_rule: str, type_rule: str) -> bool:
    """Whether a card meets a pool slot's color and type rules."""
    card = card or {}
    colors = card_colors(card)
    if color_rule in MONO_COLORS:
        color_ok = colors == {color_rule}
    elif color_rule == "multicolor":
        color_ok = len(colors) >= 2
    elif color_rule == "colorless":
        color_ok = not colors
    else:
        color_ok = True
    card_type = f"{card.get('supertype') or ''} {card.get('type') or ''}".lower()
    is_creature, is_land = "creature" in card_type, "land" in card_type
    type_ok = {"creature": is_creature, "land": is_land,
               "noncreature": not is_creature and not is_land}.get(type_rule, True)
    return color_ok and type_ok


def slot_rule_text(color_rule: str, type_rule: str) -> str:
    """Human description of a slot's rules, e.g. "Blue creature", "Any card"."""
    color = _COLOR_NAMES.get(color_rule)
    kind = _TYPE_NAMES.get(type_rule)
    if color and kind:
        return f"{color} {kind}"
    if color:
        return f"{color} card"
    if kind:
        return f"Any {kind}"
    return "Any card"


def clean_pool_slots(slots) -> list[dict]:
    """Validated [{label, color_rule, type_rule}] from the API's [{label, colorRule, typeRule}]."""
    if not isinstance(slots, list) or not slots:
        raise StorageError(400, "A pool needs at least one slot")
    if len(slots) > POOL_MAX_SLOTS:
        raise StorageError(400, f"A pool can have at most {POOL_MAX_SLOTS} slots")
    cleaned = []
    for i, slot in enumerate(slots, start=1):
        if not isinstance(slot, dict):
            raise StorageError(400, f"Slot {i} must be an object")
        color_rule = slot.get("colorRule", "any")
        type_rule = slot.get("typeRule", "any")
        if color_rule not in COLOR_RULES:
            raise StorageError(400, f"Slot {i}: color rule must be one of {', '.join(COLOR_RULES)}")
        if type_rule not in TYPE_RULES:
            raise StorageError(400, f"Slot {i}: type rule must be one of {', '.join(TYPE_RULES)}")
        label = slot.get("label")
        label = label.strip() if isinstance(label, str) else ""
        if not label:
            label = slot_rule_text(color_rule, type_rule)
        if len(label) > POOL_SLOT_LABEL_MAX:
            raise StorageError(400, f"Slot {i}: label must be at most {POOL_SLOT_LABEL_MAX} characters")
        cleaned.append({"label": label, "color_rule": color_rule, "type_rule": type_rule})
    return cleaned


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


COMMANDER_SLOT_CMC = {1: 3, 2: 4, 3: 5}
_MANA_SYMBOL_RE = re.compile(r"\{([^}]+)\}")


def commander_slot_params(params: dict, slot: int) -> dict:
    """A commander set slot's card params: a Legendary Creature costing 3, 4 or 5 mana.

    Keeps the requested colored/hybrid/Phyrexian pips, drops generic and {X}, and pads
    generic to the slot's mana value. Typed P/T is dropped so the stat curve sets each
    version's body. Pips worth more than 3 mana are a StorageError 400."""
    pips = [s for s in _MANA_SYMBOL_RE.findall(params.get("manaCost") or "")
            if not s.isdigit() and s.upper() not in ("X", "Y", "Z")]
    pip_value = sum(2 if s.startswith("2/") else 1 for s in pips)
    if pip_value > COMMANDER_SLOT_CMC[1]:
        raise StorageError(400, f"A commander set's colored pips can add up to at most "
                                f"{COMMANDER_SLOT_CMC[1]} mana (the first version costs 3)")
    cmc = COMMANDER_SLOT_CMC[slot]
    generic = cmc - pip_value
    cost = (f"{{{generic}}}" if generic else "") + "".join(f"{{{s}}}" for s in pips)
    out = {k: v for k, v in params.items() if k not in ("power", "toughness")}
    out.update(type="Creature", supertype="Legendary", manaCost=cost, cmc=cmc)
    return out


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
        self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Columns added after a table first shipped (CREATE TABLE IF NOT EXISTS skips them)."""
        card_cols = {r[1] for r in conn.execute("PRAGMA table_info(cards)")}
        if "shared_at" not in card_cols:
            conn.execute("ALTER TABLE cards ADD COLUMN shared_at TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS cards_shared ON cards(shared_at) "
                     "WHERE shared_at IS NOT NULL")

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
    # card_path, status, text_ready, art_ready, error, created_at, finished_at, shared_at
    _CARD_COLS = ("id, user_id, set_id, slot, replaced, prompt, card_params_json, card_json, "
                  "art_path, card_path, status, text_ready, art_ready, error, created_at, "
                  "finished_at, shared_at")
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

    # ----- knowledge pool -----
    # Pool row keys: id, name, status, max_entries_per_user, created_at, closed_at
    # Slot row keys: id, pool_id, position, label, color_rule, type_rule
    # Entry row keys: id, pool_id, slot_id, card_id, user_id, created_at
    _POOL_COLS = "id, name, status, max_entries_per_user, created_at, closed_at"
    _SLOT_COLS = "id, pool_id, position, label, color_rule, type_rule"
    _ENTRY_COLS = "id, pool_id, slot_id, card_id, user_id, created_at"

    def create_pool(self, name: str, max_entries_per_user: int, slots: list) -> dict:
        """New open pool with its slots (see clean_pool_slots). 409 if a pool is already open."""
        if not isinstance(name, str) or not name.strip():
            raise StorageError(400, "Pool name is required")
        name = name.strip()
        if len(name) > POOL_NAME_MAX:
            raise StorageError(400, f"Pool name must be at most {POOL_NAME_MAX} characters")
        if (type(max_entries_per_user) is not int
                or not 1 <= max_entries_per_user <= POOL_MAX_ENTRIES_CAP):
            raise StorageError(
                400, f"Submissions per player must be a whole number from 1 to {POOL_MAX_ENTRIES_CAP}")
        cleaned = clean_pool_slots(slots)
        pool_id = _new_id()
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM pools WHERE status = 'open'").fetchone():
                raise StorageError(409, "A Knowledge Pool is already open")
            conn.execute(
                "INSERT INTO pools (id, name, status, max_entries_per_user, created_at) "
                "VALUES (?, ?, 'open', ?, ?)", (pool_id, name, max_entries_per_user, _now()))
            for position, slot in enumerate(cleaned, start=1):
                conn.execute(
                    "INSERT INTO pool_slots (id, pool_id, position, label, color_rule, type_rule) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (_new_id(), pool_id, position, slot["label"], slot["color_rule"],
                     slot["type_rule"]))
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

    def pool_slots(self, pool_id: str) -> list[dict]:
        """Slots in position order."""
        return self._all(f"SELECT {self._SLOT_COLS} FROM pool_slots WHERE pool_id = ? "
                         "ORDER BY position", (pool_id,))

    def pool_entries(self, pool_id: str) -> list[dict]:
        """All entries of the pool, oldest first."""
        return self._all(f"SELECT {self._ENTRY_COLS} FROM pool_entries WHERE pool_id = ? "
                         "ORDER BY created_at, rowid", (pool_id,))

    def pool_medal_counts(self, pool_id: str) -> dict[str, dict[str, int]]:
        """entry_id -> {"gold": n, "silver": n, "bronze": n}, only entries with medals."""
        counts: dict[str, dict[str, int]] = {}
        for r in self._conn().execute(
                "SELECT entry_id, medal, COUNT(*) AS n FROM pool_medals WHERE pool_id = ? "
                "GROUP BY entry_id, medal", (pool_id,)):
            counts.setdefault(r["entry_id"], dict.fromkeys(MEDAL_POINTS, 0))[r["medal"]] = r["n"]
        return counts

    def pool_ban_counts(self, pool_id: str) -> dict[str, int]:
        """entry_id -> number of bans, only banned entries."""
        rows = self._conn().execute(
            "SELECT entry_id, COUNT(*) AS n FROM pool_bans WHERE pool_id = ? GROUP BY entry_id",
            (pool_id,)).fetchall()
        return {r["entry_id"]: r["n"] for r in rows}

    def my_pool_medals(self, voter_id: str, pool_id: str) -> dict[str, str]:
        """entry_id -> medal for this voter's medals in the pool."""
        rows = self._conn().execute(
            "SELECT entry_id, medal FROM pool_medals WHERE pool_id = ? AND voter_id = ?",
            (pool_id, voter_id)).fetchall()
        return {r["entry_id"]: r["medal"] for r in rows}

    def my_pool_bans(self, voter_id: str, pool_id: str) -> set[str]:
        """Entry ids this voter banned in the pool."""
        rows = self._conn().execute(
            "SELECT entry_id FROM pool_bans WHERE pool_id = ? AND voter_id = ?",
            (pool_id, voter_id)).fetchall()
        return {r["entry_id"] for r in rows}

    @staticmethod
    def _open_pool_slot(conn: sqlite3.Connection, slot_id: str) -> sqlite3.Row:
        """The slot joined with its pool's status; 404 unknown slot, 409 pool closed."""
        slot = conn.execute(
            "SELECT s.id, s.pool_id, s.color_rule, s.type_rule, s.label, p.status, "
            "p.max_entries_per_user FROM pool_slots s JOIN pools p ON p.id = s.pool_id "
            "WHERE s.id = ?", (slot_id,)).fetchone()
        if slot is None:
            raise StorageError(404, "Slot not found")
        if slot["status"] != "open":
            raise StorageError(409, "This Knowledge Pool is closed")
        return slot

    def submit_pool_entry(self, user_id: str, slot_id: str, card_id: str) -> dict:
        """Put one of the user's finished cards into an open pool's slot.
        403 not the owner, 400 card unfinished or not fitting the slot, 409 pool closed,
        card already in the pool, the user already has a card in the slot, or the cap reached."""
        entry_id = _new_id()
        with self._tx() as conn:
            slot = self._open_pool_slot(conn, slot_id)
            card = self._decode_card(conn.execute(
                f"SELECT {self._CARD_COLS} FROM cards WHERE id = ?", (card_id,)).fetchone())
            if card is None:
                raise StorageError(404, "Card not found")
            if card["user_id"] != user_id:
                raise StorageError(403, "You can only submit your own cards")
            if card["status"] != "done":
                raise StorageError(400, "Only finished cards can be submitted")
            data = card["card"] if card["card"] is not None else card["card_params"]
            if not card_fits_slot(data, slot["color_rule"], slot["type_rule"]):
                raise StorageError(
                    400, f"That card doesn't fit this slot (needs: "
                         f"{slot_rule_text(slot['color_rule'], slot['type_rule'])})")
            if conn.execute("SELECT 1 FROM pool_entries WHERE pool_id = ? AND card_id = ?",
                            (slot["pool_id"], card_id)).fetchone():
                raise StorageError(409, "That card is already in the pool")
            if conn.execute("SELECT 1 FROM pool_entries WHERE slot_id = ? AND user_id = ?",
                            (slot_id, user_id)).fetchone():
                raise StorageError(
                    409, "You already have a card in this slot - withdraw it first")
            count = conn.execute(
                "SELECT COUNT(*) FROM pool_entries WHERE pool_id = ? AND user_id = ?",
                (slot["pool_id"], user_id)).fetchone()[0]
            if count >= slot["max_entries_per_user"]:
                raise StorageError(
                    409, f"You've used all {slot['max_entries_per_user']} of your submissions - "
                         "withdraw one to submit another")
            conn.execute(
                "INSERT INTO pool_entries (id, pool_id, slot_id, card_id, user_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entry_id, slot["pool_id"], slot_id, card_id, user_id, _now()))
        return self._one(f"SELECT {self._ENTRY_COLS} FROM pool_entries WHERE id = ?", (entry_id,))

    def withdraw_pool_entry(self, entry_id: str, user_id: str) -> str:
        """Remove the user's entry, its medals and bans while the pool is open. Returns the pool id."""
        with self._tx() as conn:
            entry = conn.execute("SELECT pool_id, slot_id, user_id FROM pool_entries WHERE id = ?",
                                 (entry_id,)).fetchone()
            if entry is None:
                raise StorageError(404, "Submission not found")
            if entry["user_id"] != user_id:
                raise StorageError(403, "You can only withdraw your own submissions")
            self._open_pool_slot(conn, entry["slot_id"])
            conn.execute("DELETE FROM pool_medals WHERE entry_id = ?", (entry_id,))
            conn.execute("DELETE FROM pool_bans WHERE entry_id = ?", (entry_id,))
            conn.execute("DELETE FROM pool_entries WHERE id = ?", (entry_id,))
        return entry["pool_id"]

    def _open_pool_entry(self, conn: sqlite3.Connection, entry_id: str, voter_id: str,
                         action: str) -> sqlite3.Row:
        """The entry of an open pool that `voter_id` may medal or ban (not their own card)."""
        entry = conn.execute("SELECT id, pool_id, slot_id, user_id FROM pool_entries WHERE id = ?",
                             (entry_id,)).fetchone()
        if entry is None:
            raise StorageError(404, "Submission not found")
        self._open_pool_slot(conn, entry["slot_id"])
        if entry["user_id"] == voter_id:
            raise StorageError(403, f"You can't {action} your own card")
        return entry

    def award_pool_medal(self, voter_id: str, entry_id: str, medal: str) -> str:
        """Give a card gold, silver or bronze. The medal moves off any other card of the slot
        it was on, and the card's previous medal from this voter is replaced.
        400 unknown medal, 403 own card, 404 unknown entry, 409 pool closed. Returns the pool id."""
        if medal not in MEDAL_POINTS:
            raise StorageError(400, "Medal must be gold, silver or bronze")
        with self._tx() as conn:
            entry = self._open_pool_entry(conn, entry_id, voter_id, "give a medal to")
            conn.execute("DELETE FROM pool_medals WHERE voter_id = ? AND "
                         "((slot_id = ? AND medal = ?) OR entry_id = ?)",
                         (voter_id, entry["slot_id"], medal, entry_id))
            conn.execute(
                "INSERT INTO pool_medals (voter_id, pool_id, slot_id, entry_id, medal, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (voter_id, entry["pool_id"], entry["slot_id"], entry_id, medal, _now()))
        return entry["pool_id"]

    def clear_pool_medal(self, voter_id: str, entry_id: str) -> str:
        """Take back this voter's medal on the card (no-op if none). Returns the pool id."""
        with self._tx() as conn:
            entry = self._open_pool_entry(conn, entry_id, voter_id, "give a medal to")
            conn.execute("DELETE FROM pool_medals WHERE voter_id = ? AND entry_id = ?",
                         (voter_id, entry_id))
        return entry["pool_id"]

    def ban_pool_entry(self, voter_id: str, entry_id: str) -> str:
        """Spend one of the voter's POOL_BANS_PER_PLAYER bans on a card (idempotent).
        403 own card, 409 pool closed or no bans left. Returns the pool id."""
        with self._tx() as conn:
            entry = self._open_pool_entry(conn, entry_id, voter_id, "ban")
            if conn.execute("SELECT 1 FROM pool_bans WHERE voter_id = ? AND entry_id = ?",
                            (voter_id, entry_id)).fetchone():
                return entry["pool_id"]
            used = conn.execute("SELECT COUNT(*) FROM pool_bans WHERE voter_id = ? AND pool_id = ?",
                                (voter_id, entry["pool_id"])).fetchone()[0]
            if used >= POOL_BANS_PER_PLAYER:
                raise StorageError(
                    409, f"You've used both of your {POOL_BANS_PER_PLAYER} bans - lift one first")
            conn.execute("INSERT INTO pool_bans (voter_id, pool_id, entry_id, created_at) "
                         "VALUES (?, ?, ?, ?)", (voter_id, entry["pool_id"], entry_id, _now()))
        return entry["pool_id"]

    def unban_pool_entry(self, voter_id: str, entry_id: str) -> str:
        """Lift this voter's ban on the card (no-op if none). Returns the pool id."""
        with self._tx() as conn:
            entry = self._open_pool_entry(conn, entry_id, voter_id, "ban")
            conn.execute("DELETE FROM pool_bans WHERE voter_id = ? AND entry_id = ?",
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


def pool_standings(entry_ids: list[str], medals: dict[str, dict[str, int]],
                   bans: dict[str, int], apply_bans: bool) -> dict[str, dict]:
    """Per entry of one slot: {gold, silver, bronze, points, disqualified, leader, tied}.

    Points are gold 3, silver 2, bronze 1; more golds breaks a points tie. With apply_bans
    (a closed pool) entries with POOL_BAN_THRESHOLD+ bans are disqualified and can't lead.
    The best (points, golds) with points > 0 held by one entry -> leader; by several -> tied.
    """
    table = {}
    for eid in entry_ids:
        counts = {m: int((medals.get(eid) or {}).get(m, 0)) for m in MEDAL_POINTS}
        table[eid] = {**counts,
                      "points": sum(MEDAL_POINTS[m] * n for m, n in counts.items()),
                      "disqualified": apply_bans and bans.get(eid, 0) >= POOL_BAN_THRESHOLD}
    ranked = {eid: (row["points"], row["gold"]) for eid, row in table.items()
              if row["points"] > 0 and not row["disqualified"]}
    best = max(ranked.values(), default=None)
    top = [eid for eid, key in ranked.items() if key == best]
    for eid, row in table.items():
        row["leader"] = len(top) == 1 and eid in top
        row["tied"] = len(top) > 1 and eid in top
    return table
