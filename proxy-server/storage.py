"""
SQLite persistence for AI Night: users, events, commander sets, cards and votes.

Spec: docs/superpowers/specs/2026-09-28-ai-night-design.md §3.
Rows are returned as plain dicts with snake_case keys; JSON columns
(`card_params`, `card`) are decoded to dicts. IDs are uuid4 strings and
timestamps are UTC ISO-8601 strings.
"""
from __future__ import annotations

from pathlib import Path


class StorageError(Exception):
    """A rule violation that maps directly to an HTTP status (400/403/404/409)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Storage:
    def __init__(self, db_path: str | Path):
        """Open (creating if needed) the database; WAL mode; one connection per thread."""
        raise NotImplementedError

    # ----- users -----
    def login(self, username: str) -> dict:
        """Create-or-get by case-insensitive username. Trimmed, 1-24 chars of [A-Za-z0-9 _-], else 400.
        Returns {id, username} with the originally stored casing."""
        raise NotImplementedError

    def get_user(self, user_id: str) -> dict | None:
        raise NotImplementedError

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
        raise NotImplementedError

    def close_event(self, event_id: str) -> dict:
        """409 if already closed, 404 if unknown."""
        raise NotImplementedError

    def current_event(self) -> dict | None:
        raise NotImplementedError

    def get_event(self, event_id: str) -> dict | None:
        raise NotImplementedError

    def list_events(self) -> list[dict]:
        """Newest first."""
        raise NotImplementedError

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
