"""
AI Night HTTP API (users, generations, sets, events, votes, queue status, media).

Spec: docs/superpowers/specs/2026-09-28-ai-night-design.md §4.
The blueprint receives its dependencies so it can be tested without importing app.py.

CardView.card is the final card dict when set, otherwise the card's card_params,
so pending cards already show their name and type.
"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint

from generation_queue import GenerationQueue
from storage import Storage


def create_api_blueprint(storage: Storage, gen_queue: GenerationQueue, data_dir: Path,
                         admin_pin: str) -> Blueprint:
    """All spec §4 endpoints; app.py registers the result with url_prefix="/api/v1"."""
    raise NotImplementedError
