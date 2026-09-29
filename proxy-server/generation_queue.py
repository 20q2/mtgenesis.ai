"""
Two-stage generation queue: one text worker (Ollama) and one image worker (GPU).

Spec: docs/superpowers/specs/2026-09-28-ai-night-design.md §5.

Position semantics: queuePosition is the 1-based index in the image wait list,
0 while the card's art is being painted, None once art is ready or the card
finished/failed. etaSeconds = remaining time on the image currently painting
+ avg * queuePosition, where avg = mean of the last 10 image durations
(default_image_seconds when there is no history) and remaining =
max(avg - elapsed, 0), or 0 when nothing is painting. status()["etaSeconds"]
is the ETA a newly submitted card would get.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from storage import Storage

TextFn = Callable[[str, dict], "str | None"]
"""(prompt, card_params) -> raw generated text. Production: app.createCardContent."""

ArtFn = Callable[[str, dict], str]
"""(prompt, card_params) -> raw base64 PNG at config.ART_BOX_SIZE. Production: image_generation.generate_art."""

RenderFn = Callable[[dict, "str | None", "str | None", "str | None"], "tuple[dict, str | None]"]
"""(card_params, text, art_b64, force_name) -> (final card dict, rendered card base64).
Production: app.finalize_card. force_name is the set's commander_name for set cards, else None."""


class GenerationQueue:
    def __init__(self, storage: Storage, data_dir: Path, text_fn: TextFn, art_fn: ArtFn,
                 render_fn: RenderFn, default_image_seconds: float = 10.0,
                 clock: Callable[[], float] = time.monotonic, start_workers: bool = True):
        raise NotImplementedError

    def enqueue(self, card_id: str) -> None:
        raise NotImplementedError

    def process_next_text(self) -> bool:
        """Run text generation for the next waiting card. False if nothing was waiting."""
        raise NotImplementedError

    def process_next_image(self) -> bool:
        """Run art generation for the next waiting card. False if nothing was waiting."""
        raise NotImplementedError

    def position(self, card_id: str) -> tuple[int | None, float | None]:
        """(queuePosition, etaSeconds) per the module docstring."""
        raise NotImplementedError

    def status(self) -> dict:
        """{"busy": bool, "cardsAhead": int, "generatingNow": int, "avgImageSeconds": float, "etaSeconds": float}"""
        raise NotImplementedError

    def recover_on_startup(self) -> None:
        """Mark queued/generating/rendering cards failed with "Server restarted - please reroll"."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
