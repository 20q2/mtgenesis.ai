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

Card lifecycle: enqueue() puts a card in both the text and the image wait list.
Whichever worker picks it up first moves it to 'generating'; each worker sets
text_ready / art_ready when its half finishes. The worker that finishes second
moves the card to 'rendering', calls render_fn and marks it 'done'. Any failure
marks the card 'failed' with a readable error and drops it from both lists.

status() splits the in-flight cards in two, without overlap: cardsAhead is the
image wait list (cards whose art has not started), generatingNow is every other
in-flight card (art painting, art done and waiting on text, or rendering).
"""
from __future__ import annotations

import base64
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
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

IDLE_SLEEP_SECONDS = 0.2
AVERAGE_WINDOW = 10
MAX_ERROR_DETAIL = 200
RESTART_ERROR = "Server restarted - please reroll"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_b64_png(data: str) -> bytes:
    """Raw base64 (tolerating a data: URL prefix) -> PNG bytes."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def _error_detail(exc: BaseException) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return detail[:MAX_ERROR_DETAIL]


def _log(message: str) -> None:
    """Best-effort console logging: an unprintable message must never break a worker."""
    try:
        print(message)
    except Exception:
        pass


def _log_exception() -> None:
    """Best-effort traceback of the exception being handled."""
    try:
        traceback.print_exc()
    except Exception:
        pass


def _seconds(value: float) -> float:
    return round(value, 1)


class GenerationQueue:
    def __init__(self, storage: Storage, data_dir: Path, text_fn: TextFn, art_fn: ArtFn,
                 render_fn: RenderFn, default_image_seconds: float = 10.0,
                 clock: Callable[[], float] = time.monotonic, start_workers: bool = True):
        self.storage = storage
        self.data_dir = Path(data_dir)
        self.art_dir = self.data_dir / "art"
        self.cards_dir = self.data_dir / "cards"
        self.art_dir.mkdir(parents=True, exist_ok=True)
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        self.text_fn = text_fn
        self.art_fn = art_fn
        self.render_fn = render_fn
        self.default_image_seconds = default_image_seconds
        self.clock = clock

        self._lock = threading.Lock()
        self._text_waiting: deque[str] = deque()
        self._image_waiting: deque[str] = deque()
        # card_id -> {"text", "art", "text_done", "art_done"} until both halves finish
        self._partials: dict[str, dict] = {}
        self._rendering: set[str] = set()
        self._painting: str | None = None
        self._paint_started = 0.0
        self._durations: deque[float] = deque(maxlen=AVERAGE_WINDOW)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        if start_workers:
            for name, step in (("gen-text-worker", self.process_next_text),
                               ("gen-image-worker", self.process_next_image)):
                thread = threading.Thread(target=self._worker_loop, args=(step,), name=name,
                                          daemon=True)
                thread.start()
                self._threads.append(thread)

    # ----- public API -----

    def enqueue(self, card_id: str) -> None:
        with self._lock:
            if card_id in self._partials or card_id in self._rendering:
                return
            self._partials[card_id] = {"text": None, "art": None,
                                       "text_done": False, "art_done": False}
            self._text_waiting.append(card_id)
            self._image_waiting.append(card_id)

    def process_next_text(self) -> bool:
        """Run text generation for the next waiting card. False if nothing was waiting."""
        card_id, card = self._take(self._text_waiting)
        if card_id is None:
            return False
        if card is None:
            return True
        try:
            text = self.text_fn(card["prompt"], dict(card["card_params"]))
            if text is None:
                # app.createCardContent swallows Ollama errors and returns None
                raise RuntimeError("no rules text was returned (is Ollama running?)")
        except Exception as exc:
            self._fail(card_id, f"Text generation failed: {_error_detail(exc)}")
            _log_exception()
            return True
        self._half_done(card_id, "text", text)
        return True

    def process_next_image(self) -> bool:
        """Run art generation for the next waiting card. False if nothing was waiting."""
        card_id, card = self._take(self._image_waiting, painting=True)
        if card_id is None:
            return False
        if card is None:
            return True
        try:
            art_b64 = self.art_fn(card["prompt"], dict(card["card_params"]))
            if not art_b64:
                raise RuntimeError("no image was returned")
            self._finish_painting(record_duration=True)
            art_path = self.art_dir / f"{card_id}.png"
            art_path.write_bytes(_decode_b64_png(art_b64))
        except Exception as exc:
            self._finish_painting(record_duration=False)
            self._fail(card_id, f"Artwork generation failed: {_error_detail(exc)}")
            _log_exception()
            return True
        self._half_done(card_id, "art", art_b64, art_path=str(art_path))
        return True

    def position(self, card_id: str) -> tuple[int | None, float | None]:
        """(queuePosition, etaSeconds) per the module docstring."""
        with self._lock:
            avg = self._average()
            remaining = self._remaining(avg)
            if card_id == self._painting:
                return 0, _seconds(remaining)
            try:
                queue_position = self._image_waiting.index(card_id) + 1
            except ValueError:
                return None, None
            return queue_position, _seconds(remaining + avg * queue_position)

    def status(self) -> dict:
        """{"busy": bool, "cardsAhead": int, "generatingNow": int, "avgImageSeconds": float, "etaSeconds": float}"""
        with self._lock:
            avg = self._average()
            remaining = self._remaining(avg)
            cards_ahead = len(self._image_waiting)
            in_flight = len(self._partials) + len(self._rendering)
            return {
                "busy": in_flight > 0,
                "cardsAhead": cards_ahead,
                "generatingNow": in_flight - cards_ahead,
                "avgImageSeconds": _seconds(avg),
                "etaSeconds": _seconds(remaining + avg * (cards_ahead + 1)),
            }

    def recover_on_startup(self) -> None:
        """Mark queued/generating/rendering cards failed with "Server restarted - please reroll"."""
        with self._lock:
            live = set(self._partials) | self._rendering
        recovered = 0
        for card_id in self.storage.unfinished_card_ids():
            if card_id in live:
                continue
            self.storage.update_card(card_id, status="failed", error=RESTART_ERROR,
                                     finished_at=_now_iso())
            recovered += 1
        if recovered:
            _log(f"♻️ Marked {recovered} interrupted card(s) failed: {RESTART_ERROR}")

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads = []

    # ----- internals -----

    def _worker_loop(self, step: Callable[[], bool]) -> None:
        while not self._stop.is_set():
            try:
                worked = step()
            except Exception:
                # Per-card failures are handled inside step(); this guards the thread itself.
                _log_exception()
                worked = False
            if not worked:
                self._stop.wait(IDLE_SLEEP_SECONDS)

    def _average(self) -> float:
        """Mean of the last AVERAGE_WINDOW image durations (caller holds the lock)."""
        if not self._durations:
            return self.default_image_seconds
        return sum(self._durations) / len(self._durations)

    def _remaining(self, avg: float) -> float:
        """Estimated seconds left on the image being painted (caller holds the lock)."""
        if self._painting is None:
            return 0.0
        return max(avg - (self.clock() - self._paint_started), 0.0)

    def _take(self, waiting: deque, painting: bool = False) -> tuple[str | None, dict | None]:
        """Pop the next card id from a wait list and mark the card 'generating'.
        Returns (None, None) if the list is empty, (card_id, None) if the card is gone."""
        with self._lock:
            if not waiting:
                return None, None
            card_id = waiting.popleft()
            if painting:
                self._painting = card_id
                self._paint_started = self.clock()
        card = self.storage.get_card(card_id)
        with self._lock:
            if card is None:
                self._drop(card_id)
            if card is None or card_id not in self._partials:
                if painting:
                    self._painting = None
                return card_id, None
            self.storage.update_card(card_id, status="generating")
        return card_id, card

    def _finish_painting(self, record_duration: bool) -> None:
        with self._lock:
            if self._painting is None:
                return
            if record_duration:
                self._durations.append(self.clock() - self._paint_started)
            self._painting = None

    def _drop(self, card_id: str) -> None:
        """Forget a card (caller holds the lock)."""
        self._partials.pop(card_id, None)
        self._rendering.discard(card_id)
        for waiting in (self._text_waiting, self._image_waiting):
            try:
                waiting.remove(card_id)
            except ValueError:
                pass

    def _fail(self, card_id: str, message: str) -> None:
        with self._lock:
            self._drop(card_id)
            self.storage.update_card(card_id, status="failed", error=message,
                                     finished_at=_now_iso())
        _log(f"❌ Card {card_id} failed: {message}")

    def _half_done(self, card_id: str, half: str, value: str, **fields) -> None:
        """Record a finished half; the second half to finish renders the card."""
        with self._lock:
            partial = self._partials.get(card_id)
            if partial is None:
                return  # failed meanwhile
            partial[half] = value
            partial[f"{half}_done"] = True
            both_done = partial["text_done"] and partial["art_done"]
            self.storage.update_card(card_id, **{f"{half}_ready": 1}, **fields)
            if both_done:
                self._partials.pop(card_id)
                self._rendering.add(card_id)
                self.storage.update_card(card_id, status="rendering")
        if both_done:
            self._render(card_id, partial["text"], partial["art"])

    def _render(self, card_id: str, text: str, art_b64: str) -> None:
        try:
            card = self.storage.get_card(card_id)
            if card is None:
                with self._lock:
                    self._rendering.discard(card_id)
                return
            card_params = dict(card["card_params"])
            force_name = None
            if card.get("set_id"):
                card_set = self.storage.get_set(card["set_id"])
                if card_set and card_set.get("commander_name"):
                    force_name = card_set["commander_name"]
                    card_params["name"] = force_name
            final_card, rendered_b64 = self.render_fn(card_params, text, art_b64, force_name)
            if not rendered_b64:
                raise RuntimeError("the renderer returned no image")
            card_path = self.cards_dir / f"{card_id}.png"
            card_path.write_bytes(_decode_b64_png(rendered_b64))
        except Exception as exc:
            self._fail(card_id, f"Card rendering failed: {_error_detail(exc)}")
            _log_exception()
            return
        with self._lock:
            self.storage.update_card(card_id, status="done", card=final_card,
                                     card_path=str(card_path), finished_at=_now_iso())
            self._rendering.discard(card_id)
