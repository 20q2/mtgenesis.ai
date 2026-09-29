"""GenerationQueue: two-stage (text + image) card pipeline, spec §5.

Uses an in-memory FakeStorage implementing only the Storage contract methods the
queue calls (get_card, get_set, update_card, unfinished_card_ids), fake
text/art/render functions, and start_workers=False so tests drive the workers
by calling process_next_text / process_next_image directly.
"""
import base64
import copy
import io
import threading
import time
import uuid

import pytest
from PIL import Image

from generation_queue import GenerationQueue


def _png_b64(color="gray", size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


ART_B64 = _png_b64("gray")
CARD_B64 = _png_b64("black")

CARD_UPDATE_FIELDS = {"status", "text_ready", "art_ready", "card", "art_path", "card_path",
                      "error", "finished_at"}


class FakeStorage:
    """In-memory stand-in for storage.Storage (same signatures and row shapes)."""

    def __init__(self):
        self.cards = {}
        self.sets = {}
        self._lock = threading.Lock()

    # helpers used by the tests only
    def add_set(self, commander_name):
        set_id = str(uuid.uuid4())
        self.sets[set_id] = {"id": set_id, "user_id": "u1", "event_id": None,
                             "commander_name": commander_name, "prompt": "p",
                             "card_params": {}, "status": "draft",
                             "created_at": "2026-09-28T00:00:00+00:00", "locked_at": None}
        return set_id

    def add_card(self, prompt="a storm dragon", card_params=None, set_id=None, slot=None,
                 status="queued"):
        card_id = str(uuid.uuid4())
        params = card_params or {"name": "Stormwing", "manaCost": "{3}{U}", "colors": ["U"],
                                 "type": "Creature", "rarity": "rare", "cmc": 4}
        self.cards[card_id] = {"id": card_id, "user_id": "u1", "set_id": set_id, "slot": slot,
                               "replaced": 0, "prompt": prompt, "card_params": params,
                               "card": None, "art_path": None, "card_path": None,
                               "status": status, "text_ready": 0, "art_ready": 0,
                               "error": None, "created_at": "2026-09-28T00:00:00+00:00",
                               "finished_at": None}
        return card_id

    # Storage contract
    def get_card(self, card_id):
        with self._lock:
            return copy.deepcopy(self.cards.get(card_id))

    def get_set(self, set_id):
        with self._lock:
            return copy.deepcopy(self.sets.get(set_id))

    def update_card(self, card_id, **fields):
        unknown = set(fields) - CARD_UPDATE_FIELDS
        assert not unknown, f"update_card got fields outside the contract: {unknown}"
        with self._lock:
            self.cards[card_id].update(copy.deepcopy(fields))

    def unfinished_card_ids(self):
        with self._lock:
            return [c["id"] for c in self.cards.values()
                    if c["status"] in ("queued", "generating", "rendering")]


class Recorder:
    """Fake text/art/render functions that record their calls."""

    def __init__(self, text="Flying", art=ART_B64):
        self.text = text
        self.art = art
        self.text_calls = []
        self.art_calls = []
        self.render_calls = []

    def text_fn(self, prompt, card_params):
        self.text_calls.append((prompt, card_params))
        return self.text

    def art_fn(self, prompt, card_params):
        self.art_calls.append((prompt, card_params))
        return self.art

    def render_fn(self, card_params, text, art_b64, force_name):
        self.render_calls.append({"card_params": copy.deepcopy(card_params), "text": text,
                                  "art_b64": art_b64, "force_name": force_name})
        card = dict(card_params)
        card["description"] = text
        return card, CARD_B64


@pytest.fixture
def storage():
    return FakeStorage()


@pytest.fixture
def rec():
    return Recorder()


def make_queue(storage, data_dir, rec, **kwargs):
    kwargs.setdefault("start_workers", False)
    return GenerationQueue(storage, data_dir, rec.text_fn, rec.art_fn, rec.render_fn, **kwargs)


def _is_png_file(path):
    with Image.open(path) as im:
        im.verify()
        return im.format == "PNG"


# ----- B2: lifecycle -----

def test_lifecycle(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec)
    card_id = storage.add_card()

    q.enqueue(card_id)
    assert storage.get_card(card_id)["status"] == "queued"

    assert q.process_next_text() is True
    row = storage.get_card(card_id)
    assert row["status"] == "generating"
    assert row["text_ready"] == 1
    assert row["art_ready"] == 0
    assert rec.render_calls == []

    assert q.process_next_image() is True
    row = storage.get_card(card_id)
    assert row["status"] == "done"
    assert row["art_ready"] == 1
    assert row["finished_at"]
    assert row["error"] is None
    assert row["card"]["description"] == "Flying"

    art = tmp_path / "art" / f"{card_id}.png"
    rendered = tmp_path / "cards" / f"{card_id}.png"
    assert art.exists() and _is_png_file(art)
    assert rendered.exists() and _is_png_file(rendered)
    assert row["art_path"] == str(art)
    assert row["card_path"] == str(rendered)
    assert len(rec.render_calls) == 1
    assert rec.render_calls[0]["art_b64"] == ART_B64

    # nothing left to do
    assert q.process_next_text() is False
    assert q.process_next_image() is False


def test_text_and_art_get_prompt_and_params(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec)
    card_id = storage.add_card(prompt="a fiery phoenix")
    q.enqueue(card_id)
    q.process_next_text()
    q.process_next_image()
    params = storage.get_card(card_id)["card_params"]
    assert rec.text_calls == [("a fiery phoenix", params)]
    assert rec.art_calls == [("a fiery phoenix", params)]


def test_image_first_order(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec)
    card_id = storage.add_card()
    q.enqueue(card_id)

    assert q.process_next_image() is True
    row = storage.get_card(card_id)
    assert row["status"] == "generating"
    assert row["art_ready"] == 1
    assert row["text_ready"] == 0
    assert row["art_path"] == str(tmp_path / "art" / f"{card_id}.png")
    assert rec.render_calls == []

    assert q.process_next_text() is True
    row = storage.get_card(card_id)
    assert row["status"] == "done"
    assert row["text_ready"] == 1
    assert len(rec.render_calls) == 1


def test_set_card_forces_commander_name(storage, rec, tmp_path):
    rec.text = '{"name": "Other", "description": "Flying"}'
    q = make_queue(storage, tmp_path, rec)
    set_id = storage.add_set("Zur'ka, Élan of Ash")
    card_id = storage.add_card(set_id=set_id, slot=1)
    q.enqueue(card_id)
    q.process_next_text()
    q.process_next_image()

    call = rec.render_calls[0]
    assert call["force_name"] == "Zur'ka, Élan of Ash"
    assert call["card_params"]["name"] == "Zur'ka, Élan of Ash"
    assert call["text"] == '{"name": "Other", "description": "Flying"}'
    assert storage.get_card(card_id)["card"]["name"] == "Zur'ka, Élan of Ash"


def test_free_play_card_has_no_force_name(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec)
    card_id = storage.add_card()
    q.enqueue(card_id)
    q.process_next_text()
    q.process_next_image()
    call = rec.render_calls[0]
    assert call["force_name"] is None
    assert call["card_params"]["name"] == "Stormwing"


def test_text_failure_marks_failed(storage, rec, tmp_path):
    def boom(prompt, card_params):
        raise ConnectionError("ollama down")

    q = GenerationQueue(storage, tmp_path, boom, rec.art_fn, rec.render_fn, start_workers=False)
    card_id = storage.add_card()
    q.enqueue(card_id)

    assert q.process_next_text() is True
    row = storage.get_card(card_id)
    assert row["status"] == "failed"
    assert "Text generation failed" in row["error"]
    assert "ollama down" in row["error"]

    assert q.process_next_image() is False
    assert rec.art_calls == []
    assert storage.get_card(card_id)["status"] == "failed"
    assert rec.render_calls == []


def test_text_none_marks_failed(storage, rec, tmp_path):
    """app.createCardContent returns None when Ollama errors (it swallows the exception)."""
    rec.text = None
    q = make_queue(storage, tmp_path, rec)
    card_id = storage.add_card()
    q.enqueue(card_id)
    q.process_next_text()
    row = storage.get_card(card_id)
    assert row["status"] == "failed"
    assert "Text generation failed" in row["error"]


def test_art_failure_marks_failed(storage, rec, tmp_path):
    def boom(prompt, card_params):
        raise RuntimeError("CUDA out of memory")

    q = GenerationQueue(storage, tmp_path, rec.text_fn, boom, rec.render_fn, start_workers=False)
    card_id = storage.add_card()
    q.enqueue(card_id)

    assert q.process_next_image() is True
    row = storage.get_card(card_id)
    assert row["status"] == "failed"
    assert "Artwork generation failed" in row["error"]

    assert q.process_next_text() is False
    assert rec.text_calls == []
    assert storage.get_card(card_id)["status"] == "failed"


def test_art_failure_after_text_done(storage, rec, tmp_path):
    def boom(prompt, card_params):
        raise RuntimeError("CUDA out of memory")

    q = GenerationQueue(storage, tmp_path, rec.text_fn, boom, rec.render_fn, start_workers=False)
    card_id = storage.add_card()
    q.enqueue(card_id)
    q.process_next_text()
    q.process_next_image()
    row = storage.get_card(card_id)
    assert row["status"] == "failed"
    assert "Artwork generation failed" in row["error"]
    assert rec.render_calls == []


def test_render_failure_marks_failed(storage, rec, tmp_path):
    def bad_render(card_params, text, art_b64, force_name):
        raise ValueError("font missing")

    q = GenerationQueue(storage, tmp_path, rec.text_fn, rec.art_fn, bad_render,
                        start_workers=False)
    card_id = storage.add_card()
    q.enqueue(card_id)
    q.process_next_text()
    q.process_next_image()
    row = storage.get_card(card_id)
    assert row["status"] == "failed"
    assert "Card rendering failed" in row["error"]


def test_render_returning_no_image_marks_failed(storage, rec, tmp_path):
    q = GenerationQueue(storage, tmp_path, rec.text_fn, rec.art_fn,
                        lambda p, t, a, f: (dict(p), None), start_workers=False)
    card_id = storage.add_card()
    q.enqueue(card_id)
    q.process_next_text()
    q.process_next_image()
    row = storage.get_card(card_id)
    assert row["status"] == "failed"
    assert "Card rendering failed" in row["error"]


def test_fifo_order(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec)
    ids = [storage.add_card(prompt=f"card {i}") for i in range(3)]
    for card_id in ids:
        q.enqueue(card_id)
    while q.process_next_image():
        pass
    assert [prompt for prompt, _ in rec.art_calls] == ["card 0", "card 1", "card 2"]
    while q.process_next_text():
        pass
    assert [prompt for prompt, _ in rec.text_calls] == ["card 0", "card 1", "card 2"]
    assert all(storage.get_card(i)["status"] == "done" for i in ids)


def test_missing_card_is_skipped(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec)
    q.enqueue(str(uuid.uuid4()))
    q.process_next_text()
    q.process_next_image()
    assert rec.render_calls == []


def test_background_workers_finish_card(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec, start_workers=True)
    try:
        card_id = storage.add_card()
        q.enqueue(card_id)
        deadline = time.monotonic() + 5
        while storage.get_card(card_id)["status"] != "done" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert storage.get_card(card_id)["status"] == "done"
    finally:
        q.stop()


def test_background_workers_many_cards_with_failure(storage, tmp_path):
    def text_fn(prompt, card_params):
        time.sleep(0.005)
        if prompt == "card 3":
            raise ConnectionError("ollama down")
        return "Flying"

    def art_fn(prompt, card_params):
        time.sleep(0.01)
        return ART_B64

    rec = Recorder()
    q = GenerationQueue(storage, tmp_path, text_fn, art_fn, rec.render_fn)
    try:
        ids = [storage.add_card(prompt=f"card {i}") for i in range(8)]
        for card_id in ids:
            q.enqueue(card_id)
        deadline = time.monotonic() + 10
        while q.status()["busy"] and time.monotonic() < deadline:
            time.sleep(0.02)
        statuses = [storage.get_card(i)["status"] for i in ids]
        assert statuses == ["done"] * 3 + ["failed"] + ["done"] * 4
        assert len(rec.render_calls) == 7
        status = q.status()
        assert (status["busy"], status["cardsAhead"], status["generatingNow"]) == (False, 0, 0)
    finally:
        q.stop()


# ----- B3: positions, ETA, status, recovery -----

def fake_clock():
    """A list-backed clock: read with clock(), move time with t[0] += seconds."""
    t = [0.0]
    return t, (lambda: t[0])


def timed_art(t, durations):
    """art_fn that advances the fake clock by the next duration on each call."""
    durations = iter(durations)

    def art_fn(prompt, card_params):
        t[0] += next(durations)
        return ART_B64

    return art_fn


def test_positions_and_eta_default(storage, rec, tmp_path):
    t, clock = fake_clock()
    q = make_queue(storage, tmp_path, rec, clock=clock)
    ids = [storage.add_card() for _ in range(3)]
    for card_id in ids:
        q.enqueue(card_id)

    assert [q.position(i) for i in ids] == [(1, 10), (2, 20), (3, 30)]
    status = q.status()
    assert status["cardsAhead"] == 3
    assert status["busy"] is True
    assert status["generatingNow"] == 0
    assert status["avgImageSeconds"] == 10
    assert status["etaSeconds"] == 40


def test_eta_uses_rolling_average(storage, tmp_path):
    t, clock = fake_clock()
    seen = {}
    ids = {}

    def art_fn(prompt, card_params):
        if prompt == "x":
            t[0] += 1
            seen["x"] = q.position(ids["x"])
            seen["next"] = q.position(ids["next"])
            seen["status"] = q.status()
            t[0] += 4
        else:
            t[0] += {"first": 4, "second": 6}[prompt]
        return ART_B64

    rec = Recorder()
    q = GenerationQueue(storage, tmp_path, rec.text_fn, art_fn, rec.render_fn,
                        clock=clock, start_workers=False)
    for name in ("first", "second", "x", "next"):
        ids[name] = storage.add_card(prompt=name)
        q.enqueue(ids[name])

    q.process_next_image()
    q.process_next_image()
    assert q.status()["avgImageSeconds"] == 5
    assert q.position(ids["x"]) == (1, 5)

    q.process_next_image()  # paints x, sampling positions 1s in
    assert seen["x"] == (0, 4)
    assert seen["next"] == (1, 4 + 5)
    assert seen["status"]["cardsAhead"] == 1
    assert seen["status"]["etaSeconds"] == 4 + 5 * 2


def test_remaining_never_negative(storage, tmp_path):
    t, clock = fake_clock()
    seen = {}
    ids = {}

    def art_fn(prompt, card_params):
        t[0] += 25  # far longer than the 10s default
        seen["slow"] = q.position(ids["slow"])
        seen["next"] = q.position(ids["next"])
        return ART_B64

    rec = Recorder()
    q = GenerationQueue(storage, tmp_path, rec.text_fn, art_fn, rec.render_fn,
                        clock=clock, start_workers=False)
    for name in ("slow", "next"):
        ids[name] = storage.add_card(prompt=name)
        q.enqueue(ids[name])
    q.process_next_image()
    assert seen["slow"] == (0, 0)
    assert seen["next"] == (1, 10)


def test_average_window_10(storage, tmp_path):
    t, clock = fake_clock()
    rec = Recorder()
    q = GenerationQueue(storage, tmp_path, rec.text_fn, timed_art(t, range(1, 13)),
                        rec.render_fn, clock=clock, start_workers=False)
    for _ in range(12):
        q.enqueue(storage.add_card())
    for _ in range(12):
        q.process_next_image()
    # durations 1..12; only the last 10 (3..12) count
    assert q.status()["avgImageSeconds"] == pytest.approx(7.5)


def test_failed_image_not_counted_in_average(storage, tmp_path):
    t, clock = fake_clock()

    def art_fn(prompt, card_params):
        t[0] += 100
        raise RuntimeError("CUDA out of memory")

    rec = Recorder()
    q = GenerationQueue(storage, tmp_path, rec.text_fn, art_fn, rec.render_fn,
                        clock=clock, start_workers=False)
    q.enqueue(storage.add_card())
    q.process_next_image()
    status = q.status()
    assert status["avgImageSeconds"] == 10
    assert status["busy"] is False


def test_position_none_after_art(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec)
    card_id = storage.add_card()
    q.enqueue(card_id)
    q.process_next_image()
    assert storage.get_card(card_id)["text_ready"] == 0
    assert q.position(card_id) == (None, None)
    q.process_next_text()
    assert q.position(card_id) == (None, None)
    assert q.position(str(uuid.uuid4())) == (None, None)


def test_position_none_after_failure(storage, rec, tmp_path):
    def boom(prompt, card_params):
        raise ConnectionError("ollama down")

    q = GenerationQueue(storage, tmp_path, boom, rec.art_fn, rec.render_fn, start_workers=False)
    card_id = storage.add_card()
    q.enqueue(card_id)
    q.process_next_text()
    assert q.position(card_id) == (None, None)
    assert q.status()["busy"] is False


def test_idle_status(storage, rec, tmp_path):
    q = make_queue(storage, tmp_path, rec)
    status = q.status()
    assert status == {"busy": False, "cardsAhead": 0, "generatingNow": 0,
                      "avgImageSeconds": 10, "etaSeconds": 10}


def test_status_splits_waiting_and_generating(storage, rec, tmp_path):
    t, clock = fake_clock()
    q = GenerationQueue(storage, tmp_path, rec.text_fn, timed_art(t, [12, 12]), rec.render_fn,
                        clock=clock, start_workers=False)
    first, second = storage.add_card(), storage.add_card()
    q.enqueue(first)
    q.enqueue(second)
    q.process_next_image()  # first: art done, text pending
    status = q.status()
    assert status["busy"] is True
    assert status["cardsAhead"] == 1
    assert status["generatingNow"] == 1
    assert status["etaSeconds"] == 24  # nothing painting; 12s avg x new card at position 2

    q.process_next_text()  # first done
    status = q.status()
    assert status["cardsAhead"] == 1
    assert status["generatingNow"] == 0


def test_recover_on_startup(storage, rec, tmp_path):
    ids = {status: storage.add_card(status=status)
           for status in ("queued", "generating", "rendering", "done", "failed")}
    storage.cards[ids["done"]]["finished_at"] = "2026-09-28T01:00:00+00:00"
    done_before = storage.get_card(ids["done"])
    failed_before = storage.get_card(ids["failed"])

    q = make_queue(storage, tmp_path, rec)
    q.recover_on_startup()

    for status in ("queued", "generating", "rendering"):
        row = storage.get_card(ids[status])
        assert row["status"] == "failed"
        assert row["error"] == "Server restarted - please reroll"
    assert storage.get_card(ids["done"]) == done_before
    assert storage.get_card(ids["failed"]) == failed_before
