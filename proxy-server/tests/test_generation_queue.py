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
