"""The queue's brief stage: the director's brief comes before rules text and art.

Spec: docs/superpowers/specs/2026-10-02-card-director-design.md §3.
"""
import pytest

from generation_queue import GenerationQueue
from test_generation_queue import Recorder, add_card, add_set, storage  # noqa: F401 (fixture)


def brief(tag):
    return {"identity": f"identity {tag}", "mechanic": f"mechanic {tag}",
            "art": {"subject": f"subject {tag}", "action": "a", "setting": "s", "framing": "f",
                    "light": "l"}}


class Director:
    """Fake brief_fn: returns briefs tagged b1, b2 ... and records its calls."""

    def __init__(self, result="briefs"):
        self.calls = []
        self.result = result

    def __call__(self, params, count, avoid):
        self.calls.append({"name": params.get("name"), "count": count, "avoid": avoid})
        if self.result == "none":
            return None
        if self.result == "raise":
            raise ConnectionError("ollama is down")
        start = sum(c["count"] for c in self.calls[:-1])
        return [brief(f"b{start + i + 1}") for i in range(count)]


@pytest.fixture
def rec():
    return Recorder()


def queue(storage, tmp_path, rec, director):
    return GenerationQueue(storage, tmp_path, rec.text_fn, rec.art_fn, rec.render_fn,
                           start_workers=False, brief_fn=director)


def set_cards(storage, n=3):
    set_id = add_set(storage, "Zur")
    return set_id, [add_card(storage, set_id=set_id, slot=slot) for slot in range(1, n + 1)]


def test_brief_runs_before_text_and_art(storage, rec, tmp_path):
    director = Director()
    q = queue(storage, tmp_path, rec, director)
    card_id = add_card(storage)
    q.enqueue(card_id)
    assert q.process_next_text() is False
    assert q.process_next_image() is False

    assert q.process_next_brief() is True
    assert director.calls == [{"name": "Stormwing", "count": 1, "avoid": []}]
    assert storage.get_card(card_id)["brief"] == brief("b1")
    assert q.process_next_text() is True
    assert q.process_next_image() is True
    assert rec.text_calls[0][1]["brief"] == brief("b1")
    assert rec.art_calls[0][1]["brief"] == brief("b1")
    assert storage.get_card(card_id)["status"] == "done"


def test_ollama_worker_serves_briefs_first(storage, rec, tmp_path):
    director = Director()
    q = queue(storage, tmp_path, rec, director)
    a = add_card(storage, card_params={"name": "A", "type": "Creature", "colors": [], "cmc": 1})
    q.enqueue(a)
    assert q.process_next_ollama() is True  # A's brief
    b = add_card(storage, card_params={"name": "B", "type": "Creature", "colors": [], "cmc": 1})
    q.enqueue(b)
    assert q.process_next_ollama() is True  # B's brief before A's text
    assert [c["name"] for c in director.calls] == ["A", "B"]
    assert rec.text_calls == []
    assert q.process_next_ollama() is True  # now A's text
    assert len(rec.text_calls) == 1


def test_set_makes_one_call_and_maps_slots(storage, rec, tmp_path):
    director = Director()
    q = queue(storage, tmp_path, rec, director)
    _, (s1, s2, s3) = set_cards(storage)
    for card_id in (s2, s1, s3):  # taken out of slot order on purpose
        q.enqueue(card_id)
    while q.process_next_brief():
        pass
    assert [c["count"] for c in director.calls] == [3]
    assert [storage.get_card(c)["brief"] for c in (s1, s2, s3)] == [brief("b1"), brief("b2"), brief("b3")]


def test_reroll_avoids_siblings(storage, rec, tmp_path):
    director = Director()
    q = queue(storage, tmp_path, rec, director)
    _, (s1, s2, s3) = set_cards(storage)
    for card_id in (s1, s2, s3):
        q.enqueue(card_id)
    while q.process_next_brief():
        pass
    for card_id in (s1, s2, s3):
        storage.update_card(card_id, status="done")
    q._partials.clear()  # the three originals finished
    new = storage.reroll_card(s2, storage.get_card(s2)["user_id"])["id"]
    q.enqueue(new)
    assert q.process_next_brief() is True
    assert director.calls[-1]["count"] == 1
    assert director.calls[-1]["avoid"] == [brief("b1"), brief("b3")]
    assert storage.get_card(new)["brief"] == brief("b4")


def test_reroll_after_failed_set_brief(storage, rec, tmp_path):
    director = Director(result="none")
    q = queue(storage, tmp_path, rec, director)
    _, (s1, s2, s3) = set_cards(storage)
    for card_id in (s1, s2, s3):
        storage.update_card(card_id, status="done")
    new = storage.reroll_card(s2, storage.get_card(s2)["user_id"])["id"]
    q.enqueue(new)
    q.process_next_brief()
    assert director.calls == [{"name": "Stormwing", "count": 1, "avoid": []}]


@pytest.mark.parametrize("result", ["none", "raise"])
def test_failed_brief_continues_without_one(storage, rec, tmp_path, result):
    q = queue(storage, tmp_path, rec, Director(result=result))
    _, cards = set_cards(storage)
    for card_id in cards:
        q.enqueue(card_id)
    assert q.process_next_brief() is True
    assert q.process_next_brief() is False  # one call covered the whole set
    for _ in cards:
        q.process_next_text()
        q.process_next_image()
    assert all("brief" not in params for _, params in rec.text_calls + rec.art_calls)
    assert [storage.get_card(c)["status"] for c in cards] == ["done"] * 3


def test_director_off_is_todays_queue(storage, rec, tmp_path):
    q = queue(storage, tmp_path, rec, None)
    card_id = add_card(storage)
    q.enqueue(card_id)
    assert q.process_next_brief() is False
    assert q.process_next_text() is True
    assert "brief" not in rec.text_calls[0][1]


def test_position_counts_the_brief_stage(storage, rec, tmp_path):
    q = queue(storage, tmp_path, rec, Director())
    first, second = add_card(storage), add_card(storage)
    q.enqueue(first)
    q.enqueue(second)
    assert q.position(first)[0] == 1
    position, eta = q.position(second)
    assert position == 2 and eta is not None
    assert q.status()["cardsAhead"] == 2
    q.process_next_brief()
    assert q.position(second)[0] == 2  # first is now in the image list, second still briefing


def test_position_while_the_brief_is_being_written(storage, rec, tmp_path):
    seen = []

    def director(params, count, avoid):
        seen.append(q.position(card_id))
        return None

    q = queue(storage, tmp_path, rec, director)
    card_id = add_card(storage)
    q.enqueue(card_id)
    q.process_next_brief()
    assert seen[0][0] == 1 and seen[0][1] is not None


def test_enqueue_many_keeps_a_set_in_one_call(storage, rec, tmp_path):
    director = Director()
    q = queue(storage, tmp_path, rec, director)
    _, cards = set_cards(storage)
    q.enqueue_many(cards)
    while q.process_next_brief():
        pass
    assert [c["count"] for c in director.calls] == [3]


def test_set_split_across_calls_still_avoids_the_briefed_sibling(storage, rec, tmp_path):
    director = Director()
    q = queue(storage, tmp_path, rec, director)
    _, (s1, s2, s3) = set_cards(storage)
    q.enqueue(s1)
    q.process_next_brief()          # slot 1 briefed alone (the worker woke early)
    q.enqueue_many([s2, s3])
    q.process_next_brief()
    assert director.calls[-1]["count"] == 2
    assert director.calls[-1]["avoid"] == [brief("b1")]
