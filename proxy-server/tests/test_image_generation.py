import base64
import io
import sys

from PIL import Image

import config
import image_generation


def test_placeholder_mode(monkeypatch):
    monkeypatch.setattr(config, "MODEL_SIZE", "placeholder")
    torch_loaded_before = "torch" in sys.modules

    result = image_generation.generate_art("x", None)

    assert not result.startswith("data:")
    image = Image.open(io.BytesIO(base64.b64decode(result)))
    assert image.format == "PNG"
    assert image.size == (408, 336)
    assert image.convert("RGB").getpixel((0, 0)) == (50, 50, 50)
    # When this test runs alone torch is never imported; in a full run another
    # test may have imported it first, so only require that we didn't.
    if not torch_loaded_before:
        assert "torch" not in sys.modules


class _FakePipe:
    """Stands in for the SDXL pipeline: each call 'renders' the next solid color."""

    class _Tokenizer:
        def __call__(self, text):
            return type("T", (), {"input_ids": [0] * (len(text.split()) + 2)})()

    tokenizer = _Tokenizer()

    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        color = (self.calls * 40, 0, 0)
        return type("R", (), {"images": [Image.new("RGB", (64, 64), color=color)]})()


def _setup(monkeypatch, scores):
    """Fake pipeline plus a classifier that returns `scores` in order (one per render)."""
    pipe = _FakePipe()
    monkeypatch.setattr(config, "MODEL_SIZE", "heavy")
    monkeypatch.setattr(config, "NSFW_CHECK", True)
    monkeypatch.setattr(config, "NSFW_ATTEMPTS", 3)
    monkeypatch.setattr(image_generation, "_get_pipeline", lambda: pipe)
    remaining = list(scores)
    monkeypatch.setattr(image_generation, "nsfw_score", lambda image: remaining.pop(0))
    return pipe


def _pixel(result):
    return Image.open(io.BytesIO(base64.b64decode(result))).convert("RGB").getpixel((0, 0))


def test_clean_art_is_kept(monkeypatch):
    pipe = _setup(monkeypatch, [0.01])
    result = image_generation.generate_art("a knight", None)
    assert pipe.calls == 1
    assert _pixel(result) == (40, 0, 0)


def test_flagged_art_is_rerolled(monkeypatch):
    pipe = _setup(monkeypatch, [0.97, 0.80, 0.02])
    result = image_generation.generate_art("a knight", None)
    assert pipe.calls == 3
    assert _pixel(result) == (120, 0, 0)  # the third render, the only clean one


def test_always_flagged_gives_placeholder(monkeypatch):
    pipe = _setup(monkeypatch, [0.9, 0.9, 0.9])
    result = image_generation.generate_art("a knight", None)
    assert pipe.calls == 3
    assert _pixel(result) == (50, 50, 50)


def test_check_failure_lets_art_through(monkeypatch):
    pipe = _setup(monkeypatch, [])
    def broken(image):
        raise RuntimeError("no classifier")
    monkeypatch.setattr(image_generation, "nsfw_score", broken)
    result = image_generation.generate_art("a knight", None)
    assert pipe.calls == 1
    assert _pixel(result) == (40, 0, 0)


def test_check_disabled(monkeypatch):
    pipe = _setup(monkeypatch, [])
    monkeypatch.setattr(config, "NSFW_CHECK", False)
    image_generation.generate_art("a knight", None)
    assert pipe.calls == 1
