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
