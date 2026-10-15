"""Profile pictures (docs/superpowers/specs/2026-10-14-profile-pictures-design.md).

The browser frames the picture and sends a small square JPEG; the server still treats the body
as untrusted: it must open as a JPEG, PNG or WebP no larger than MAX_SIDE on a side, and it is
re-encoded to an AVATAR_SIZE square JPEG, which drops every bit of metadata (GPS included).
"""
from __future__ import annotations

import io
import warnings

from PIL import Image, ImageOps

from storage import StorageError

AVATAR_SIZE = 256
MAX_SIDE = 4096
FORMATS = ("JPEG", "PNG", "WEBP")
NOT_AN_IMAGE = "That file isn't a picture - use a JPEG, PNG or WebP image"


def process_avatar(data: bytes) -> bytes:
    """The picture as a centered AVATAR_SIZE square JPEG; StorageError(400) if it isn't one."""
    if not data:
        raise StorageError(400, "No picture was sent")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            if img.format not in FORMATS:
                raise StorageError(400, NOT_AN_IMAGE)
            if max(img.size) > MAX_SIDE:
                raise StorageError(400, f"That picture is too big - at most {MAX_SIDE} pixels on a side")
            img.load()
    except StorageError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        raise StorageError(400, NOT_AN_IMAGE) from None

    img = ImageOps.exif_transpose(img)  # phones store "rotate me" in EXIF rather than the pixels
    if img.mode not in ("RGB", "L"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.getchannel("A"))
        img = background
    square = ImageOps.fit(img.convert("RGB"), (AVATAR_SIZE, AVATAR_SIZE), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    square.save(out, "JPEG", quality=85, optimize=True)
    return out.getvalue()
