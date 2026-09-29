"""
Card art generation (SDXL Lightning fine-tune) and art prompt construction.

Spec: docs/superpowers/specs/2026-09-28-ai-night-design.md §6.
torch/diffusers are imported lazily so placeholder mode and tests never load them.
"""
from __future__ import annotations


def build_art_prompt(prompt: str, card_data: dict | None) -> tuple[str, str]:
    """(positive prompt of at most 75 CLIP tokens, subject first; negative prompt)."""
    raise NotImplementedError


def generate_art(prompt: str, card_data: dict | None) -> str:
    """Raw base64 PNG (no data: prefix) at config.ART_BOX_SIZE."""
    raise NotImplementedError
