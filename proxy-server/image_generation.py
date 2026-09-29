"""
Card art generation (SDXL Lightning fine-tune) and art prompt construction.

Spec: docs/superpowers/specs/2026-09-28-ai-night-design.md §6.
torch/diffusers are imported lazily so placeholder mode and tests never load them.
"""
from __future__ import annotations

import base64
import io
import re
import threading
import time

from PIL import Image

import config

# ===== PROMPT CONSTRUCTION =====

MAX_PROMPT_TOKENS = 75       # CLIP's 77 minus the start/end tokens
SUBJECT_MIN_TOKENS = 30      # the subject always keeps at least this much room

# DreamShaper leans toward glossy, high-contrast digital art. MTG art reads as traditional
# painting: visible brushwork, muted earthy palettes, soft atmospheric light, no pure whites.
# The style sits right after the subject (CLIP weights early tokens most). No "highly
# detailed" or "dramatic lighting": they pull toward photoreal HDR with blown highlights.
ART_STYLE = ("oil painting, Magic: The Gathering card art, painterly brushstrokes, "
             "rich earthy colors, soft atmospheric light")
# Photo/3D terms push away from realism and the glossy CG look; the exposure terms stop the
# blown-white skies and halos; nsfw/nudity because SDXL has no safety checker and
# DreamShaper drifts toward nudity on humanoid subjects.
NEGATIVE_PROMPT = ("photograph, photorealistic, 3d render, cgi, overexposed, blown highlights, "
                   "harsh contrast, oversaturated, oversharpened, glowing halo, text, watermark, "
                   "signature, border, frame, card, UI, blurry, lowres, deformed, extra limbs, nsfw, nudity")
GENERIC_CONTEXT = "fantasy scene"

WUBRG = "WUBRG"
_COLOR_NAMES = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G", "colorless": "C"}

# Mood hints per color. Mono-colored cards get both words; multicolor cards get the first
# word of each color so the hint stays short. Nothing about light or radiance: those words
# make every white card a glowing, overexposed halo.
COLOR_MOODS = {
    "W": "noble, hallowed",
    "U": "arcane, mysterious",
    "B": "ominous, decay",
    "R": "fiery, wild",
    "G": "verdant, primal",
}

# Color palettes keyed by the set of WUBRG colors. Painted, mid-value colors: "pure white"
# or "electric" hues render as clipped highlights.
COLOR_PALETTES = {
    frozenset(): "weathered bronze, stone gray",
    # mono
    frozenset("W"): "warm ivory, pale gold, soft sky blue",
    frozenset("U"): "deep blue, slate gray, silver",
    frozenset("B"): "dark violet, charcoal, bone",
    frozenset("R"): "crimson, ember orange, smoky brown",
    frozenset("G"): "moss green, earthy brown",
    # guilds
    frozenset("WU"): "ivory, slate blue",
    frozenset("WB"): "ivory, charcoal",
    frozenset("WR"): "ivory, crimson",
    frozenset("WG"): "ivory, moss green",
    frozenset("UB"): "deep blue, charcoal",
    frozenset("UR"): "deep blue, crimson",
    frozenset("UG"): "teal, moss green",
    frozenset("BR"): "charcoal, crimson",
    frozenset("BG"): "charcoal, moss green",
    frozenset("RG"): "crimson, moss green",
    # shards and wedges
    frozenset("WUG"): "ivory, slate blue, moss green",       # Bant
    frozenset("UBR"): "deep blue, charcoal, crimson",        # Grixis
    frozenset("BRG"): "charcoal, crimson, moss green",       # Jund
    frozenset("RGW"): "crimson, moss green, ivory",          # Naya
    frozenset("WBG"): "ivory, charcoal, moss green",         # Abzan
    frozenset("URW"): "slate blue, crimson, ivory",          # Jeskai
    frozenset("BGU"): "charcoal, moss green, deep blue",     # Sultai
    frozenset("RWB"): "crimson, ivory, charcoal",            # Mardu
    frozenset("GUR"): "moss green, deep blue, crimson",      # Temur
    # all five
    frozenset(WUBRG): "rich jewel tones",
}
_PALETTE_FALLBACK_BY_COUNT = {
    3: "rich jewel tones",
    4: "rich jewel tones",
}

# Type contexts. Checked in order, so an "Artifact Creature" gets the creature context.
TYPE_CONTEXTS = {
    "creature": "creature portrait, full figure",
    "instant": "spell in motion, dynamic action",
    "sorcery": "powerful spell being cast",
    "artifact": "ornate artifact, object focus",
    "enchantment": "enchanted scene, mystical atmosphere",
    "land": "sweeping landscape, atmospheric perspective",
    "planeswalker": "powerful planeswalker, character focus",
    "battle": "epic battle scene, warfare",
}


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of CLIP tokens - CLIP tokenizer splits on spaces and punctuation.
    This is a conservative estimate to stay under the 77 token limit.
    """
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return len(tokens)


def truncate_prompt_smartly(prompt: str, max_tokens: int = 75) -> str:
    """
    Intelligently truncate prompt while preserving the most important elements.
    Priority: subject > style > color palette > lighting
    """
    estimated_tokens = estimate_tokens(prompt)

    if estimated_tokens <= max_tokens:
        return prompt

    print(f"⚠️  Prompt too long ({estimated_tokens} tokens), truncating...")

    parts = prompt.split(', ')

    # Prioritize parts: Subject > Color > Style > Magic context > Lighting
    subject_parts = []
    color_parts = []
    style_parts = []
    magic_parts = []
    lighting_parts = []

    for part in parts:
        part_lower = part.lower()
        if 'color palette' in part_lower:
            color_parts.append(part)
        elif any(keyword in part_lower for keyword in ['magic: the gathering', 'card art']):
            magic_parts.append(part)
        elif any(keyword in part_lower for keyword in ['fantasy art', 'style', 'detailed', 'illustration', 'artwork']):
            style_parts.append(part)
        elif any(keyword in part_lower for keyword in ['lighting', 'contrast', 'dramatic']):
            lighting_parts.append(part)
        else:
            subject_parts.append(part)

    final_parts = subject_parts
    test_prompt = ', '.join(final_parts)
    for group in (color_parts, style_parts, magic_parts, lighting_parts):
        for part in group:
            if estimate_tokens(test_prompt + ', ' + part) <= max_tokens:
                final_parts.append(part)
                test_prompt = ', '.join(final_parts)
                break

    final_prompt = ', '.join(final_parts)
    print(f"✂️  Truncated to {estimate_tokens(final_prompt)} tokens: {final_prompt[:100]}...")
    return final_prompt


def _normalize_colors(card_data: dict | None) -> tuple[list[str], bool]:
    """(WUBRG colors in canonical order, explicitly colorless?)."""
    raw = (card_data or {}).get("colors") or []
    letters = set()
    for color in raw:
        if not isinstance(color, str):
            continue
        c = color.strip()
        letters.add(_COLOR_NAMES.get(c.lower(), c.upper()))
    return [c for c in WUBRG if c in letters], "C" in letters


def _type_context(card_data: dict | None) -> str:
    card_type = str((card_data or {}).get("type") or "").lower()
    for keyword, context in TYPE_CONTEXTS.items():
        if keyword in card_type:
            return context
    return GENERIC_CONTEXT


def _color_part(card_data: dict | None) -> str:
    """Mood hints plus palette, or '' when the card has no color information."""
    colors, colorless = _normalize_colors(card_data)
    if not colors:
        card_type = str((card_data or {}).get("type") or "").lower()
        if colorless or "artifact" in card_type:
            return f"{COLOR_PALETTES[frozenset()]} palette"
        return ""
    if len(colors) == 1:
        mood = COLOR_MOODS[colors[0]]
    else:
        mood = ", ".join(COLOR_MOODS[c].split(",")[0] for c in colors)
    palette = COLOR_PALETTES.get(frozenset(colors)) or _PALETTE_FALLBACK_BY_COUNT.get(len(colors), "")
    return f"{mood}, {palette} palette" if palette else mood


def _trim_to_tokens(text: str, max_tokens: int, count_tokens=estimate_tokens) -> str:
    """Keep the leading words of text that fit in max_tokens."""
    if count_tokens(text) <= max_tokens:
        return text
    kept: list[str] = []
    for word in text.split():
        if count_tokens(" ".join(kept + [word])) > max_tokens:
            break
        kept.append(word)
    return " ".join(kept).rstrip(" ,;:.-")


def build_art_prompt(prompt: str, card_data: dict | None, count_tokens=estimate_tokens) -> tuple[str, str]:
    """(positive prompt of at most 75 CLIP tokens, subject first; negative prompt).

    count_tokens defaults to the rough estimate_tokens, which undercounts rare words and
    names; generate_art passes the pipeline's real CLIP tokenizer so the tail isn't cut off.

    The positive prompt is ordered: subject, art style, type context, color mood and
    palette. When the budget is tight the subject is trimmed (keeping its first words,
    never below SUBJECT_MIN_TOKENS), then the palette, type context and style are
    dropped in that order.
    """
    subject = " ".join(str(prompt or "").split()).strip(" ,;")
    if not subject:
        subject = str((card_data or {}).get("name") or "").strip() or "a fantasy scene"

    type_context = _type_context(card_data)
    color_part = _color_part(card_data)
    tail = [p for p in (ART_STYLE, type_context, color_part) if p]

    def tail_tokens(parts: list[str]) -> int:
        return sum(count_tokens(p) + 1 for p in parts)  # +1 for the joining comma

    subject = _trim_to_tokens(subject, max(MAX_PROMPT_TOKENS - tail_tokens(tail), SUBJECT_MIN_TOKENS),
                              count_tokens)

    # Drop lowest-priority tail parts until everything fits.
    for part in [p for p in (color_part, type_context, ART_STYLE) if p]:
        if count_tokens(subject) + tail_tokens(tail) <= MAX_PROMPT_TOKENS:
            break
        tail.remove(part)

    positive = ", ".join([subject] + tail)
    # Final guard; a no-op because the budget above already holds.
    positive = truncate_prompt_smartly(positive, max_tokens=MAX_PROMPT_TOKENS)
    return positive, NEGATIVE_PROMPT


# ===== PIPELINE =====

_pipeline = None
_load_lock = threading.Lock()
# The pipeline's scheduler holds per-run state, so only one inference runs at a time
# (the legacy routes can call createCardImage from several threads).
_inference_lock = threading.Lock()


def _get_pipeline():
    """Lazily load the SDXL Lightning pipeline once. Raises if loading fails, so the
    next call retries instead of caching the failure."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _load_lock:
        if _pipeline is not None:
            return _pipeline

        import torch
        from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

        use_cuda = config.USE_CUDA and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        print(f"🔄 Loading image model {config.IMAGE_MODEL_ID} on {device} ({dtype})... "
              f"(the first run downloads it)")
        start = time.perf_counter()

        extra = {}
        if config.IMAGE_VAE_ID:
            extra["vae"] = AutoencoderKL.from_pretrained(config.IMAGE_VAE_ID, torch_dtype=dtype)

        pipe = None
        if use_cuda:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    config.IMAGE_MODEL_ID, torch_dtype=dtype, variant="fp16", **extra)
            except (OSError, ValueError) as e:
                print(f"⚠️ No fp16 variant for {config.IMAGE_MODEL_ID} ({e}); loading default weights as fp16")
        if pipe is None:
            pipe = StableDiffusionXLPipeline.from_pretrained(config.IMAGE_MODEL_ID, torch_dtype=dtype, **extra)

        # Lightning models are distilled on trailing timesteps. Euler ancestral paints softer
        # edges and gradients than DPM++ 2M SDE Karras, which looked crunchy and over-sharpened.
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing")
        if use_cuda and config.IMAGE_CPU_OFFLOAD:
            # Must replace .to("cuda"); see config.IMAGE_CPU_OFFLOAD for why.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        pipe.enable_vae_slicing()
        pipe.set_progress_bar_config(disable=True)

        _pipeline = pipe
        print(f"✅ Image model loaded on {device} in {time.perf_counter() - start:.1f}s")
        return _pipeline


def _png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_art(prompt: str, card_data: dict | None) -> str:
    """Raw base64 PNG (no data: prefix) at config.ART_BOX_SIZE.

    Placeholder mode returns a solid gray image without importing torch. Any model
    failure raises, so callers can mark the card failed.
    """
    if config.MODEL_SIZE == "placeholder":
        return _png_base64(Image.new("RGB", tuple(config.ART_BOX_SIZE), color=(50, 50, 50)))

    pipe = _get_pipeline()

    def count_tokens(text: str) -> int:
        return len(pipe.tokenizer(text).input_ids) - 2  # minus the start/end tokens

    positive, negative = build_art_prompt(prompt, card_data, count_tokens)
    width, height = config.IMAGE_GEN_SIZE
    print(f"🎨 Generating art ({count_tokens(positive)} tokens): {positive}")

    with _inference_lock:
        start = time.perf_counter()
        image = pipe(
            prompt=positive,
            negative_prompt=negative,
            num_inference_steps=config.IMAGE_STEPS,
            guidance_scale=config.IMAGE_GUIDANCE,
            width=width,
            height=height,
        ).images[0]
        print(f"⚡ Art inference took {time.perf_counter() - start:.2f}s")

    return _png_base64(image.convert("RGB").resize(tuple(config.ART_BOX_SIZE), Image.LANCZOS))
