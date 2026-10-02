"""
Card director: a short brief per card that steers both the rules text and the art.

Spec: docs/superpowers/specs/2026-10-02-card-director-design.md.
The brief says who the card is (identity), what its abilities revolve around (mechanic)
and what its painting shows (art). A commander set gets three briefs in one call, each
with a different mechanic. Players never see a brief. Any failure returns None, and the
card is then generated exactly as it was before the director existed.
"""
from __future__ import annotations

import itertools
import json
import re

import power_level as power
from rules_text import type_line

MECHANIC_MAX_OVERLAP = 0.5   # content-word Jaccard at or above this = "the same mechanic"
ART_FIELDS = ("subject", "action", "setting", "framing", "light")
IDENTITY_MAX_WORDS = 20
MECHANIC_MAX_WORDS = 15
ART_MAX_WORDS = 12
ATTEMPTS = 2

_STOPWORDS = {"the", "and", "for", "with", "your", "you", "each", "that", "this", "from",
              "into", "are", "its", "when", "whenever", "card", "cards", "creature", "creatures"}
# Belt and braces with image_generation.NEGATIVE_PROMPT: never hand these to SDXL.
_UNSAFE_ART = re.compile(r"\b(?:nude|naked|topless|shirtless|bare[- ]chest(?:ed)?|cleavage|nsfw)\b",
                         re.I)

SYSTEM_PROMPT = """You are the creative director for a custom Magic: The Gathering card. From the card's name, type line, colors, cost and rarity, invent who or what this card is and what its rules text should revolve around, then describe its painting.
Return JSON only: {"briefs": [{"identity": "...", "mechanic": "...", "art": {"subject": "...", "action": "...", "setting": "...", "framing": "...", "light": "..."}}]}.
- identity: who or what the card is, in at most 20 words. Draw on the name and subtype.
- mechanic: what its abilities revolve around, in at most 15 words, in Magic terms (for example "sacrifice tokens to drain each opponent"). Fit the power budget: cheap or common cards get small, simple mechanics.
- art: a painting brief. subject names the creature type and what it looks like; action is what it is doing; setting is where; framing is the camera (for example "low angle, close"); light is the light source and mood. Each at most 12 words. People are always fully clothed.
Make each brief specific to this card. Avoid generic fantasy filler."""


def _log(message: str) -> None:
    """Print without ever raising (a console that can't encode the emoji must not lose a brief)."""
    try:
        print(message)
    except Exception:
        pass


def content_words(text: str) -> set[str]:
    """Lowercase words of 3+ letters, minus filler, for comparing mechanics and rules text."""
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in _STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def set_overlap(texts: list[str]) -> float:
    """Mean pairwise content-word Jaccard of several texts (0.0 for fewer than two). The e2e
    tools use it to measure how alike a commander set's three versions read."""
    pairs = list(itertools.combinations([content_words(t) for t in texts], 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs) if pairs else 0.0


def _schema(count: int) -> dict:
    art = {"type": "object", "properties": {f: {"type": "string"} for f in ART_FIELDS},
           "required": list(ART_FIELDS)}
    item = {"type": "object",
            "properties": {"identity": {"type": "string"}, "mechanic": {"type": "string"}, "art": art},
            "required": ["identity", "mechanic", "art"]}
    return {"type": "object",
            "properties": {"briefs": {"type": "array", "items": item, "minItems": count,
                                      "maxItems": count}},
            "required": ["briefs"]}


def _messages(card: dict, count: int, avoid: list[dict]) -> list[dict]:
    card = card or {}
    colors = [c for c in (card.get("colors") or []) if c in "WUBRG"]
    facts = [f"Name: {(card.get('name') or '').strip() or 'Untitled'}",
             f"Type line: {type_line(card) or 'Creature'}",
             "Colors: " + (", ".join({"W": "white", "U": "blue", "B": "black", "R": "red",
                                      "G": "green"}[c] for c in colors) or "colorless"),
             f"Mana cost: {card.get('manaCost') or '{0}'} (mana value {card.get('cmc', 0)})",
             f"Rarity: {(card.get('rarity') or 'common').lower()}",
             power.describe_budget(card)]
    asks = [f"Write {count} brief{'s' if count > 1 else ''}."]
    if count > 1:
        asks.append("Each brief must use a different mechanic.")
    for other in avoid:
        if other.get("mechanic"):
            asks.append(f"Use a mechanic different from: {other['mechanic']}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(facts) + "\n\n" + "\n".join(asks)}]


def _words(text: str, limit: int) -> str:
    return " ".join(text.split()[:limit])


def _clean_art(text: str) -> str:
    text = _UNSAFE_ART.sub("", text)
    text = re.sub(r"\s+,", ",", re.sub(r"\s{2,}", " ", text))
    return text.strip(" ,")


def _clean(raw: dict, subtype: str) -> dict | None:
    """One validated, trimmed brief, or None if a field is missing or empty."""
    if not isinstance(raw, dict) or not isinstance(raw.get("art"), dict):
        return None
    texts = [raw.get("identity"), raw.get("mechanic"), *(raw["art"].get(f) for f in ART_FIELDS)]
    if not all(isinstance(t, str) and t.strip() for t in texts):
        return None
    art = {f: _clean_art(raw["art"][f]) for f in ART_FIELDS}
    sub_words = re.findall(r"[a-z]+", subtype.lower())
    if sub_words and not set(sub_words) & set(re.findall(r"[a-z]+", art["subject"].lower())):
        art["subject"] = f"a {' '.join(sub_words)}, {art['subject']}"
    if not all(art.values()):
        return None
    return {"identity": _words(raw["identity"], IDENTITY_MAX_WORDS),
            "mechanic": _words(raw["mechanic"], MECHANIC_MAX_WORDS),
            "art": {f: _words(v, ART_MAX_WORDS) for f, v in art.items()}}


def _distinct(briefs: list[dict], avoid: list[dict]) -> bool:
    words = [content_words(b["mechanic"]) for b in briefs]
    avoided = [content_words(a.get("mechanic") or "") for a in avoid]
    for a, b in itertools.combinations(words, 2):
        if jaccard(a, b) >= MECHANIC_MAX_OVERLAP:
            return False
    return all(jaccard(w, x) < MECHANIC_MAX_OVERLAP for w in words for x in avoided)


def write_briefs(card: dict, count: int, avoid: list[dict] | None, client, model: str) -> list[dict] | None:
    """`count` briefs for the card (one per commander-set version), each with a mechanic
    different from the others and from `avoid`. None if two attempts fail; never raises."""
    avoid = avoid or []
    subtype = (card or {}).get("subtype") or ""
    for _ in range(ATTEMPTS):
        try:
            resp = client.chat(
                model=model, messages=_messages(card, count, avoid), format=_schema(count),
                think=False, keep_alive="30m",
                # Small context, like the rules-text call: SDXL must still fit beside the model.
                options={"temperature": 0.9, "top_p": 0.95, "num_predict": 200 * count + 100,
                         "num_ctx": 2560})
            raw = json.loads(resp["message"]["content"])
            briefs = [_clean(b, subtype) for b in raw.get("briefs") or []]
        except Exception as exc:  # timeouts, connection errors, bad JSON
            _log(f"🎬 Director call failed: {exc}")
            continue
        if len(briefs) == count and all(briefs) and _distinct(briefs, avoid):
            _log(f"🎬 Director briefs: {[b['mechanic'] for b in briefs]}")
            return briefs
        _log("🎬 Director reply rejected (missing fields or repeated mechanics)")
    return None
