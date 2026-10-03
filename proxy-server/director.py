"""
Card director: a short brief per card that steers both the rules text and the art.

Spec: docs/superpowers/specs/2026-10-02-card-director-design.md.
The brief says who the card is (identity), what its abilities revolve around (mechanic)
and what its painting shows (art). Each brief gets random on-color ingredients (two design
hooks, an ability shape, sometimes a twist) so similar cards still come out different. A
commander set gets three briefs in one call, each with a different mechanic. Players never
see a brief. Any failure returns None, and the
card is then generated exactly as it was before the director existed.
"""
from __future__ import annotations

import itertools
import json
import random
import re

import power_level as power
from rules_text import COLOR_HOOKS, COLORLESS_HOOKS, DESIGN_GUIDE, POWER_GUIDE, type_line

MECHANIC_MAX_OVERLAP = 0.5   # content-word Jaccard at or above this = "the same mechanic"
ART_FIELDS = ("subject", "action", "setting", "framing", "light")
IDENTITY_MAX_WORDS = 20
MECHANIC_MAX_WORDS = 15
ART_MAX_WORDS = 12
ATTEMPTS = 2

# Ingredients: which kind of ability carries the mechanic, and an optional twist. Drawn per
# brief (no repeats within a set) so the model can't settle on the same few favorites. The
# shape skips the director and goes to the rules-text writer with the brief: shown to the
# director it made mechanics read like rules text ("When you cast a spell, ...").
SHAPES = ("a trigger when it enters", "an attack or combat damage trigger",
          "a trigger when it dies or leaves the battlefield", "an activated ability with a real cost",
          "a static ability", "a trigger at the beginning of your upkeep or end step",
          "a trigger when you cast a spell")
LAND_SHAPES = ("a trigger when it enters", "an activated ability with a real cost", "a static ability")
# No "named counter type" (the model wrote "Whenever an Ash Counter exists, ...") and no
# "choice between two modes" (it stacked two effects on one ability and went over budget).
TWISTS = ("add a real drawback or extra cost",
          "interact with opponents' creatures or cards", "scale with something you count",
          "involve the graveyard")
TWIST_CHANCE = 0.5

_STOPWORDS = {"the", "and", "for", "with", "your", "you", "each", "that", "this", "from",
              "into", "are", "its", "when", "whenever", "card", "cards", "creature", "creatures"}
# Belt and braces with image_generation.NEGATIVE_PROMPT: never hand these to SDXL.
_UNSAFE_ART = re.compile(r"\b(?:nude|naked|topless|shirtless|bare[- ]chest(?:ed)?|cleavage|nsfw)\b",
                         re.I)

SYSTEM_PROMPT = """You are the creative director for a custom Magic: The Gathering card. From the card's name, type line, colors, cost and rarity, invent who or what this card is and what its rules text should revolve around, then describe its painting.
Return JSON only: {"briefs": [{"identity": "...", "mechanic": "...", "art": {"subject": "...", "action": "...", "setting": "...", "framing": "...", "light": "..."}}]}.
- identity: who or what the card is, in at most 20 words. Draw on the name and subtype.
- mechanic: a short theme for its abilities, at most 8 words, not rules text (for example "sacrifice tokens to drain opponents"). Build it from that brief's ingredients: one of its themes, with its twist if it has one. It must fit the power budget: cheap or common cards get small, simple mechanics.
- The mechanic grows out of who the card is: a Gravecaller raises the dead, a Stormherald rewards casting spells. Fuse the name and subtype with the ingredients into one flavorful idea.
- Use only real Magic game objects: creatures, tokens, counters, cards, life, mana, the graveyard, the library. Never invent new zones, realms or rules.
- art: a painting brief. subject starts with who they are (for people: age and gender, e.g. "an old elf woman"), names the creature type, then what it looks like; action is what it is doing; setting is where; framing is the camera (for example "low angle, close"); light is the light source and mood. Each at most 12 words. Never put the card's name in the art fields. For people, describe their clothing or armor, never bare skin or their body.
Make each brief specific to this card. Avoid generic fantasy filler.

The rules-text writer turns your mechanic into abilities under these rules, so design within them:
""" + POWER_GUIDE + "\n\n" + DESIGN_GUIDE


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


def _shapes(card: dict) -> tuple[str, ...]:
    """Ability shapes that make sense for the card type (none for spells and planeswalkers)."""
    card_type = (card.get("type") or "").lower()
    subtype = (card.get("subtype") or "").lower().split()
    if any(t in card_type for t in ("instant", "sorcery", "planeswalker")):
        return ()
    if "land" in card_type:
        return LAND_SHAPES
    if "creature" in card_type or {"vehicle", "equipment", "aura"} & set(subtype):
        return SHAPES
    return tuple(s for s in SHAPES if "attack" not in s)


def draw_ingredients(card: dict, count: int, rng: random.Random) -> list[dict]:
    """Per brief: two on-color design hooks, an ability shape (None for spells and
    planeswalkers) and sometimes a twist. Within a set no hook, shape or twist repeats while
    there are fresh ones left. Multicolor cards take a hook from each of two of their colors."""
    card = card or {}
    colors = [c for c in (card.get("colors") or []) if c in "WUBRG"]
    used: set[str] = set()

    def pick(pool: list[str], taken: list[str]) -> str:
        fresh = [h for h in pool if h not in used and h not in taken]
        choice = rng.choice(fresh or [h for h in pool if h not in taken] or pool)
        used.add(choice)
        return choice

    shapes = list(_shapes(card))
    rng.shuffle(shapes)
    twists = list(TWISTS)
    rng.shuffle(twists)
    result = []
    for n in range(count):
        if not colors:
            pools = [COLORLESS_HOOKS, COLORLESS_HOOKS]
        elif len(colors) == 1:
            pools = [COLOR_HOOKS[colors[0]]] * 2
        else:
            pools = [COLOR_HOOKS[c] for c in rng.sample(colors, 2)]
        hooks: list[str] = []
        for pool in pools:
            hooks.append(pick(pool, hooks))
        shape = shapes[n % len(shapes)] if shapes else None
        twist = twists.pop() if twists and rng.random() < TWIST_CHANCE else None
        result.append({"hooks": hooks, "shape": shape, "twist": twist})
    return result


def _describe(n: int, ingredients: dict) -> str:
    parts = [f"Brief {n}: build the mechanic on one of these themes: {'; '.join(ingredients['hooks'])}."]
    if ingredients["twist"]:
        parts.append(f"Twist: {ingredients['twist']}.")
    return " ".join(parts)


def _messages(card: dict, count: int, avoid: list[dict], rng: random.Random | None = None,
              ingredients: list[dict] | None = None) -> list[dict]:
    card = card or {}
    colors = [c for c in (card.get("colors") or []) if c in "WUBRG"]
    facts = [f"Name: {(card.get('name') or '').strip() or 'Untitled'}",
             f"Type line: {type_line(card) or 'Creature'}",
             "Colors: " + (", ".join({"W": "white", "U": "blue", "B": "black", "R": "red",
                                      "G": "green"}[c] for c in colors) or "colorless"),
             f"Mana cost: {card.get('manaCost') or '{0}'} (mana value {card.get('cmc', 0)})",
             f"Rarity: {(card.get('rarity') or 'common').lower()}",
             power.describe_budget(card)]
    ingredients = ingredients or draw_ingredients(card, count, rng or random.Random())
    asks = [f"Write {count} brief{'s' if count > 1 else ''}, in this order. Ingredients, rolled at "
            "random so this card is unlike any other:"]
    asks += [_describe(n, ing) for n, ing in enumerate(ingredients, 1)]
    if count > 1:
        asks.append("Each brief must use a different mechanic.")
        asks.append("All briefs show the same character: give every brief the same art.subject "
                    "(identical appearance) and vary only action, setting, framing and light. "
                    "Each brief takes place in a clearly different setting.")
    for other in avoid:
        if other.get("mechanic"):
            asks.append(f"Use a mechanic different from: {other['mechanic']}")
        if isinstance(other.get("art"), dict) and other["art"].get("setting"):
            asks.append(f"Use a setting different from: {other['art']['setting']}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(facts) + "\n\n" + "\n".join(asks)}]


def _words(text: str, limit: int) -> str:
    return " ".join(text.split()[:limit])


def _clean_art(text: str) -> str:
    text = _UNSAFE_ART.sub("", text)
    text = re.sub(r"\s+,", ",", re.sub(r"\s{2,}", " ", text))
    return text.strip(" ,")


def _strip_name(text: str, name: str) -> str:
    """Drop a leading card name ("Zur'ka, Élan of Ash, a cleric" -> "a cleric"): CLIP can't use it."""
    for candidate in (name, name.split(",")[0]):
        candidate = candidate.strip()
        if candidate and text.lower().startswith(candidate.lower()):
            return text[len(candidate):].lstrip(" ,:-")
    return text


def _clean(raw: dict, subtype: str, name: str = "") -> dict | None:
    """One validated, trimmed brief, or None if a field is missing or empty."""
    if not isinstance(raw, dict) or not isinstance(raw.get("art"), dict):
        return None
    texts = [raw.get("identity"), raw.get("mechanic"), *(raw["art"].get(f) for f in ART_FIELDS)]
    if not all(isinstance(t, str) and t.strip() for t in texts):
        return None
    art = {f: _clean_art(_strip_name(raw["art"][f], name)) for f in ART_FIELDS}
    sub_words = re.findall(r"[a-z]+", subtype.lower())
    if sub_words and not set(sub_words) & set(re.findall(r"[a-z]+", art["subject"].lower())):
        art["subject"] = f"a {' '.join(sub_words)}, {art['subject']}"
    if not all(art.values()):
        return None
    return {"identity": _words(raw["identity"], IDENTITY_MAX_WORDS),
            "mechanic": _words(raw["mechanic"], MECHANIC_MAX_WORDS),
            "art": {f: _words(v, ART_MAX_WORDS) for f, v in art.items()}}


def _distinct(briefs: list[dict], avoid: list[dict]) -> bool:
    """Mechanics and settings differ between the briefs and from `avoid` (versions already
    written): one character, but each version plays and looks different."""
    def differ(texts: list[str], avoided: list[str]) -> bool:
        words = [content_words(t) for t in texts]
        others = [content_words(t) for t in avoided if t]
        if any(jaccard(a, b) >= MECHANIC_MAX_OVERLAP for a, b in itertools.combinations(words, 2)):
            return False
        return all(jaccard(w, x) < MECHANIC_MAX_OVERLAP for w in words for x in others)

    def setting(b: dict) -> str:
        return (b.get("art") or {}).get("setting") or "" if isinstance(b.get("art"), dict) else ""

    return (differ([b["mechanic"] for b in briefs], [a.get("mechanic") or "" for a in avoid])
            and differ([b["art"]["setting"] for b in briefs], [setting(a) for a in avoid]))


def _same_character(briefs: list[dict], avoid: list[dict]) -> None:
    """A commander set is one character: every version keeps one appearance (the set's first
    brief, or for a reroll the existing versions'), so only action, setting, framing and light vary."""
    kept = [a["art"]["subject"] for a in avoid if isinstance(a.get("art"), dict) and a["art"].get("subject")]
    subject = kept[0] if kept else briefs[0]["art"]["subject"]
    for b in briefs:
        b["art"]["subject"] = subject


def write_briefs(card: dict, count: int, avoid: list[dict] | None, client, model: str,
                 rng: random.Random | None = None) -> list[dict] | None:
    """`count` briefs for the card (one per commander-set version), each with a mechanic
    different from the others and from `avoid`. None if two attempts fail; never raises.
    `rng` draws the ingredients (seed it for repeatable runs); a retry draws fresh ones."""
    avoid = avoid or []
    rng = rng or random.Random()
    subtype = (card or {}).get("subtype") or ""
    for _ in range(ATTEMPTS):
        ingredients = draw_ingredients(card, count, rng)
        try:
            resp = client.chat(
                model=model, messages=_messages(card, count, avoid, ingredients=ingredients),
                format=_schema(count),
                think=False, keep_alive="30m",
                # Small context, like the rules-text call: SDXL must still fit beside the model.
                options={"temperature": 0.9, "top_p": 0.95, "num_predict": 200 * count + 100,
                         "num_ctx": 2560})
        except Exception as exc:  # timeout, connection refused, unknown model
            # Art waits for the brief, so a slow or missing Ollama is not retried.
            _log(f"🎬 Director call failed: {exc}")
            return None
        try:
            raw = json.loads(resp["message"]["content"])
            briefs = [_clean(b, subtype, (card or {}).get("name") or "") for b in raw.get("briefs") or []]
        except Exception as exc:  # bad JSON or shape: worth one more try
            _log(f"🎬 Director reply unreadable: {exc}")
            continue
        if len(briefs) == count and all(briefs) and _distinct(briefs, avoid):
            _same_character(briefs, avoid)
            for b, ing in zip(briefs, ingredients):
                if ing["shape"]:
                    b["shape"] = ing["shape"]
            _log(f"🎬 Director briefs: {[b['mechanic'] for b in briefs]}")
            return briefs
        _log("🎬 Director reply rejected (missing fields or repeated mechanics)")
    return None
