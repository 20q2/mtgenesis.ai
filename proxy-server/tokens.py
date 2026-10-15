"""
Token Maker: a token's params, its funny flavor text, and the finished token.

A token is stored like any card (cards table, generation queue, gallery) with
card_params["kind"] == "token". It skips the card director and the rules-text writer: the
player types the name, type line, P/T and any abilities, picks a frame
(token_renderer.FRAMES), and the AI paints the art and writes a short funny flavor text
(and a name, when the name is left blank). A flavor reply that fails never fails the
token: it is simply printed without flavor text.
"""
from __future__ import annotations

import json
import random
import re

import token_renderer


def _log(message: str) -> None:
    """Best-effort console logging: an unprintable message (emoji on a cp1252 console) must
    never break a token."""
    try:
        print(message)
    except Exception:
        pass

KIND = "token"
TYPES = ("Creature", "Artifact Creature", "Enchantment Creature", "Artifact", "Enchantment", "Land")
COLORS = "WUBRG"
NAME_MAX = 30
SUBTYPE_MAX = 40
TEXT_MAX = 300
FLAVOR_MAX = 160
ART_MAX = 400
PT_RE = re.compile(r"^(\d{1,2}|\*|X)$")
# Art render sizes: SDXL sizes near one megapixel, multiples of 64.
SDXL_SIZES = [(832, 1216), (896, 1152), (1024, 1024), (1088, 896), (1152, 832), (1216, 832)]
ART_SCALE = 0.5   # stored art: half the frame's template art box (about 690x940 full bleed)

FLAVOR_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "quote": {"type": "string"},
                   "speaker": {"type": "string"}},
    "required": ["name", "quote", "speaker"],
}

# Picked at random per token so a night of Goblin tokens doesn't get one joke.
FLAVOR_ANGLES = [
    "a deadpan complaint from the token itself",
    "an exasperated comment from the wizard who summoned it",
    "a misunderstanding about what the token is for",
    "a workplace or bureaucracy joke set in a fantasy world",
    "an overconfident boast that is obviously about to go wrong",
    "a tiny, petty grudge treated as an epic saga",
    "a field guide entry that gets distracted",
    "an absurdly specific rule or warning sign",
    "a pun on the token's name or creature type",
    "a parent, teacher or boss scolding it",
]

TONE_EXAMPLES = [
    "The squirrels have unionized. Their first demand is acorns. Their second demand is also acorns.",
    "“It's not a cage. It's a commitment.” —ogre zookeeper",
    "Every zombie remembers one thing from life. Gerald remembers where he parked.",
    "“I asked for one spirit. The form said 'list all that apply.'” —junior necromancer",
]
COPY_OVERLAP = 0.5   # word overlap with a tone example at or above this = a copied joke
# Creature types the model likes to slip into names ("Goblin Gold Hoarder" for a Treasure)
NAME_TYPES = {"goblin", "elf", "zombie", "dragon", "spirit", "squirrel", "soldier", "angel", "demon",
              "vampire", "wizard", "knight", "beast", "cat", "dog", "drake", "saproling", "elemental",
              "faerie", "rat", "bird", "insect", "ogre", "giant", "human", "dwarf", "merfolk", "pirate"}

SYSTEM_PROMPT = """You write flavor text for custom Magic: The Gathering tokens. Flavor text is the short italic line at the bottom of a card: it never explains the rules, it adds character.
Make it genuinely funny: witty, dry and specific to this token, in the voice of real Magic flavor text (think Unhinged and the funniest Goblin cards). One or two short sentences, at most 22 words. No rules text or game jargon (mana, graveyard, battlefield), no hashtags or emoji, nothing crude.
Return JSON only: {"name": "...", "quote": "...", "speaker": "..."}.
- quote: the flavor text itself, without quotation marks.
- speaker: who says it, if it is a quote ("Krenko, mob boss", "a tired summoner"), at most 5 words; "" when the line is narration.
- name: only when asked for one: a short, playful token name, 1 to 3 words, built on its subtype. Never add a creature type it doesn't have. Otherwise "".
The tone, in lines about other creatures (write a new joke about this token, never one of these):
""" + "\n".join(f"- {line}" for line in TONE_EXAMPLES) + "\n"


class TokenError(ValueError):
    """A request the Token Maker can't make; the message is shown to the player."""


def is_token(params: dict | None) -> bool:
    return bool(params) and params.get("kind") == KIND


def _text(value, limit: int, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TokenError(f"{label} must be text")
    value = " ".join(value.split()) if "\n" not in value else "\n".join(
        " ".join(line.split()) for line in value.split("\n") if line.strip())
    if len(value) > limit:
        raise TokenError(f"{label} must be {limit} characters or less")
    return value


def token_params(data: dict) -> dict:
    """Validate the Token Maker form into card params. Raises TokenError."""
    if not isinstance(data, dict):
        raise TokenError("token must be an object")
    frame = data.get("frame") or token_renderer.DEFAULT_FRAME
    if frame not in token_renderer.FRAMES_BY_ID:
        raise TokenError("Unknown frame")
    card_type = data.get("type") or "Creature"
    if card_type not in TYPES:
        raise TokenError(f"type must be one of: {', '.join(TYPES)}")
    raw_colors = data.get("colors") or []
    if not isinstance(raw_colors, list) or any(c not in COLORS for c in raw_colors):
        raise TokenError("colors must be a list of W, U, B, R, G")
    params = {
        "kind": KIND,
        "frame": frame,
        "name": _text(data.get("name"), NAME_MAX, "Name"),
        "type": card_type,
        "subtype": _text(data.get("subtype"), SUBTYPE_MAX, "Subtype"),
        "colors": [c for c in COLORS if c in raw_colors],
        "description": _text(data.get("description"), TEXT_MAX, "Abilities"),
        "flavorText": _text(data.get("flavorText"), FLAVOR_MAX, "Flavor text"),
        "artPrompt": _text(data.get("artPrompt"), ART_MAX, "Art description").replace("\n", " "),
        # Whether the AI fills a blank name / flavor text; off, the field stays blank on the card
        "aiName": data.get("aiName") is not False,
        "aiFlavor": data.get("aiFlavor") is not False,
        # The rest of the app reads these from every card
        "manaCost": "", "cmc": 0, "rarity": "common",
    }
    if "creature" in card_type.lower():
        # Optional: both or neither (a creature token without them prints no P/T box)
        pt = [str(data.get(k) if data.get(k) is not None else "").strip().upper()
              for k in ("power", "toughness")]
        if any(pt):
            if not all(PT_RE.match(v) for v in pt):
                raise TokenError("Power and toughness go together: 0-99, * or X")
            params["power"], params["toughness"] = pt
    return params


def preview_params(data: dict) -> dict:
    """Lenient params for the live preview: anything unfinished is just left off. A blank
    name the AI will fill shows the subtype, which the finished token prints if the AI's
    name doesn't come through."""
    params = _preview_params(data)
    if not params.get("name") and params.get("aiName", True):
        params["name"] = params.get("subtype") or "Token"
    return params


def _preview_params(data: dict) -> dict:
    try:
        return token_params(data)
    except TokenError:
        pass
    data = data if isinstance(data, dict) else {}
    card_type = data.get("type") if data.get("type") in TYPES else "Creature"
    frame = data.get("frame") if data.get("frame") in token_renderer.FRAMES_BY_ID else token_renderer.DEFAULT_FRAME
    colors = data.get("colors") if isinstance(data.get("colors"), list) else []
    clip = lambda key, limit: str(data.get(key) or "")[:limit]
    params = {"kind": KIND, "frame": frame, "type": card_type,
              "colors": [c for c in COLORS if c in colors],
              "name": clip("name", NAME_MAX), "subtype": clip("subtype", SUBTYPE_MAX),
              "description": clip("description", TEXT_MAX), "flavorText": clip("flavorText", FLAVOR_MAX),
              "aiName": data.get("aiName") is not False, "aiFlavor": data.get("aiFlavor") is not False}
    if "creature" in card_type.lower():
        for key in ("power", "toughness"):
            value = str(data.get(key) or "").strip().upper()
            params[key] = value if PT_RE.match(value) else ""
    return params


def art_prompt(params: dict) -> str:
    """The subject handed to image_generation.build_art_prompt."""
    if params.get("artPrompt"):
        return params["artPrompt"]
    card_type = (params.get("type") or "Creature").lower()
    subject = params.get("subtype") or params.get("name") or (
        "mysterious creature" if "creature" in card_type else f"mysterious {card_type}")
    return f"{subject}, {params.get('name')}" if params.get("name") and params.get("subtype") else subject


def art_sizes(params: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    """(SDXL render size, stored art size) shaped like the frame's art box."""
    frame = token_renderer.FRAMES_BY_ID.get(params.get("frame"), token_renderer.FRAMES_BY_ID[token_renderer.DEFAULT_FRAME])
    aspect = token_renderer.art_aspect(frame.id)
    render = min(SDXL_SIZES, key=lambda s: abs(s[0] / s[1] - aspect))
    x0, y0, x1, y1 = frame.art_box
    return render, (round((x1 - x0) * ART_SCALE), round((y1 - y0) * ART_SCALE))


def art_card_data(params: dict) -> dict:
    """What build_art_prompt reads (type context, colors) for a token."""
    return {"type": params.get("type") or "Creature", "subtype": params.get("subtype") or "",
            "colors": params.get("colors") or [], "name": params.get("name") or ""}


# ----- flavor text -----

def build_messages(params: dict, rng: random.Random) -> list[dict]:
    facts = [f"Type line: {token_renderer.type_line(params)}"]
    if params.get("name"):
        facts.insert(0, f"Name: {params['name']}")
    colors = params.get("colors") or []
    facts.append("Colors: " + (", ".join(colors) if colors else "colorless"))
    if params.get("type", "").lower().find("creature") >= 0 and params.get("power"):
        facts.append(f"Power/toughness: {params['power']}/{params['toughness']}")
    if params.get("description"):
        facts.append("Abilities: " + params["description"].replace("\n", "; "))
    if params.get("artPrompt"):
        facts.append(f"Its painting shows: {params['artPrompt']}")
    wants_name = not params.get("name") and params.get("aiName", True)
    ask = "Also give it a name." if wants_name else "Leave name empty."
    angle = rng.choice(FLAVOR_ANGLES)
    user = "\n".join(facts) + f"\n\nWrite the flavor text as {angle}. {ask}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def copies_an_example(quote: str) -> bool:
    words = _words(quote)
    for example in TONE_EXAMPLES:
        other = _words(example)
        if words and len(words & other) / len(words | other) >= COPY_OVERLAP:
            return True
    return False


def clean_flavor(quote: str, speaker: str) -> str:
    """Curly quotes and an em-dash attribution line, or '' when the reply is unusable."""
    quote = " ".join(re.sub(r"[*_]", "", str(quote or "")).split()).strip().strip("\"“”").strip()
    if len(quote) > 1 and quote[0] in "'‘" and quote[-1] in "'’":
        quote = quote[1:-1].strip()
    speaker = " ".join(str(speaker or "").split()).strip().lstrip("-—–~ ").strip("\"'“”")
    if not quote or len(quote) > FLAVOR_MAX:
        return ""
    if "{" in quote or re.search(r"\b(mana|graveyard|battlefield|this creature)\b", quote, re.I):
        return ""  # rules text, not flavor
    if copies_an_example(quote):
        return ""
    if speaker and len(speaker) <= 40 and len(quote) + len(speaker) <= FLAVOR_MAX:
        return f"“{quote}”\n—{speaker}"
    return quote


def clean_name(name: str, subtype: str = "") -> str:
    """The AI's name, or '' when it is too long or adds a creature type the token lacks."""
    name = " ".join(str(name or "").split()).strip("\"'“”.,!")
    if not 0 < len(name) <= NAME_MAX:
        return ""
    extra = (_words(name) & NAME_TYPES) - {w.rstrip("s") for w in _words(subtype)} - _words(subtype)
    return "" if extra else name


def generate_flavor(params: dict, client, model: str, *, rng=None, keep_alive="30m") -> str:
    """JSON {"name", "flavorText"} for the queue's text half. Never raises and never empty:
    a failed reply gives "{}" and the token prints without flavor (or named by its type)."""
    rng = rng or random.Random()
    want_flavor = not params.get("flavorText") and params.get("aiFlavor", True)
    want_name = not params.get("name") and params.get("aiName", True)
    out: dict = {}
    if want_flavor or want_name:
        for _ in range(2):
            try:
                resp = client.chat(
                    model=model, messages=build_messages(params, rng), format=FLAVOR_SCHEMA,
                    think=False, keep_alive=keep_alive,
                    # The same num_ctx as rules text and the director: another value makes
                    # Ollama reload the model between calls.
                    options={"temperature": 1.0, "top_p": 0.95, "num_predict": 120, "num_ctx": 2560})
                raw = resp["message"]["content"]
                _log(f"🃏 Token flavor raw reply: {raw!r}")
                reply = json.loads(raw)
            except Exception as e:
                _log(f"⚠️ Token flavor failed: {e}")
                continue
            flavor = clean_flavor(reply.get("quote"), reply.get("speaker")) if want_flavor else ""
            name = clean_name(reply.get("name"), params.get("subtype") or "") if want_name else ""
            if flavor:
                out["flavorText"] = flavor
            if name:
                out["name"] = name
            if (flavor or not want_flavor) and (name or not want_name):
                break
    return json.dumps(out)


def finalize_token(params: dict, generated_text: str | None, art_b64: str | None,
                   render=token_renderer.render_token) -> tuple[dict, str]:
    """The queue's render half: merge the AI's name and flavor, then draw the token."""
    card = dict(params)
    try:
        reply = json.loads(generated_text or "{}")
    except (TypeError, ValueError):
        reply = {}
    if isinstance(reply, dict):
        if not card.get("name") and reply.get("name"):
            card["name"] = reply["name"]
        if not card.get("flavorText") and reply.get("flavorText"):
            card["flavorText"] = reply["flavorText"]
    if not card.get("name") and card.get("aiName", True):
        card["name"] = card.get("subtype") or "Token"  # the AI's name didn't come through
    return card, render(card, art_b64)
