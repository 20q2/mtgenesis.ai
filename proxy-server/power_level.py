"""
Power level for generated cards: the creature stat curve, a rough "mana value" estimate
of a card's rules text, and the budget a card of a given mana value and rarity gets.

The estimate is a heuristic built from real Limited design rates (a 2-mana 2/2 is fair,
2 damage to a creature costs about 1 mana, "draw a card" about 1 mana, a repeating
trigger is worth about twice a one-shot one). It is deliberately simple: its job is to
catch cards that are clearly too strong for their cost (a 2-mana common that makes three
tokens every attack), not to price every card exactly.

Used by rules_text (the prompt states the budget; the linter flags cards far over it)
and by app.generate_creature_stats (the body the prompt promised is the body printed).
"""
from __future__ import annotations

import re

# Extra value a card may have over its mana value, by rarity (in mana).
RARITY_BONUS = {'common': 0.25, 'uncommon': 0.75, 'rare': 1.5, 'mythic': 3.0}

# Total power + toughness by mana value: slightly under a vanilla creature, so a
# creature's abilities have room in its budget (a 3-drop is a 2/3 or 3/2 with a bonus).
STAT_TOTAL = {0: 2, 1: 3, 2: 4, 3: 5, 4: 7, 5: 9, 6: 10, 7: 11, 8: 12}

# How far over budget a card may drift before it is flagged.
WARN_OVER = 1.0
ERROR_OVER = 2.0

_WORD_NUMS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
              'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'x': 3, 'that many': 2, 'up to one': 1,
              'up to two': 2, 'up to three': 3}

KEYWORD_VALUE = {
    'flying': 1.0, 'first strike': 0.5, 'double strike': 1.5, 'deathtouch': 0.75, 'lifelink': 0.5,
    'trample': 0.4, 'vigilance': 0.25, 'reach': 0.2, 'menace': 0.5, 'haste': 0.5, 'hexproof': 1.0,
    'indestructible': 1.5, 'flash': 0.5, 'defender': -1.0, 'ward': 0.5, 'protection from': 1.0,
    'prowess': 0.5, 'shroud': 0.75, 'skulk': 0.4, 'fear': 0.75, 'intimidate': 0.75, 'infect': 1.0,
}


def _num(word: str) -> int:
    word = (word or '').lower().strip()
    if word.isdigit():
        return int(word)
    return _WORD_NUMS.get(word, 1)


def creature_stats(card: dict) -> tuple[int, int]:
    """
    The printed power/toughness for a creature (or Vehicle) of this mana value and
    rarity. Deterministic, so the prompt can tell the model the body before it writes
    the abilities and finalize_card prints the same body.
    """
    card = card or {}
    mv = int(card.get('cmc') or 0)
    rarity = (card.get('rarity') or 'common').lower()
    total = STAT_TOTAL.get(mv, 12)
    total += 1 if rarity == 'mythic' else 0
    subtype = (card.get('subtype') or '').lower()
    if 'vehicle' in subtype:
        total += 1  # vehicles need crewing, so they get a slightly better body
    power = total // 2
    toughness = total - power
    # Evasive or aggressive creature types lean toward power, defensive ones toward toughness
    if re.search(r'\b(wall|treefolk|golem|construct|turtle)\b', subtype) and power > 1:
        power, toughness = power - 1, toughness + 1
    elif re.search(r'\b(goblin|berserker|warrior|dragon|demon|cat|rogue)\b', subtype) and toughness > 1:
        power, toughness = power + 1, toughness - 1
    return max(power, 0), max(toughness, 1)


def body_value(power: int, toughness: int) -> float:
    """Mana a vanilla creature with these stats is worth (a 2/2 is 2 mana, a 4/4 is 4)."""
    return (power + toughness) / 2


def budget(card: dict) -> float:
    """Mana-equivalent value the card's rules text may have."""
    card = card or {}
    mv = float(card.get('cmc') or 0)
    rarity = (card.get('rarity') or 'common').lower()
    card_type = (card.get('type') or '').lower()
    total = mv + RARITY_BONUS.get(rarity, 0.5)
    if 'land' in card_type:
        total = 1.0 + RARITY_BONUS.get(rarity, 0.5)  # a land's mana ability is free
    if 'creature' in card_type or 'vehicle' in (card.get('subtype') or '').lower():
        p, t = _stats(card)
        total -= body_value(p, t)
        # Real creatures almost always carry a small bonus on a fair body (a 2-mana 2/2
        # with "When this enters, scry 2"), so never budget a creature at nothing
        total = max(total, {'common': 0.75, 'uncommon': 1.0}.get(rarity, 1.25))
    return round(total, 2)


def _stats(card: dict) -> tuple[int, int]:
    try:
        return int(card.get('power')), int(card.get('toughness'))
    except (TypeError, ValueError):
        return creature_stats(card)


def ability_value(line: str) -> float:
    """Rough mana value of one ability line."""
    s = line.strip()
    low = s.lower()
    value = 0.0

    # A pure keyword line: every comma-separated part is a keyword
    parts = [p.strip() for p in low.rstrip('.').split(',') if p.strip()]
    kw_values = []
    for part in parts:
        match = next((v for kw, v in KEYWORD_VALUE.items() if part == kw or part.startswith(kw + ' ')), None)
        if match is None and not re.fullmatch(r'(equip|crew|enchant|cycling|kicker|flashback)\b.*', part):
            break
        kw_values.append(match or 0.0)
    else:
        if ':' not in s and len(parts) <= 6:
            return sum(kw_values)

    # Keywords granted inside a line ("has flying and first strike")
    for kw, v in KEYWORD_VALUE.items():
        if re.search(r'\b(has|have|gains?)\b[^.]*\b' + re.escape(kw) + r'\b', low):
            value += v * 0.6

    # Damage and life loss
    for m in re.finditer(r'deals? (\d+|x|that much) damage to ([^.,]+)', low):
        n = 2 if m.group(1) == 'that much' else _num(m.group(1))
        target = m.group(2)
        per = 0.5
        if 'each opponent' in target or 'each player' in target:
            per = 0.6
        elif 'each creature' in target or 'each other creature' in target:
            per = 1.0
        elif 'any target' in target or 'player' in target:
            per = 0.6
        value += n * per
    for m in re.finditer(r'loses? (\d+|x) life', low):
        value += _num(m.group(1)) * (0.5 if 'each opponent' in low else 0.35)
    for m in re.finditer(r'gain (\d+|x) life', low):
        value += _num(m.group(1)) * 0.15
    if re.search(r'gain life equal to', low):
        value += 0.6

    # Cards
    for m in re.finditer(r'draws? (a|one|two|three|four|\d+|x) cards?', low):
        n = _num(m.group(1))
        value += 1.0 + (n - 1) * 1.3
    if re.search(r'exile the top card of (your|their|that player\'s) library[^.]*(\.\s*)?[^.]*you may (play|cast)', low):
        value += 1.0
    if re.search(r'\bscry (\d+|x)', low):
        value += 0.3
    if re.search(r'\bsurveil (\d+|x)', low):
        value += 0.3
    if re.search(r'search your library for a (basic )?land', low):
        value += 1.0
    elif re.search(r'search your library for a', low):
        value += 2.5

    # Tokens
    for m in re.finditer(r'create (a|an|one|two|three|four|five|x|\d+|that many) (?:[a-z]+ ){0,3}?(\d+)/(\d+)', low):
        k = _num(m.group(1))
        value += k * (int(m.group(2)) + int(m.group(3))) * 0.5  # a 1/1 token is about 1 mana
    for m in re.finditer(r'create (a|an|one|two|three|x|\d+) (treasure|food|clue|blood)', low):
        value += _num(m.group(1)) * 0.6

    # Removal and interaction
    if re.search(r'destroy all|exile all|each creature gets -', low):
        value += 4.0
    elif re.search(r'(destroy|exile) target (creature|permanent|nonland permanent)', low):
        value += 1.5 if re.search(r'with (power|mana value) \d+ or less', low) else 2.5
    elif re.search(r'(destroy|exile) target (artifact|enchantment)', low):
        value += 1.5
    if re.search(r'counter target spell', low):
        value += 1.2 if 'unless' in low else 2.0
    if re.search(r"return (up to (?:one|two|three) )?target [^.]*to (?:its|their) owners?'s? hands?", low):
        value += 1.0
    for m in re.finditer(r'return (up to (one|two|three) )?target [^.]*card[^.]*from (?:your|a) graveyard to (the battlefield|your hand)', low):
        k = _num(m.group(2)) if m.group(2) else 1
        value += k * (2.5 if m.group(3) == 'the battlefield' else 1.0)
    if re.search(r'gain control of', low):
        value += 1.5 if 'until end of turn' in low else 4.0
    if re.search(r'\btap (up to (one|two) )?target', low):
        value += 0.4
    if re.search(r"can't block", low):
        value += 0.3
    if re.search(r'fights?', low):
        value += 1.5

    # Pumps and counters
    for m in re.finditer(r'gets? \+(\d+|x)/\+(\d+|x)', low):
        n = (_num(m.group(1)) + _num(m.group(2))) / 2
        team = re.search(r'creatures you control get', low)
        value += n * (2.0 if team else 0.5 if 'until end of turn' in low else 0.8)
    for m in re.finditer(r'put (a|an|one|two|three|x|\d+) \+1/\+1 counters? on', low):
        value += _num(m.group(1)) * 0.75  # permanent growth, unlike a pump until end of turn
    if re.search(r"can't be blocked", low):
        value += 0.8 if 'except' in low or 'power' in low else 1.5

    # Things that break the game at any cost
    if re.search(r'without paying (its|their) mana cost', low):
        value += 4.0
    if re.search(r'extra turn', low):
        value += 6.0
    if 'emblem' in low:
        value += 3.0
    if re.search(r'\badd one mana of any color|\badd \{', low):
        value += 0.5 if not low.startswith('{t}') else 0.3

    # How often it happens
    if re.match(r'^(at the beginning of (the|each)|whenever (a|another) creature (you control )?(attacks|enters|dies))', low):
        value *= 3.0   # every turn for both players, or once per creature
    elif re.match(r'^at the beginning of your (upkeep|end step|combat|draw step)', low):
        value *= 2.5   # every one of your turns, unconditionally
    elif re.match(r'^whenever [^,]*\b(attacks|blocks|deals combat damage)', low):
        value *= 1.5   # needs combat
    elif re.match(r'^whenever', low):
        value *= 2.0
    elif ':' in s and re.match(r'^\{t\}:', low):
        value *= 2.5  # costs nothing but a tap: once every turn, like an upkeep trigger
    elif ':' in s and re.match(r'^\{t\}, ', low):
        value *= 1.4  # repeatable, but with an extra cost
    elif re.match(r'^(when|whenever)\b.*\b(dies|leaves)', low):
        value *= 0.8
    # A cost paid inside the effect makes it cheaper
    if re.search(r'you may pay \{', low):
        value *= 0.7
    return round(value, 2)


def trim_to_budget(lines: list[str], card: dict, keep_first: int = 0) -> list[str]:
    """
    Drop the most valuable ability until the card is within ERROR_OVER of its budget,
    keeping at least one ability and never touching the first keep_first lines (an
    Enchant line or a */* definition) or Equip/Crew lines. The last resort when the model
    ignored the budget on every attempt.
    """
    lines = list(lines)
    allowed = budget(card)
    protected = lambda i, l: i < keep_first or re.match(r'^(equip|crew|enchant)\b', l, re.I)
    while estimate('\n'.join(lines), card) - allowed > ERROR_OVER:
        candidates = [(ability_value(l), i) for i, l in enumerate(lines) if not protected(i, l)]
        if len(candidates) <= 1:
            break
        _, worst = max(candidates)
        del lines[worst]
    return lines


def estimate(text: str, card: dict | None = None) -> float:
    """Rough mana value of a card's whole rules text."""
    lines = [l for l in (text or '').split('\n') if l.strip()]
    if 'planeswalker' in ((card or {}).get('type') or '').lower():
        # One loyalty ability a turn: the best one counts fully, the rest partly
        values = sorted((ability_value(l.split(':', 1)[1]) if ':' in l else ability_value(l) for l in lines), reverse=True)
        return round((values[0] if values else 0) + 0.4 * sum(values[1:]), 2)
    return round(sum(ability_value(l) for l in lines), 2)


def assess(text: str, card: dict) -> tuple[float, float]:
    """(estimated value, budget) for a finished card."""
    return estimate(text, card), budget(card)


def describe_budget(card: dict) -> str:
    """One line for the prompt: what this card's cost and rarity can afford."""
    card = card or {}
    b = budget(card)
    mv = int(card.get('cmc') or 0)
    rarity = (card.get('rarity') or 'common').lower()
    card_type = (card.get('type') or '').lower()
    if 'land' in card_type:
        return 'Power budget: a land. Its mana ability is free; any other ability is small and costs mana to use.'
    if b <= 0.5:
        size = ('almost nothing beyond its body: at most one minor keyword (vigilance, reach) or a tiny '
                'one-time effect like "When this creature enters, you gain 2 life"')
    elif b <= 1.0:
        size = ('one small bonus: a single keyword like first strike or lifelink, or a one-time effect like '
                '"When this creature enters, scry 2" or "When this creature dies, create a Treasure token"')
    elif b <= 1.5:
        size = 'one small effect, about the size of "draw a card", "2 damage to a creature", or flying'
    elif b <= 2.5:
        size = ('one medium effect, about "3 damage to a creature", "draw two cards", '
                'or flying plus a small one-time trigger')
    elif b <= 3.5:
        size = ('one strong effect or two medium ones, about "destroy target creature with power 3 or less" '
                'or a small effect that repeats every turn')
    elif b <= 5:
        size = 'a strong effect, about "destroy target creature" or "draw three cards"'
    else:
        size = 'a powerful, game-changing effect, like a repeating engine or a one-sided board effect'
    return f'Power budget: a {mv}-mana {rarity} card, so its abilities together are worth {size}.'
