"""
Rules text for generated cards: the LLM prompt, the cleanup that turns the model's
abilities into legal, legible Magic templating, and a linter that scores the result.

The model returns JSON ({"abilities": [...]}), one string per ability, so nothing here
has to guess where one ability ends and the next begins. Cleanup works on that list:
each fixer takes and returns list[str], and format_rules_text() joins the final list
into the text the renderer draws (one ability per line; keywords share the first line).

lint_rules_text() is used by the e2e harness (tools/e2e_rules_text.py) to score whole
runs, and by generation to decide whether a reply is worth a retry.
"""
from __future__ import annotations

import re

import power_level as power

# ===== Vocabulary =====

# Keyword abilities a card may simply list ("Flying, trample"). Lowercase. Keywords that
# take a parameter (ward {2}, protection from red, toxic 2, ...) are in PARAM_KEYWORDS.
SIMPLE_KEYWORDS = {
    'flying', 'first strike', 'double strike', 'deathtouch', 'defender', 'haste',
    'hexproof', 'indestructible', 'lifelink', 'menace', 'reach', 'trample', 'vigilance',
    'flash', 'prowess', 'shroud', 'fear', 'intimidate', 'skulk', 'shadow', 'horsemanship',
    'flanking', 'exalted', 'infect', 'wither', 'persist', 'undying', 'changeling', 'devoid',
    'convoke', 'delve', 'cascade', 'storm', 'rebound', 'split second', 'living weapon',
    'phasing', 'battle cry', 'soulbond', 'evolve', 'extort', 'dethrone', 'melee',
    'myriad', 'partner', 'improvise', 'ascend', 'mentor', 'riot', 'afterlife', 'decayed',
    'training', 'daybound', 'nightbound', 'ravenous', 'backup', 'for mirrodin!', 'enlist',
    'read ahead', 'unleash', 'undaunted', 'sunburst', 'provoke', 'fabricate',
}

# Keywords followed by a parameter: "Ward {2}", "Protection from black", "Toxic 1".
PARAM_KEYWORDS = {
    'ward', 'protection from', 'toxic', 'annihilator', 'afflict', 'bushido', 'rampage',
    'crew', 'equip', 'enchant', 'cycling', 'kicker', 'flashback', 'madness', 'escape',
    'dash', 'evoke', 'echo', 'bestow', 'embalm', 'eternalize', 'unearth', 'ninjutsu',
    'suspend', 'vanishing', 'fading', 'modular', 'renown', 'outlast', 'scavenge',
    'bloodthirst', 'devour', 'tribute', 'amplify', 'graft', 'absorb', 'frenzy',
    'landwalk', 'islandwalk', 'swampwalk', 'mountainwalk', 'forestwalk', 'plainswalk',
    'megamorph', 'morph', 'disguise', 'blitz', 'casualty', 'squad', 'prototype',
    'reconfigure', 'buyback', 'multikicker', 'mutate', 'foretell', 'encore', 'overload',
    'surge', 'emerge', 'spectacle', 'plot', 'offspring', 'saddle', 'station', 'impending',
    'affinity for', 'backup', 'boast', 'craft with', 'hideaway', 'splice onto',
}

# Evergreen keywords in Comprehensive Rules order (702.2 deathtouch ... 702.21 ward), then
# common later ones: the order keywords are printed in.
KEYWORD_ORDER = [
    'deathtouch', 'defender', 'double strike', 'first strike', 'flash', 'flying', 'haste', 'hexproof',
    'indestructible', 'intimidate', 'lifelink', 'protection from', 'reach', 'shroud', 'trample',
    'vigilance', 'ward', 'menace', 'prowess', 'fear', 'shadow', 'skulk', 'infect', 'wither',
]

# Creature combat keywords that make no sense printed on an instant or sorcery.
CREATURE_KEYWORDS = {
    'flying', 'first strike', 'double strike', 'deathtouch', 'defender', 'haste',
    'lifelink', 'menace', 'reach', 'trample', 'vigilance', 'skulk', 'fear', 'intimidate',
    'shadow', 'horsemanship', 'flanking', 'infect', 'wither',
}

# Subtypes that don't fly (a Human with flying reads wrong); style warning only.
GROUNDED_SUBTYPES = {
    'human', 'dwarf', 'elf', 'orc', 'goblin', 'zombie', 'skeleton', 'beast', 'bear', 'wolf',
    'cat', 'dog', 'plant', 'treefolk', 'wall', 'golem', 'giant', 'minotaur', 'centaur',
    'knight', 'warrior', 'soldier', 'rogue', 'assassin', 'berserker', 'barbarian', 'archer',
    'scout', 'ooze', 'elephant', 'rhino', 'boar', 'dinosaur', 'hydra', 'troll', 'kithkin',
}

# Mana/tap symbols allowed inside braces.
_SYMBOL = r'(?:\d{1,2}|[WUBRGCXYZSTQE]|[WUBRG2C]/[WUBRGP])'
SYMBOL_RE = re.compile(r'\{' + _SYMBOL + r'\}', re.IGNORECASE)
BRACE_RE = re.compile(r'\{[^{}]*\}')

# Words that only belong in the cost of an activated ability (left of the colon).
_COST_WORDS = r'(?:\{[^}]*\}|sacrifice|discard|pay|exile|remove|tap|untap|return|reveal|put|collect|forage|mill|,|\s|and|an?|another|untapped|this|creature|artifact|land|card|cards|from|your|hand|graveyard|life|counters?|[+\-−]?\d+/[+\-−]?\d+|[\w\'’]+ counters?|\d+|X|one|two|three|of|it|permanent|enchantment|token|nontoken|other|you|control|with|mana|value|power|named|~)+'

# A permanent's line that starts like a spell's instruction ("Draw a card.") has lost its
# trigger or cost; static abilities describe a state instead ("Creatures you control get").
IMPERATIVE_START = re.compile(
    r'^(draw|scry|surveil|exile|destroy|return|create|deal|gain|lose|put|search|counter|tap|'
    r'untap|discard|mill|look|reveal|choose|add|copy|shuffle|investigate|proliferate|fight)\b', re.I)

# Every real ability mentions at least one game object or action; a sentence with none of
# these ("Vyraxa exhales a molten torrent that scorches the battlefield.") is flavor text.
RULES_WORDS = re.compile(
    r"\{|\b(you|your|target|creature|creatures|card|cards|damage|life|counter|counters|token|tokens|"
    r"mana|gets?|has|have|can't|each|player|opponent|spell|spells|land|lands|draw|turn|control|"
    r"power|toughness|enters|dies|attacks|blocks|library|graveyard|hand|equipped|enchanted|"
    r"artifact|enchantment|permanent|sacrifice|exile|destroy|tap|untap|flying|protection|equal|"
    r"create|scry|surveil|mill|return|counter|search|reveal|discard|fight|copy|cast)\b", re.I)

# Abilities that are never acceptable. The linter flags them (so the model retries), and if
# every attempt still has one, generation drops that line rather than print it.
FORBIDDEN = [
    (re.compile(r'^(At the beginning of|\{T\}:)[^.]*\b(destroy|exile) (target|each|all)\b', re.I),
     'removal that repeats every turn'),
    (re.compile(r'\bwithout paying (its|their) mana costs?\b', re.I), 'free spells break any budget'),
    (re.compile(r'\bextra turn\b', re.I), 'extra turns'),
    (re.compile(r'\bspells? you (control|cast) gets? [+\-]', re.I), 'spells don\'t get +N/+N or cost changes this way'),
    (re.compile(r'\byou get (a|an|one|two) (charge|\+1/\+1|-1/-1|land|loyalty) (counter|card)', re.I),
     'players don\'t get counters or cards like that'),
    (re.compile(r'\bthis (enchantment|artifact|land) deals combat damage\b', re.I), 'only creatures deal combat damage'),
    (re.compile(r'\{X\} mana\b|\badd \{X\}', re.I), 'variable mana with no defined X'),
    (re.compile(r'^(?:Whenever|At the beginning of) [^,]+, (?:you may )?add \{[^}]+\}(?: or \{[^}]+\})?\.$', re.I),
     'filler: mana as the only reward'),
    (re.compile(r'\b(exile|destroy|return) any target\b', re.I), '"any target" only works for damage'),
    (re.compile(r'\bpay \{([WUBRGC])\}\. If you do, add \{\1\}', re.I), 'pointless mana exchange'),
    (re.compile(r'\bgets? \+\d+/\+\d+ [a-z]+ [A-Za-z ]*tokens?\b'), 'garbled stat change'),
]

RARITY_ABILITY_CAP = {'common': 2, 'uncommon': 3, 'rare': 4, 'mythic': 5}
MAX_TEXT_CHARS = 340  # beyond this the renderer's font gets hard to read


# ===== Classification helpers =====

def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())


def keyword_head(part: str) -> str | None:
    """The keyword a comma-separated fragment names, or None if it isn't a keyword."""
    p = _norm(part).rstrip('.').lower()
    if not p:
        return None
    if p in SIMPLE_KEYWORDS:
        return p
    for kw in PARAM_KEYWORDS:
        if p == kw or p.startswith(kw + ' ') or (kw.endswith(' from') and p.startswith(kw)):
            # A parameter keyword must be short ("equip {2}", "protection from red"),
            # not a sentence that happens to start with the word ("enchant ... gets").
            if len(p.split()) <= 5 and ':' not in p:
                return kw
    return None


def is_keyword_line(line: str) -> bool:
    """True for a pure keyword line: 'Flying', 'Flying, trample', 'Ward {2}', 'Equip {3}'."""
    line = _norm(line).rstrip('.')
    if not line or ':' in line:
        return False
    parts = [p for p in line.split(',')]
    return all(keyword_head(p) for p in parts)


def ability_kind(line: str) -> str:
    """keyword | loyalty | activated | triggered | static (for ordering and linting)."""
    s = _norm(line)
    if is_keyword_line(s):
        return 'keyword'
    if re.match(r'^[+\-−]?\s*(\d+|X)\s*:', s):
        return 'loyalty'
    if re.match(r'^(when|whenever|at the beginning|at the end|at end of)\b', s, re.I):
        return 'triggered'
    colon = _colon_outside_quotes(s)
    if colon > 0:
        return 'activated'
    return 'static'


def _colon_outside_quotes(s: str) -> int:
    depth = False
    for i, ch in enumerate(s):
        if ch in '"“”':
            depth = not depth
        elif ch == ':' and not depth:
            return i
    return -1


def type_line(card: dict) -> str:
    card = card or {}
    parts = [card.get('supertype') or '', card.get('type') or '']
    head = ' '.join(p.strip() for p in parts if p and p.strip())
    sub = (card.get('subtype') or '').strip()
    return f'{head} — {sub}' if sub else head


# ===== Linting =====

def lint_rules_text(text: str, card: dict | None) -> list[tuple[str, str]]:
    """
    Problems with finished rules text, as (severity, message) pairs.
    severity 'error' = illegal or garbled; 'warn' = legal but off-template or hard to read.
    """
    card = card or {}
    issues: list[tuple[str, str]] = []

    def err(msg):
        issues.append(('error', msg))

    def warn(msg):
        issues.append(('warn', msg))

    text = (text or '').strip()
    if not text:
        return [('error', 'empty rules text')]

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    card_type = (card.get('type') or '').lower()
    subtype = (card.get('subtype') or '').lower()
    name = (card.get('name') or '').strip()
    rarity = (card.get('rarity') or 'common').lower()
    mana_cost = (card.get('manaCost') or '').upper()
    is_spell = 'instant' in card_type or 'sorcery' in card_type
    is_pw = 'planeswalker' in card_type
    is_creature = 'creature' in card_type

    # Meta commentary, labels, markdown, leftovers
    for l in lines:
        low = l.lower()
        if re.search(r'\b(rules text|here is|here are|this card (?:is|has)|ability:|keywords?:|triggered ability|activated ability|static ability)\b', low):
            err(f'meta/label text: {l!r}')
        if re.search(r'(\*\*|^\s*[-*•]\s|^#)', l):
            err(f'markdown in text: {l!r}')
        if '~' in l or 'cardname' in low:
            err(f'unreplaced self-reference: {l!r}')
        if l[0].islower():
            warn(f'line starts lowercase: {l!r}')

    # Type-line / name contamination
    tl = type_line(card).lower()
    for l in lines:
        low = l.lower().rstrip('.')
        if tl and (low == tl or low.endswith(tl) or low.startswith(tl)):
            err(f'type line in rules text: {l!r}')
        elif name and re.fullmatch(re.escape(name.lower()) + r'[,:\-— ]+.*(legendary|creature|artifact|enchantment|instant|sorcery)\b.*', low):
            err(f'name/type header in rules text: {l!r}')
        if re.fullmatch(r'\d+/\d+', low):
            err(f'power/toughness in rules text: {l!r}')

    # Keywords
    for l in lines:
        if ability_kind(l) == 'keyword':
            if l.rstrip() != l.rstrip('.'):
                warn(f'keyword line should not end with a period: {l!r}')
            continue
        # A comma list that looks like keywords but contains unknown ones (made-up keywords)
        if ':' not in l and ',' in l and len(l.split()) <= 8 and not re.search(r'\b(you|target|each|gets?|gains?|has|have|is|are|deals?|draws?|creates?|put|return)\b', l.lower()):
            unknown = [p.strip() for p in l.rstrip('.').split(',') if not keyword_head(p)]
            if unknown and len(unknown) < len(l.split(',')) + 1:
                err(f'unknown keyword(s) {unknown} in {l!r}')

    # Symbols and costs
    for l in lines:
        for b in BRACE_RE.findall(l):
            if not SYMBOL_RE.fullmatch(b):
                err(f'invalid symbol {b} in {l!r}')
        if l.count('{') != l.count('}'):
            err(f'unbalanced braces in {l!r}')
        if re.search(r'(?<!\{)\bTap\s*:', l) or re.search(r'\(T\)|\[T\]', l):
            err(f'tap symbol not written as {{T}}: {l!r}')
        kind = ability_kind(l)
        if kind == 'activated':
            cost = l[:_colon_outside_quotes(l)]
            if re.search(r'\b(add|draw|deal|deals|create|destroy|counter target|gain|gains|search)\b', cost, re.I):
                err(f'effect words inside activation cost: {l!r}')
            elif not re.search(r'\{|\b(sacrifice|discard|pay|exile|remove|tap|untap|return|reveal|collect|forage|mill)\b', cost, re.I):
                err(f'label used as an activation cost: {l!r}')
        if re.search(r'[.!]\s+(\{[^}]+\}[^.]*:|When\b|Whenever\b|At the beginning\b)', l):
            err(f'several abilities run together on one line: {l!r}')
        if l.count('"') % 2:
            err(f'unbalanced quote in {l!r}')
        if re.search(r'\b(deals?|gets?|draws?|creates?|gains?|loses?|mills?)\s+\{[WUBRGC]\}', l) or \
                re.search(r'\{[^}]+\}\s+(damage|life|cards?|creatures?|counters?)\b', l):
            err(f'mana symbol used as a number: {l!r}')
        if kind == 'activated' and len(re.findall(r'\{[^}]+\}\s*(?:,[^:]*)?:', l)) > 1 and l.count(':') > 1:
            err(f'two activated abilities on one line: {l!r}')

    # Templating
    for l in lines:
        if re.search(r'\bgains? [+\-−]\d+/[+\-−]\d+', l, re.I):
            err(f'"gains +N/+N" should be "gets +N/+N": {l!r}')
        if re.search(r'\benters the battlefield\b', l, re.I):
            warn(f'old template "enters the battlefield" (now "enters"): {l!r}')
        if re.search(r'\bconverted mana cost\b', l, re.I):
            warn(f'obsolete term "converted mana cost" (now "mana value"): {l!r}')
        if re.search(r"\bit's controller\b|\btheir owner's hand\b(?! s)", l, re.I):
            warn(f'grammar: {l!r}')
        if not l.endswith(('.', '"', '”', ')')) and ability_kind(l) not in ('keyword',) and not re.match(r'^(equip|crew|enchant)\b', l, re.I):
            warn(f'missing final period: {l!r}')

    # Card-type rules
    if is_spell:
        for l in lines:
            if ability_kind(l) == 'keyword' and any(keyword_head(p) in CREATURE_KEYWORDS for p in l.split(',')):
                err(f'creature keyword printed on an instant/sorcery: {l!r}')
            if ability_kind(l) == 'activated':
                err(f'activated ability on an instant/sorcery: {l!r}')
            if ability_kind(l) == 'triggered' and not re.match(r'^when you cast this spell\b', l, re.I):
                err(f'triggered ability on an instant/sorcery: {l!r}')
            if re.search(r'\bwhen (?:this|~|' + re.escape(name.lower() or '~') + r')\b.*\benters\b', l.lower()):
                err(f'enters trigger on an instant/sorcery: {l!r}')
    if is_pw:
        loyalty = [l for l in lines if ability_kind(l) == 'loyalty']
        if len(loyalty) < 2:
            err('planeswalker needs at least two loyalty abilities')
        costs = [l.split(':')[0].replace(' ', '') for l in loyalty]
        if loyalty and not any(c.startswith(('-', '−')) for c in costs):
            err('planeswalker has no minus loyalty ability')
        if len(set(costs)) < len(costs):
            err(f'planeswalker repeats a loyalty cost: {costs}')
    if not is_spell and not is_pw:
        for l in lines:
            if ability_kind(l) == 'static' and re.match(IMPERATIVE_START, l):
                err(f'effect with no trigger or cost on a permanent: {l!r}')
            if ability_kind(l) == 'static' and re.search(r'\buntil end of turn\b', l) and '"' not in l:
                err(f'static ability with a duration (needs a trigger or cost): {l!r}')
            if ability_kind(l) == 'static' and re.search(r'\bdeals? (\d+|X) damage\b', l) and '"' not in l \
                    and not re.search(r'\bwould\b|\bwhenever\b|\beach time\b', l, re.I):
                err(f'damage with no trigger or cost on a permanent: {l!r}')
    if 'equipment' not in subtype and any(re.match(r'^equip\b', l, re.I) for l in lines):
        err('Equip on a card that isn\'t Equipment')
    if not is_creature and 'vehicle' not in subtype:
        for l in lines:
            if re.search(r'\bthis (artifact|enchantment|land)\b gets [+\-−]\d', l, re.I):
                err(f'noncreature permanent getting +N/+N: {l!r}')
    if sum(1 for l in lines if re.match(r'^crew\b', l, re.I)) > 1:
        err('more than one Crew ability')
    for l in lines:
        if re.search(r'\bmana pool\b', l, re.I):
            warn(f'obsolete "mana pool" wording: {l!r}')
        m = re.match(r'^([A-Z][a-z]+), ([A-Z])', l)
        if m and not keyword_head(m.group(1)):
            err(f'invented keyword prefix {m.group(1)!r}: {l!r}')
        if re.match(r'^(When|Whenever) [\w\']+,', l) and not re.match(r'^(When|Whenever) (?:you|it|this)\b', l):
            err(f'trigger condition has no event: {l!r}')
        if ability_kind(l) == 'triggered' and _colon_outside_quotes(l) > 0 and not re.search(r'"[^"]*:', l):
            err(f'colon inside a triggered ability: {l!r}')
        if 'emblem' in l.lower() and not is_pw:
            err(f'emblem on a non-planeswalker: {l!r}')
        if re.search(r'\bequipped creature\b', l, re.I) and 'equipment' not in subtype:
            err(f'"equipped creature" on a card that isn\'t Equipment: {l!r}')
        if re.search(r'\benchanted creature\b', l, re.I) and 'aura' not in subtype:
            err(f'"enchanted creature" on a card that isn\'t an Aura: {l!r}')
        if re.search(r'\bthis creature\b', l, re.I) and not is_creature and 'vehicle' not in subtype \
                and '"' not in l:
            err(f'"this creature" on a noncreature card: {l!r}')
        if ability_kind(l) == 'activated' and l.count(':') - len(re.findall(r'"[^"]*:[^"]*"', l)) > 1:
            err(f'two colons in one activated ability: {l!r}')
        if re.search(r'\b(gets?|gains?) \+\d+ (power|toughness)\b', l):
            err(f'stat change not written as +N/+N: {l!r}')
        if re.fullmatch(r"[A-Z][\w'’-]+\.?", l) and not keyword_head(l):
            err(f'invented keyword: {l!r}')
        m = re.search(r'\b(?:has|have|gains?) ([a-z]+)(?: until end of turn)?\.$', l)
        if m and m.group(1) not in SIMPLE_KEYWORDS and m.group(1) not in ('it', 'them', 'this', 'that'):
            err(f'unknown keyword {m.group(1)!r} granted: {l!r}')
        if re.search(r'\battached\b', l) and not ('aura' in subtype or 'equipment' in subtype):
            err(f'"attached" on a card that can\'t be attached: {l!r}')
        if ability_kind(l) != 'activated' and re.search(r'\b(mana ability|this ability (?:can|any number))\b', l, re.I):
            err(f'meta sentence about abilities: {l!r}')
        if re.search(r', [A-Z][a-z]+, (create|draw|put|deal|deals|return|exile|destroy)\b', l):
            err(f'name inserted mid-sentence: {l!r}')
        m = re.search(r'\bthat (creature|player|card|spell|permanent|land|artifact)\b', l)
        if m and not re.search(r'\b(target|a|an|another|each|enchanted|equipped|blocking|attacking|blocked|dies|cast|one|it|the top)\b', l[:m.start()]):
            err(f'dangling "{m.group(0)}" with nothing it refers to: {l!r}')
        if re.search(r'\b(bounce|tutor|mana pool|counter any target|at the end of the game|if this ability is used|remove [^,.]* from battle)\b', l, re.I):
            err(f'slang or invalid wording: {l!r}')
        if re.search(r'\bany target\b', l) and not re.search(r'\bdamage\b', l):
            err(f'"any target" without damage: {l!r}')
        if re.match(r'^This \w+ has (?:\w+ )?counters?\b', l):
            err(f'meaningless counter statement: {l!r}')
        for m in re.finditer(r'\bcreate (?:a|an|one|two|three|four|five|X|\d+|that many)\b([^.]*?)(?:\.|$)', l, re.I):
            made = m.group(1)
            if not re.search(r'\btokens?\b', made):
                err(f'creates something that isn\'t a token: {l!r}')
            elif not re.search(r"\bcreature tokens?\b|\b(treasure|food|clue|blood|map|powerstone|junk|gold|incubator)\b|\bcopy\b", made, re.I):
                err(f'undefined token: {l!r}')
        if re.search(r'\bpay (\{[^}]+\})+ (?:more )?to add\b|\bpay [^.]* to add \{', l, re.I):
            err(f'pointless mana exchange: {l!r}')
        if re.search(r'\bdrain\b', l, re.I):
            err(f'"drain" is not a game action: {l!r}')
        if re.match(r'^(When|Whenever) (?!you\b)[\w\'’ ]+? (create|draw|put|deal|destroy|exile|return|gain|shuffle|sacrifice)\b', l) or \
                re.search(r'\bif \w+ (shuffle|create|draw|put|deal)\b', l):
            err(f'trigger or condition missing its verb: {l!r}')
        if re.search(r'\bis attacked\b', l):
            err(f'creatures are never "attacked": {l!r}')
        for pattern, why in FORBIDDEN:
            if pattern.search(l):
                err(f'{why}: {l!r}')
        if ability_kind(l) == 'activated' and re.match(r'^(?:\{(?:\d+|[WUBRGC])\})+:\s*Add\b[^.]*\.$', l):
            err(f'pointless mana exchange: {l!r}')
        if not is_spell and ability_kind(l) == 'static' and re.match(r'^You may\b', l):
            err(f'optional effect with no trigger or cost on a permanent: {l!r}')
        if ability_kind(l) == 'static' and not RULES_WORDS.search(l):
            err(f'reads like flavor text, not an ability: {l!r}')
        if re.search(r"\bthis (land|enchantment|artifact) can't be blocked\b", l, re.I):
            err(f'a noncreature permanent can\'t be blocked: {l!r}')
        if re.search(r'\btokens? (?:on top of|into|in) (?:your|a|its owner\'s) library\b', l, re.I) or \
                re.search(r'\bactivate this (enchantment|artifact|creature|land)\b', l, re.I):
            err(f'nonsense game action: {l!r}')
        if re.search(r'\benters with\b(?![^.]*\bcounters?\b)', l, re.I):
            err(f'"enters with" something that isn\'t counters: {l!r}')
        if re.search(r'\+(?=[,.]|\s|$)', l):
            err(f'stray "+" (garbled stat or counter): {l!r}')
        m = re.search(r'\b(?:has|have|gains?) ([a-z ,]+?)(?: until end of turn)?(?:\.|$)', l)
        # "protection from red and white" is one keyword, not two
        granted_text = re.sub(r'protection from [a-z]+(?: and [a-z]+)?', 'protection', m.group(1)) if m else ''
        granted_text = re.sub(r'\s*(?:,\s*|\band\s+)?loses [a-z ]+$', '', granted_text)  # "and loses trample"
        if m and re.search(r',| and ', granted_text):
            granted = [p.strip() for p in re.split(r',\s*|\s+and\s+', granted_text) if p.strip()]
            if len(granted) > 1:
                unknown = [g for g in granted if not keyword_head(g) and not g.startswith(('protection', "can't", 'hexproof from'))]
                if unknown:
                    err(f'unknown keyword(s) {unknown} granted: {l!r}')
        if re.search(r'\bdestroy any target\b', l, re.I):
            err(f'"destroy any target" is not a legal target: {l!r}')
    if 'equipment' in subtype and not any(re.match(r'^equip\b', l, re.I) for l in lines):
        err('Equipment without an Equip ability')
    if 'vehicle' in subtype and not any(re.match(r'^crew \d+', l, re.I) for l in lines):
        err('Vehicle without Crew N')
    if 'aura' in subtype and not re.match(r'^enchant\b', lines[0], re.I):
        err('Aura must start with "Enchant ..."')
    if not is_creature and not ('equipment' in subtype or 'aura' in subtype or 'vehicle' in subtype):
        for l in lines:
            if ability_kind(l) == 'keyword' and any(keyword_head(p) in CREATURE_KEYWORDS for p in l.split(',')):
                err(f'creature keyword on a noncreature card: {l!r}')
    if is_creature and 'flying' in text.lower() and any(g in subtype.split() for g in GROUNDED_SUBTYPES):
        if any(ability_kind(l) == 'keyword' and 'flying' in l.lower() for l in lines):
            warn(f'grounded subtype {subtype!r} has flying')

    # X costs and variable P/T
    has_x_cost = '{X}' in mana_cost
    uses_x = bool(re.search(r'\bX\b', text))
    if has_x_cost and not uses_x:
        err('X in mana cost but not used in rules text')
    if uses_x and not has_x_cost and not re.search(r'\bwhere X is\b|\bX is\b|\{X\}', text) and not is_pw:
        err('X used but never defined')
    for stat in ('power', 'toughness'):
        v = str(card.get(stat) or '')
        if '*' in v or 'X' in v.upper():
            if not re.search(r'\b' + stat + r'\b[^.]*\bequal to\b', text, re.I) and \
                    not re.search(r'\bpower and toughness are each\b', text, re.I):
                err(f'variable {stat} ({v}) never defined')

    # Power: far over what the mana value and rarity can afford
    value, allowed = power.assess(text, card)
    if value - allowed > power.ERROR_OVER:
        err(f'too strong for its cost: worth ~{value} mana of abilities, budget {allowed}')
    elif value - allowed > power.WARN_OVER:
        warn(f'strong for its cost: worth ~{value} mana of abilities, budget {allowed}')

    # Size and duplicates
    count = sum(len(l.split(',')) if ability_kind(l) == 'keyword' else 1 for l in lines)
    cap = RARITY_ABILITY_CAP.get(rarity, 4)
    if is_pw:
        cap += 1
    if count > cap:
        warn(f'{count} abilities for a {rarity} (cap {cap})')
    if len(text) > MAX_TEXT_CHARS:
        warn(f'{len(text)} characters of rules text (legibility cap {MAX_TEXT_CHARS})')
    seen = set()
    for l in lines:
        key = l.lower().rstrip('.')
        if key in seen:
            err(f'duplicate ability: {l!r}')
        seen.add(key)

    return issues


def error_count(issues) -> int:
    return sum(1 for s, _ in issues if s == 'error')


# ===== Prompt =====

SYSTEM_PROMPT = """You are a senior Magic: The Gathering card designer. You write the rules text for one new card at a time, in exact modern Oracle templating, and every card you write is legal, playable and has a clear, memorable hook.

Output JSON only: {"abilities": ["...", "..."]}. Each array item is ONE ability exactly as printed in the text box. Never include the card name as a header, the mana cost, the type line, power/toughness, flavor text, reminder text, explanations or labels.

Templating rules:
- Keyword abilities go in a single item, comma-separated, first letter capitalized only: "Flying, lifelink". Use real keywords only (flying, first strike, double strike, deathtouch, defender, haste, hexproof, indestructible, lifelink, menace, reach, trample, vigilance, flash, prowess, ward {N}, protection from [quality], and set mechanics like cycling {N}, flashback {cost}, kicker {cost}). Never invent keywords.
- Activated ability: "[cost]: [effect]." Costs go left of the colon and contain only mana symbols, {T}, {Q}, or costs like "Sacrifice a creature", "Discard a card", "Pay 2 life", "Remove a +1/+1 counter from this creature". One activated ability per item.
- Triggered ability: "When...", "Whenever...", or "At the beginning of...", then a comma and the effect.
- Symbols: {W} {U} {B} {R} {G} {C} {X} {T} and generic {1} {2} {3}. Never write "Tap:" or "1R".
- Modern wording: "enters" (not "enters the battlefield"), "gets +2/+2" for stat changes, "gains flying" for keywords, "put a +1/+1 counter on", "any target", "mana value", "its owner's hand", "then shuffle".
- Refer to the card itself as "this creature", "this artifact", "this enchantment" or "this land". A legendary card may use the first part of its name instead (for "Vyraxa, Ember Sovereign" write "Vyraxa").
- Tokens are created: "create a Treasure token", "create a 1/1 white Soldier creature token". Never "put a token onto the battlefield". Never mention a mana pool.
- Stat changes are always written "+N/+N" ("gets +2/+0", never "+2 power"). Use contractions: "can't", "doesn't".
- Only planeswalkers make emblems. Only Equipment says "equipped creature"; only Auras say "enchanted creature"; a Vehicle refers to itself as "this Vehicle".
- End every non-keyword ability with a period.

Every ability on a permanent (creature, artifact, enchantment, land, battle) is exactly one of:
1. keywords: "Flying, vigilance"
2. a static ability that states a continuous fact with a subject: "Creatures you control get +1/+0.", "Enchanted creature has hexproof.", "This creature can't be blocked by creatures with power 2 or less."
3. a triggered ability: "Whenever this creature attacks, ..."
4. an activated ability: "{2}{G}, {T}: ..."
Never write a bare instruction like "Draw a card." or "Gain protection from red." on a permanent: give it a trigger or a cost. Noncreature permanents never "get +N/+N" themselves.

Card-type rules:
- Instant/sorcery: only the spell's effect, one item of one or two sentences, e.g. "This spell deals 3 damage to any target. Scry 1." No triggered or activated abilities, no keywords, nothing about entering or attacking.
- Planeswalker: 3 loyalty abilities (4 at mythic, the last an ultimate), each item "+1: ...", "0: ..." or "−3: ...", every cost different and at least one minus. Do not write starting loyalty.
- Aura: the first item is "Enchant creature"; the rest describe "enchanted creature".
- Equipment: effects on "equipped creature", and the last item is exactly one "Equip {N}".
- Vehicle: a creature-like artifact; the last item is exactly one "Crew N" such as "Crew 2".
- Land: it has no mana cost; usually "{T}: Add {C}." or colored mana, plus at most one utility ability with a cost.
- Noncreature cards do not get creature keywords like flying or trample themselves.
- An X in the mana cost must be used in the rules text.

Examples of correct output:
Uncommon green creature: {"abilities": ["Reach", "When this creature enters, put a +1/+1 counter on target creature you control.", "Whenever a creature you control with a +1/+1 counter on it attacks, it gains trample until end of turn."]}
Rare legendary black-red creature named "Kethra, Ash Regent": {"abilities": ["Menace", "Whenever another creature you control dies, Kethra deals 1 damage to each opponent.", "{1}{B}, Sacrifice another creature: Put two +1/+1 counters on Kethra."]}
Common blue instant: {"abilities": ["Counter target spell unless its controller pays {3}. If that spell is countered this way, scry 2."]}
Uncommon white Aura: {"abilities": ["Enchant creature", "Enchanted creature gets +2/+1 and has first strike.", "When enchanted creature dies, return this card to its owner's hand."]}
Rare Equipment: {"abilities": ["Equipped creature gets +1/+0 and has \\"Whenever this creature deals combat damage to a player, create a Treasure token.\\"", "Equip {2}"]}
Mythic planeswalker: {"abilities": ["+1: Scry 2, then draw a card.", "−2: Return target creature to its owner's hand.", "−7: You get an emblem with \\"Instant and sorcery spells you cast cost {2} less to cast.\\""]}
Uncommon land: {"abilities": ["This land enters tapped.", "{T}: Add {U} or {R}.", "{2}, {T}, Sacrifice this land: Draw a card."]}

Power level (the most common mistake is making cheap cards far too strong):
- Price effects like a Limited designer. Typical costs: "draw a card" ~1 mana; "draw two cards" ~3; 2 damage to a creature ~1; 3 damage to any target ~2; "destroy target creature" ~4 at common (3 at rare); "counter target spell" ~2; a +2/+2 pump until end of turn ~1; one 1/1 token ~1; flying ~1 on a creature.
- A creature's body is already paid for: the power/toughness you are given is about what its mana value buys, so its abilities must fit in the stated power budget.
- An effect that repeats (Whenever..., At the beginning of your upkeep..., {T}: ...) costs about twice a one-time effect (When this creature enters...). A 2-mana card never makes tokens or draws cards every turn.
- Rarity: commons do one simple thing and are modest; uncommons are a bit stronger or do two related things; rares can be strong and build-around; mythics can be splashy. Protection, hexproof and indestructible are uncommon or higher.
- Never: extra turns, casting spells without paying their mana cost, "destroy all" below rare, loops, removal that repeats every turn, or more than three tokens from one ability. A repeating ability makes at most one token.
- Avoid filler: paying mana to add mana, "you may add {U}" on attack, or restating what a keyword already does. Every ability should matter in a game.

Design: make it feel like a real card from a premier set. Build every ability around one idea that fits the card's name, colors and concept, and do not copy the examples. Prefer specific, flavorful effects, but a small effect done well beats a big one: most real cards are simple. Keep it concise: real cards rarely exceed 60 words."""


# Design hooks per color. One is suggested at random so similar requests still diverge.
COLOR_HOOKS = {
    'W': ['go wide with 1/1 tokens', 'lifegain payoffs', 'tap down or detain opposing creatures',
          'protect your creatures', '+1/+1 counters on a team', 'exile removal until this leaves',
          'enchantments matter', 'attack-with-a-group rewards'],
    'U': ['card selection with scry or surveil', 'bounce and tempo', 'tap or freeze permanents',
          'instants and sorceries matter', 'copy a spell or ability', 'artifacts matter',
          'flash tricks', 'mill yourself for value', 'draw-your-second-card-each-turn payoffs'],
    'B': ['sacrifice for value', 'return creatures from the graveyard', 'drain each opponent',
          'removal with a cost', 'discard', '-1/-1 counters', 'pay life for power',
          'creatures dying matters'],
    'R': ['attack triggers', 'direct damage', 'haste and aggression', 'exile the top card, you may play it',
          'Treasure tokens', 'sacrifice for a burst of power', 'instants and sorceries matter',
          'damage to each opponent'],
    'G': ['+1/+1 counters', 'extra land drops or ramp', 'fight another creature',
          'creatures with power 4 or greater matter', 'landfall', 'creature tokens that grow',
          'defense with big blockers', 'search your library for a land'],
}
COLORLESS_HOOKS = ['artifacts matter', 'charge counters', 'mana abilities with a twist',
                   'sacrifice this for an effect', 'colorless spells matter']

ABILITY_BUDGET = {'common': (1, 2), 'uncommon': (2, 3), 'rare': (2, 3), 'mythic': (3, 4)}

RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {'abilities': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 6}},
    'required': ['abilities'],
}


def self_reference(card: dict) -> str:
    """How the card names itself in its own rules text."""
    card = card or {}
    name = (card.get('name') or '').strip()
    if 'legendary' in (card.get('supertype') or '').lower() and name:
        return name.split(',')[0].strip()
    t = (card.get('type') or '').lower()
    sub = (card.get('subtype') or '').lower()
    for word in ('Equipment', 'Vehicle', 'Aura'):
        if word.lower() in sub.split():
            return f'this {word}'
    for word in ('creature', 'planeswalker', 'artifact', 'enchantment', 'land', 'battle'):
        if word in t:
            return f'this {word}'
    if 'instant' in t or 'sorcery' in t:
        return 'this spell'
    return 'this permanent'


def build_messages(prompt: str, card: dict | None, rng) -> list[dict]:
    """System + user messages for one card. rng is a random.Random (seedable in tests)."""
    card = card or {}
    name = (card.get('name') or '').strip() or 'Untitled'
    rarity = (card.get('rarity') or 'common').lower()
    card_type = (card.get('type') or '').lower()
    subtype = (card.get('subtype') or '').lower()
    colors = [c for c in (card.get('colors') or []) if c in 'WUBRG']
    mana_cost = card.get('manaCost') or ''
    mv = card.get('cmc', 0)

    facts = [f'Name: {name}', f'Type line: {type_line(card) or "Creature"}', f'Rarity: {rarity}']
    if 'land' in card_type and not mana_cost:
        facts.append('Mana cost: none (land)')
    else:
        facts.append(f'Mana cost: {mana_cost or "{0}"} (mana value {mv})')
    facts.append('Colors: ' + (', '.join({'W': 'white', 'U': 'blue', 'B': 'black', 'R': 'red', 'G': 'green'}[c] for c in colors) or 'colorless'))
    is_body = 'creature' in card_type or 'vehicle' in subtype
    if card.get('power') or card.get('toughness'):
        facts.append(f"Power/toughness: {card.get('power') or '?'}/{card.get('toughness') or '?'}")
    elif is_body:
        # The body is fixed before the text (finalize_card prints the same stats), so the
        # model can spend the budget knowing what the creature already is
        p, t = power.creature_stats(card)
        facts.append(f'Power/toughness: {p}/{t} (fixed; design the abilities around this body)')
    facts.append(power.describe_budget(card))
    brief = card.get('brief')
    concept = re.sub(r',?\s*(detailed digital art|magic: the gathering style|fantasy art of)', '', prompt or '', flags=re.I).strip(' ,')
    if concept and not brief:  # the director's brief replaces the auto-built art string
        facts.append(f'Concept: {concept}')

    # How many abilities: the rarity's range, trimmed to what the power budget can afford
    lo, hi = ABILITY_BUDGET.get(rarity, (2, 3))
    affordable = power.budget(card)
    hi = min(hi, 1 if affordable <= 0.75 else 2 if affordable <= 2.5 else hi)
    lo = min(lo, hi)
    count = rng.randint(lo, hi)
    musts = []
    if 'planeswalker' in card_type:
        count = 4 if rarity == 'mythic' else 3
        musts.append(f'Write exactly {count} loyalty abilities.')
    elif 'instant' in card_type or 'sorcery' in card_type:
        musts.append('Write the spell as one item of one or two sentences.')
    elif count == 1:
        musts.append('Write 1 ability: a small triggered or activated ability that shows off the concept '
                     '(a lone keyword only if it is the most flavorful choice).')
    else:
        musts.append(f'Write {count} abilities (a keyword item of up to two keywords counts as one).')
    if 'aura' in subtype:
        musts.append('Start with "Enchant creature".')
    if 'equipment' in subtype:
        musts.append(f'End with "Equip {{{max(1, min(4, int(mv or 1)))}}}" or a similar equip cost.')
    if 'vehicle' in subtype:
        musts.append('End with exactly one Crew ability, such as "Crew 2".')
    if '{X}' in mana_cost.upper():
        musts.append('Use X in the effect (for example "X damage" or "X target creatures").')
    variable = [stat for stat in ('power', 'toughness') if re.search(r'[*X]', str(card.get(stat) or ''), re.I)]
    if variable:
        ref = self_reference(card)
        ref = ref[0].upper() + ref[1:]
        what = 'power and toughness are each' if len(variable) == 2 else f'{variable[0]} is'
        musts.append(f'Its {" and ".join(variable)} {"are" if len(variable) == 2 else "is"} variable, so the first ability '
                     f'must define it, e.g. "{ref}\'s {what} equal to the number of lands you control."')
    if 'legendary' in (card.get('supertype') or '').lower():
        musts.append(f'Refer to the card as "{self_reference(card)}".')
    grounded = [w for w in subtype.split() if w in GROUNDED_SUBTYPES]
    if 'creature' in card_type and grounded:
        musts.append(f'A {grounded[0].capitalize()} does not fly: no flying; prefer grounded keywords like vigilance, first strike, reach or menace.')

    hooks = [h for c in colors for h in COLOR_HOOKS[c]] or COLORLESS_HOOKS
    hook = rng.choice(hooks)  # drawn either way so seeded runs stay comparable
    if brief:
        musts.append(f"Card idea: {brief['identity']}. Build the abilities around this mechanic, "
                     f"scaled to the power budget: {brief['mechanic']}.")
    else:
        musts.append(f'Design hook to consider, scaled to the power budget: {hook}.')

    user = '\n'.join(facts) + '\n\n' + '\n'.join(f'- {m}' for m in musts)
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]


# ===== Cleanup =====

# Bullets, numbering and labels the model sometimes prefixes. A "-" before a number and
# colon is a loyalty cost ("-2: ..."), not a bullet.
_LABEL_RE = re.compile(r'^\s*(?:(?:[*•]|-(?!\s*(?:\d+|X)\s*:))+|\d+[.)]|\*\*[^*]+\*\*:?|(?:keywords?|static|triggered|activated|keyword|passive|loyalty)\s*(?:ability|abilities)?\s*:)\s*', re.I)
_TRIGGER_START = r'(?:When\b|Whenever\b|At the beginning\b|At the end\b)'


def _strip_wrappers(line: str) -> str:
    s = _norm(line)
    for _ in range(3):
        s = _LABEL_RE.sub('', s).strip()
        s = s.strip('*').strip()
        if len(s) >= 2 and s[0] in '"“\'' and s[-1] in '"”\'' and s[1:-1].count(s[0]) == 0:
            s = s[1:-1].strip()
    # A lone stray quote at either end (the model's old quoting habit)
    if s.count('"') % 2 == 1:
        s = s.strip('"').strip()
    if s[:1] in "'’" and s.count("'") % 2 == 1:
        s = s[1:].strip()
    if s[-1:] in "'’" and not re.search(r"s'$", s) and s.count("'") % 2 == 1:
        s = s[:-1].strip()
    return s


def _split_run_ons(line: str) -> list[str]:
    """'Flying. {T}: Draw a card. Whenever ...' -> separate abilities."""
    # "Cycling {2}, {T}: ..." -> keyword with its cost, then the activated ability
    line = re.sub(r'^((?:Cycling|Kicker|Flashback|Ward|Equip) (?:\{[^}]+\})+), (?=\{)', r'\1. ', line)
    # "Flash, {T}: Add {U}{U}." -> keywords, then the activated ability
    m = re.match(r'^([^:{]+?), (\{[^}]+\}.*:.*)$', line)
    if m and is_keyword_line(m.group(1)):
        line = f'{m.group(1)}. {m.group(2)}'
    # "Protection from blue, when this enters, ..." -> keyword, then the trigger
    m = re.match(r'^([^,:.]+), (when(?:ever)?\b.*)$', line, re.I)
    if m and is_keyword_line(m.group(1)):
        line = f'{m.group(1)}. {m.group(2)[0].upper()}{m.group(2)[1:]}'
    # Split after a sentence end (optionally closing a quote) before anything that starts
    # a new ability: a trigger, a cost ("{2}:", "Pay 2 life:", "Sacrifice ...:"), Equip or Crew
    parts = re.split(r'(?:(?<=[.!])|(?<=[.!]["”]))\s+(?=' + _TRIGGER_START +
                     r'|\{[^}]+\}(?:[^.:]{0,40}):|(?:Pay \d+ life|Sacrifice [^:.]{1,40}|Discard [^:.]{1,30}):|Equip\b|Crew \d)', line)
    out = []
    for p in parts:
        # "Flying, trample. When ..." -> keyword head split off
        m = re.match(r'^([A-Za-z][A-Za-z ,{}0-9]+)\.\s+(.*)$', p)
        if m and is_keyword_line(m.group(1)) and m.group(2):
            out += [m.group(1), m.group(2)]
        else:
            out.append(p)
    return [p.strip() for p in out if p.strip()]


def _fix_symbols(s: str) -> str:
    s = re.sub(r'\{(tap|t)\}', '{T}', s, flags=re.I)
    s = re.sub(r'(?<![{\w])(?:\(T\)|\[T\])', '{T}', s)
    s = re.sub(r'(^|[.;]\s*)Tap\s*:', r'\1{T}:', s)
    s = re.sub(r'\{([wubrgcxtq])\}', lambda m: '{' + m.group(1).upper() + '}', s)
    # {2U} / {1RR} -> {2}{U} / {1}{R}{R}
    s = re.sub(r'\{(\d+)([WUBRG]+)\}', lambda m: '{' + m.group(1) + '}' + ''.join('{' + c + '}' for c in m.group(2)), s)
    s = re.sub(r'\{([WUBRG]{2,})\}', lambda m: ''.join('{' + c + '}' for c in m.group(1)), s)
    return s


def _fix_templating(s: str, card: dict) -> str:
    ref = self_reference(card)
    name = (card.get('name') or '').strip()
    # Self-references: full name, ~, CARDNAME -> modern self-reference
    if name:
        s = re.sub(re.escape(name) + r'(?=\b|\W|$)', '~', s)
    s = re.sub(r'\bCARDNAME\b', '~', s)
    s = re.sub(r"(^|[.:]\s+|\n)~", lambda m: m.group(1) + ref[0].upper() + ref[1:], s)
    s = s.replace('~', ref)
    s = re.sub(r'\bthis (creature|artifact|enchantment|land|spell|planeswalker|permanent)\b', r'this \1', s, flags=re.I)
    # A legendary card uses its own name, outside quoted granted abilities
    if not ref.startswith('this ') and re.search(r'creature|planeswalker', (card.get('type') or ''), re.I):
        parts = s.split('"')
        parts[::2] = [re.sub(r'\b[Tt]his (?:creature|planeswalker)\b', ref, p) for p in parts[::2]]
        s = '"'.join(parts)
    s = re.sub(r'(^|[.:]\s+)this ', lambda m: m.group(1) + 'This ', s)

    s = re.sub(r'\benters the battlefield\b', 'enters', s, flags=re.I)
    s = re.sub(r'\b(gain)(s?) ([+\-−]\d+/[+\-−]\d+)', r'get\2 \3', s, flags=re.I)
    s = re.sub(r'\bconverted mana cost\b', 'mana value', s, flags=re.I)
    s = re.sub(r'\b(?:target creature or player|target player or creature|any target creature or player)\b', 'any target', s, flags=re.I)
    s = re.sub(r', then shuffle your library\b', ', then shuffle', s)
    s = re.sub(r'\bshuffle your library\b', 'shuffle', s)
    s = re.sub(r"\bit's controller\b", 'its controller', s)
    s = re.sub(r"\btheir owner's hand\b", "their owners' hands", s)
    s = re.sub(r'^Tap (?:it |this \w+ )?to add\b', '{T}: Add', s, flags=re.I)
    # "{2}: This creature gains/gets a +1/+1 counter." -> "{2}: Put a +1/+1 counter on this creature."
    s = re.sub(r'(:\s*)(This \w+|' + re.escape(ref) + r') (?:gains|gets|receives) (a|an|one|two|three|\d+) ([+\-−]\d+/[+\-−]\d+ counters?)',
               lambda m: f'{m.group(1)}Put {m.group(3)} {m.group(4)} on {m.group(2)[0].lower() + m.group(2)[1:] if m.group(2).startswith("This") else m.group(2)}', s)
    # "Enters with three +1/+1 counters." -> "This creature enters with three +1/+1 counters on it."
    s = re.sub(r'^[Ee]nters (with|tapped)', lambda m: f'{ref[0].upper() + ref[1:]} enters {m.group(1)}', s)
    # "This land has 3 charge counters." -> "This land enters with three charge counters on it."
    s = re.sub(r'^This (\w+) has (a|an|one|two|three|four|five|\d+) ([\w+/\-−]+) (counters?)\.?$', r'This \1 enters with \2 \3 \4 on it.', s)
    # Counters are removed, not paid: "{T}, Pay 1 charge counter:" -> "{T}, Remove a charge counter from this land:"
    s = re.sub(r'\b[Pp]ay (a|an|one|\d+|X) ([\w+/\-−]+) (counters?)\b',
               lambda m: f'Remove {"a" if m.group(1) in ("1", "one", "a", "an") else m.group(1)} {m.group(2)} {m.group(3)} from {ref}', s)
    s = re.sub(r'\byou get (\d+ |X )?life\b', lambda m: f'you gain {m.group(1) or ""}life', s)
    s = re.sub(r'\bfrom graveyard to hand\b', 'from your graveyard to your hand', s)
    s = re.sub(r'\bfrom graveyard\b', 'from your graveyard', s)
    s = re.sub(r'(?<!your )(?<!its owner\'s )(?<!their owners\' )\bto hand\b', 'to your hand', s)
    # "Add {R} and {U}" -> "Add {R}{U}"; small counts are spelled out ("with three counters")
    s = re.sub(r'\b([Aa]dd (?:\{[WUBRGC]\})+)(?: and |, (?:and )?)(\{[WUBRGC]\})(?!\s*or)', r'\1\2', s)
    s = re.sub(r'\b(enters with|put) ([2-9]|10) ([\w+/\-−]+ counters)\b',
               lambda m: f"{m.group(1)} {['two','three','four','five','six','seven','eight','nine','ten'][int(m.group(2)) - 2]} {m.group(3)}", s)
    # Generic mana can't be produced: "add {1}" -> "add {C}"
    s = re.sub(r'\b([Aa]dd) \{1\}', r'\1 {C}', s)
    s = re.sub(r'\b([Aa]dd) \{2\}', r'\1 {C}{C}', s)
    s = re.sub(r'(\benters with (?:a|an|one|two|three|four|five|X|\d+) [+\-−]\d+/[+\-−]\d+ counters?)(?! on)', r'\1 on it', s)
    # Tokens are created, not put onto the battlefield; artifact token names are capitalized
    s = re.sub(r'\bput (a|an|one|two|three|X|\d+) ([\w/ ]+?) tokens? (?:on|onto) the battlefield',
               lambda m: f'create {m.group(1)} {m.group(2)} token' + ('' if m.group(1).lower() in ('a', 'an', 'one') else 's'),
               s, flags=re.I)
    s = re.sub(r'\b(treasure|food|clue|blood|map|powerstone|junk) (tokens?)\b', lambda m: m.group(1).capitalize() + ' ' + m.group(2), s)
    s = re.sub(r'\s+(?:to|into) your mana pool\b', '', s, flags=re.I)
    # A tacked-on "You may pay {B}{G} to add {B} or {G}." sentence does nothing: drop it
    s = re.sub(r'\s*You may pay (?:\{[^}]+\})+ (?:more )?to add [^.]*\.', '', s).strip()
    # "you may pay {3} to add {R}. If you do, ..." -> "you may pay {3}. If you do, ..." (the mana is noise)
    s = re.sub(r'(\bpay (?:\{[^}]+\})+) to add (?:\{[^}]+\})+(?: or (?:\{[^}]+\})+)?(?=\. If you do\b)', r'\1', s)
    # Runaway repetition from a looping reply: "{C} {C} {C} {C} ..." -> "{C}{C}"
    s = re.sub(r'((\{[^}]+\})\s*)\2(?:\s*\2){3,}', r'\2\2', s)
    # An undefined token ("create a storm token") becomes a 1/1 creature token in the card's colors
    def define_token(m):
        word = m.group(2)
        if word.lower() in ('treasure', 'food', 'clue', 'blood', 'map', 'powerstone', 'junk', 'gold', 'incubator', 'creature'):
            return m.group(0)
        names = {'W': 'white', 'U': 'blue', 'B': 'black', 'R': 'red', 'G': 'green'}
        colors = [names[c] for c in (card.get('colors') or []) if c in names]
        color = ' and '.join(colors[:2]) if colors else 'colorless'
        plural = m.group(3) == 'tokens'
        return f"{m.group(1)} 1/1 {color} {word.capitalize()} creature token{'s' if plural else ''}"
    s = re.sub(r'\b([Cc]reate (?:a|an|one|two|three|X|\d+)) ([A-Za-z]+) (tokens?)\b', define_token, s)
    # Triggers: a card's own entering happens once ("When"); casting spells repeats ("Whenever")
    s = re.sub(r'^Whenever (this \w+|' + re.escape(ref) + r') enters\b', r'When \1 enters', s)
    s = re.sub(r'^When you cast (a|an|another|your|each)\b', r'Whenever you cast \1', s)
    s = re.sub(r'^When(?:ever)? (?:equipped|this equipment is attached),',
               'Whenever this Equipment becomes attached to a creature,', s, flags=re.I)
    s = re.sub(r'\bthis (vehicle|equipment|aura)\b', lambda m: 'this ' + m.group(1).capitalize(), s, flags=re.I)
    # Stat changes are always +N/+N; contractions are Oracle style
    s = re.sub(r'\b(gets?|gains?) \+(\d+|X) power\b', r'gets +\2/+0', s)
    s = re.sub(r'\b(gets?|gains?) \+(\d+|X) toughness\b', r'gets +0/+\2', s)
    s = re.sub(r'\bcannot\b', "can't", s)
    s = re.sub(r'\bCannot\b', "Can't", s)
    s = re.sub(r'\bdoes not\b', "doesn't", s)
    # Token colors are lowercase: "three 1/1 Red Fire Elemental" -> "three 1/1 red Fire Elemental"
    s = re.sub(r'(\d+/\d+) (White|Blue|Black|Red|Green|Colorless)\b', lambda m: f'{m.group(1)} {m.group(2).lower()}', s)
    s = re.sub(r'\b(?:When|Whenever) you play a land\b', 'Whenever a land you control enters', s)
    # Events that can happen again trigger with "Whenever"; the card's own events keep "When"
    s = re.sub(r'^When (?!this\b|you cast this\b|enchanted\b|equipped\b)(you|an opponent|a player|another|a creature|a land|an artifact|one or more)\b',
               r'Whenever \1', s)
    s = re.sub(r'^When (this \w+|equipped creature|enchanted creature|' + re.escape(ref) + r') (?!enters|dies|leaves|is put into|becomes attached|is turned face up)(attacks|blocks|deals|becomes blocked|becomes tapped)\b',
               r'Whenever \1 \2', s)
    s = re.sub(r'\b([Pp]ays?) \{(\d+|X)\} life\b', r'\1 \2 life', s)
    s = re.sub(r'\bDrain each opponent for (\d+|X) life\b', r'Each opponent loses \1 life and you gain \1 life', s)
    s = re.sub(r'\)\.$', ')', s)
    # Legendary short names: the model truncates ("Vyrax" for "Vyraxa") and inserts the
    # name mid-sentence ("At the beginning of your end step, Vyraxa, create ...")
    if ref and not ref.startswith('this '):
        s = re.sub(r'\b(\w{4,})\b', lambda m: ref if m.group(1) != ref and ref.startswith(m.group(1)) and ' ' not in ref else m.group(1), s)
        s = re.sub(r'(, )' + re.escape(ref) + r', (?=(create|draw|put|deal|return|exile|destroy|you|each|target)\b)', r'\1', s)
        # "..., Vyraxa, deals 3 damage" -> "..., Vyraxa deals 3 damage" (the name is the subject)
        s = re.sub(r'(, )' + re.escape(ref) + r', (?=[a-z]+s\b)', r'\1' + ref + ' ', s)
        # "Vyraxa, the dragon queen, can't be blocked" -> "Vyraxa can't be blocked" (anywhere in the line)
        s = re.sub(r'\b' + re.escape(ref) + r', the [^,]+, ', ref + ' ', s)
        s = re.sub(r'\b' + re.escape(ref) + r', the [A-Z][\w\'-]+\b', ref, s)
    # A permanent's damage has a source: ", deal 2 damage" -> ", this enchantment deals 2 damage"
    if not ref.startswith('this spell'):
        s = re.sub(r'(, (?:you may )?)deal (\d+|X) damage', lambda m: f'{m.group(1)}{ref} deals {m.group(2)} damage'.replace('you may ' + ref + ' deals', 'you may have ' + ref + ' deal'), s)
    # Slang and invalid targets
    s = re.sub(r'\bbounce (target|a|an) (creature|permanent|artifact|enchantment)( you control)?',
               lambda m: f"return {m.group(1)} {m.group(2)}{m.group(3) or ''} to its owner's hand", s)
    s = re.sub(r'\btutor for (a|an) ([\w ]+?card)\b', r'search your library for \1 \2', s)
    s = re.sub(r'\b[Cc]ounter any target\b', lambda m: m.group(0)[0] + 'ounter target spell', s)
    s = re.sub(r'\bcreate (a|an|one|two|three|\d+) ([+\-−]\d+/[+\-−]\d+|[a-z]+) (counters?) on\b', r'put \1 \2 \3 on', s)
    # An invented label before a cost: "Roots, {T}: Add {G}." -> "{T}: Add {G}."
    m = re.match(r'^([A-Z][a-z]+),\s+(?=\{|[A-Z])', s)
    if m and not keyword_head(m.group(1)) and not re.match(r'(Sacrifice|Discard|Exile|Pay|Remove|Tap|Untap|Return|Reveal)$', m.group(1)):
        s = s[m.end():]
    # "{Q}, {T}: Sacrifice this land: Add ..." -> "{Q}, {T}, Sacrifice this land: Add ..."
    s = re.sub(r'^((?:\{[^}]+\}(?:,\s*)?)+):\s*((?:Sacrifice|Discard|Exile|Pay|Remove)[^:.]*):', r'\1, \2:', s)
    # Statics say "has"; "gains" is for effects with a duration
    s = re.sub(r'^(Equipped|Enchanted) creature gains\b(?![^.]*\buntil\b)', r'\1 creature has', s)
    s = re.sub(r'\b([Pp])rotection from (White|Blue|Black|Red|Green)\b', lambda m: m.group(1) + 'rotection from ' + m.group(2).lower(), s)
    # Invalid or made-up game actions
    s = re.sub(r'\badd (a|an|one|two|three|\d+) ([+\-−]\d+/[+\-−]\d+|[a-z]+) (counters?) on\b', r'put \1 \2 \3 on', s)
    s = re.sub(r'\bgains? \+?(\d+)/\+?(\d+)\b', r'gets +\1/+\2', s)
    # A repeating ability makes at most one token ("Whenever a creature you control attacks,
    # create three 1/1 ..." is the most common way a cheap card breaks)
    if re.match(r'^(Whenever|At the beginning|\{T\}:)', s):
        def one_token(m):
            art = 'an' if re.match(r'[aeiou]', m.group(3), re.I) else 'a'
            return f'{m.group(1)}reate {art} {m.group(3)}token'
        s = re.sub(r'\b([Cc])reate (two|three|four|five|X|[2-9]) ([^.]*?)tokens\b', one_token, s)
    # Truncated or garbled forms of a legendary name: "Vyraxa Sovereign" -> "Vyraxa"
    full_words = set((card.get('name') or '').replace(',', ' ').split())
    if ref and not ref.startswith('this ') and len(full_words) > 1:
        s = re.sub(r'\b' + re.escape(ref) + r'((?: [A-Z][\w\'-]+)+)\b',
                   lambda m: ref if set(m.group(1).split()) <= full_words else m.group(0), s)
    # Statics say "have": "Artifacts you control gain trample." -> "... have trample."
    if ability_kind(s) == 'static' and 'until end of turn' not in s:
        s = re.sub(r'^((?:Other )?(?:[A-Z]\w* )?(?:creatures|artifacts|permanents|[A-Z]\w+s) you control) gain\b', r'\1 have', s)
    # Graveyard targets are cards: "Return target creature from your graveyard" -> "target creature card"
    s = re.sub(r'\btarget (creature|artifact|enchantment|land|instant|sorcery|permanent) from (your|a|an opponent\'s) graveyard',
               r'target \1 card from \2 graveyard', s)
    s = re.sub(r'\bAt the beginning of end step\b', 'At the beginning of your end step', s)
    s = re.sub(r'^(When|Whenever) it\b', lambda m: f'{m.group(1)} {ref}', s)
    s = re.sub(r'^Discard:', 'Discard a card:', s)
    # "Vyraxa deals 2 damage to each opponent during your end step." -> a real trigger
    m = re.match(r'^(.+?),? (?:during|at) (?:the beginning of )?your (upkeep|end step|draw step)\.?$', s)
    if m and ability_kind(s) == 'static':
        body = m.group(1)
        if not body.startswith(ref):
            body = body[0].lower() + body[1:]
        s = f'At the beginning of your {m.group(2)}, {body}.'
    # Counters are put on things, never gained: "it gains a charge counter" -> "put a charge counter on it"
    s = re.sub(r'\b(it|this \w+|' + re.escape(ref) + r') (?:gains|gets|receives) (a|an|one|two|three|\d+) ([\w+/\-−]+ counters?)\b',
               r'put \2 \3 on \1', s)
    # A payment-free colon inside a trigger: "When ~ dies, return it to your hand: You gain 3 life."
    if ability_kind(s) == 'triggered' and _colon_outside_quotes(s) > 0 and not re.search(r'\bpay\b[^:]*:', s):
        i = _colon_outside_quotes(s)
        rest_of = s[i + 1:].strip()
        s = s[:i] + '. ' + rest_of[:1].upper() + rest_of[1:]
    # No reminder text: it crowds the text box ("Cycling {2} (You may pay ...)")
    s = re.sub(r'\s*\((?:[^()"]|"[^"]*")*\)', '', s).strip()
    # Only players gain life: "Kaelis gains 1 life" -> "you gain 1 life"
    s = re.sub(r'\b(?:' + re.escape(ref) + r'|this creature|it) gains (\d+|X) life\b', r'you gain \1 life', s)
    # "Bligma has menace and first strike." -> keyword line
    m = re.match(r'^(?:' + re.escape(ref) + r'|This creature|It) has ([a-z ,]+)\.$', s, re.I)
    if m:
        parts = [p.strip() for p in re.split(r',\s*|\s+and\s+', m.group(1)) if p.strip()]
        if parts and all(p in SIMPLE_KEYWORDS for p in parts):
            s = ', '.join(parts)
            s = s[0].upper() + s[1:]
    if ability_kind(s) in ('activated', 'loyalty'):
        s = re.sub(r'^([^:"]+:\s*)it\b', lambda m: m.group(1) + (ref[0].upper() + ref[1:]), s)
        s = re.sub(r'^([^:"]+:\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)
    s = re.sub(r'\bprotection from (artifact|creature|enchantment|instant|sorcery|planeswalker)\b(?!s)', r'protection from \1s', s)
    s = re.sub(r'\bhas ([^.,"]+?) and has\b', r'has \1 and', s)
    s = re.sub(r'\bDestroy any target\b', 'Destroy target creature', s)
    s = re.sub(r'\bdestroy any target\b', 'destroy target creature', s)
    # "Add {C} or {U} or {R} or {G} or {B}" -> "Add one mana of any color"
    if len(re.findall(r'\{[WUBRG]\}', s)) >= 4 and re.search(r'\bAdd (\{[WUBRGC]\}(?:,| or| and)?\s*){4,}', s):
        s = re.sub(r'\bAdd (\{[WUBRGC]\}(?:,| or| and)?\s*){4,}', 'Add one mana of any color', s).replace('color.', 'color.').strip()
        s = re.sub(r'any color(?=[A-Za-z])', 'any color ', s)
    # A payment inside a trigger: "..., you may pay {R}{R}: Effect." -> "..., you may pay {R}{R}. If you do, effect."
    if ability_kind(s) == 'triggered':
        def if_you_do(m):
            word = m.group(2)
            # lowercase ordinary words ("Target" -> "target"), keep names ("Vyraxa")
            if word.split()[0] in ('Target', 'Put', 'Create', 'Draw', 'You', 'Each', 'It', 'This', 'That', 'Return',
                                   'Destroy', 'Exile', 'Deal', 'Tap', 'Untap', 'Scry', 'Add', 'Gain', 'Search'):
                word = word[0].lower() + word[1:]
            return f'{m.group(1)}. If you do, {word}'
        s = re.sub(r'(\bpay (?:\{[^}]+\})+):\s*(\w+)', if_you_do, s)
    s = re.sub(r'\s+([.,;:])', r'\1', s)
    s = re.sub(r'\.{2,}', '.', s)
    return s


def _keyword_items(line: str) -> list[str] | None:
    """The keywords in a keyword-ish item, or None if it isn't one. Unknown fragments are dropped."""
    s = line.strip().rstrip('.')
    if ':' in s or not s or len(s.split()) > 12:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    known = [p for p in parts if keyword_head(p)]
    # Mostly keywords -> treat as a keyword item and drop invented ones ("primal growth")
    if known and len(known) >= len(parts) - 1 and not re.search(r'\b(you|target|each|gets?|has|have|deals?|draws?|creates?)\b', s.lower()):
        return known
    return None


def clean_abilities(abilities: list[str], card: dict | None) -> list[str]:
    """Turn the model's abilities into legal, consistently templated lines (see module doc)."""
    card = card or {}
    card_type = (card.get('type') or '').lower()
    subtype = (card.get('subtype') or '').lower()
    rarity = (card.get('rarity') or 'common').lower()
    name = (card.get('name') or '').strip().lower()
    tl = type_line(card).lower()
    is_spell = 'instant' in card_type or 'sorcery' in card_type
    is_pw = 'planeswalker' in card_type
    is_creature = 'creature' in card_type
    grants = 'equipment' in subtype or 'aura' in subtype or 'vehicle' in subtype

    # 1. Unwrap, split run-ons, drop junk
    lines: list[str] = []
    for raw in abilities or []:
        if not isinstance(raw, str):
            continue
        for chunk in re.split(r'\n+', raw):
            s = _strip_wrappers(chunk)
            if not s:
                continue
            lines.extend(_split_run_ons(s))
    kept = []
    for s in lines:
        low = s.lower().rstrip('.')
        if re.search(r'\b(rules text|here is|here are|starting loyalty|loyalty:)\b', low) and not re.match(r'^[+\-−]?\d+:', s):
            continue
        if re.fullmatch(r'\d+/\d+', low) or (tl and (low == tl or low.startswith(tl))):
            continue
        if name and (low == name or re.fullmatch(re.escape(name) + r'[,:\-— ]+.*\b(legendary|creature|artifact|enchantment|instant|sorcery|land)\b.*', low)):
            continue
        kept.append(s)

    # 2. Symbols and templating, one line at a time; then drop near-duplicates
    # ("+1: ... deals 1 damage" / "+1: ... deals 2 damage" -> keep the first)
    kept = [_fix_templating(_fix_symbols(s), card) for s in kept]
    # Flavor text posing as an ability ("Vyraxa exhales a molten torrent ...") has no rules meaning
    kept = [s for s in kept if ability_kind(s) != 'static' or RULES_WORDS.search(s) or keyword_head(s.split(',')[0])]
    # A lone invented name ("Miststep.", "Giant Growth.") or verbless fragment ("Charge counters.")
    # is a made-up keyword with no rules meaning
    kept = [s for s in kept if not (
        (re.fullmatch(r"(?:[A-Z][\w'’-]+ ){0,2}[A-Z][\w'’-]+\.?", s) or
         (re.fullmatch(r"[A-Z][\w'’-]+(?: [\w'’-]+){0,2}\.?", s) and not IMPERATIVE_START.match(s)
          and not re.search(r'\b(gets?|has|have|can\'t|deals?|is|are)\b', s)))
        and not keyword_head(s))]
    seen_keys, unique = set(), []
    for s in kept:
        key = re.sub(r'[\d+\-−{}X\s]+', ' ', s.lower()).strip()[:60]
        if key and key not in seen_keys:
            seen_keys.add(key)
            unique.append(s)
    kept = unique

    # 3. Keywords: gather into one line, validate, drop redundancy
    simple_kw, param_first, param_last, rest = [], [], [], []
    for s in kept:
        kws = _keyword_items(s)
        if kws is None:
            rest.append(s)
            continue
        for k in kws:
            head = keyword_head(k)
            text = k[0].upper() + k[1:]
            if head == 'enchant':
                param_first.append(text)
            elif head in ('equip', 'crew'):
                param_last.append(text)
            elif head in SIMPLE_KEYWORDS:
                simple_kw.append(k.lower())
            else:
                simple_kw.append(k[0].lower() + k[1:])
    simple_kw = list(dict.fromkeys(simple_kw))
    # Printed order follows the Comprehensive Rules ("Flying, trample", "Deathtouch, lifelink")
    simple_kw.sort(key=lambda k: KEYWORD_ORDER.index(keyword_head(k)) if keyword_head(k) in KEYWORD_ORDER else len(KEYWORD_ORDER))
    if 'flying' in simple_kw and 'reach' in simple_kw:
        simple_kw.remove('reach')
    if 'double strike' in simple_kw and 'first strike' in simple_kw:
        simple_kw.remove('first strike')
    if not is_creature and not grants:
        simple_kw = [k for k in simple_kw if keyword_head(k) not in CREATURE_KEYWORDS]
    if is_spell:
        simple_kw = [k for k in simple_kw if keyword_head(k) not in CREATURE_KEYWORDS]

    # 4. Card-type rules
    if is_spell:
        spell_lines = []
        for s in rest:
            m = re.match(r'^When(?:ever)? (?:you cast (?:this spell|it)|this spell is cast|\w[\w ,\']* is cast),\s*(.+)$', s, re.I)
            if m:
                s = m.group(1)[0].upper() + m.group(1)[1:]
            kind = ability_kind(s)
            if kind in ('activated', 'triggered') and spell_lines:
                continue  # permanent-style abilities don't belong on a spell
            if kind == 'triggered' and re.search(r'\benters\b|\battacks\b|\bdies\b', s):
                continue
            if kind == 'activated':
                # "{R}: Deal 2 damage..." as the spell's only text: the cost is spurious
                s = s[_colon_outside_quotes(s) + 1:].strip()
                s = s[:1].upper() + s[1:]
            # A spell's damage has a source: "Deal 3 damage" -> "This spell deals 3 damage"
            s = re.sub(r'^Deals? (X|\d+) damage', r'This spell deals \1 damage', s)
            spell_lines.append(s)
        # A spell is one effect block: keep the first two sentences' worth of items
        rest = spell_lines[:2]
    if is_pw:
        loyalty = []
        for s in rest:
            m = re.match(r'^([+\-−–—]?)\s*(\d+|X)\s*:\s*(.+)$', s)
            if m:
                sign = {'-': '−', '–': '−', '—': '−'}.get(m.group(1), m.group(1))
                if m.group(2) == '0':
                    sign = ''
                elif not sign:
                    sign = '+'
                loyalty.append(f'{sign}{m.group(2)}: {m.group(3)[0].upper() + m.group(3)[1:]}')
        others = [s for s in rest if not re.match(r'^[+\-−–—]?\s*(\d+|X)\s*:', s)]
        # One ability per loyalty cost (real planeswalkers never repeat one)
        by_cost = {}
        for ab in loyalty:
            by_cost.setdefault(ab.split(':')[0], ab)
        loyalty = list(by_cost.values())
        loyalty.sort(key=lambda x: 0 if x.startswith('+') else 1 if x.startswith('0') else 2)
        rest = others[:1] + loyalty[:4]
    # A non-Equipment, non-Aura card that talks about "equipped/enchanted creature" (a
    # "Banner" enchantment) most plausibly means the team: "Creatures you control get ..."
    if 'equipment' not in subtype and 'aura' not in subtype:
        def team(s):
            s = re.sub(r'^Whenever (?:equipped|enchanted) creature\b', 'Whenever a creature you control', s)
            s = re.sub(r'^(?:Equipped|Enchanted) creature gets\b', 'Creatures you control get', s)
            s = re.sub(r'^(?:Equipped|Enchanted) creature has\b', 'Creatures you control have', s)
            s = re.sub(r'^(?:Equipped|Enchanted) creature can\'t\b', "Creatures you control can't", s)
            return s
        rest = [team(s) for s in rest]
    # Meaningless "X has charge counters." statements
    rest = [s for s in rest if not re.match(r'^[\w ]+ has (?:\w+ )?counters?\.?$', s)]
    param_first = list(dict.fromkeys(p.rstrip('.') for p in param_first))
    param_last = list(dict.fromkeys(p.rstrip('.') for p in param_last))
    # Enchant / Equip / Crew only belong on Auras / Equipment / Vehicles
    if 'aura' not in subtype:
        param_first = []
    if 'equipment' not in subtype:
        param_last = [p for p in param_last if not p.lower().startswith('equip')]
    if 'vehicle' not in subtype:
        param_last = [p for p in param_last if not p.lower().startswith('crew')]
    if 'aura' in subtype and not param_first:
        param_first = ['Enchant creature']
    if 'equipment' in subtype and not param_last:
        param_last = [f'Equip {{{max(1, min(4, int(card.get("cmc") or 1)))}}}']
    # Exactly one numeric Crew ("Crew 1, Crew 2, ... Crew N" happens)
    crews = [p for p in param_last if re.match(r'^crew \d+$', p, re.I)]
    param_last = [p for p in param_last if not p.lower().startswith('crew')] + crews[:1]
    if 'vehicle' in subtype and not crews:
        param_last.append(f'Crew {max(1, min(4, int(card.get("cmc") or 2) - 1))}')

    # Variable power/toughness must be defined, and first
    ref = self_reference(card)
    ref_cap = ref[0].upper() + ref[1:]
    star_p = bool(re.search(r'[*X]', str(card.get('power') or ''), re.I))
    star_t = bool(re.search(r'[*X]', str(card.get('toughness') or ''), re.I))
    if is_creature and (star_p or star_t):
        both = star_p and star_t
        for i, s in enumerate(rest):
            if re.search(r"\b(power|toughness)\b[^.]*\bequal to\b", s, re.I):
                if both and not re.search(r'power and toughness', s, re.I):
                    s = re.sub(r"'s (power|toughness) (?:is|are) equal to", "'s power and toughness are each equal to", s)
                rest.pop(i)
                rest.insert(0, s)
                break
        else:
            colors = card.get('colors') or []
            counted = ('the number of lands you control' if 'G' in colors else
                       'the number of creature cards in your graveyard' if 'B' in colors else
                       'the number of cards in your hand' if 'U' in colors else
                       'the number of creatures you control')
            stat = 'power and toughness are each' if both else ('power is' if star_p else 'toughness is')
            rest.insert(0, f"{ref_cap}'s {stat} equal to {counted}.")

    # 5. Periods and capitals
    def finish(s):
        s = s.strip()
        if not s:
            return s
        s = s[0].upper() + s[1:]
        if not s.endswith(('.', '"', '”', ')', '!')):
            s += '.'
        return s

    rest = [finish(s) for s in rest if s.strip(' .')]
    rest = list(dict.fromkeys(rest))

    # 6. Budget: rarity cap and legibility, dropping from the end (usually the most complex)
    cap = RARITY_ABILITY_CAP.get(rarity, 4) + (1 if is_pw else 0)

    def assemble():
        out = []
        out += param_first
        if simple_kw:
            kw_line = ', '.join(simple_kw)
            out.append(kw_line[0].upper() + kw_line[1:])
        out += rest
        out += param_last
        return out

    def weight():
        return len(simple_kw) + len(rest) + len(param_first)

    while rest and (weight() > cap or len('\n'.join(assemble())) > MAX_TEXT_CHARS) and len(rest) > (2 if is_pw else 1):
        rest.pop()
    while len(simple_kw) > 1 and weight() > cap:
        simple_kw.pop()
    return [l for l in assemble() if l]


def format_rules_text(lines: list[str]) -> str:
    return '\n'.join(l.strip() for l in lines if l and l.strip())


# ===== Generation =====

def parse_reply(raw: str) -> list[str]:
    """The abilities from a model reply: JSON {"abilities": [...]}, or plain lines as a fallback."""
    import json
    raw = (raw or '').strip()
    raw = re.sub(r'^```(?:json)?|```$', '', raw).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            items = data.get('abilities') or data.get('rules') or data.get('text') or []
            if isinstance(items, str):
                items = [items]
            return [str(x) for x in items if str(x).strip()]
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
    except (ValueError, TypeError):
        pass
    # Truncated JSON (the reply hit the length limit): keep every complete string
    if raw.startswith('{') and '"abilities"' in raw:
        body = raw.split('"abilities"', 1)[1]
        items = [json.loads(f'"{x}"') for x in re.findall(r'"((?:[^"\\]|\\.)*)"(?=\s*[,\]])', body)]
        return [x for x in items if x.strip()]
    # Not JSON: one quoted ability per "..." or one per line
    quoted = re.findall(r'"([^"\n]{3,})"', raw)
    if len(quoted) >= 1 and sum(len(q) for q in quoted) > len(raw) * 0.6:
        return quoted
    return [l for l in raw.split('\n') if l.strip()]


def generate_rules_text(prompt: str, card: dict | None, client, model: str, *,
                        attempts: int = 2, temperature: float = 0.85, keep_alive='30m', rng=None,
                        think: bool = False) -> str | None:
    """
    Ask the model for rules text, clean it, and keep the attempt with the fewest lint
    errors, then the least over its power budget (stopping early on a clean one).
    Returns None if every call failed.
    """
    import random
    rng = rng or random.Random()
    best, best_score = None, None
    for _ in range(max(1, attempts)):
        messages = build_messages(prompt, card, rng)
        try:
            resp = client.chat(
                model=model, messages=messages, format=RESPONSE_SCHEMA, think=think,
                options={"temperature": temperature, "top_p": 0.95, "repeat_penalty": 1.1,
                         "num_predict": 2000 if think else 300,
                         # ~1.5k-token prompt + short reply. A small context keeps qwen3:8b at ~5.3 GB so
                         # SDXL still fits beside it on a 12 GB GPU (at 4096 it spills: 5s -> 120s art).
                         "num_ctx": 4096 if think else 2560},
                keep_alive=keep_alive,
            )
        except Exception as e:  # timeouts, connection errors, unknown model
            print(f'❌ Rules text model call failed: {e}')
            if best is None:
                raise
            break
        raw = resp['message']['content'] if isinstance(resp, dict) or hasattr(resp, '__getitem__') else str(resp)
        print(f'📜 Rules text raw reply: {raw!r}')
        text = format_rules_text(clean_abilities(parse_reply(raw), card))
        errors = error_count(lint_rules_text(text, card)) if text else 99
        value, allowed = power.assess(text, card or {})
        over = max(0.0, value - allowed)
        print(f'🧹 Cleaned rules text ({errors} lint errors, power {value}/{allowed}): {text!r}')
        if best is None or (errors, over) < best_score:
            best, best_score = text, (errors, over)
        if errors == 0 and over <= power.WARN_OVER:
            break
    # Last resorts. A line that still has a lint error after every attempt is dropped, as
    # long as the card keeps at least one real ability and its Enchant/Equip/Crew lines
    if best and best_score[0]:
        lines = best.split('\n')
        messages = [m for s, m in lint_rules_text(best, card) if s == 'error']
        bad = [l for l in lines if any(repr(l) in m for m in messages)
               and not re.match(r'^(Enchant|Equip|Crew)\b', l)]
        kept_lines = [l for l in lines if l not in bad]
        if bad and any(not re.match(r'^(Enchant|Equip|Crew)\b', l) for l in kept_lines):
            print(f'🚫 Dropped lines that stayed broken: {bad!r}')
            best = format_rules_text(kept_lines)
            best_score = (error_count(lint_rules_text(best, card)),
                          max(0.0, power.estimate(best, card or {}) - power.budget(card or {})))
    # Never print a forbidden ability: drop it if anything else remains
    if best:
        lines = best.split('\n')
        allowed_lines = [l for l in lines if not any(p.search(l) for p, _ in FORBIDDEN)]
        if allowed_lines and allowed_lines != lines:
            print(f'🚫 Dropped forbidden abilities: {[l for l in lines if l not in allowed_lines]!r}')
            best = format_rules_text(allowed_lines)
            best_score = (best_score[0], max(0.0, power.estimate(best, card or {}) - power.budget(card or {})))
    # The model ignored the budget every time, so drop its most valuable ability
    if best and best_score[1] > power.ERROR_OVER:
        lines = best.split('\n')
        protect = 1 if lines and re.match(r"^(Enchant\b|.*'s (power|toughness)\b.*\bequal to)", lines[0]) else 0
        trimmed = format_rules_text(power.trim_to_budget(lines, card or {}, keep_first=protect))
        print(f'✂️ Trimmed to its power budget: {trimmed!r}')
        best = trimmed
    return best
