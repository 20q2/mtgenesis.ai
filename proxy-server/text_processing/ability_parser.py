"""
Ability parsing and classification for Magic: The Gathering cards.

This module handles parsing quoted abilities, classifying abilities by type,
and reordering abilities according to Magic card conventions.
"""

import re


def parse_abilities(card_text):
    """
    Parse abilities from quoted format where each ability is wrapped in double quotes.
    Input: '"Flying, trample" "When this enters, draw a card" "{T}: Add one mana"'
    Output: ['Flying, trample', 'When this enters, draw a card', '{T}: Add one mana']
    
    Falls back to traditional parsing if no quoted abilities are found.
    """
    from .text_sanitizer import smart_split_by_periods
    
    if not card_text:
        return []
    
    # Pattern to match quoted abilities: "ability text"
    quoted_pattern = r'"([^"]+)"'
    quoted_matches = re.findall(quoted_pattern, card_text)
    
    if quoted_matches:
        # Found quoted abilities - use them
        abilities = [ability.strip() for ability in quoted_matches if ability.strip()]
        return abilities
    else:
        # No quotes found - fall back to traditional parsing
        print(f"   ⚠️  No quoted abilities found, falling back to traditional parsing")
        if '\n' in card_text:
            abilities = [ability.strip() for ability in card_text.split('\n') if ability.strip()]
            return abilities
        else:
            # Smart period splitting that respects quotes
            abilities = smart_split_by_periods(card_text)
            return abilities


def classify_ability(ability, keywords):
    """
    Classify an ability as keyword, active, passive, or triggered
    """
    clean_ability = ability.rstrip('.,!?').lower()
    
    # Check if it's a keyword ability (exact match or starts with keyword)
    for keyword in keywords:
        if keyword.lower() == clean_ability or clean_ability.startswith(keyword.lower() + ' '):
            return 'keyword'
    
    # Check for triggered abilities (start with trigger words)
    trigger_patterns = [
        r'\bwhen\b', r'\bwhenever\b', r'\bat the beginning of\b', r'\bat end of\b',
        r'\bat the beginning of each\b', r'\bduring\b', r'\bif\b.*\bthen\b'
    ]
    
    for pattern in trigger_patterns:
        if re.search(pattern, clean_ability):
            return 'triggered'
    
    # Check for activated abilities (contain costs)
    cost_patterns = [
        r'\{[^}]*\}:', r'\{[^}]*\},', r'pay\s+\d+', r'sacrifice\s+', r'discard\s+',
        r'tap\s+.*:', r'exile\s+.*:', r'return\s+.*to.*hand.*:'
    ]
    
    for pattern in cost_patterns:
        if re.search(pattern, clean_ability):
            return 'active'
    
    # Default to passive
    return 'passive'


def reorder_abilities_properly(card_text, card_data=None):
    """
    Reorder abilities in proper Magic order based on card type
    """
    from .text_sanitizer import clean_ability_quotes, clean_ability_arrays
    
    # Determine card type for appropriate processing
    card_type = card_data.get('type', '').lower() if card_data else ''
    is_creature = 'creature' in card_type
    is_instant_sorcery = any(t in card_type for t in ['instant', 'sorcery'])
    is_artifact = 'artifact' in card_type
    is_enchantment = 'enchantment' in card_type
    is_planeswalker = 'planeswalker' in card_type
    
    # For instant/sorcery cards, don't apply creature-style ability reordering
    if is_instant_sorcery:
        # Just clean up the text and ensure proper periods (but not after quotes)
        cleaned_text = card_text.strip()
        if cleaned_text and not cleaned_text.endswith(('.', '!', '?', '"', "'")):
            cleaned_text += '.'
        return cleaned_text
    
    # For planeswalker cards, preserve loyalty abilities order
    if is_planeswalker:
        print(f"🔮 Planeswalker detected - preserving loyalty ability order")
        # Clean up text but don't reorder planeswalker abilities
        lines = [line.strip() for line in card_text.split('\n') if line.strip()]
        cleaned_lines = []
        for line in lines:
            if line and not line.endswith(('.', '!', '?', '"', "'")):
                line += '.'
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    # For creatures and other permanent types, use ability classification
    # Common Magic keywords (not exhaustive, but covers most used ones)
    keywords = {
        'flying', 'trample', 'vigilance', 'haste', 'reach', 'deathtouch', 'lifelink',
        'first strike', 'double strike', 'hexproof', 'shroud', 'indestructible',
        'flash', 'defender', 'menace', 'prowess', 'skulk', 'madness',
        'ward', 'protection', 'flanking', 'shadow', 'phasing', 'horsemanship',
        'fear', 'intimidate', 'landwalk', 'islandwalk', 'mountainwalk', 'swampwalk',
        'forestwalk', 'plainswalk', 'bushido', 'ninjutsu', 'bloodthirst', 'devour',
        'persist', 'undying', 'cascade', 'storm', 'buyback', 'flashback', 'escape',
        'cycling', 'kicker', 'multikicker', 'suspend', 'vanishing', 'fading',
        'echo', 'morph', 'megamorph', 'bestow', 'enchant', 'equip', 'crew', 'regenerate',
        'reinforce', 'retrace', 'rebound', 'totem armor', 'living weapon',
        'battle cry', 'undaunted', 'surge', 'emerge', 'escalate', 'melee',
        'fabricate', 'improvise', 'revolt', 'explore', 'enrage', 'raid', 'landfall',
        'prowl', 'champion', 'amplify', 'annihilator', 'exalted', 'wither', 'infect',
        'soulbond', 'miracle', 'overload', 'scavenge', 'unleash', 'cipher', 'evolve',
        'extort', 'battalion', 'tribute', 'inspired', 'heroic', 'constellation',
        'ferocious', 'outlast', 'dash', 'exploit', 'renown', 'awaken', 'ingest',
        'devoid', 'cohort', 'delirium', 'transform', 'meld', 'energy', 'aftermath',
        'embalm', 'eternalize', 'afflict', 'rampage', 'changeling', 'convoke', 'delve', 'splice'
    }
    
    # Parse abilities using the new quote-aware parser
    abilities = parse_abilities(card_text)
    
    keyword_abilities = []
    passive_abilities = []
    triggered_abilities = []
    active_abilities = []
    
    for ability in abilities:
        ability_type = classify_ability(ability, keywords)
        
        # Debug individual ability classification
        print(f"   🔍 Classifying: '{ability}' → {ability_type}")
        
        # Clean outer quotes from ability
        ability = clean_ability_quotes(ability.strip())
        
        # Ensure ability ends with a period (unless it's a keyword or ends with other punctuation or quotes)
        if ability and ability_type != 'keyword' and not ability.endswith(('.', '!', '?', ':', '"', "'")):
            ability += '.'
        
        if ability_type == 'keyword':
            keyword_abilities.append(ability)
        elif ability_type == 'passive':
            passive_abilities.append(ability)
        elif ability_type == 'triggered':
            triggered_abilities.append(ability)
        elif ability_type == 'active':
            active_abilities.append(ability)
    
    # Clean up typeline contamination from passive and triggered abilities
    # Note: This function is still in app.py and needs to be extracted
    if card_data:
        # TODO: Extract remove_typeline_contamination to avoid circular imports
        pass  # Temporarily disabled until we extract this function
    
    # Remove AI generation artifacts that mention "rules text" or meta-commentary
    def filter_rules_text_artifacts(abilities_list):
        """Remove abilities that are AI generation artifacts mentioning 'rules text' or meta-commentary"""
        meta_commentary_patterns = [
            'rules text',
            'here is a',
            'here are the',
            'potential rules text',
            'guidelines',
            'based on your',
            'i\'ll generate',
            'let me create',
            'card based on'
        ]
        
        filtered = []
        for ability in abilities_list:
            ability_lower = ability.lower()
            is_meta_commentary = any(pattern in ability_lower for pattern in meta_commentary_patterns)
            
            if is_meta_commentary:
                print(f"🗑️  Removed AI meta-commentary: '{ability}'")
            else:
                filtered.append(ability)
        return filtered
    
    # Apply rules text artifact filtering to all ability types
    keyword_abilities = filter_rules_text_artifacts(keyword_abilities)
    passive_abilities = filter_rules_text_artifacts(passive_abilities)
    triggered_abilities = filter_rules_text_artifacts(triggered_abilities)
    active_abilities = filter_rules_text_artifacts(active_abilities)
    
    # Clean unwanted formatting characters from all ability arrays
    abilities_dict = {
        'keyword': keyword_abilities,
        'passive': passive_abilities, 
        'triggered': triggered_abilities,
        'active': active_abilities
    }
    cleaned_abilities = clean_ability_arrays(abilities_dict)
    keyword_abilities = cleaned_abilities['keyword']
    passive_abilities = cleaned_abilities['passive']
    triggered_abilities = cleaned_abilities['triggered']
    active_abilities = cleaned_abilities['active']
    
    # Combine in proper order: Keywords -> Passive -> Triggered -> Active
    # Keywords should be comma-separated on one line, other abilities use newlines
    final_ability_blocks = []
    
    # Add keywords as a single comma-separated line
    if keyword_abilities:
        keywords_line = ', '.join(keyword_abilities)
        final_ability_blocks.append(keywords_line)
    
    # Add passive abilities (each on its own line)
    final_ability_blocks.extend(passive_abilities)
    
    # Add triggered abilities (each on its own line)
    final_ability_blocks.extend(triggered_abilities)
    
    # Add active abilities (each on its own line) 
    final_ability_blocks.extend(active_abilities)
    
    # Use newlines to separate different ability BLOCKS (not individual keywords)
    result = '\n'.join(final_ability_blocks)
    
    return result