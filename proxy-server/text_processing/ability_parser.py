"""
Ability parsing and classification for Magic: The Gathering cards.

This module handles parsing quoted abilities, classifying abilities by type,
and reordering abilities according to Magic card conventions.
"""

import re


def _is_keyword_like_structure(full_text, parts, known_keyword_count, ability_index=0):
    """
    Enhanced structural analysis to identify keyword-like patterns even with hallucinated keywords.
    
    Args:
        full_text: The full cleaned ability text
        parts: List of comma-separated parts 
        known_keyword_count: Number of parts that match known keywords
        ability_index: Position of this ability in the card (0 = first)
    
    Returns:
        bool: True if this looks like a keyword line structure
    """
    
    # Exclusion rules: Definitely NOT keywords if these patterns are present
    exclusion_patterns = [
        r'\{[^}]*\}',           # Mana symbols: {T}, {1}, {W}
        r':',                   # Activation costs: "{T}:", "Pay 1 life:"
        r'\bwhen\b|\bwhenever\b|\bat\s+the\s+beginning\b', # Triggered ability words
        r'\btarget\b|\bchoose\b|\bsearch\b',               # Ability verbs
        r'\bto\s+your\s+hand\b|\bonto\s+the\s+battlefield\b', # Long ability phrases
        r'\bdraw\s+\w+\s+card|\bdeal\s+\d+\s+damage',      # Specific ability patterns
        r'\byou\s+may\b|\bif\s+you\s+do\b',                # Modal/conditional language
    ]
    
    for pattern in exclusion_patterns:
        if re.search(pattern, full_text, re.IGNORECASE):
            return False
    
    # Word count validation: Each part should be 1-3 words
    for part in parts:
        word_count = len(part.split())
        if word_count > 3:  # Keywords rarely exceed 3 words
            return False
        if word_count == 0:  # Empty parts
            return False
    
    # Length heuristics: Keyword lines are typically shorter
    if len(full_text) > 80:  # Very long lines unlikely to be just keywords
        return False
    
    # Positional scoring: Keywords more likely at start of abilities
    position_bonus = max(0, 1.0 - (ability_index * 0.2))  # First ability gets full bonus
    
    # Structure scoring
    structure_score = 0.0
    
    # Base score for comma-separated structure
    structure_score += 0.3
    
    # Bonus for known keywords present
    if known_keyword_count > 0:
        known_ratio = known_keyword_count / len(parts)
        structure_score += known_ratio * 0.4  # Up to 0.4 bonus
    
    # Bonus for typical keyword characteristics
    avg_word_count = sum(len(part.split()) for part in parts) / len(parts)
    if avg_word_count <= 2.0:  # Most keywords are 1-2 words
        structure_score += 0.2
    
    # Bonus for proper capitalization patterns (Title Case is common for keywords)
    capitalization_score = 0
    for part in parts:
        words = part.split()
        if all(word.istitle() or word.islower() for word in words):
            capitalization_score += 1
    if capitalization_score == len(parts):
        structure_score += 0.1
    
    # Apply positional bonus
    final_score = structure_score * (0.7 + 0.3 * position_bonus)
    
    # Decision threshold: 0.6 means we're fairly confident this is keyword-like
    confidence_threshold = 0.6
    
    # Special case: If we have at least one known keyword and good structure, be more lenient
    if known_keyword_count >= 1 and len(parts) <= 4:
        confidence_threshold = 0.5
    
    # Additional fallback: Very short lists with good known keyword ratio
    if len(parts) <= 3 and known_keyword_count >= len(parts) // 2:
        confidence_threshold = 0.4
    
    # Debug scoring (can be removed later)
    # print(f"     📊 Keyword structure scoring: parts={len(parts)}, known={known_keyword_count}, "
    #       f"structure={structure_score:.2f}, position={position_bonus:.2f}, final={final_score:.2f}, "
    #       f"threshold={confidence_threshold:.2f}")
    
    return final_score >= confidence_threshold


def _is_single_keyword_like(text):
    """
    Check if a single word/phrase looks like it could be a keyword.
    Very conservative - only matches obvious keyword patterns.
    """
    # Must be short (1-2 words)
    words = text.split()
    if len(words) > 2:
        return False
    
    # Exclude common ability words that aren't keywords
    ability_exclusions = [
        'enters', 'dies', 'attacks', 'blocks', 'deals', 'takes', 'becomes',
        'target', 'choose', 'search', 'draw', 'discard', 'exile', 'return',
        'destroy', 'sacrifice', 'create', 'put', 'gain', 'lose', 'add', 'remove'
    ]
    
    for word in words:
        if word.lower() in ability_exclusions:
            return False
    
    # Must not contain ability syntax
    if ':' in text or '{' in text or '}' in text:
        return False
    
    # Length check: Keywords are typically 3-15 characters
    if len(text) < 3 or len(text) > 15:
        return False
    
    # Pattern check: Should look like a proper noun or adjective
    # Examples of good patterns: "Shroud", "Battle Cry", "Snow Ward"
    # Examples of bad patterns: "the", "and", "of"
    
    # Exclude common English function words
    function_words = ['the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by']
    if text.lower() in function_words:
        return False
    
    # Must be primarily alphabetic (allows for things like "Ward 2" if needed later)
    if not any(c.isalpha() for c in text):
        return False
    
    return True


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


def classify_ability(ability, keywords, ability_index=0):
    """
    Classify an ability as keyword, active, passive, or triggered
    Enhanced with structural pattern recognition for hallucinated keywords
    """
    clean_ability = ability.rstrip('.,!?').lower()
    
    # Check if it's a comma-separated list (enhanced with structural analysis)
    if ',' in clean_ability:
        # Split by commas and check if all parts are keywords
        parts = [part.strip() for part in clean_ability.split(',')]
        if all(parts):  # Make sure no empty parts
            # First check: Are ALL parts known keywords? (existing logic)
            all_known_keywords = True
            known_keyword_count = 0
            
            for part in parts:
                is_known_keyword = False
                for keyword in keywords:
                    if keyword.lower() == part or part.startswith(keyword.lower() + ' '):
                        is_known_keyword = True
                        known_keyword_count += 1
                        break
                if not is_known_keyword:
                    all_known_keywords = False
            
            # If all are known keywords, immediately return 'keyword'
            if all_known_keywords:
                return 'keyword'
            
            # Enhanced check: Structural pattern recognition for mixed known/unknown
            if _is_keyword_like_structure(clean_ability, parts, known_keyword_count, ability_index):
                print(f"   ✨ Structural keyword detection: '{clean_ability}' ({known_keyword_count}/{len(parts)} known)")
                return 'keyword'
    
    # Check if it's a single keyword ability (exact match or starts with keyword)
    for keyword in keywords:
        if keyword.lower() == clean_ability or clean_ability.startswith(keyword.lower() + ' '):
            return 'keyword'
    
    # Fallback: Single unknown word that might be a hallucinated keyword
    # Only if it's early in the abilities list and matches keyword patterns
    if ability_index <= 1 and _is_single_keyword_like(clean_ability):
        print(f"   🤔 Potential hallucinated single keyword: '{clean_ability}' (position {ability_index})")
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


def filter_rules_text_artifacts(abilities_list):
    """Remove abilities that are AI generation artifacts mentioning 'rules text' or meta-commentary"""
    meta_commentary_patterns = [
        'rules text',
        'here is a',
        'here are the',
        'these abilities',
        'ability list',
        'card abilities',
        'generated abilities',
        'mtg abilities',
        'magic abilities',
        'rules for',
        'text for',
        'abilities for',
        'card rules',
        'game rules'
    ]
    
    cleaned_abilities = []
    for ability in abilities_list:
        ability_lower = ability.lower().strip()
        is_meta_commentary = False
        
        for pattern in meta_commentary_patterns:
            if pattern in ability_lower:
                print(f"🗑️  Removed AI artifact: '{ability}' (contains '{pattern}')")
                is_meta_commentary = True
                break
        
        if not is_meta_commentary:
            cleaned_abilities.append(ability)
    
    return cleaned_abilities


def reorder_abilities_properly_array(abilities_array, card_data=None):
    """
    Reorder abilities array in proper Magic order based on card type.
    This version works directly with arrays to avoid parsing issues.
    """
    from .text_sanitizer import clean_ability_quotes, clean_ability_arrays
    
    if not abilities_array:
        return []
    
    # Determine card type for appropriate processing
    card_type = card_data.get('type', '').lower() if card_data else ''
    is_creature = 'creature' in card_type
    is_instant_sorcery = any(t in card_type for t in ['instant', 'sorcery'])
    is_artifact = 'artifact' in card_type
    is_enchantment = 'enchantment' in card_type
    is_planeswalker = 'planeswalker' in card_type
    
    # For instant/sorcery cards, don't apply creature-style ability reordering
    if is_instant_sorcery:
        # Just clean up abilities and ensure proper periods
        cleaned_abilities = []
        for ability in abilities_array:
            cleaned = ability.strip()
            if cleaned and not cleaned.endswith(('.', '!', '?', '"', "'")):
                cleaned += '.'
            cleaned_abilities.append(cleaned)
        return cleaned_abilities
    
    # For planeswalker cards, preserve loyalty abilities order
    if is_planeswalker:
        print(f"🔮 Planeswalker detected - preserving loyalty ability order")
        cleaned_abilities = []
        for ability in abilities_array:
            if ability and not ability.endswith(('.', '!', '?', '"', "'")):
                ability += '.'
            cleaned_abilities.append(ability)
        return cleaned_abilities
    
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
    
    # Classify abilities directly from the array (no re-parsing needed!)
    keyword_abilities = []
    passive_abilities_with_order = []  # Track passive abilities with their original position
    triggered_abilities = []
    active_abilities = []
    
    for i, ability in enumerate(abilities_array):
        ability_type = classify_ability(ability, keywords, ability_index=i)
        
        # Debug individual ability classification
        print(f"   🔍 Classifying: '{ability}' → {ability_type}")
        
        # Clean outer quotes from ability
        ability = clean_ability_quotes(ability.strip())
        
        # Ensure ability ends with a period (unless it's a keyword or ends with other punctuation or quotes)
        # Strip trailing whitespace and check for existing periods to avoid double periods
        ability = ability.rstrip()
        if (ability and ability_type != 'keyword' and 
            not ability.endswith(('.', '!', '?', ':', '"', "'"))):
            ability += '.'
        
        if ability_type == 'keyword':
            keyword_abilities.append(ability)
        elif ability_type == 'passive':
            # Store passive abilities with their original position to preserve order
            passive_abilities_with_order.append((i, ability))
        elif ability_type == 'triggered':
            triggered_abilities.append(ability)
        elif ability_type == 'active':
            active_abilities.append(ability)
    
    # Apply filtering and cleaning (reusing existing logic)
    # Extract passive abilities for filtering (preserving order info)
    passive_abilities = [ability for _, ability in passive_abilities_with_order]
    
    # Apply rules text artifact filtering to all ability types
    keyword_abilities = filter_rules_text_artifacts(keyword_abilities)
    passive_abilities_filtered = filter_rules_text_artifacts(passive_abilities)
    triggered_abilities = filter_rules_text_artifacts(triggered_abilities)
    active_abilities = filter_rules_text_artifacts(active_abilities)
    
    # Reconstruct passive abilities with order, keeping only the filtered ones
    passive_abilities_with_order_filtered = []
    passive_index = 0
    for original_order, original_ability in passive_abilities_with_order:
        if passive_index < len(passive_abilities_filtered) and passive_abilities_filtered[passive_index] in original_ability:
            passive_abilities_with_order_filtered.append((original_order, passive_abilities_filtered[passive_index]))
            passive_index += 1
    
    # Clean unwanted formatting characters from all ability arrays
    abilities_dict = {
        'keyword': keyword_abilities,
        'passive': passive_abilities_filtered, 
        'triggered': triggered_abilities,
        'active': active_abilities
    }
    cleaned_abilities = clean_ability_arrays(abilities_dict)
    keyword_abilities = cleaned_abilities['keyword']
    passive_abilities_cleaned = cleaned_abilities['passive']
    triggered_abilities = cleaned_abilities['triggered']
    active_abilities = cleaned_abilities['active']
    
    # Combine in proper order: Keywords first, then abilities in original order but grouped by type
    # Keywords should be comma-separated on one line, other abilities use newlines
    final_ability_blocks = []
    
    # Add keywords as a single comma-separated line (always first)
    if keyword_abilities:
        keywords_line = ', '.join(keyword_abilities)
        final_ability_blocks.append(keywords_line)
    
    # Create a mixed list preserving passive ability order while keeping triggered/active grouped
    # We'll build this by going through all abilities in original order and placing them appropriately
    all_abilities_ordered = []
    
    # Add passive abilities in their original relative positions
    passive_abilities_by_order = {order: ability for order, ability in passive_abilities_with_order_filtered}
    
    # Insert passive abilities in their original relative positions
    # For simplicity, we'll add passive abilities first, then triggered, then active
    # But passive abilities maintain their original order
    passive_abilities_sorted = sorted(passive_abilities_by_order.items())
    for _, ability in passive_abilities_sorted:
        final_ability_blocks.append(ability)
    
    # Then add triggered abilities
    final_ability_blocks.extend(triggered_abilities)
    
    # Finally add active abilities
    final_ability_blocks.extend(active_abilities)
    
    # Remove empty abilities
    final_abilities = [ability for ability in final_ability_blocks if ability and ability.strip()]
    
    return final_abilities


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
    passive_abilities_with_order = []  # Track passive abilities with their original position
    triggered_abilities = []
    active_abilities = []
    
    for i, ability in enumerate(abilities):
        ability_type = classify_ability(ability, keywords, ability_index=i)
        
        # Debug individual ability classification
        print(f"   🔍 Classifying: '{ability}' → {ability_type}")
        
        # Clean outer quotes from ability
        ability = clean_ability_quotes(ability.strip())
        
        # Ensure ability ends with a period (unless it's a keyword or ends with other punctuation or quotes)
        # Strip trailing whitespace and check for existing periods to avoid double periods
        ability = ability.rstrip()
        if (ability and ability_type != 'keyword' and 
            not ability.endswith(('.', '!', '?', ':', '"', "'"))):
            ability += '.'
        
        if ability_type == 'keyword':
            keyword_abilities.append(ability)
        elif ability_type == 'passive':
            # Store passive abilities with their original position to preserve order
            passive_abilities_with_order.append((i, ability))
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
    # (Function is now defined globally above)
    
    # Extract passive abilities for filtering (preserving order info)
    passive_abilities = [ability for _, ability in passive_abilities_with_order]
    
    # Apply rules text artifact filtering to all ability types
    keyword_abilities = filter_rules_text_artifacts(keyword_abilities)
    passive_abilities_filtered = filter_rules_text_artifacts(passive_abilities)
    triggered_abilities = filter_rules_text_artifacts(triggered_abilities)
    active_abilities = filter_rules_text_artifacts(active_abilities)
    
    # Reconstruct passive abilities with order, keeping only the filtered ones
    passive_abilities_with_order_filtered = []
    passive_index = 0
    for original_order, original_ability in passive_abilities_with_order:
        if passive_index < len(passive_abilities_filtered) and passive_abilities_filtered[passive_index] in original_ability:
            passive_abilities_with_order_filtered.append((original_order, passive_abilities_filtered[passive_index]))
            passive_index += 1
    
    # Clean unwanted formatting characters from all ability arrays
    abilities_dict = {
        'keyword': keyword_abilities,
        'passive': passive_abilities_filtered, 
        'triggered': triggered_abilities,
        'active': active_abilities
    }
    cleaned_abilities = clean_ability_arrays(abilities_dict)
    keyword_abilities = cleaned_abilities['keyword']
    passive_abilities_cleaned = cleaned_abilities['passive']
    triggered_abilities = cleaned_abilities['triggered']
    active_abilities = cleaned_abilities['active']
    
    # Combine in proper order: Keywords first, then abilities in original order but grouped by type
    # Keywords should be comma-separated on one line, other abilities use newlines
    final_ability_blocks = []
    
    # Add keywords as a single comma-separated line (always first)
    if keyword_abilities:
        keywords_line = ', '.join(keyword_abilities)
        final_ability_blocks.append(keywords_line)
    
    # Create a mixed list preserving passive ability order while keeping triggered/active grouped
    # We'll build this by going through all abilities in original order and placing them appropriately
    all_abilities_ordered = []
    
    # Add passive abilities in their original positions (relative to each other)
    passive_abilities_by_order = {order: ability for order, ability in passive_abilities_with_order_filtered}
    
    # Add triggered abilities (maintaining their relative order but grouped together)
    for ability in triggered_abilities:
        all_abilities_ordered.append(('triggered', ability))
    
    # Add active abilities (maintaining their relative order but grouped together)
    for ability in active_abilities:
        all_abilities_ordered.append(('active', ability))
    
    # Insert passive abilities in their original relative positions
    # For simplicity, we'll add passive abilities first, then triggered, then active
    # But passive abilities maintain their original order
    passive_abilities_sorted = sorted(passive_abilities_by_order.items())
    for _, ability in passive_abilities_sorted:
        final_ability_blocks.append(ability)
    
    # Then add triggered abilities
    final_ability_blocks.extend(triggered_abilities)
    
    # Finally add active abilities
    final_ability_blocks.extend(active_abilities)
    
    # Use newlines to separate different ability BLOCKS (not individual keywords)
    result = '\n'.join(final_ability_blocks)
    
    return result