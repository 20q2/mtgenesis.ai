"""
Rules text processing and sanitization utilities for Magic: The Gathering card generation.

This module handles cleaning, validating, and sanitizing rules text for different card types
to ensure they follow MTG conventions and don't contain type line contamination.
"""

import re
import ollama


def convert_to_legal_mtg_text(rules_text, card_data=None):
    """
    Pass rules text through context model to convert it to proper legal MTG card text format.
    
    Args:
        rules_text: The raw rules text to be converted
        card_data: Optional card data for context
        
    Returns:
        Legally formatted MTG rules text
    """
    if not rules_text or not rules_text.strip():
        return rules_text
    
    # Build conversion prompt
    card_type = card_data.get('type', '').lower() if card_data else 'card'
    card_name = card_data.get('name', '[cardname]') if card_data else '[cardname]'
    
    conversion_prompt = f"""Convert the following text into proper legal Magic: The Gathering card text format.

REQUIREMENTS:
- Use official MTG templating and wording
- Follow proper rules text conventions
- Fix any grammar, punctuation, or formatting issues
- Ensure abilities use correct Magic terminology
- Output ONLY the corrected rules text, no explanations
- Do not add card name, type line, or any other card elements
- This is for a {card_type} card

TEXT TO CONVERT:
{rules_text}

LEGAL MTG RULES TEXT:"""

    try:
        print(f"🔧 Converting rules text to legal MTG format...")
        response = ollama.chat(
            model='mistral:latest',
            messages=[{
                'role': 'user', 
                'content': conversion_prompt
            }]
        )
        
        converted_text = response['message']['content'].strip()
        print(f"   Original: {repr(rules_text[:100])}...")
        print(f"   Converted: {repr(converted_text[:100])}...")
        
        # Basic validation - ensure we got actual rules text back
        if len(converted_text) < 10 or 'LEGAL MTG RULES TEXT:' in converted_text:
            print(f"⚠️  Conversion failed, using original text")
            return rules_text
            
        return converted_text
        
    except Exception as e:
        print(f"❌ Legal MTG conversion failed: {e}")
        return rules_text


def limit_creature_active_abilities(card_text):
    """
    Ensure creature cards don't have more than 3 active abilities.
    Active abilities are those with activation costs like {T}, {1}, {2}, etc.
    Passive abilities (Flying, Vigilance) and triggered abilities don't count.
    """
    
    # Pattern to match active abilities: {COST}: effect
    # This matches things like {T}, {1}, {2}, {W}, {U}{R}, etc. followed by a colon
    active_ability_pattern = r'\{[^}]+\}:'
    
    # Find all active abilities
    active_abilities = re.findall(active_ability_pattern, card_text)
    
    if len(active_abilities) <= 3:
        return card_text  # Already within limit
    
    print(f"Warning: Found {len(active_abilities)} active abilities in creature text, limiting to 3")
    
    # Split the text into sentences and abilities
    sentences = [s.strip() for s in card_text.split('.') if s.strip()]
    
    # Rebuild the text, keeping only the first 3 active abilities
    filtered_sentences = []
    active_count = 0
    
    for sentence in sentences:
        sentence_active_abilities = re.findall(active_ability_pattern, sentence)
        
        # If this sentence would push us over the limit, skip it
        if active_count + len(sentence_active_abilities) > 3:
            # If we haven't hit the limit yet, try to keep part of this sentence
            if active_count < 3:
                # Keep passive abilities and triggered abilities from this sentence
                words = sentence.split()
                filtered_words = []
                temp_active_count = active_count
                
                for word in words:
                    if re.search(active_ability_pattern, word):
                        if temp_active_count < 3:
                            filtered_words.append(word)
                            temp_active_count += 1
                        # Skip remaining active abilities in this sentence
                    else:
                        filtered_words.append(word)
                
                if filtered_words:
                    filtered_sentence = ' '.join(filtered_words)
                    if filtered_sentence.strip():
                        filtered_sentences.append(filtered_sentence)
                        active_count = temp_active_count
            break
        else:
            filtered_sentences.append(sentence)
            active_count += len(sentence_active_abilities)
    
    # Rebuild the card text
    result = '. '.join(filtered_sentences)
    if result and not result.endswith('.'):
        result += '.'
    
    print(f"Limited creature abilities from {len(active_abilities)} to {active_count} active abilities")
    return result


def remove_typeline_contamination(abilities_list, card_data, ability_type="abilities"):
    """
    Remove any abilities that exactly match the card's typeline
    """
    # Build the complete typeline
    typeline_parts = []
    
    # Add supertype if present
    supertype = (card_data.get('supertype') or '').strip()
    if supertype:
        typeline_parts.append(supertype)
    
    # Add main type
    main_type = (card_data.get('type') or '').strip()
    if main_type:
        typeline_parts.append(main_type)
    
    # Add subtype if present
    subtype = (card_data.get('subtype') or '').strip()
    if subtype:
        typeline_parts.extend(['—', subtype])  # or use '-' depending on what's used
    
    # Create possible typeline variations
    possible_typelines = []
    if typeline_parts:
        # Version with em dash
        typeline_em = ' '.join(typeline_parts)
        possible_typelines.append(typeline_em)
        
        # Version with regular dash 
        typeline_dash = typeline_em.replace('—', '-')
        possible_typelines.append(typeline_dash)
    
    # Also add just the subtype alone (common contamination pattern)
    if subtype:
        possible_typelines.append(subtype)
        
    # Add main type alone too
    if main_type:
        possible_typelines.append(main_type)
    
    # Remove any abilities that exactly match a typeline variation
    cleaned_abilities = []
    contamination_found = False
    
    for ability in abilities_list:
        ability_clean = ability.strip().rstrip('.,!?')
        if not any(ability_clean.lower() == typeline.lower() for typeline in possible_typelines):
            cleaned_abilities.append(ability)
        else:
            print(f"🗑️  Removed typeline contamination from {ability_type}: '{ability}'")
            contamination_found = True
    
    return cleaned_abilities


def ensure_periods_on_abilities(card_text):
    """
    Ensure each ability/line ends with a period
    """
    if not card_text:
        return card_text
    
    # Split by newlines to handle each ability separately
    abilities = card_text.split('\n')
    fixed_abilities = []
    
    for ability in abilities:
        ability = ability.strip()
        if ability:  # Skip empty lines
            # Only add period if it doesn't already end with punctuation or quotes
            if not ability.endswith(('.', '!', '?', ':', '"', "'")):
                ability += '.'
        fixed_abilities.append(ability)
    
    return '\n'.join(fixed_abilities)


def sanitize_planeswalker_abilities(card_text):
    """
    Sanitize planeswalker abilities to ensure proper loyalty format
    """
    
    # Remove any non-loyalty abilities from planeswalkers
    lines = card_text.split('\n')
    valid_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a proper loyalty ability (+X:, -X:, 0:) or starting loyalty
        loyalty_pattern = r'^[+\-]?\d+:|^Starting loyalty:'
        if re.match(loyalty_pattern, line, re.IGNORECASE) or 'loyalty' in line.lower():
            valid_lines.append(line)
        elif any(keyword in line.lower() for keyword in ['emblem', 'ultimate', 'target', 'draw', 'deal', 'destroy', 'create']):
            # Common planeswalker ability words - likely valid
            valid_lines.append(line)
    
    result = '\n'.join(valid_lines)
    return result if result else card_text


def sanitize_spell_abilities(card_text):
    """
    Sanitize instant/sorcery abilities to focus on spell effects
    """
    
    # Remove keywords that don't make sense on spells
    invalid_spell_keywords = ['flying', 'trample', 'vigilance', 'haste', 'deathtouch', 'lifelink', 'first strike', 'double strike']
    
    lines = card_text.split('\n')
    valid_lines = []
    
    for line in lines:
        line_lower = line.lower()
        # Skip lines that are just creature keywords
        if any(f"{keyword}," in line_lower or line_lower.strip() == keyword for keyword in invalid_spell_keywords):
            continue
        valid_lines.append(line)
    
    # Limit to 2-3 effects maximum for spells
    if len(valid_lines) > 3:
        valid_lines = valid_lines[:3]
    
    return '\n'.join(valid_lines)


def sanitize_land_abilities(card_text):
    """
    Sanitize land abilities to focus on mana production and utility
    """
    
    # Remove creature-only keywords from lands
    invalid_land_keywords = ['flying', 'trample', 'vigilance', 'haste', 'deathtouch', 'lifelink', 'first strike', 'menace']
    
    lines = card_text.split('\n')
    valid_lines = []
    
    for line in lines:
        line_lower = line.lower()
        # Skip creature keywords
        if any(keyword in line_lower for keyword in invalid_land_keywords):
                continue
        valid_lines.append(line)
    
    return '\n'.join(valid_lines)


def sanitize_permanent_abilities(card_text):
    """
    Sanitize artifact/enchantment abilities for balanced effects
    """
    
    # Artifacts and enchantments can have most abilities, but limit complexity
    lines = card_text.split('\n')
    
    # Limit to 4 abilities maximum
    if len(lines) > 4:
        lines = lines[:4]
    
    return '\n'.join(lines)


def apply_universal_complexity_limits(card_text, card_data):
    """
    Apply complexity limits based on mana cost and rarity for ALL card types
    """
    
    cmc = card_data.get('cmc', 0)
    rarity = card_data.get('rarity', 'common').lower()
    
    lines = [line.strip() for line in card_text.split('\n') if line.strip()]
    max_abilities = 6  # Global maximum
    
    # Adjust limits based on mana cost and rarity
    if cmc <= 1:
        max_abilities = 2 if rarity in ['rare', 'mythic'] else 1
    elif cmc <= 2:
        max_abilities = 3 if rarity in ['rare', 'mythic'] else 2
    elif cmc <= 3:
        max_abilities = 4 if rarity in ['mythic'] else 3
    elif cmc <= 5:
        max_abilities = 5 if rarity == 'mythic' else 4
    else:
        max_abilities = 6 if rarity == 'mythic' else 5
    
    # Count meaningful abilities (not just keywords on one line)
    ability_count = 0
    keyword_line_count = 0
    
    for line in lines:
        # Check if it's a keyword line (contains only keywords separated by commas)
        if ',' in line and not any(symbol in line for symbol in ['{', '}', ':', 'when', 'whenever', 'target', 'draw', 'deal']):
            keyword_line_count += 1
            ability_count += 1  # Keywords count as 1 ability regardless of how many
        else:
            ability_count += 1
    
    if ability_count > max_abilities:
        print(f"Complexity limit exceeded: {ability_count} > {max_abilities} (CMC:{cmc}, {rarity})")
        # Keep the most important abilities (keywords first, then others)
        limited_lines = lines[:max_abilities]
        result = '\n'.join(limited_lines)
        print(f"Reduced from {len(lines)} to {len(limited_lines)} ability lines")
        return result
    
    return card_text


def validate_rules_text(rules_text, card_data):
    """
    Validate that rules text doesn't contain type line elements
    Returns True if valid, False if contaminated
    """
    if not card_data or not rules_text:
        return True
        
    # Build type line components to check against
    type_components = []
    
    # Add supertype if present
    supertype = (card_data.get('supertype') or '').strip()
    if supertype:
        type_components.append(supertype)
    
    # Add main type
    main_type = (card_data.get('type') or '').strip()
    if main_type:
        type_components.append(main_type)
    
    # Add subtype if present
    subtype = (card_data.get('subtype') or '').strip()
    if subtype:
        type_components.append(subtype)
    
    # Create full type line variations to check (ONLY full type lines, not individual words)
    type_checks = []
    
    if type_components:
        # Only check for complete type line combinations that shouldn't appear in rules text
        if supertype and main_type and subtype:
            # Full 3-part type line: "Legendary Creature — Beast"
            type_checks.append(f"{supertype} {main_type} — {subtype}")
            type_checks.append(f"{supertype} {main_type} - {subtype}")
            # Also check for the supertype + main type combo: "Legendary Creature"
            type_checks.append(f"{supertype} {main_type}")
        elif main_type and subtype:
            # 2-part type line: "Creature — Beast"  
            type_checks.append(f"{main_type} — {subtype}")
            type_checks.append(f"{main_type} - {subtype}")
        
        # NOTE: We deliberately don't check individual words like "creature", "beast", "legendary"
        # since these are valid in rules text (e.g., "Target creature gains flying")
    
    # Check if any type line elements appear in rules text
    rules_text_lower = rules_text.lower()
    for type_check in type_checks:
        if type_check.lower() in rules_text_lower:
            print(f"⚠️  Rules text contaminated with type line element: '{type_check}'")
            return False
    
    # Check for incorrect mana symbol format (square brackets instead of curly braces)
    square_bracket_pattern = r'\[([WUBRGCTXYZ0-9]+)\]'
    square_bracket_matches = re.findall(square_bracket_pattern, rules_text, re.IGNORECASE)
    
    if square_bracket_matches:
        print(f"⚠️  Rules text has incorrect mana symbol format: {square_bracket_matches} (should use curly braces {{}} not square brackets [])")
        return False
    
    # Check for card name contamination (only at the beginning of rules text)
    card_name = (card_data.get('name') or '').strip()
    if card_name and len(card_name) > 2:  # Only check meaningful names
        rules_text_lower = rules_text.lower().strip()
        card_name_lower = card_name.lower()
        
        # Check if card name appears at the very beginning (likely a title)
        if rules_text_lower.startswith(card_name_lower):
            # Check what follows the card name at the start
            after_name = rules_text_lower[len(card_name_lower):]
            if after_name.startswith(('\n', ',', ':', ' -', ' —')):
                print(f"⚠️  Rules text starts with card name: '{card_name}' (likely title instead of rules)")
                return False
    
    # Check for X mana cost requirement
    mana_cost = card_data.get('manaCost', '')
    if mana_cost and ('X' in mana_cost.upper() or '{X}' in mana_cost.upper()):
        # Card has X in mana cost, must use X in rules text
        rules_text_upper = rules_text.upper()
        # Enhanced validation: X cards now MUST include explanations
        x_explanation_patterns = [
            r'WHERE\s+X\s+IS\s+THE\s+(AMOUNT\s+OF\s+MANA|NUMBER\s+OF)',  # "where X is the amount of mana" or "where X is the number of"
            r'ENTERS\s+(\w+\s+)*WITH\s+X\s+\+1/\+1\s+COUNTERS?',  # "enters with X +1/+1 counters"
            r'X\s+IS\s+(EQUAL\s+TO|THE\s+(AMOUNT|NUMBER))',  # "X is equal to" or "X is the amount/number"
            r'AMOUNT\s+OF\s+MANA\s+(PAID|SPENT).*CAST',  # "amount of mana paid/spent to cast"
            r'MANA\s+(PAID|SPENT).*FOR\s+X',  # "mana paid/spent for X"
            r'COMES\s+INTO\s+PLAY\s+WITH\s+X',  # "comes into play with X"
            r'X\s+COUNTERS?\s+(ON\s+IT|PLACED)',  # "X counters on it" or "X counters placed"
        ]
        
        x_basic_usage_patterns = [
            r'DEAL\s+X\s+DAMAGE',  # Deal X damage
            r'DRAW\s+X\s+CARDS?',  # Draw X cards
            r'CREATE\s+X\s+.*TOKENS?',  # Create X tokens
            r'X\s+TARGET\s+CREATURES?',  # X target creatures
            r'[+\-]X/[+\-]X',  # Power/toughness modifiers like +X/+X
            r'GAIN\s+X\s+LIFE',  # Gain X life
            r'ADD\s+X\s+MANA',  # Add X mana
            r'DESTROY\s+UP\s+TO\s+X',  # Destroy up to X
            r'RETURN\s+UP\s+TO\s+X',  # Return up to X
        ]
        
        # Check for basic X usage
        has_x_usage = any(re.search(pattern, rules_text_upper) for pattern in x_basic_usage_patterns)
        
        # Check for X explanation
        has_x_explanation = any(re.search(pattern, rules_text_upper) for pattern in x_explanation_patterns)
        
        if not has_x_usage:
            print(f"⚠️  Card has X in mana cost ({mana_cost}) but rules text doesn't use X meaningfully: '{rules_text}'")
            return False
        elif not has_x_explanation:
            print(f"⚠️  Card has X in mana cost ({mana_cost}) but rules text doesn't explain what X represents: '{rules_text}'")
            print(f"    Required: Must include explanation like 'where X is the amount of mana spent' or 'enters with X +1/+1 counters'")
            return False
        else:
            pass  # X validation passed
    
    # Check for duplicate tap symbols (invalid in Magic rules)
    duplicate_tap_patterns = [
        r'\{T\},\s*\{T\}',  # {T}, {T}
        r'Tap,\s*Tap',      # Tap, Tap (case sensitive)
        r'tap,\s*tap',      # tap, tap (lowercase)
    ]
    
    for pattern in duplicate_tap_patterns:
        if re.search(pattern, rules_text, re.IGNORECASE):
            match = re.search(pattern, rules_text, re.IGNORECASE)
            print(f"⚠️  Rules text has invalid duplicate tap symbols: '{match.group()}' (you can't tap twice)")
            return False
    
    # Check for unwanted card elements (mana costs, type lines, titles)
    unwanted_patterns = [
        r'\{\d+\}\{[WUBRG]\}',  # Mana costs like {3}{U}
        r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # Titles like "Lightning Bolt"
        r'(Instant|Sorcery|Creature|Artifact|Enchantment|Land|Planeswalker)\s*[-—]\s*',  # Type lines
        r'^\d+/\d+$',  # Power/toughness like "3/3"
        r'Mana Cost:|Type:|Power/Toughness:',  # Card formatting labels
    ]
    
    for pattern in unwanted_patterns:
        if re.search(pattern, rules_text, re.MULTILINE):
            match = re.search(pattern, rules_text, re.MULTILINE)
            print(f"⚠️  Rules text contains unwanted card elements: '{match.group()}' (should only contain abilities)")
            return False
    
    # Check for Aura enchantment requirements
    card_type = card_data.get('type', '').lower() if card_data else ''
    subtype = (card_data.get('subtype') or '').lower() if card_data else ''
    
    if 'enchantment' in card_type and 'aura' in subtype:
        # Aura enchantments must have "Enchant" as their first ability
        rules_text_upper = rules_text.upper()
        enchant_patterns = [
            r'^ENCHANT\s+CREATURE',  # Most common: "Enchant creature"
            r'^ENCHANT\s+ARTIFACT',  # Less common: "Enchant artifact" 
            r'^ENCHANT\s+LAND',      # Rare: "Enchant land"
            r'^ENCHANT\s+ENCHANTMENT', # Very rare: "Enchant enchantment"
            r'^ENCHANT\s+PERMANENT',   # Flexible: "Enchant permanent"
            r'^ENCHANT\s+PLAYER',      # Rare: "Enchant player"
        ]
        
        has_enchant_ability = any(re.search(pattern, rules_text_upper.strip()) for pattern in enchant_patterns)
        
        if not has_enchant_ability:
            print(f"⚠️  Aura enchantment missing 'Enchant' ability at start of rules text: '{rules_text}'")
            print(f"    Required: Auras must start with 'Enchant creature' (or other valid target)")
            return False
        
        # Also check that it has effects on the enchanted permanent (not always required, but common)
        enchanted_effect_patterns = [
            r'ENCHANTED\s+CREATURE',   # "Enchanted creature gets/has..."
            r'ENCHANTED\s+ARTIFACT',   # "Enchanted artifact..."
            r'ENCHANTED\s+LAND',       # "Enchanted land..."
            r'ENCHANTED\s+PERMANENT',  # "Enchanted permanent..."
        ]
        
        has_enchanted_effects = any(re.search(pattern, rules_text_upper) for pattern in enchanted_effect_patterns)
        
        if has_enchant_ability and not has_enchanted_effects:
            print(f"⚠️  Aura has 'Enchant' but no 'Enchanted [target]' effects: '{rules_text}'")
            print(f"    Recommended: Include effects like 'Enchanted creature gets +1/+1' or similar")
            # Don't fail validation for this, just warn, as some Auras might have other effects
        else:
            pass  # Aura validation passed
    
    # Check for overuse of "draw cards" effects (encourage variety)
    draw_patterns = [
        r'\bDRAW\s+(\d+|X|A|ONE|TWO|THREE|FOUR|FIVE)\s+(CARD|CARDS)\b',  # "Draw X cards", "Draw three cards", etc.
        r'\bDRAW\s+A\s+CARD\b',  # "Draw a card" - word boundaries to avoid "CREW" matches
    ]
    
    rules_text_upper = rules_text.upper()
    draw_matches = []
    for pattern in draw_patterns:
        matches = re.findall(pattern, rules_text_upper)
        draw_matches.extend(matches)
    
    if len(draw_matches) > 0:
        print(f"⚠️  Card draw effect detected in rules text: '{rules_text}'")
        # Don't fail validation, but add a note about variety
        # This is more of a stylistic preference than a rule violation
        # Check for multiple draw effects or large draw amounts (3+)
        large_draw_match = re.search(r'DRAW\s+(\d+)', rules_text_upper)
        large_draw = large_draw_match and int(large_draw_match.group(1)) >= 3
        
        if len(draw_matches) > 1 or large_draw:
            print(f"    💡 Consider more variety - card draw effects are very common. Try damage, tokens, buffs, removal, etc.")
        else:
            print(f"    ℹ️  Card draw noted - variety is good for Magic diversity")
    
    # Check for asterisk power/toughness definition requirements
    power = card_data.get('power')
    toughness = card_data.get('toughness') 
    has_asterisk_power = power and '*' in str(power)
    has_asterisk_toughness = toughness and '*' in str(toughness)
    
    if has_asterisk_power or has_asterisk_toughness:
        print(f"⭐ Validating asterisk P/T definitions for {power}/{toughness}")
        
        # Common asterisk definition patterns
        asterisk_patterns = [
            r"power.*equal.*to.*the.*number",
            r"toughness.*equal.*to.*the.*number", 
            r"power.*and.*toughness.*are.*each.*equal.*to.*the.*number",
            r"power.*equal.*to.*your.*life",
            r"toughness.*equal.*to.*your.*life"
        ]
        
        rules_text_lower = rules_text.lower()
        has_definition = any(re.search(pattern, rules_text_lower) for pattern in asterisk_patterns)
        
        if not has_definition:
            print(f"⚠️  Card has asterisk P/T ({power}/{toughness}) but no definition in rules text: '{rules_text}'")
            print(f"    Required: Must define what * equals (e.g., 'power equal to the number of cards in your hand')")
            return False
    
    return True


def process_card_description_text(card_data, generated_card_text=None):
    """
    Comprehensive text processing pipeline for card descriptions.
    Handles JSON parsing, text cleanup, formatting fixes, and rules text processing.
    
    Args:
        card_data: Dictionary containing card data (including original description)
        generated_card_text: Optional new generated text to parse and process
        
    Returns:
        Dictionary with updated card data containing processed description
    """
    from text_processing.text_sanitizer import clean_ability_text, fix_markdown_bullet_points
    import json
    
    updated_card_data = card_data.copy()
    print(f"🔍 Original card data keys: {list(card_data.keys())}")
    print(f"🔍 Original description: {repr(card_data.get('description', 'NO DESCRIPTION'))}")
    
    # Step 1: Parse and integrate generated text if provided
    if generated_card_text:
        try:
            # Try to parse structured card data
            parsed_text = json.loads(generated_card_text)
            if isinstance(parsed_text, dict):
                # Update with parsed structured data
                if 'description' in parsed_text:
                    updated_card_data['description'] = parsed_text['description']
                if 'name' in parsed_text and parsed_text['name']:
                    updated_card_data['name'] = parsed_text['name']
                if 'flavorText' in parsed_text:
                    updated_card_data['flavorText'] = parsed_text['flavorText']
                print(f"Updated card data with parsed structured content")
            else:
                # If it's a JSON string, use the string content
                updated_card_data['description'] = str(parsed_text)
                print(f"Updated card data with JSON string content")
        except json.JSONDecodeError:
            # If not JSON, treat as plain description text
            updated_card_data['description'] = generated_card_text
            print(f"Updated card data with plain text content")
    
    # Step 2: Apply comprehensive text processing to the description
    if 'description' in updated_card_data and updated_card_data['description']:
        original_text = updated_card_data['description']
        
        # Apply the text processing pipeline
        processed_text = original_text
        print(f"🔍 Step 0 - Original: {repr(processed_text)}")
        
        # Step 2.1: Clean up text formatting
        processed_text = processed_text.replace('\n\n', '\n')  # Double newlines to single
        processed_text = processed_text.replace(' ~ ', f' {updated_card_data.get("name", "~")} ')  # Replace ~ with card name
        processed_text = processed_text.replace('~', updated_card_data.get("name", "~"))  # Replace any remaining ~
        print(f"🔍 Step 1 - After cleanup: {repr(processed_text)}")
        
        # Step 2.1.5: Convert to legal MTG text format using context model
        processed_text = convert_to_legal_mtg_text(processed_text, updated_card_data)
        print(f"🔍 Step 1.75 - After legal MTG conversion: {repr(processed_text)}")
        
        # Step 2.2: Clean individual ability lines of quotes and special characters
        lines = processed_text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():
                cleaned_line = clean_ability_text(line)
                if cleaned_line:  # Only add non-empty lines
                    cleaned_lines.append(cleaned_line)
        processed_text = '\n'.join(cleaned_lines)
        print(f"🔍 Step 1.25 - After quote cleanup: {repr(processed_text)}")
        
        # Step 2.3: Fix markdown bullet points (convert "* item" to "item")
        processed_text = fix_markdown_bullet_points(processed_text)
        print(f"🔍 Step 1.5 - After bullet fix: {repr(processed_text)}")
        
        # Step 2.4: Remove typeline contamination (card names, types appearing in rules text)
        lines = processed_text.split('\n')
        cleaned_lines = remove_typeline_contamination(lines, updated_card_data, "rules text")
        processed_text = '\n'.join(cleaned_lines)
        print(f"🔍 Step 2.5 - After typeline cleanup: {repr(processed_text)}")
        
        # Step 2.5: Ensure periods on abilities
        processed_text = ensure_periods_on_abilities(processed_text)
        print(f"🔍 Step 3 - After period fix: {repr(processed_text)}")
        
        updated_card_data['description'] = processed_text
        print(f"🔧 Content model parsed output: {repr(processed_text)}")
    
    return updated_card_data