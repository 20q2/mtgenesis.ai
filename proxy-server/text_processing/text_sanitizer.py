"""
Text sanitization and formatting utilities for Magic: The Gathering cards.

This module handles cleaning, formatting, and sanitizing card text including
removing unwanted characters, fixing formatting issues, and processing abilities.
"""

import re


def clean_ability_quotes(ability: str) -> str:
    """
    Remove outer double quotes from abilities while preserving intentional quotes within the text.
    
    Examples:
    - '"Flying"' -> 'Flying'  
    - '"{T}: Add one mana"' -> '{T}: Add one mana'
    - '"Whenever you cast a spell, say "Hello""' -> 'Whenever you cast a spell, say "Hello"'
    """
    ability = ability.strip()
    
    # Check if ability starts and ends with double quotes
    if len(ability) >= 2 and ability.startswith('"') and ability.endswith('"'):
        # Remove outer quotes
        inner_text = ability[1:-1]
        
        # Check if there are intentional quotes inside (odd number of quotes means unmatched quote)
        inner_quote_count = inner_text.count('"')
        
        if inner_quote_count == 0:
            # No internal quotes, just return cleaned text
            return inner_text
        elif inner_quote_count % 2 == 0:
            # Even number of internal quotes (properly paired), return cleaned text
            return inner_text
        else:
            # Odd number of internal quotes, the outer quote was probably intentional closing quote
            # Keep the closing quote
            return inner_text + '"'
    
    return ability


def smart_split_by_periods(text):
    """
    Split text by periods, but respect quoted sections.
    Periods inside quotes should not cause splits.
    Also split on trigger words that clearly start new abilities.
    """
    if not text:
        return []
    
    # First, handle trigger word splits within quoted sections
    # Look for patterns like ". Whenever" or ". When" or ". At the beginning"
    
    # Pattern to find period + space + trigger words that should start new abilities
    trigger_split_pattern = r'(\.\s+)(When(?:ever)?|At\s+the\s+beginning|At\s+end\s+of|During|While)'
    
    # Replace with period + SPLIT_MARKER + trigger word
    SPLIT_MARKER = "||ABILITY_SPLIT||"
    text_with_markers = re.sub(trigger_split_pattern, r'\1' + SPLIT_MARKER + r'\2', text, flags=re.IGNORECASE)
    
    parts = []
    current_part = ""
    in_quotes = False
    quote_char = None
    
    i = 0
    while i < len(text_with_markers):
        # Check for split marker
        if text_with_markers[i:i+len(SPLIT_MARKER)] == SPLIT_MARKER:
            # This is a split point - finish current part
            if current_part.strip():
                parts.append(current_part.strip())
            current_part = ""
            i += len(SPLIT_MARKER)
            continue
            
        char = text_with_markers[i]
        
        # Track quote state
        if char in ['"', "'"] and (i == 0 or text_with_markers[i-1] != '\\'):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        
        # Only split on periods if not inside quotes
        if char == '.' and not in_quotes:
            # Check if this period should cause a split
            # Don't split on periods that are part of numbers or abbreviations
            next_char = text_with_markers[i+1] if i+1 < len(text_with_markers) else ''
            
            # Add the period to current part and potentially split
            current_part += char
            
            # Split if next char is whitespace (normal sentence ending)
            if next_char.isspace():
                parts.append(current_part.strip())
                current_part = ""
                i += 1
                # Skip the whitespace
                while i < len(text_with_markers) and text_with_markers[i].isspace():
                    i += 1
                continue
        
        current_part += char
        i += 1
    
    # Add the last part if it exists
    if current_part.strip():
        parts.append(current_part.strip())
    
    return [part for part in parts if part.strip()]


def clean_ability_text(ability):
    """Remove unwanted characters like bullets, stars, quotes, etc. from ability text"""
    if not ability:
        return ability
        
    # Remove leading/trailing whitespace
    ability = ability.strip()
    
    # Remove leading bullets, stars, quotes, or dashes
    ability = re.sub(r'^[*•\-"\'\s]+', '', ability)
    
    # Remove trailing bullets, stars, quotes, or dashes  
    ability = re.sub(r'[*•\-"\'\s]+$', '', ability)
    
    # Clean up extra whitespace
    ability = ' '.join(ability.split())
    
    return ability


def clean_ability_arrays(abilities_dict):
    """Clean unwanted formatting from ability arrays"""
    cleaned = {}
    
    for ability_type, abilities_list in abilities_dict.items():
        cleaned_list = []
        for ability in abilities_list:
            if ability:  # Skip empty abilities
                cleaned_ability = clean_ability_text(ability)                
                if cleaned_ability:
                    cleaned_list.append(cleaned_ability)
        cleaned[ability_type] = cleaned_list
    
    return cleaned


def fix_markdown_bullet_points(card_text):
    """
    Convert markdown bullet points (* item) to proper Magic formatting.
    Removes stars and bullets that shouldn't be in MTG rules text.
    """
    if not card_text:
        return card_text
    
    # Pattern to match lines that start with * followed by space and text
    bullet_pattern = r'^\*\s+(.+)$'
    
    lines = card_text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check if line starts with markdown bullet
        match = re.match(bullet_pattern, line.strip())
        if match:
            # Extract the content after the *
            content = match.group(1)
            fixed_lines.append(content)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def format_ability_newlines(card_text):
    """
    Add newlines before ability costs that come after periods (not in quotes).
    This helps separate abilities that were concatenated together.
    """
    if not card_text:
        return card_text
    
    # Pattern to find periods followed by ability costs
    # Look for: period + optional space + cost pattern (like {T}:, {1}:, etc.)
    cost_pattern = r'(\.)(\s*)(\{[^}]*\}:)'
    
    def replacement(match):
        period = match.group(1)
        whitespace = match.group(2)
        cost = match.group(3)
        # Add newline after period, before the cost
        return f"{period}\n{cost}"
    
    # Apply the replacement
    result = re.sub(cost_pattern, replacement, card_text)
    
    # Clean up any double newlines that might have been created
    result = re.sub(r'\n\s*\n', '\n', result)
    
    return result


def strip_non_rules_text(rules_text, card_data):
    """
    Strip everything that isn't actual rules text from the generated content.
    """
    if not rules_text:
        return ""
    
    # Get card info for contamination detection
    card_name = card_data.get('name', '') if card_data else ''
    card_type = card_data.get('type', '') if card_data else ''
    supertype = card_data.get('supertype', '') if card_data else ''
    subtype = card_data.get('subtype', '') if card_data else ''
    mana_cost = card_data.get('manaCost', '') if card_data else ''
    
    # Build full type line variations for detection
    type_line_parts = []
    if supertype:
        type_line_parts.append(supertype)
    if card_type:
        type_line_parts.append(card_type)
    if subtype:
        type_line_parts.extend(['—', subtype])  # Em dash
    
    type_line_variations = []
    if type_line_parts:
        type_line_em = ' '.join(type_line_parts)
        type_line_variations.append(type_line_em)
        # Also try with regular dash
        type_line_dash = type_line_em.replace('—', '-')
        type_line_variations.append(type_line_dash)
    
    # Clean the text line by line
    lines = re.split(r'\.|\n', rules_text)
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove card name if it appears alone
        if line == card_name:
            print(f"🗑️  Stripped card name: {line}")
            continue
            
        # Remove type line variations
        type_line_found = False
        for type_line in type_line_variations:
            if line.lower() == type_line.lower():
                print(f"🗑️  Stripped type line: {line}")
                type_line_found = True
                break
        
        if type_line_found:
            continue
            
        # Remove mana cost if it appears alone
        if mana_cost and line == mana_cost:
            print(f"🗑️  Stripped mana cost: {line}")
            continue
            
        # Remove lines that are just punctuation or formatting
        if re.match(r'^["\'\*\-\s]*$', line):
            print(f"🗑️  Stripped formatting line: '{line}'")
            continue
            
        # Keep the line - it appears to be rules text
        clean_lines.append(line)
    
    result = '\n'.join(clean_lines).strip()
    return result