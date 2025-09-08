"""
Image prompt generation utilities for Magic: The Gathering card art.

This module handles converting card data (colors, types, etc.) into optimized
image generation prompts for AI models like SDXL-Turbo.
"""

import re


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of CLIP tokens - CLIP tokenizer splits on spaces and punctuation.
    This is a conservative estimate to stay under the 77 token limit.
    """
    # Split on spaces, punctuation, and common word boundaries
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    # Add padding for safety since CLIP tokenization can be complex
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
    
    # Split prompt into components
    parts = prompt.split(', ')
    
    subject_parts = []  # Unique subject matter (HIGHEST priority)
    color_parts = []    # Color palette (HIGH priority for visual consistency)
    style_parts = []    # Art style (MEDIUM priority) 
    magic_parts = []    # Generic Magic context (LOWER priority)  
    lighting_parts = []
    
    for part in parts:
        part_lower = part.lower()
        if 'color palette' in part_lower:
            color_parts.append(part)
        elif any(keyword in part_lower for keyword in ['magic: the gathering', 'card art']):
            magic_parts.append(part)  # Deprioritize generic Magic terms
        elif any(keyword in part_lower for keyword in ['fantasy art', 'style', 'detailed', 'illustration', 'artwork']):
            style_parts.append(part)
        elif any(keyword in part_lower for keyword in ['lighting', 'contrast', 'dramatic']):
            lighting_parts.append(part)
        else:
            subject_parts.append(part)  # The unique subject matter gets highest priority
    
    # Rebuild prompt with subject-first priority order
    final_parts = subject_parts  # Start with the unique subject matter
    
    # Add color palette if space allows (high priority for visual consistency)
    test_prompt = ', '.join(final_parts)
    if color_parts:
        for color_part in color_parts:
            if estimate_tokens(test_prompt + ', ' + color_part) <= max_tokens:
                final_parts.append(color_part)
                test_prompt = ', '.join(final_parts)
                break
    
    # Add style if space allows
    if style_parts:
        for style_part in style_parts:
            if estimate_tokens(test_prompt + ', ' + style_part) <= max_tokens:
                final_parts.append(style_part)
                test_prompt = ', '.join(final_parts)
                break
    
    # Add Magic context if space allows (lower priority)
    if magic_parts:
        for magic_part in magic_parts:
            if estimate_tokens(test_prompt + ', ' + magic_part) <= max_tokens:
                final_parts.append(magic_part)
                test_prompt = ', '.join(final_parts)
                break
    
    # Add lighting if space allows (lowest priority)
    if lighting_parts:
        for lighting_part in lighting_parts:
            if estimate_tokens(test_prompt + ', ' + lighting_part) <= max_tokens:
                final_parts.append(lighting_part)
                test_prompt = ', '.join(final_parts)
                break
    
    result = ', '.join(final_parts)
    print(f"✅ Truncated to {estimate_tokens(result)} tokens: {result}")
    return result


def generate_color_palette(colors):
    """
    Convert MTG color combinations to art-appropriate color palette descriptions.
    
    Args:
        colors: List of MTG color letters ['W', 'U', 'B', 'R', 'G']
    
    Returns:
        String describing color palette for image generation
    """
    if not colors:
        return ", color palette: metallic silver, steel gray"
    
    # Handle multicolor combinations first
    if len(colors) >= 5:
        # Five colors - WUBRG (all colors)
        return ", color palette: rainbow prismatic, all five mana colors"
    
    elif len(colors) == 4:
        # Four color combinations - simplified
        return ", color palette: four-color convergence, rich jewel tones"
    
    elif len(colors) == 3:
        # Three color combinations (Shards and Wedges) - simplified
        colors_set = set(colors)
        if colors_set == {'W', 'U', 'G'}:  # Bant
            return ", color palette: white marble, blue sapphire, green emerald"
        elif colors_set == {'U', 'B', 'R'}:  # Grixis
            return ", color palette: dark blues, void black, burning red"
        elif colors_set == {'B', 'R', 'G'}:  # Jund
            return ", color palette: shadow black, flame red, wild green"
        elif colors_set == {'R', 'G', 'W'}:  # Naya
            return ", color palette: burning red, emerald green, pure white"
        elif colors_set == {'G', 'W', 'U'}:  # Same as Bant, reordered
            return ", color palette: emerald green, pure white, sapphire blue"
        elif colors_set == {'W', 'B', 'G'}:  # Abzan
            return ", color palette: ivory white, deep black, forest green"
        elif colors_set == {'U', 'R', 'W'}:  # Jeskai
            return ", color palette: sapphire blue, flame red, pure white"
        elif colors_set == {'B', 'G', 'U'}:  # Sultai
            return ", color palette: shadow black, wild green, deep blue"
        elif colors_set == {'R', 'W', 'B'}:  # Mardu
            return ", color palette: burning red, bone white, void black"
        elif colors_set == {'G', 'U', 'R'}:  # Temur
            return ", color palette: emerald green, ocean blue, molten red"
        else:
            return ", color palette: three-color blend, rich jewel tones"
    
    elif len(colors) == 2:
        # Two color guild combinations - simplified
        if 'W' in colors and 'U' in colors:
            return ", color palette: pristine white, sapphire blue"
        elif 'W' in colors and 'B' in colors:
            return ", color palette: pure white, deep black"
        elif 'W' in colors and 'R' in colors:
            return ", color palette: ivory white, burning red"
        elif 'W' in colors and 'G' in colors:
            return ", color palette: marble white, forest green"
        elif 'U' in colors and 'B' in colors:
            return ", color palette: midnight blue, void black"
        elif 'U' in colors and 'R' in colors:
            return ", color palette: electric blue, molten red"
        elif 'U' in colors and 'G' in colors:
            return ", color palette: ocean blue, living green"
        elif 'B' in colors and 'R' in colors:
            return ", color palette: shadow black, blood red"
        elif 'B' in colors and 'G' in colors:
            return ", color palette: decay black, wild green"
        elif 'R' in colors and 'G' in colors:
            return ", color palette: flame red, primal green"
    
    # Single color palettes - simplified
    elif len(colors) == 1:
        if 'W' in colors:
            return ", color palette: pure white, warm gold"
        elif 'U' in colors:
            return ", color palette: sapphire blue, silver"
        elif 'B' in colors:
            return ", color palette: void black, dark purple"
        elif 'R' in colors:
            return ", color palette: burning red, molten orange"
        elif 'G' in colors:
            return ", color palette: forest green, earth brown"
    
    # Colorless - simplified
    elif 'C' in colors or not colors:
        return ", color palette: metallic silver, steel gray"
    
    return ", color palette: metallic silver, steel gray"


def generate_art_type_context(card_type, colors):
    """
    Generate type-specific art prompt context based on card type and colors.
    
    Args:
        card_type: String card type (e.g., 'Creature', 'Instant', etc.)
        colors: List of MTG color letters
        
    Returns:
        String describing art context for the card type
    """
    card_type_lower = card_type.lower()
    
    if 'creature' in card_type_lower:
        # Creatures should show the actual creature/being
        # Special handling for blue creatures to diversify away from wizards/mages
        if colors and 'U' in colors and len(colors) == 1:  # Pure blue creatures
            return ", detailed creature portrait, living being, aquatic creature, flying creature, sea monster, elemental being, sphinx, merfolk, bird, octopus, dragon, character focus"
        else:
            return ", detailed creature portrait, living being, character focus"
    
    elif 'instant' in card_type_lower:
        # Instants should show magical effects in action
        return ", magical spell effect in action, dynamic energy, casting magic"
    
    elif 'sorcery' in card_type_lower:
        # Sorceries should show larger magical scenes or rituals
        return ", grand magical ritual, powerful sorcery scene, mystical ceremony"
    
    elif 'artifact' in card_type_lower:
        # Artifacts should show detailed mechanical/magical objects
        return ", detailed magical artifact, intricate device, mystical technology"
    
    elif 'enchantment' in card_type_lower:
        # Enchantments should show mystical auras or magical environments
        return ", mystical enchantment aura, magical atmosphere, ethereal energy"
    
    elif 'land' in card_type_lower:
        # Lands should show landscapes appropriate to their colors
        return ", detailed landscape, magical terrain, fantasy environment"
    
    elif 'planeswalker' in card_type_lower:
        # Planeswalkers should show powerful magical beings
        return ", powerful planeswalker character, magical being, mystical portrait"
    
    else:
        # Generic fallback
        return ", magical fantasy scene"


def build_enhanced_art_prompt(base_prompt, card_data):
    """
    Build complete enhanced art prompt from card data.
    
    Args:
        base_prompt: Base user prompt
        card_data: Dictionary containing card information (colors, type, etc.)
        
    Returns:
        Complete enhanced prompt optimized for image generation
    """
    # Extract card data
    colors = card_data.get('colors', []) if card_data else []
    card_type = card_data.get('type', '') if card_data else ''
    
    print(f"Debug - Colors received: {colors}")
    
    # Generate color palette and art context
    color_palette = generate_color_palette(colors)
    art_type_context = generate_art_type_context(card_type, colors)
    
    # Create enhanced art prompt with subject FIRST for better CLIP attention
    # Format: Subject first, then type context, then style, then Magic context, then color palette
    art_prompt = f"{base_prompt}{art_type_context}{color_palette}, fantasy art, Magic: The Gathering style, detailed illustration, dramatic lighting"
    
    # Apply smart truncation to stay within CLIP's 77 token limit
    final_prompt = truncate_prompt_smartly(art_prompt, max_tokens=75)
    print(f"Debug - Final art prompt ({estimate_tokens(final_prompt)} tokens): {final_prompt}")
    
    return final_prompt