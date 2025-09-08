"""
Power and toughness calculation utilities for creature cards.

This module handles creature stats generation, asterisk power/toughness patterns,
and validation of creature abilities related to power and toughness.
"""

import re
import random


def generate_creature_stats(card_data: dict) -> dict:
    """
    Generate balanced power/toughness stats for creature cards based on CMC, abilities, and rarity.
    """
    cmc = card_data.get('cmc', 1)
    rarity = card_data.get('rarity', 'common').lower()
    abilities_text = card_data.get('description', '')
    
    # Base stat total based on converted mana cost
    if cmc <= 1:
        base_total = 2  # 1-cost creatures like 1/1, 2/0, 0/2
    elif cmc == 2:
        base_total = 3  # 2-cost creatures like 2/1, 1/2
    elif cmc == 3:
        base_total = 5  # 3-cost creatures like 2/3, 3/2
    elif cmc == 4:
        base_total = 6  # 4-cost creatures like 3/3, 4/2
    elif cmc == 5:
        base_total = 7  # 5-cost creatures like 4/3, 3/4
    elif cmc == 6:
        base_total = 8  # 6-cost creatures like 4/4, 5/3
    else:
        base_total = min(cmc + 2, 12)  # Higher cost creatures, cap at 12
    
    # Adjust for abilities complexity (more abilities = lower stats)
    ability_count = len([line for line in abilities_text.split('\n') if line.strip()])
    if ability_count >= 3:
        base_total -= 1  # Complex creatures get -1 total stats
    elif ability_count >= 5:
        base_total -= 2  # Very complex creatures get -2 total stats
    
    # Adjust for rarity (higher rarity can be slightly more efficient)
    if rarity == 'rare':
        base_total += 1
    elif rarity == 'mythic':
        base_total += 2
    
    # Ensure minimum viable stats
    base_total = max(base_total, 1)
    
    # Distribute stats between power and toughness
    # Favor slightly more toughness for defensive play
    if base_total == 1:
        distributions = [(0, 1), (1, 0)]
    elif base_total == 2:
        distributions = [(1, 1), (2, 0), (0, 2)]
    elif base_total == 3:
        distributions = [(1, 2), (2, 1), (0, 3), (3, 0)]
    elif base_total == 4:
        distributions = [(2, 2), (1, 3), (3, 1), (0, 4)]
    elif base_total == 5:
        distributions = [(2, 3), (3, 2), (1, 4), (4, 1)]
    elif base_total == 6:
        distributions = [(3, 3), (2, 4), (4, 2), (1, 5)]
    elif base_total == 7:
        distributions = [(3, 4), (4, 3), (2, 5), (5, 2)]
    elif base_total == 8:
        distributions = [(4, 4), (3, 5), (5, 3), (2, 6)]
    else:
        # For higher totals, create reasonable distributions
        power = base_total // 2
        toughness = base_total - power
        distributions = [(power, toughness), (toughness, power)]
        if power > 1:
            distributions.extend([(power - 1, toughness + 1), (power + 1, toughness - 1)])
    
    # Choose a random distribution
    power, toughness = random.choice(distributions)
    
    return {
        'power': str(power),
        'toughness': str(toughness)
    }


def should_generate_asterisk_pt(card_data: dict) -> dict:
    """
    Determine if this creature should have asterisk (*) power/toughness and what pattern to use.
    """
    cmc = card_data.get('cmc', 1)
    rarity = card_data.get('rarity', 'common').lower()
    colors = card_data.get('colors', [])
    
    # Asterisk creatures are more interesting at higher CMC and rarity
    asterisk_chance = 0
    
    if cmc >= 3 and rarity in ['rare', 'mythic']:
        asterisk_chance = 0.15  # 15% chance for high-cost rare/mythic
    elif cmc >= 4:
        asterisk_chance = 0.08  # 8% chance for high-cost cards
    elif rarity in ['rare', 'mythic']:
        asterisk_chance = 0.05  # 5% chance for rare/mythic
    
    should_asterisk = random.random() < asterisk_chance
    
    if should_asterisk:
        return choose_asterisk_pattern(card_data)
    else:
        return {'has_asterisk': False}


def choose_asterisk_pattern(card_data: dict) -> dict:
    """
    Choose an appropriate asterisk pattern based on card colors and properties.
    """
    colors = card_data.get('colors', [])
    cmc = card_data.get('cmc', 1)
    
    # Color-based asterisk patterns
    patterns = []
    
    if 'G' in colors:
        patterns.extend([
            {'type': 'lands', 'stat': 'both', 'pattern': 'power and toughness'},
            {'type': 'creatures', 'stat': 'power', 'pattern': 'power'},
            {'type': 'forests', 'stat': 'toughness', 'pattern': 'toughness'}
        ])
    
    if 'U' in colors:
        patterns.extend([
            {'type': 'cards_in_hand', 'stat': 'both', 'pattern': 'power and toughness'},
            {'type': 'artifacts', 'stat': 'power', 'pattern': 'power'},
            {'type': 'instants_sorceries', 'stat': 'toughness', 'pattern': 'toughness'}
        ])
    
    if 'B' in colors:
        patterns.extend([
            {'type': 'graveyard', 'stat': 'both', 'pattern': 'power and toughness'},
            {'type': 'creature_cards_graveyard', 'stat': 'power', 'pattern': 'power'},
            {'type': 'swamps', 'stat': 'toughness', 'pattern': 'toughness'}
        ])
    
    if 'R' in colors:
        patterns.extend([
            {'type': 'mountains', 'stat': 'both', 'pattern': 'power and toughness'},
            {'type': 'instant_sorcery_graveyard', 'stat': 'power', 'pattern': 'power'}
        ])
    
    if 'W' in colors:
        patterns.extend([
            {'type': 'creatures', 'stat': 'both', 'pattern': 'power and toughness'},
            {'type': 'plains', 'stat': 'toughness', 'pattern': 'toughness'},
            {'type': 'enchantments', 'stat': 'power', 'pattern': 'power'}
        ])
    
    # Fallback patterns if no colors or no specific patterns
    if not patterns:
        patterns = [
            {'type': 'cards_in_hand', 'stat': 'both', 'pattern': 'power and toughness'},
            {'type': 'lands', 'stat': 'power', 'pattern': 'power'}
        ]
    
    # Choose random pattern
    chosen_pattern = random.choice(patterns)
    chosen_pattern['has_asterisk'] = True
    
    return chosen_pattern


def generate_asterisk_stats(card_data: dict, pattern: dict) -> dict:
    """
    Generate asterisk-based power/toughness stats with proper ability text requirements.
    """
    stat_type = pattern.get('stat', 'both')
    pattern_type = pattern.get('type', 'cards_in_hand')
    
    # Generate the stats
    if stat_type == 'both':
        power = '*'
        toughness = '*'
    elif stat_type == 'power':
        power = '*'
        # Generate normal toughness
        toughness = str(max(1, card_data.get('cmc', 1)))
    elif stat_type == 'toughness':
        toughness = '*'
        # Generate normal power
        power = str(max(1, card_data.get('cmc', 1) - 1))
    else:
        power = '*'
        toughness = '*'
    
    # Define ability text templates for asterisk definitions
    definitions = {
        'cards_in_hand': "This creature's {stat} is equal to the number of cards in your hand.",
        'lands': "This creature's {stat} is equal to the number of lands you control.",
        'creatures': "This creature's {stat} is equal to the number of creatures you control.",
        'graveyard': "This creature's {stat} is equal to the number of cards in your graveyard.",
        'creature_cards_graveyard': "This creature's {stat} is equal to the number of creature cards in your graveyard.",
        'forests': "This creature's {stat} is equal to the number of Forests you control.",
        'swamps': "This creature's {stat} is equal to the number of Swamps you control.",
        'mountains': "This creature's {stat} is equal to the number of Mountains you control.",
        'plains': "This creature's {stat} is equal to the number of Plains you control.",
        'islands': "This creature's {stat} is equal to the number of Islands you control.",
        'artifacts': "This creature's {stat} is equal to the number of artifacts you control.",
        'enchantments': "This creature's {stat} is equal to the number of enchantments you control.",
        'instant_sorcery_graveyard': "This creature's {stat} is equal to the number of instant and sorcery cards in your graveyard.",
        'instants_sorceries': "This creature's {stat} is equal to the number of instant and sorcery cards in your graveyard."
    }
    
    definition_template = definitions.get(pattern_type, "This creature's {stat} is equal to the number of cards in your hand.")
    
    return {
        'power': power,
        'toughness': toughness,
        'asterisk_definition': definition_template.format(stat=pattern.get('pattern', 'power and toughness'))
    }


def validate_asterisk_abilities(rules_text, card_data):
    """
    Validate that asterisk power/toughness creatures have proper defining abilities in their rules text.
    """
    if not card_data:
        return True
    
    power = card_data.get('power')
    toughness = card_data.get('toughness')
    
    # Check if this creature has asterisk stats
    has_asterisk_power = power and '*' in str(power)
    has_asterisk_toughness = toughness and '*' in str(toughness)
    
    if not (has_asterisk_power or has_asterisk_toughness):
        return True  # No asterisk stats, no validation needed
    
    if not rules_text:
        return False  # Has asterisk but no rules text to define it
    
    # Patterns that indicate asterisk definition
    asterisk_patterns = [
        r"power\s+(?:is\s+)?equal\s+to",
        r"toughness\s+(?:is\s+)?equal\s+to", 
        r"power\s+and\s+toughness\s+are\s+(?:each\s+)?equal\s+to",
        r"gets\s+[+-]\d+/[+-]\d+",
        r"base\s+power\s+(?:is\s+)?equal\s+to",
        r"base\s+toughness\s+(?:is\s+)?equal\s+to"
    ]
    
    # Check if rules text contains asterisk definition
    has_definition = any(re.search(pattern, rules_text, re.IGNORECASE) for pattern in asterisk_patterns)
    
    if not has_definition:
        return False
    
    return True