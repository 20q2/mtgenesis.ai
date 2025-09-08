"""
Text processing modules for MTGenesis AI card generation.

This package contains modules for parsing, cleaning, and processing Magic: The Gathering
card rules text, including ability classification, text sanitization, and power calculation.
"""

# Import main functions for easy access
from .ability_parser import (
    parse_abilities,
    classify_ability,
    reorder_abilities_properly
)

from .text_sanitizer import (
    strip_non_rules_text,
    fix_markdown_bullet_points,
    clean_ability_text,
    clean_ability_arrays,
    format_ability_newlines,
    smart_split_by_periods,
    clean_ability_quotes
)

from .power_calculator import (
    generate_creature_stats,
    should_generate_asterisk_pt,
    choose_asterisk_pattern,
    generate_asterisk_stats,
    validate_asterisk_abilities
)

__all__ = [
    # Ability parsing
    'parse_abilities',
    'classify_ability', 
    'reorder_abilities_properly',
    
    # Text sanitization
    'strip_non_rules_text',
    'fix_markdown_bullet_points',
    'clean_ability_text',
    'clean_ability_arrays',
    'format_ability_newlines',
    'smart_split_by_periods',
    'clean_ability_quotes',
    
    # Power calculation
    'generate_creature_stats',
    'should_generate_asterisk_pt',
    'choose_asterisk_pattern', 
    'generate_asterisk_stats',
    'validate_asterisk_abilities'
]