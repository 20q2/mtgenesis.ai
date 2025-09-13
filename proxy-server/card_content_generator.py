"""
Card content generation utilities for Magic: The Gathering cards.

This module handles the core text generation logic for MTG card abilities and rules text
using LLM models via Ollama. It includes comprehensive guidance for different card types,
colors, rarities, and special mechanics.
"""

import re
import ollama
import time
from text_processing import (
    strip_non_rules_text, format_ability_newlines, reorder_abilities_properly, reorder_abilities_properly_array,
    smart_split_by_periods
)
from rules_text_processor import (
    validate_rules_text, limit_creature_active_abilities, sanitize_planeswalker_abilities,
    sanitize_spell_abilities, sanitize_land_abilities, sanitize_permanent_abilities,
    apply_universal_complexity_limits, ensure_periods_on_abilities
)
from config import (
    FLYING_RESTRICTED_TYPES as flying_restricted_types,
    FLYING_ENCOURAGED_TYPES as flying_encouraged_types
)


def build_prompt(card_data: dict) -> str:
    """
    Build a modular prompt for card content generation.
    Only includes relevant sections based on card properties.
    """
    card_type = card_data.get('type', '').lower()
    colors = card_data.get('colors', [])
    rarity = card_data.get('rarity', 'common').lower()
    cmc = card_data.get('cmc', 0)
    subtype = card_data.get('subtype', '')
    card_name = card_data.get('name', '')
    
    # Build prompt sections
    sections = []
    
    # Core task
    sections.append(_get_core_task())
    
    # Card type guidance
    sections.append(_get_type_guidance(card_type, subtype, rarity))
    
    # Color guidance
    if colors:
        sections.append(_get_color_guidance(colors, card_type))
    
    # Rarity guidance
    sections.append(_get_rarity_guidance(rarity, card_type, cmc))
    
    # Name inspiration
    if card_name:
        sections.append(_get_name_inspiration(card_name))
    
    # Special restrictions (flying, etc.)
    if 'creature' in card_type and subtype:
        sections.append(_get_creature_restrictions(subtype))
    
    # Output format rules
    sections.append(_get_output_rules())
    
    # Few-shot examples
    sections.append(_get_examples(card_type))
    
    return ' '.join(sections)

def _get_core_task() -> str:
    """Core task description."""
    return ("Generate Magic: The Gathering rules text for a card. "
            "Focus on creating unique, cohesive abilities that work together thematically. "
            "CRITCAL: Ensure that the generated rules text only includes abilities and effects for this card, and does not include the card title, typeline, mana cost, power/toughness, or flavor text."
            "CRITAL: Do not start your output with anything like \"here are some potential abilities\". Simply output the rules text on their own.")

def _get_type_guidance(card_type: str, subtype: str, rarity: str) -> str:
    """Get type-specific guidance."""
    if 'creature' in card_type:
        return _get_creature_guidance(subtype, rarity)
    elif 'artifact' in card_type:
        return _get_artifact_guidance(subtype)
    elif 'enchantment' in card_type:
        return _get_enchantment_guidance(subtype)
    elif 'land' in card_type:
        return _get_land_guidance(subtype)
    elif 'planeswalker' in card_type:
        return _get_planeswalker_guidance(rarity)
    elif 'instant' in card_type or 'sorcery' in card_type:
        return "Generate spell effects that happen when cast. Keep effects concise and focused."
    else:
        return "Generate appropriate abilities for this card type."

def _get_creature_guidance(subtype: str, rarity: str) -> str:
    """Creature-specific guidance."""
    guidance = "CREATURE: Generate abilities that enhance combat or provide utility. "
    
    # Rarity limits
    if rarity == 'mythic':
        guidance += "Mythic: Maximum 3-4 abilities. "
    elif rarity == 'rare':
        guidance += "Rare: Maximum 3 abilities. "
    elif rarity == 'uncommon':
        guidance += "Uncommon: Maximum 2 abilities. "
    else:
        guidance += "Common: Maximum 1 ability. "
    
    # Subtype-specific advice with detailed flavorful guidance
    if subtype:
        subtype_lower = subtype.lower()
        
        # Flying creatures
        if 'dragon' in subtype_lower:
            guidance += "DRAGON: Flying (almost always), damage-dealing abilities, treasure creation, breath weapon effects. Examples: 'Flying' + 'When this attacks, deal 2 damage to any target', '{R}: Deal 1 damage to any target', 'Whenever this deals combat damage, create a Treasure token'. "
        elif 'angel' in subtype_lower:
            guidance += "ANGEL: Flying OR vigilance (not both), lifegain, protection effects, helping other creatures. Examples: 'Flying' + 'Whenever you gain life, put a +1/+1 counter on target creature', 'Other creatures you control have lifelink', 'When this enters, you gain 3 life'. "
        elif 'bird' in subtype_lower:
            guidance += "BIRD: Flying (always), evasion, mobility, scouting effects. Examples: 'Flying', 'Flying' + 'When this enters, scry 1', 'Flying' + '{T}: Look at the top card of your library', 'Whenever this deals combat damage to a player, draw a card'. "
        elif 'spirit' in subtype_lower:
            guidance += "SPIRIT: Flying, ethereal abilities, graveyard interactions, phasing/flickering. Examples: 'Flying', 'Flying' + 'When this enters, return target creature to its owner's hand', 'Flying' + 'Whenever a creature dies, you may exile this card and return it to the battlefield', '{1}: This creature phases out'. "
        elif 'phoenix' in subtype_lower:
            guidance += "PHOENIX: Flying, recursion from graveyard, fire/damage themes. Examples: 'Flying', 'Flying' + 'When this dies, return it to your hand at the beginning of the next end step', 'Flying' + 'Whenever you cast an instant or sorcery spell, this deals 1 damage to any target'. "
        
        # Aggressive creatures  
        elif 'goblin' in subtype_lower:
            guidance += "GOBLIN: Haste, direct damage, chaotic effects, artifact interactions, swarm tactics. Examples: 'Haste', 'Haste' + 'When this enters, deal 1 damage to any target', '{T}, Sacrifice an artifact: Deal 2 damage to any target', 'Other Goblins you control get +1/+1'. "
        elif 'orc' in subtype_lower:
            guidance += "ORC: Aggressive combat abilities, tribal synergies, menace, raid effects. Examples: 'Menace', 'Trample', 'Whenever this attacks, other Orcs you control get +1/+0 until end of turn', '{1}: This creature gets +2/+0 until end of turn'. "
        elif 'warrior' in subtype_lower:
            guidance += "WARRIOR: First strike, combat bonuses, equipment synergies, battle tactics. Examples: 'First strike', 'Vigilance', 'Whenever this attacks, target creature can't block this turn', 'Equipped creatures you control have first strike'. "
        elif 'berserker' in subtype_lower:
            guidance += "BERSERKER: Aggressive abilities, rampage effects, damage-based triggers, reckless combat. Examples: 'Trample', 'Whenever this deals combat damage, it deals that much damage to you', '{R}: This creature gets +1/+0 and gains trample until end of turn'. "
        
        # Undead creatures
        elif 'zombie' in subtype_lower:
            guidance += "ZOMBIE: Deathtouch, graveyard recursion, sacrifice synergies, decay effects. Examples: 'Deathtouch', 'When this dies, create a 2/2 black Zombie creature token', 'Whenever a creature dies, put a +1/+1 counter on this creature', '{B}, Sacrifice a creature: This creature gets +2/+2 until end of turn'. "
        elif 'skeleton' in subtype_lower:
            guidance += "SKELETON: Recursion abilities, self-sacrifice effects, bone/death themes. Examples: '{1}{B}: Return this card from your graveyard to the battlefield tapped', 'When this dies, you may pay {2}. If you do, return it to your hand', '{B}, Sacrifice this creature: Target creature gets -1/-1 until end of turn'. "
        elif 'vampire' in subtype_lower:
            guidance += "VAMPIRE: Lifelink, flying, life drain, +1/+1 counter growth, blood themes. Examples: 'Lifelink', 'Flying' + 'Lifelink', 'Whenever this deals combat damage to a player, put a +1/+1 counter on it', 'When this enters, target opponent loses 2 life and you gain 2 life'. "
        elif 'wraith' in subtype_lower or 'specter' in subtype_lower:
            guidance += "WRAITH/SPECTER: Flying, discard effects, fear abilities, intangible themes. Examples: 'Flying', 'Flying' + 'Whenever this deals combat damage to a player, that player discards a card', 'This creature can't be blocked except by artifact creatures'. "
        
        # Large creatures
        elif 'beast' in subtype_lower:
            guidance += "BEAST: Trample, fighting abilities, +1/+1 counters, primal/nature effects. Examples: 'Trample', 'When this enters, fight target creature', 'Whenever this attacks, put a +1/+1 counter on it', '{G}: This creature gets +1/+1 until end of turn'. "
        elif 'giant' in subtype_lower:
            guidance += "GIANT: Large stats, reach, land-based abilities, massive effects. Examples: 'Reach', 'Trample', 'When this enters, destroy target land', 'Whenever a Mountain enters the battlefield under your control, this creature gets +1/+1 until end of turn'. "
        elif 'troll' in subtype_lower:
            guidance += "TROLL: Regeneration, +1/+1 counter growth, sacrifice for benefits, resilience. Examples: '{G}: Regenerate this creature', 'Whenever this creature is dealt damage, put a +1/+1 counter on it', 'Sacrifice a land: This creature gets +2/+2 until end of turn'. "
        elif 'elephant' in subtype_lower:
            guidance += "ELEPHANT: Trample, memory/graveyard effects, large presence, herd mentality. Examples: 'Trample', 'When this enters, return target card from your graveyard to your hand', 'Other Elephants you control get +1/+1', 'Whenever this attacks, create a 3/3 green Elephant creature token'. "
        
        # Magical creatures
        elif 'wizard' in subtype_lower:
            guidance += "WIZARD: Spell synergies, card draw, instant/sorcery effects, magical knowledge. Examples: 'When this enters, draw a card', 'Whenever you cast an instant or sorcery spell, scry 1', '{T}: Add one mana of any color. Spend this mana only to cast instant or sorcery spells', 'Instant and sorcery spells you cast cost {1} less to cast'. "
        elif 'shaman' in subtype_lower:
            guidance += "SHAMAN: Mana abilities, elemental effects, nature magic, tribal synergies. Examples: '{T}: Add one mana of any color', 'When this enters, search your library for a basic land card and put it onto the battlefield tapped', 'Whenever you cast a creature spell, this deals 1 damage to any target'. "
        elif 'cleric' in subtype_lower:
            guidance += "CLERIC: Lifegain, protection effects, enchantment synergies, divine magic. Examples: 'When this enters, you gain 3 life', 'Other creatures you control have lifelink', '{T}: Target creature gains protection from the color of your choice until end of turn', 'Whenever you gain life, scry 1'. "
        elif 'druid' in subtype_lower:
            guidance += "DRUID: Mana production, land abilities, creature tokens, nature harmony. Examples: '{T}: Add {G}', 'When this enters, create a 1/1 green Saproling creature token', '{T}: Search your library for a Forest card and put it onto the battlefield tapped', 'Creatures you control with power 1 or less get +1/+1'. "
        
        # Sneaky creatures
        elif 'rogue' in subtype_lower or 'assassin' in subtype_lower:
            guidance += "ROGUE/ASSASSIN: Deathtouch, unblockable, card advantage, removal effects. Examples: 'Deathtouch', 'This creature can't be blocked', 'When this deals combat damage to a player, draw a card', 'When this enters, destroy target creature with power 3 or less'. "
        elif 'ninja' in subtype_lower:
            guidance += "NINJA: Ninjutsu, evasion, saboteur effects, stealth abilities. Examples: 'This creature can't be blocked', 'Whenever this deals combat damage to a player, draw a card and discard a card', 'When this enters, return target creature to its owner's hand'. "
        
        # Defensive creatures
        elif 'knight' in subtype_lower:
            guidance += "KNIGHT: First strike, vigilance, protection effects, honor-based abilities. Examples: 'First strike', 'Vigilance', 'Protection from black', 'When this enters, destroy target artifact or enchantment', 'Other Knights you control get +1/+1'. "
        elif 'soldier' in subtype_lower:
            guidance += "SOLDIER: Vigilance, combat bonuses, token creation, military tactics. Examples: 'Vigilance', 'When this enters, create a 1/1 white Soldier creature token', 'Other Soldiers you control get +1/+1', 'Whenever this attacks, tap target creature'. "
        elif 'wall' in subtype_lower:
            guidance += "WALL: Defender, high toughness abilities, protective effects, utility functions. Examples: 'Defender', 'Defender' + '{T}: Add one mana of any color', 'Defender' + 'When this enters, draw a card', 'Creatures can't attack you unless their controller pays {1} for each creature they control that's attacking you'. "
        
        # Elemental creatures
        elif 'elemental' in subtype_lower:
            guidance += "ELEMENTAL: Abilities matching their element colors, basic land synergies, primal forces. Examples: For red - 'When this enters, deal 2 damage to any target'; for blue - 'When this enters, draw a card'; for green - 'Trample' + 'Whenever a Forest enters under your control, this gets +1/+1 until end of turn'. "
        
        # Demons and devils
        elif 'demon' in subtype_lower:
            guidance += "DEMON: Flying, menace, sacrifice effects, life drain, powerful with drawbacks. Examples: 'Flying' + 'Menace', 'Flying' + 'At the beginning of your upkeep, sacrifice a creature', 'When this enters, each opponent loses 3 life', 'Whenever a creature dies, you gain 1 life'. "
        elif 'devil' in subtype_lower:
            guidance += "DEVIL: Direct damage, sacrifice triggers, aggressive abilities, chaotic effects. Examples: 'When this dies, deal 1 damage to any target', 'Whenever you sacrifice a permanent, this deals 1 damage to any target', 'Haste', '{R}, Sacrifice this creature: Deal 2 damage to any target'. "
        
        # Small creatures
        elif 'elf' in subtype_lower:
            guidance += "ELF: Mana generation, creature synergies, forest effects, tribal bonuses. Examples: '{T}: Add {G}', 'Other Elves you control get +1/+1', 'When this enters, search your library for a Forest card and put it onto the battlefield tapped', 'Whenever you cast an Elf spell, create a 1/1 green Elf Warrior creature token'. "
        elif 'human' in subtype_lower:
            guidance += "HUMAN: Versatile abilities, equipment synergies, cooperation, adaptability. Examples: 'When this enters, create a Treasure token', 'Whenever you cast a noncreature spell, this gets +1/+1 until end of turn', 'Other Humans you control get +1/+1', '{T}: Target creature gains first strike until end of turn'. "
        elif 'halfling' in subtype_lower or 'kithkin' in subtype_lower:
            guidance += "HALFLING/KITHKIN: Small creature synergies, evasion, community effects, resourcefulness. Examples: 'This creature can't be blocked by creatures with power 3 or greater', 'Whenever a creature with power 2 or less enters the battlefield, draw a card', 'Other creatures you control with power 2 or less get +1/+1'. "
        
        # Artifact creatures
        elif 'construct' in subtype_lower or 'golem' in subtype_lower:
            guidance += "CONSTRUCT/GOLEM: Artifact synergies, utility abilities, colorless effects. Examples: 'When this enters, create a Treasure token', 'Artifacts you control have hexproof', '{T}: Add one mana of any color', 'Whenever you cast an artifact spell, this gets +1/+1 until end of turn'. "
        elif 'thopter' in subtype_lower:
            guidance += "THOPTER: Flying, artifact synergies, small flying utility. Examples: 'Flying', 'Flying' + 'When this enters, create a 1/1 colorless Thopter artifact creature token with flying', 'Whenever you cast an artifact spell, create a 1/1 colorless Thopter artifact creature token with flying'. "
    
    return guidance

def _get_artifact_guidance(subtype: str) -> str:
    """Artifact-specific guidance."""
    guidance = "ARTIFACT: Generate utility effects or activated abilities. "
    
    if subtype:
        subtype_lower = subtype.lower()
        if 'equipment' in subtype_lower:
            guidance += "EQUIPMENT: Must have 'Equip {cost}' ability. Focus on 'Equipped creature gets/has...' effects. "
        elif 'vehicle' in subtype_lower:
            guidance += "VEHICLE: Must have 'Crew X' ability. Focus on combat stats and abilities. "
        elif 'food' in subtype_lower:
            guidance += "FOOD: Must have '{2}, {T}, Sacrifice: You gain 2 life.' "
    
    return guidance

def _get_enchantment_guidance(subtype: str) -> str:
    """Enchantment-specific guidance."""
    guidance = "ENCHANTMENT: Generate ongoing effects that modify the game. "
    
    if subtype:
        subtype_lower = subtype.lower()
        if 'aura' in subtype_lower:
            guidance += "AURA: Must start with 'Enchant creature'. Focus on 'Enchanted creature gets/has...' effects. "
        elif 'saga' in subtype_lower:
            guidance += "SAGA: Must have chapter abilities (I, II, III) that tell a story. "
    
    return guidance

def _get_land_guidance(subtype: str) -> str:
    """Land-specific guidance."""
    guidance = "LAND: Focus on mana abilities and utility effects. "
    
    if subtype:
        subtype_lower = subtype.lower()
        if 'forest' in subtype_lower:
            guidance += "FOREST: Should produce green mana. Consider green-themed effects. "
        elif 'island' in subtype_lower:
            guidance += "ISLAND: Should produce blue mana. Consider blue-themed effects. "
        elif 'mountain' in subtype_lower:
            guidance += "MOUNTAIN: Should produce red mana. Consider red-themed effects. "
        elif 'plains' in subtype_lower:
            guidance += "PLAINS: Should produce white mana. Consider white-themed effects. "
        elif 'swamp' in subtype_lower:
            guidance += "SWAMP: Should produce black mana. Consider black-themed effects. "
        elif 'gate' in subtype_lower:
            guidance += "GATE: Typically dual mana but enters tapped. "
        elif 'desert' in subtype_lower:
            guidance += "DESERT: Colorless mana with desert-themed abilities. "
    
    return guidance

def _get_planeswalker_guidance(rarity: str) -> str:
    """Planeswalker-specific guidance."""
    guidance = "PLANESWALKER: Generate 2-4 loyalty abilities. Format: '+X: effect', '-X: effect'. Include starting loyalty. "
    
    if rarity in ['rare', 'mythic']:
        guidance += "Rare/Mythic: 3-4 abilities including ultimate. Starting loyalty 3-5. "
    else:
        guidance += "Uncommon: 2-3 abilities, simpler effects. Starting loyalty 2-4. "
    
    return guidance

def _get_color_guidance(colors: list, card_type: str) -> str:
    """Color-specific guidance."""
    guidance = "COLOR THEMES: "
    
    if 'W' in colors:
        guidance += "White: lifegain, protection, tokens, combat buffs. "
    if 'U' in colors:
        guidance += "Blue: card draw, counterspells, flying, library manipulation. "
    if 'B' in colors:
        guidance += "Black: destruction, life drain, graveyard, sacrifice. "
    if 'R' in colors:
        guidance += "Red: damage, haste, chaos, artifact destruction. "
    if 'G' in colors:
        guidance += "Green: big creatures, mana, +1/+1 counters, nature. "
    
    return guidance

def _get_rarity_guidance(rarity: str, card_type: str, cmc: int) -> str:
    """Rarity-based complexity guidance."""
    if rarity == 'mythic':
        return "MYTHIC: Complex, build-around effects. Can be game-changing. "
    elif rarity == 'rare':
        return "RARE: Unique mechanics, moderate complexity. "
    elif rarity == 'uncommon':
        return "UNCOMMON: Moderate effects, some complexity. "
    else:
        return "COMMON: Simple, straightforward effects. "

def _get_name_inspiration(card_name: str) -> str:
    """Extract thematic guidance from card name."""
    guidance = f"CARD NAME: This card is named '{card_name}'. "
    
    # Provide specific guidance based on name content
    name_lower = card_name.lower()
    
    # Action words that suggest abilities
    if any(word in name_lower for word in ['strike', 'slash', 'smash', 'crush', 'destroy']):
        guidance += "The name suggests aggressive/destructive abilities. "
    elif any(word in name_lower for word in ['guard', 'protect', 'shield', 'wall', 'sentinel']):
        guidance += "The name suggests defensive/protective abilities. "
    elif any(word in name_lower for word in ['weaver', 'mage', 'seer', 'sage', 'oracle']):
        guidance += "The name suggests magical/spellcaster abilities. "
    elif any(word in name_lower for word in ['shadow', 'whisper', 'stealth', 'sneak', 'rogue']):
        guidance += "The name suggests evasive/sneaky abilities. "
    elif any(word in name_lower for word in ['flame', 'fire', 'burn', 'blaze', 'ember']):
        guidance += "The name suggests fire/damage abilities. "
    elif any(word in name_lower for word in ['storm', 'wind', 'gale', 'tempest']):
        guidance += "The name suggests weather/air abilities. "
    elif any(word in name_lower for word in ['life', 'heal', 'mercy', 'blessing', 'grace']):
        guidance += "The name suggests lifegain/healing abilities. "
    elif any(word in name_lower for word in ['death', 'doom', 'bane', 'reaper', 'grave']):
        guidance += "The name suggests death/sacrifice abilities. "
    elif any(word in name_lower for word in ['growth', 'bloom', 'flourish', 'verdant']):
        guidance += "The name suggests +1/+1 counter/growth abilities. "
    elif any(word in name_lower for word in ['treasure', 'hoard', 'riches', 'gold', 'coin']):
        guidance += "The name suggests treasure/mana abilities. "
    
    # Size indicators
    if any(word in name_lower for word in ['tiny', 'small', 'little', 'mini']):
        guidance += "The name suggests small/evasive creature themes. "
    elif any(word in name_lower for word in ['giant', 'massive', 'colossal', 'enormous', 'titan']):
        guidance += "The name suggests large/impactful abilities. "
    
    # Leadership/cooperation themes
    if any(word in name_lower for word in ['lord', 'master', 'chief', 'leader', 'commander']):
        guidance += "The name suggests tribal/leadership abilities that help other creatures. "
    elif any(word in name_lower for word in ['pack', 'swarm', 'horde', 'clan', 'tribe']):
        guidance += "The name suggests tribal synergies or token generation. "
    
    guidance += f"Create abilities that thematically match '{card_name}' and feel unique to this specific character. "
    
    return guidance

def _get_creature_restrictions(subtype: str) -> str:
    """Flying and other creature restrictions."""
    subtype_lower = subtype.lower()
    
    if any(restricted_type in subtype_lower for restricted_type in flying_restricted_types):
        return f"FLYING RESTRICTION: {subtype} creatures rarely have flying. Use grounded abilities instead. "
    elif any(flying_type in subtype_lower for flying_type in flying_encouraged_types):
        return f"FLYING NATURAL: {subtype} creatures are natural fliers. "
    
    return ""

def _get_output_rules() -> str:
    """Output formatting rules."""
    return ("OUTPUT FORMAT: Generate ONLY rules text. "
            "Do NOT include card name, type line, mana cost, power/toughness, or flavor text. "
            "Use {T} for tap symbol. "
            "Wrap each ability in double quotes. "
            "Example: '\"Flying, trample\" \"When this enters, gain 3 life\"' ")

def _get_examples(card_type: str) -> str:
    """Few-shot examples based on card type."""
    if 'creature' in card_type:
        return ("CORRECT EXAMPLES: "
                "'\"Flying, vigilance\"' "
                "'\"Deathtouch\" \"{T}: Deal 1 damage to any target\"' "
                "'\"When this enters the battlefield, create two 1/1 token creatures\"' ")
    elif 'artifact' in card_type:
        return ("CORRECT EXAMPLES: "
                "'\"{T}: Add one mana of any color\"' "
                "'\"{2}, {T}: Draw a card\"' "
                "'\"Equipped creature gets +2/+1. Equip {3}\"' ")
    elif 'enchantment' in card_type:
        return ("CORRECT EXAMPLES: "
                "'\"Creatures you control get +1/+1\"' "
                "'\"Whenever a creature enters the battlefield, you gain 1 life\"' "
                "'\"Enchant creature. Enchanted creature has flying\"' ")
    elif 'land' in card_type:
        return ("CORRECT EXAMPLES: "
                "'\"{T}: Add {G}\"' "
                "'\"{T}: Add one mana of any color\"' "
                "'\"{1}, {T}: Target creature gets +1/+0 until end of turn\"' ")
    else:
        return ("CORRECT EXAMPLES: "
                "'\"Destroy target creature\"' "
                "'\"Draw three cards\"' "
                "'\"Deal 4 damage to any target\"' ")

def createCardContent(prompt, card_data=None):
    """
    Generate card content using Ollama Python client with enhanced context
    Returns the response text from the LLM
    """
    start_time = time.time()
    print(f"[DEBUG] createCardContent called with:")
    print(f"   Prompt: {repr(prompt[:100])}...")
    print(f"   Card data keys: {list(card_data.keys()) if card_data else 'None'}")
    print(f"[TIMING] Function start: 0.00s")
    try:
        # Build enhanced prompt using modular system
        if card_data:
            enhanced_prompt = build_prompt(card_data)
        else:
            # Fallback for when no card data is provided
            enhanced_prompt = ("Generate Magic: The Gathering rules text for a card. "
                             "Focus on unique, cohesive abilities. "
                             "OUTPUT FORMAT: Generate ONLY rules text. "
                             "Wrap each ability in double quotes. ")
        
        prompt_time = time.time()
        print(f"[TIMING] Prompt built: {prompt_time - start_time:.2f}s")
        
        # Generate and validate rules text (retry if contaminated)
        max_attempts = 3
        card_text = ""
        
        for attempt in range(max_attempts):
            print(f"[TIMING] Starting ollama.generate attempt {attempt + 1}: {time.time() - start_time:.2f}s")
            
            ollama_start = time.time()
            response = ollama.generate(
                model='mistral:latest',
                prompt=enhanced_prompt
            )
            ollama_end = time.time()
            
            print(f"[TIMING] Ollama response received: {ollama_end - start_time:.2f}s (generation took {ollama_end - ollama_start:.2f}s)")
            
            # Clean up the response
            card_text = response['response'].strip()
            print(f"[RAW] Content model raw output: {repr(card_text)}")
            
            # Strip any non-rules text that might have been included
            strip_start = time.time()
            card_text = strip_non_rules_text(card_text, card_data)
            print(f"[CLEAN] After stripping non-rules text: {repr(card_text)}")
            print(f"[TIMING] Text stripping completed: {time.time() - start_time:.2f}s (stripping took {time.time() - strip_start:.2f}s)")
            
            # Validate the response doesn't contain type line elements
            validation_start = time.time()
            if validate_rules_text(card_text, card_data):
                print(f"[TIMING] Validation passed: {time.time() - start_time:.2f}s (validation took {time.time() - validation_start:.2f}s)")
                break
            else:
                print(f"[ERROR] Rules text validation failed on attempt {attempt + 1}, regenerating...")
                print(f"[TIMING] Validation failed: {time.time() - start_time:.2f}s (validation took {time.time() - validation_start:.2f}s)")
                if attempt < max_attempts - 1:
                    # Add additional constraint for retry
                    enhanced_prompt += f" CRITICAL: Do NOT include type line elements like '{card_data.get('type', '')}' or '{card_data.get('subtype', '')}' in the rules text. Generate ONLY the abilities text."
                else:
                    print(f"[WARN] Max validation attempts reached, using last generated text")
                    break
        
        # Remove surrounding quotes if present
        processing_start = time.time()
        if (card_text.startswith('"') and card_text.endswith('"')) or \
           (card_text.startswith("'") and card_text.endswith("'")):
            card_text = card_text[1:-1].strip()
        
        # Fix common formatting issues
        # Replace literal \\n with actual newlines
        card_text = card_text.replace('\\n', '\n')
        
        # Replace "Tap:" with "{T}:" for tap symbols
        card_text = card_text.replace('Tap:', '{T}:')
        
        # Sanitize double braces to single braces for mana/tap symbols
        card_text = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', card_text)
        
        # Remove any ability type labels that might have slipped through
        ability_label_patterns = [
            r'Triggered Ability:\s*',
            r'Active Ability:\s*', 
            r'Passive Ability:\s*',
            r'Keywords:\s*',
            r'Keyword Abilities:\s*',
            r'Activated Ability:\s*'
        ]
        
        for pattern in ability_label_patterns:
            card_text = re.sub(pattern, '', card_text, flags=re.IGNORECASE)
        
        # Remove named ability patterns like "Channel Life" - {T}: Add {G}
        # Pattern 1: "Quoted Name" - ability or "Quoted Name": ability  
        card_text = re.sub(r'"[^"]+"\s*[-:]\s*', '', card_text)
        # Pattern 2: Unquoted Name - {ability} or Name: {ability} (more precise)
        card_text = re.sub(r'^[A-Z][A-Za-z\s]+\s*[-:]\s*(?=\{)', '', card_text, flags=re.MULTILINE)
        # Pattern 3: Handle cases where ability name appears at start of line
        card_text = re.sub(r'\n[A-Z][A-Za-z\s]+\s*[-:]\s*(?=\{)', '\n{', card_text)
        
        # Condense verbose keyword descriptions to standard keywords
        # Trample variations
        card_text = re.sub(r'"?[Tt]rample over blockers"?', 'Trample', card_text)
        card_text = re.sub(r'"?[Tt]rample through blockers"?', 'Trample', card_text)
        card_text = re.sub(r'"?[Tt]rample past blockers"?', 'Trample', card_text)
        card_text = re.sub(r'"?[Tt]rample damage over"?', 'Trample', card_text)
        
        # Flying variations
        card_text = re.sub(r'"?[Ff]lying over creatures"?', 'Flying', card_text)
        card_text = re.sub(r'"?[Ff]lying above blockers"?', 'Flying', card_text)
        
        # First strike variations
        card_text = re.sub(r'"?[Ff]irst strike in combat"?', 'First strike', card_text)
        card_text = re.sub(r'"?[Ss]trikes first in combat"?', 'First strike', card_text)
        
        # Deathtouch variations
        card_text = re.sub(r'"?[Dd]eathtouch damage"?', 'Deathtouch', card_text)
        card_text = re.sub(r'"?[Ll]ethal touch"?', 'Deathtouch', card_text)
        
        print(f"[FORMAT] After initial formatting: {repr(card_text)}")
        print(f"[TIMING] Basic formatting completed: {time.time() - start_time:.2f}s (formatting took {time.time() - processing_start:.2f}s)")
        
        # Parse card text into abilities - prioritize newlines, fallback to periods
        parse_start = time.time()
        # First try splitting on newlines (most natural for Magic card abilities)
        if '\n' in card_text:
            sentences = [line.strip() for line in card_text.split('\n') if line.strip()]
            print(f"[SPLIT] Newline split result: {sentences}")
        else:
            # Fallback to smart period splitting if no newlines
            sentences = smart_split_by_periods(card_text)
            print(f"[SPLIT] Period split result: {sentences}")
        
        print(f"[TIMING] Text parsing completed: {time.time() - start_time:.2f}s (parsing took {time.time() - parse_start:.2f}s)")
        
        # Filter out single numbers and other junk elements
        filtered_sentences = []
        for sentence in sentences:
            # Skip single numbers (like "1", "2", etc.)
            if sentence.strip().isdigit():
                print(f"[FILTER] Removing single number: '{sentence}'")
                continue
            # Skip numbered list markers (like "1.", "2)", etc.)
            if re.match(r'^\d+[.)\]:]\s*$', sentence.strip()):
                print(f"[FILTER] Removing numbered marker: '{sentence}'")
                continue
            # Skip very short non-ability text (less than 3 chars unless it's a keyword)
            if len(sentence.strip()) < 3 and not re.match(r'^[A-Za-z]+$', sentence.strip()):
                print(f"[FILTER] Removing short junk: '{sentence}'")
                continue
            # Keep the sentence, but strip surrounding quotes
            cleaned_sentence = sentence.strip()
            # Remove surrounding quotes if present (both single and double)
            if (cleaned_sentence.startswith('"') and cleaned_sentence.endswith('"')) or \
               (cleaned_sentence.startswith("'") and cleaned_sentence.endswith("'")):
                cleaned_sentence = cleaned_sentence[1:-1].strip()
                print(f"[FILTER] Stripped quotes from: '{sentence}' -> '{cleaned_sentence}'")
            
            filtered_sentences.append(cleaned_sentence)
        
        sentences = filtered_sentences
        print(f"[FILTER] After filtering: {sentences}")
        
        # Create abilities array from cleaned sentences
        if len(sentences) > 4:
            print(f"[WARN] Truncating from {len(sentences)} to 4 sentences")
            abilities_array = sentences[:4]
        else:
            abilities_array = sentences

        # UNIFIED SANITATION PIPELINE - Applied to ALL card types
        if card_data:
            # Step 1: Universal ability reordering (work with abilities array)
            print(f"[BEFORE] Before reordering: {abilities_array}")
            abilities_array = reorder_abilities_properly_array(abilities_array, card_data)
            print(f"[AFTER] After reordering: {abilities_array}")
            
            # Step 2: Type-specific sanitation (convert array back to text for legacy processors)
            card_text = '\n'.join(abilities_array)
            card_type = (card_data.get('type') or '').lower()
            
            if 'creature' in card_type:
                # Creatures get more aggressive ability limiting
                card_text = limit_creature_active_abilities(card_text)
            elif 'planeswalker' in card_type:
                # Planeswalkers get loyalty ability formatting
                card_text = sanitize_planeswalker_abilities(card_text)
            elif 'instant' in card_type or 'sorcery' in card_type:
                # Spells get spell-specific sanitation
                card_text = sanitize_spell_abilities(card_text)
            elif 'land' in card_type:
                # Lands get land-specific sanitation
                card_text = sanitize_land_abilities(card_text)
            elif 'artifact' in card_type or 'enchantment' in card_type:
                # Artifacts/enchantments get general permanent sanitation
                card_text = sanitize_permanent_abilities(card_text)
            
            # Step 3: Universal complexity limits (prevent overpowered cards)
            card_text = apply_universal_complexity_limits(card_text, card_data)
        else:
            # No card data - just convert abilities array to final text
            card_text = '\n'.join(abilities_array)
        
        # Final validation after all processing (check for post-processing issues)
        final_validation_start = time.time()
        if not validate_rules_text(card_text, card_data):
            print(f"[WARN] Final validation failed after text processing - issues introduced during formatting")
            print(f"[WARN] Processed text: {repr(card_text)}")
            # For now, return the text anyway, but log the issue
            # TODO: Could implement full regeneration loop here if needed
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[TIMING] Final validation completed: {total_time:.2f}s (validation took {end_time - final_validation_start:.2f}s)")
        print(f"[TIMING] ===== TOTAL PROCESSING TIME: {total_time:.2f}s =====")
        
        print(f"[RETURN] createCardContent returning:")
        print(f"   Result: {repr(card_text)}")
        return card_text
        
    except Exception as e:
        print(f"[ERROR] Error in createCardContent: {e}")
        print("Make sure Mistral model is installed: 'ollama pull mistral:latest'")
        return prompt
