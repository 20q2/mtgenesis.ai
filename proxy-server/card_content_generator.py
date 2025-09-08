"""
Card content generation utilities for Magic: The Gathering cards.

This module handles the core text generation logic for MTG card abilities and rules text
using LLM models via Ollama. It includes comprehensive guidance for different card types,
colors, rarities, and special mechanics.
"""

import re
import ollama
from text_processing import (
    strip_non_rules_text, format_ability_newlines, reorder_abilities_properly
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


def createCardContent(prompt, card_data=None):
    """
    Generate card content using Ollama Python client with enhanced context
    Returns the response text from the LLM
    """
    print(f"🧠 createCardContent called with:")
    print(f"   📝 Prompt: {repr(prompt[:100])}...")
    print(f"   🎲 Card data keys: {list(card_data.keys()) if card_data else 'None'}")
    try:
        # Build enhanced prompt based on card properties
        enhanced_prompt = f"Generate the rules text for a Magic: the gathering card. \nOutput format: Only the rules text abilities, no explanations, no card name, no type line."
        
        if card_data:
            # Analyze mana cost for power level
            cmc = card_data.get('cmc', 0)
            colors = card_data.get('colors', [])
            card_type = (card_data.get('type') or '').lower()
            power = card_data.get('power')
            toughness = card_data.get('toughness')
            rarity = (card_data.get('rarity') or 'common').lower()
            
            # Check for asterisk (*) in power/toughness - CRITICAL VALIDATION
            # Handle cases like "*/2", "3/*", "*/*", "*+1", "2+*", etc.
            has_asterisk_power = power and '*' in str(power)
            has_asterisk_toughness = toughness and '*' in str(toughness)
            
            if has_asterisk_power or has_asterisk_toughness:
                asterisk_guidance = ""
                
                # Enhanced asterisk validation with color-appropriate suggestions
                color_examples = {
                    'G': "creatures you control, lands you control, or Forests you control",
                    'U': "cards in your hand, artifacts you control, or Islands you control", 
                    'B': "creature cards in your graveyard, cards in your graveyard, or Swamps you control",
                    'W': "creatures you control, Plains you control, or enchantments you control",
                    'R': "Mountains you control or cards in target opponent's hand"
                }
                
                # Build color-specific examples
                example_sources = []
                for color in colors:
                    if color in color_examples:
                        example_sources.append(color_examples[color])
                
                if example_sources:
                    examples_text = ", ".join(example_sources[:2])  # Use first 2 color examples
                else:
                    examples_text = "cards in your hand, creatures you control, or cards in your graveyard"
                
                if has_asterisk_power and has_asterisk_toughness:
                    asterisk_guidance = f" CRITICAL ASTERISK RULE: This creature has power {power} and toughness {toughness} - BOTH are asterisks (*). You MUST start your rules text with a definition like '[Card Name]'s power and toughness are each equal to the number of [something you count].' For {colors} colors, appropriate things to count: {examples_text}. REQUIRED FORMAT: Start with the definition, then add any other abilities. Example: 'Patrick Star's power and toughness are each equal to the number of cards in your hand. [Other abilities...]' This asterisk definition is MANDATORY and must be the first line of rules text."
                elif has_asterisk_power:
                    asterisk_guidance = f" MANDATORY ASTERISK RULE: This creature has power {power}. You MUST include an ability that defines what the * equals. For {colors} colors, consider: {examples_text}. Examples: 'This creature's power is equal to the number of cards in your hand' or 'This creature's power is equal to the number of artifacts you control'. The * power MUST be defined in the rules text."
                elif has_asterisk_toughness:
                    asterisk_guidance = f" MANDATORY ASTERISK RULE: This creature has toughness {toughness}. You MUST include an ability that defines what the * equals. For {colors} colors, consider: {examples_text}. Examples: 'This creature's toughness is equal to the number of lands you control' or 'This creature's toughness is equal to your life total'. The * toughness MUST be defined in the rules text."
                
                enhanced_prompt += asterisk_guidance
            
            # Check for regenerate keyword - CRITICAL VALIDATION
            # Regenerate requires specific formatting and rules
            has_regenerate = False
            
            # Check in existing description
            if card_data.get('description'):
                description_text = str(card_data.get('description', '')).lower()
                if 'regenerate' in description_text:
                    has_regenerate = True
            
            # Check in the original prompt
            if 'regenerate' in prompt.lower():
                has_regenerate = True
                
            if has_regenerate:
                regenerate_guidance = " MANDATORY REGENERATE RULE: This card uses regenerate abilities. You MUST use the correct format. Regenerate abilities must be written as '{cost}: Regenerate this creature' or 'Regenerate this creature' (if no cost). Examples: '{1}{G}: Regenerate this creature', '{B}: Regenerate this creature', 'Regenerate this creature'. NEVER write incorrect formats like 'can regenerate', 'has regenerate', or 'regenerates'. Regenerate means: The next time this creature would be destroyed this turn, it isn't. Instead, tap it, remove all damage from it, and remove it from combat. This is the official Magic templating and must be followed exactly."
                enhanced_prompt += regenerate_guidance
            
            supertype = (card_data.get('supertype') or '').lower()
            is_legendary = 'legendary' in supertype
            mana_cost = card_data.get('manaCost', '')
            
            # Check if card has X in mana cost - if so, must use X in rules text
            has_x_cost = 'X' in mana_cost.upper() or '{X}' in mana_cost.upper()
            x_guidance = ""
            if has_x_cost:
                x_guidance = f" CRITICAL X REQUIREMENT: This card has X in its mana cost ({mana_cost}), so the rules text MUST reference X meaningfully AND include an explanation of what X represents. REQUIRED FORMAT: The rules text must contain BOTH an X effect AND an explanation such as 'where X is the amount of mana spent to cast this spell' OR 'enters with X +1/+1 counters on it' OR 'where X is the number of [condition]'. Examples: 'Deal X damage to any target, where X is the amount of mana spent to cast this spell', 'Create X 1/1 token creatures, where X is the amount of mana spent to cast this spell', 'enters with X +1/+1 counters on it', 'X target creatures gain flying until end of turn, where X is the amount of mana spent to cast this spell'."
            
            # Power level guidance based on CMC and rarity
            base_power = ""
            if cmc <= 1:
                base_power = "very simple and low-power"
            elif cmc <= 3:
                base_power = "moderate power"
            elif cmc <= 5:
                base_power = "strong"
            else:
                base_power = "very powerful and game-changing"
            
            # Rarity adjustments with ability guidance - BALANCED POWER LEVELS
            # IMPORTANT: Keywords count as abilities and are valuable!
            if rarity == 'common':
                power_level = f"{base_power}, extremely simple with ONLY 1 total ability - either ONE keyword (Haste, Trample, Deathtouch, Reach, Menace, or Lifelink) OR one simple triggered ability, never both. Keywords are abilities and count toward the limit."
            elif rarity == 'uncommon':
                power_level = f"{base_power}, with modest complexity, maximum 2 total abilities - this can be 1-2 keywords, OR 1 keyword + 1 simple triggered ability, OR 2 simple triggered abilities, OR 1 activated ability. Keywords count as abilities."
            elif rarity == 'rare':
                power_level = f"{base_power}, with unique mechanics, maximum 3 total abilities - this can be 1-2 keywords + 1-2 other abilities, OR 3 non-keyword abilities. All keywords count toward the total ability limit."
            elif rarity == 'mythic':
                power_level = f"{base_power}, with splashy build-around effects, maximum 3-4 total abilities - this includes ALL keywords, triggered abilities, and activated abilities. Keywords are valuable and count toward limits."
            else:
                power_level = base_power
            
            # Enhanced color identity guidance with specific mechanics
            color_guidance = ""
            
            if 'W' in colors:
                color_guidance += " White mechanics: protection from colors, lifegain triggers, exile removal, prevent damage, tap creatures, +1/+1 counters on creatures, enchantment synergies, vigilance, first strike."
            
            if 'U' in colors:
                color_guidance += " Blue mechanics: counter spells, return to hand, tap/untap permanents, scry, flying creatures, mill cards, copy spells, phase out, control magic, card selection, bounce effects, temporary steal effects."
            
            if 'B' in colors:
                color_guidance += " Black mechanics: pay life for effects, destroy creatures, discard from hand, return from graveyard, sacrifice creatures for value, -1/-1 counters, drain life, deathtouch, menace."
            
            if 'R' in colors:
                color_guidance += " Red mechanics: deal damage to creatures/players, haste, can't block, sacrifice for temporary effects, random discard, artifact destruction, first strike, trample, impulse draw."
            
            if 'G' in colors:
                color_guidance += " Green mechanics: ramp (add mana), large creature stats, trample, fight other creatures, destroy artifacts/enchantments, +1/+1 counters, land tutoring, hexproof, reach."
            
            # Add specific multicolor synergies
            if len(colors) > 1:
                if 'W' in colors and 'U' in colors:
                    color_guidance += " Azorius themes: control elements, tax effects, flying creatures, artifact synergies."
                elif 'U' in colors and 'B' in colors:
                    color_guidance += " Dimir themes: mill and graveyard, card selection, unblockable creatures, surveil."
                elif 'B' in colors and 'R' in colors:
                    color_guidance += " Rakdos themes: aggressive damage, sacrifice for value, spectacle costs."
                elif 'R' in colors and 'G' in colors:
                    color_guidance += " Gruul themes: aggressive large creatures, land destruction, riot mechanics."
                elif 'G' in colors and 'W' in colors:
                    color_guidance += " Selesnya themes: token generation, populate, lifegain matters."
                elif 'W' in colors and 'B' in colors:
                    color_guidance += " Orzhov themes: lifegain/lifedrain, sacrifice/recursion, exile effects."
                elif 'U' in colors and 'R' in colors:
                    color_guidance += " Izzet themes: instant/sorcery matters, spell copying, artifact synergies."
                elif 'B' in colors and 'G' in colors:
                    color_guidance += " Golgari themes: graveyard value, creature sacrifice, +1/+1 counters."
                elif 'R' in colors and 'W' in colors:
                    color_guidance += " Boros themes: aggressive creatures, combat tricks, equipment matters."
                elif 'G' in colors and 'U' in colors:
                    color_guidance += " Simic themes: +1/+1 counters, card draw, creature evolution."
            
            # Three-color combinations (Shards and Wedges)
            elif len(colors) == 3:
                sorted_colors = sorted(colors)
                if sorted_colors == ['G', 'U', 'W']:  # Bant
                    color_guidance += " Bant themes: exalted mechanics, control with creatures, artifact interaction, noble and honorable effects."
                elif sorted_colors == ['B', 'U', 'W']:  # Esper
                    color_guidance += " Esper themes: artifact creatures, control magic, evasive threats, combining technology with magic."
                elif sorted_colors == ['B', 'R', 'U']:  # Grixis
                    color_guidance += " Grixis themes: graveyard manipulation, spell copying, creature theft, necromantic power."
                elif sorted_colors == ['G', 'R', 'W']:  # Naya
                    color_guidance += " Naya themes: large creatures, power matters, creature tokens, primal and savage effects."
                elif sorted_colors == ['B', 'G', 'R']:  # Jund
                    color_guidance += " Jund themes: devour mechanics, large threats, resource conversion, predatory nature."
                elif sorted_colors == ['R', 'U', 'W']:  # Jeskai
                    color_guidance += " Jeskai themes: prowess and noncreature spells, tempo plays, martial arts and wisdom."
                elif sorted_colors == ['B', 'G', 'W']:  # Abzan
                    color_guidance += " Abzan themes: +1/+1 counters, toughness matters, endurance and resilience, outlast mechanics."
                elif sorted_colors == ['G', 'R', 'U']:  # Temur
                    color_guidance += " Temur themes: ferocious (power 4+), morph mechanics, savage shamanism and elemental power."
                elif sorted_colors == ['B', 'R', 'W']:  # Mardu
                    color_guidance += " Mardu themes: aggressive creatures, warrior tribal, raid mechanics, honor through combat."
                elif sorted_colors == ['G', 'U', 'B']:  # Sultai
                    color_guidance += " Sultai themes: delve and graveyard, self-mill strategies, ruthless ambition and ancient knowledge."
            
            # Four-color combinations
            elif len(colors) == 4:
                missing_color = set(['W', 'U', 'B', 'R', 'G']) - set(colors)
                if 'W' in missing_color:  # UBRG (No White)
                    color_guidance += " Chaos themes: unpredictable effects, transformation, breaking rules, anti-order mechanics."
                elif 'U' in missing_color:  # WBRG (No Blue)
                    color_guidance += " Aggression themes: direct damage, large creatures, immediate threats, anti-control strategies."
                elif 'B' in missing_color:  # WURG (No Black)
                    color_guidance += " Growth themes: ramp effects, creature enhancement, positive development, anti-death mechanics."
                elif 'R' in missing_color:  # WUBG (No Red)
                    color_guidance += " Control themes: card draw, removal, strategic play, methodical and calculated effects."
                elif 'G' in missing_color:  # WUBR (No Green)
                    color_guidance += " Artifice themes: artifact synergies, constructed beings, technology over nature, precise mechanics."
            
            # Five-color (WUBRG)
            elif len(colors) == 5:
                color_guidance += " WUBRG themes: domain effects, all colors matter, converge mechanics, chromatic unity, powerful legendary effects, mana-intensive abilities that showcase mastery over all five colors of magic."
            
            # TYPE-SPECIFIC CONTEXT PIPELINES
            type_specific_guidance = ""
            
            # INSTANT PIPELINE
            if 'instant' in card_type:
                type_specific_guidance = " INSTANT DESIGN: Generate effects that provide immediate answers, reactions, or advantages. Focus on: counterspells, removal spells, combat tricks, pump spells, bounce effects, damage spells, protection, or temporary buffs. Instants should have immediate impact and be reactive in nature. COHESION FOR INSTANTS: Since instants typically have single focused effects, avoid multiple unrelated abilities. If you include multiple effects, they should be closely related (e.g., 'Deal 3 damage, then scry 1' or 'Counter target spell, draw a card'). Common patterns: 'Counter target spell', 'Destroy target creature', 'Target creature gets +X/+X until end of turn', 'Deal X damage to any target', 'Return target permanent to its owner's hand', 'Target creature gains protection from [color] until end of turn'. Keep effects simple and focused - instants are about timing and immediate utility, not complex interactions."
                
                # Instant-specific rarity scaling
                if rarity == 'common':
                    type_specific_guidance += " Common instant: Simple, focused effect with minimal complexity. Examples: basic counterspell, simple buff, or small damage spell."
                elif rarity == 'uncommon':
                    type_specific_guidance += " Uncommon instant: Moderate complexity, potentially with choices or additional effects. Examples: counterspell with card draw, conditional removal, or larger effect."
                elif rarity in ['rare', 'mythic']:
                    type_specific_guidance += " Rare/Mythic instant: Powerful unique effects, potentially game-changing. Examples: powerful counterspells with additional effects, mass effects, or unique utility."
            
            # SORCERY PIPELINE  
            elif 'sorcery' in card_type:
                type_specific_guidance = " SORCERY DESIGN: Generate proactive effects that provide significant impact on your turn. Focus on: tutoring, mass effects, creature tokens, permanent solutions, board development, reanimation, or transformation effects. Sorceries should be powerful but require planning since they're sorcery speed. COHESION FOR SORCERIES: Sorceries can have multiple effects, but they must support a unified strategy. Good themes: token creation + token buffs, reanimation + graveyard filling, ramp + expensive effects, or mass removal + card advantage. Avoid combining unrelated effects like 'Create tokens + Counter next spell + Gain life'. Common patterns: 'Search your library for...', 'Destroy all creatures', 'Create X creature tokens', 'Return target card from graveyard to hand', 'Transform target creature', 'Put a creature from your graveyard onto the battlefield'. Sorceries can be more complex than instants since timing isn't critical."
                
                # Sorcery-specific rarity scaling
                if rarity == 'common':
                    type_specific_guidance += " Common sorcery: Straightforward effects like simple creature tokens, basic removal, or minor utility effects."
                elif rarity == 'uncommon':
                    type_specific_guidance += " Uncommon sorcery: Moderate complexity with choices or multiple effects. Can affect multiple targets or have additional benefits."
                elif rarity in ['rare', 'mythic']:
                    type_specific_guidance += " Rare/Mythic sorcery: Powerful unique effects that can significantly impact the game state. Mass effects, powerful tutoring, or unique mechanics."
            
            # ARTIFACT PIPELINE
            elif 'artifact' in card_type:
                type_specific_guidance = " ARTIFACT DESIGN: Generate effects that provide ongoing utility, activated abilities, or passive benefits. Artifacts are colorless and should feel mechanical/technological. Focus on: activated abilities with costs ({T}:, {1}:, etc.), static effects that modify the game, or utility functions. COHESION FOR ARTIFACTS: Artifacts should have a clear purpose or theme. Good themes: mana production + mana sinks, sacrifice artifacts + artifact recursion, +1/+1 counters + counter synergies, or card selection + card advantage. Avoid random combinations like 'Tap for mana + Flying creatures + Graveyard removal'. Common patterns: '{T}: Add one mana of any color', '{2}, {T}: Card selection effect', 'Creatures you control get +1/+1', '{1}, Sacrifice ~: Deal 2 damage to any target'. Artifacts often have multiple modes of use or ongoing value."
                
                # Artifact-specific types - check both subtype and full typeline
                full_type = (card_data.get('typeLine') or '').lower()
                subtype = (card_data.get('subtype') or '').lower()
                
                if 'equipment' in subtype or 'equipment' in full_type:
                    type_specific_guidance += " EQUIPMENT: MANDATORY - All Equipment MUST have an 'Equip {cost}' ability (e.g., 'Equip {1}', 'Equip {2}', 'Equip {3}', etc.). Focus on 'Equipped creature gets/has...' effects that enhance creatures with stats, keywords, or abilities. The equip cost is essential and required for all Equipment. EXAMPLE FORMAT: 'Equipped creature gets +2/+1 and has flying. Equip {3}' or 'Equipped creature gets +1/+1. Whenever equipped creature deals combat damage to a player, draw a card. Equip {2}'. Notice the pattern: [Enhancement effect] + [Optional triggered/static ability] + [Equip cost]. The equip ability always comes LAST."
                elif 'vehicle' in subtype or 'vehicle' in full_type:
                    type_specific_guidance += " VEHICLE MANDATORY RULES: ALL Vehicles MUST have 'Crew X (Tap any number of creatures you control with total power X or greater: This Vehicle becomes an artifact creature until end of turn.)' ability. CRITICAL FORMAT: Crew is a STANDALONE ability - just 'Crew X', NEVER '{T}, Crew X' or any other cost additions. Examples: 'Crew 2', 'Crew 1', 'Crew 3' (standalone abilities). Common crew costs: Crew 1 (small vehicles), Crew 2 (medium vehicles), Crew 3 (large vehicles), Crew 4+ (huge vehicles). Vehicles start as non-creature artifacts and only become creatures when crewed. They should have strong creature stats (power/toughness) to justify the crew cost. Vehicle abilities should focus on: combat abilities (flying, trample, vigilance), triggered abilities when attacking, or static abilities while crewed. Balance: Higher crew cost = better stats/abilities. NEVER add additional costs to crew abilities - crew activates by tapping other creatures, not the vehicle itself."
                elif 'food' in subtype or 'food' in full_type:
                    type_specific_guidance += " FOOD TOKEN: MANDATORY - All Food tokens MUST have the ability '{2}, {T}, Sacrifice this artifact: You gain 2 life.' This is the defining characteristic of Food tokens as specified by the user. You may add one additional minor ability, but this exact sacrifice ability is required."
                else:
                    type_specific_guidance += " Generic artifact: Utility effects, activated abilities, or static benefits available to all colors."
            
            # ENCHANTMENT PIPELINE
            elif 'enchantment' in card_type:
                type_specific_guidance = " ENCHANTMENT DESIGN: Generate ongoing effects that modify game rules or provide continuous benefits. Enchantments represent magical effects that persist. Focus on: static effects ('Creatures you control have...'), triggered abilities ('Whenever/When...'), or activated abilities that represent magical powers. COHESION FOR ENCHANTMENTS: All abilities should support a unified magical theme. Good themes: creature buffs + creature synergies, graveyard effects + death triggers, mana enhancement + expensive activated abilities, or tribal effects + creature type matters. Avoid random combinations like 'Creature buffs + Land destruction + Card draw + Life gain'. Common patterns: 'Creatures you control get +1/+1', 'Whenever a creature enters the battlefield, ...', '{T}: Target creature gains flying until end of turn', 'At the beginning of your upkeep, ...' Enchantments should feel magical and provide long-term value."
                
                # Enchantment-specific types
                if 'aura' in (card_data.get('subtype') or '').lower():
                    type_specific_guidance += " AURA ENCHANTMENT: MANDATORY - ALL Auras MUST have 'Enchant creature' as their first ability (unless specifically targeting something else like artifacts or lands, but 95% should enchant creatures). REQUIRED FORMAT: Start with 'Enchant creature' followed by effects on 'Enchanted creature gets/has/gains...' Examples: 'Enchant creature. Enchanted creature gets +2/+2', 'Enchant creature. Enchanted creature has flying and vigilance', 'Enchant creature. Enchanted creature gets +1/+1 for each artifact you control'. Auras provide ongoing benefits to the creature they're attached to. Focus on stat boosts, keyword abilities, or special powers for the enchanted creature. Common patterns: +X/+X boosts, keyword abilities (flying, trample, lifelink, etc.), protection abilities, tap/untap effects, or triggered abilities that benefit the enchanted creature."
                elif 'saga' in (card_data.get('subtype') or '').lower():
                    type_specific_guidance += " SAGA: Must have chapter abilities (I, II, III) that tell a story progression. Each chapter should be a triggered ability that activates in sequence."
                else:
                    type_specific_guidance += " Generic enchantment: Ongoing magical effects that modify the game state or provide continuous benefits."
            
            # LAND PIPELINE
            elif 'land' in card_type:
                type_specific_guidance = " LAND DESIGN: Generate mana-producing abilities and/or utility effects. Lands are the foundation of Magic's resource system. Focus on: mana generation ('{T}: Add {color}'), activated abilities with costs, or utility functions. Most lands should produce mana as their primary function. Common patterns: '{T}: Add {W}', '{T}: Add one mana of any color', '{1}, {T}: Draw a card', '{T}: Target creature gets +1/+0 until end of turn'. Utility lands should have higher activation costs to balance their additional effects."
                
                # Land-specific rarity scaling
                if rarity == 'common':
                    type_specific_guidance += " Common land: Simple mana production, possibly with basic utility. Examples: basic lands, simple dual lands, or lands with minor activated abilities."
                elif rarity == 'uncommon':
                    type_specific_guidance += " Uncommon land: Dual mana production or useful activated abilities. Balance mana fixing with utility effects."
                elif rarity in ['rare', 'mythic']:
                    type_specific_guidance += " Rare/Mythic land: Powerful utility effects or unique mana abilities. Can have complex activated abilities or game-changing effects."
            
            # PLANESWALKER PIPELINE
            elif 'planeswalker' in card_type:
                type_specific_guidance = " PLANESWALKER DESIGN: Generate loyalty abilities that represent a powerful ally. Planeswalkers have starting loyalty and 2-4 abilities with loyalty costs. Format as '+X: [effect]', '-X: [effect]', and optional ultimate '-X: [powerful effect]'. First ability should be positive or neutral loyalty, middle ability(ies) should cost loyalty for stronger effects, ultimate should be game-changing but expensive. Common patterns: '+1: Draw a card', '-2: Deal 3 damage to any target', '-7: You get an emblem with...'. Each ability should feel distinct and flavorful to the character."
                
                # Planeswalker complexity by rarity
                if rarity in ['rare', 'mythic']:
                    type_specific_guidance += " Rare/Mythic planeswalker: 3-4 abilities including a powerful ultimate. Starting loyalty 3-5. Abilities should synergize and tell a story."
                else:
                    type_specific_guidance += " Uncommon planeswalker: 2-3 abilities, simpler effects. Starting loyalty 2-4. Focus on utility rather than game-ending effects."
            
            # BATTLE PIPELINE
            elif 'battle' in card_type:
                type_specific_guidance = " BATTLE DESIGN: Generate effects that trigger when the battle enters or is defeated. Battles start with defense counters and have effects when they transform or are defeated. Focus on: enter-the-battlefield effects, static effects while on battlefield, and powerful 'when this battle is defeated' triggers. Common patterns: 'When ~ enters the battlefield, ...', 'Whenever ~ loses a defense counter, ...', 'When ~ is defeated, ...'. Battles should feel like epic conflicts with meaningful rewards for defeating them."
            
            # Enhanced creature-specific guidance with stats balancing
            creature_guidance = ""
            
            # Flying restrictions for ALL creatures (not just legendary)
            creature_flying_guidance = ""
            if 'creature' in card_type:
                subtype = card_data.get('subtype', '')
                if subtype:
                    subtype_lower = subtype.lower()
                    
                    # Use flying restrictions from config
                    
                    if any(restricted_type in subtype_lower for restricted_type in flying_restricted_types):
                        creature_flying_guidance = f" FLYING RESTRICTION: {subtype} creatures should rarely have flying unless there's a compelling magical or mechanical reason. Prioritize grounded keywords like trample, vigilance, first strike, deathtouch, reach, menace, or lifelink instead of flying."
                    elif any(flying_type in subtype_lower for flying_type in flying_encouraged_types):
                        creature_flying_guidance = f" FLYING NATURAL: {subtype} creatures are natural fliers and flying is highly appropriate for this creature type."
            
            if 'creature' in card_type and power and toughness:
                try:
                    p = int(power) if power.isdigit() else 0
                    t = int(toughness) if toughness.isdigit() else 0
                    stat_total = p + t
                    
                    # More aggressive stat-based ability limiting
                    if stat_total >= 10:  # Large creatures like 6/6, 5/5, etc.
                        creature_guidance = " This creature has very large stats, so limit to AT MOST 0-2 simple abilities (prefer keywords like Trample, Vigilance). Avoid complex activated abilities."
                    elif stat_total >= 7:  # Medium-large creatures like 4/4, 3/4, etc.
                        creature_guidance = " This creature has large stats for its cost, so abilities should be minimal - prefer 0-2 keywords or one simple triggered ability. Avoid multiple activated abilities."
                    elif stat_total >= 5:  # Average creatures
                        if stat_total > cmc * 2.2:  # Above-rate stats
                            creature_guidance = " This creature has above-average stats, so limit abilities to 0-2 simple ones (mostly keywords)."
                        else:
                            creature_guidance = " This creature has moderate stats, so it can have 2-3 balanced abilities."
                    elif stat_total < cmc * 1.5:  # Below-rate stats
                        creature_guidance = " This creature has low stats for its cost, so it should have multiple powerful abilities (2-4 depending on rarity) to compensate."
                    else:
                        creature_guidance = " This creature has balanced stats, so it can have moderate utility abilities."
                        
                    # Special case for defensive creatures (high toughness, low power)
                    if t >= 5 or (t > p and t >= 3):
                        creature_guidance += " This is a defensive creature - consider abilities like Defender, Wall synergies, or activated abilities that don't require attacking."
                        
                except:
                    pass
            
            # Legendary creature limitation with name/subtype awareness
            legendary_guidance = ""
            if is_legendary and 'creature' in card_type:
                card_name = card_data.get('name', '')
                subtype = card_data.get('subtype', '')
                
                legendary_guidance = " LEGENDARY CONSTRAINT: This is a legendary creature - structure abilities as follows: "
                
                # Power level determines ability structure - FOLLOW SAME RARITY LIMITS AS NON-LEGENDARY
                # Legendary status does not grant extra abilities, just unique flavor
                if rarity == 'mythic':
                    legendary_guidance += "Mythic legendary: Maximum 3-4 total abilities including keywords, triggered, and activated abilities. Make abilities feel unique and build-around worthy."
                elif rarity == 'rare':
                    legendary_guidance += "Rare legendary: Maximum 3 total abilities including keywords. Focus on unique mechanics that feel special."
                elif rarity == 'uncommon':
                    legendary_guidance += "Uncommon legendary: Maximum 2 total abilities including keywords. Simple but memorable effects."
                else:  # common
                    legendary_guidance += "Common legendary: Maximum 1 total ability - either ONE keyword OR one simple triggered ability. Being legendary doesn't grant extra complexity."
                
                # Add name and subtype flavor guidance
                if card_name:
                    legendary_guidance += f" IMPORTANT: This creature is named '{card_name}' - design abilities that reflect this specific character's identity, personality, and lore. Make the abilities feel unique to this individual."
                
                if subtype:
                    # Add subtype-specific ability suggestions
                    subtype_lower = subtype.lower()
                    
                    if 'dragon' in subtype_lower:
                        legendary_guidance += " As a Dragon, consider abilities like flying, dealing damage, treasure generation, or breath weapon effects."
                    elif 'angel' in subtype_lower:
                        legendary_guidance += " As an Angel, consider abilities like flying OR vigilance (not both), lifegain, protection effects, flash, hexproof, or helping other creatures."
                    elif 'demon' in subtype_lower:
                        legendary_guidance += " As a Demon, consider abilities like flying, menace, sacrifice effects, life drain, or punishing opponents."
                    elif 'beast' in subtype_lower:
                        legendary_guidance += " As a Beast, consider abilities like trample, fighting other creatures, +1/+1 counters, or natural/primal effects."
                    elif 'wizard' in subtype_lower or 'mage' in subtype_lower:
                        legendary_guidance += " As a Wizard/Mage, consider abilities related to spells, card draw, instant/sorcery synergies, or magical effects."
                    elif 'warrior' in subtype_lower or 'soldier' in subtype_lower:
                        legendary_guidance += " As a Warrior/Soldier, consider abilities like first strike, vigilance, combat bonuses, or military tactics."
                    elif 'rogue' in subtype_lower or 'assassin' in subtype_lower:
                        legendary_guidance += " As a Rogue/Assassin, consider abilities like deathtouch, unblockable, card advantage through sneaky means, or removal effects."
                    elif 'knight' in subtype_lower:
                        legendary_guidance += " As a Knight, consider abilities like first strike, vigilance, protection effects, or honor-based abilities."
                    elif 'spirit' in subtype_lower:
                        legendary_guidance += " As a Spirit, consider abilities like flying, phasing, graveyard interactions, or ethereal effects."
                    elif 'elemental' in subtype_lower:
                        legendary_guidance += " As an Elemental, consider abilities related to basic lands, elemental forces, or effects that match your colors (fire=damage, water=card draw, etc.)."
                    elif 'vampire' in subtype_lower:
                        legendary_guidance += " As a Vampire, consider abilities like lifelink, flying, life drain effects, or graveyard recursion."
                    elif 'zombie' in subtype_lower:
                        legendary_guidance += " As a Zombie, consider abilities like deathtouch, graveyard recursion, sacrifice synergies, or undeath effects."
                    elif 'elf' in subtype_lower:
                        legendary_guidance += " As an Elf, consider abilities like mana generation, creature synergies, forest/nature effects, or tribal bonuses."
                    elif 'goblin' in subtype_lower:
                        legendary_guidance += " As a Goblin, consider abilities like haste, direct damage, artifact interactions, or chaotic/random effects."
                    elif 'human' in subtype_lower:
                        legendary_guidance += " As a Human, consider versatile abilities that could represent leadership, innovation, adaptability, or cooperation with other creatures."
                    
                    legendary_guidance += f" The subtype '{subtype}' should strongly influence the flavor and mechanics of the abilities."
                
                # Use flying restrictions from config
                
                if any(restricted_type in subtype_lower for restricted_type in flying_restricted_types):
                    legendary_guidance += f" FLYING RESTRICTION: {subtype} creatures rarely have flying unless there's a specific magical reason (enchantment, spell effect, etc.). Consider grounded abilities like trample, first strike, vigilance, deathtouch, or reach instead."
                elif any(flying_type in subtype_lower for flying_type in flying_encouraged_types):
                    legendary_guidance += f" FLYING ENCOURAGED: {subtype} creatures naturally fly and should strongly consider having flying as an ability."
            
            # Add card name inspiration for creative abilities
            name_inspiration = ""
            card_name = (card_data.get('name') or '').strip()
            if card_name and len(card_name) > 2:
                name_inspiration += f" CARD NAME INSPIRATION: This card is named '{card_name}' - use this name as creative inspiration for unique abilities. Extract thematic concepts from the name: if it mentions elements (fire, ice, storm), create elemental effects; if it mentions creatures (dragon, angel, demon), incorporate those creature themes; if it mentions objects (sword, tome, crown), design abilities that reflect those items; if it mentions actions (strike, whisper, shatter), create abilities based on those verbs; if it mentions places (tower, grove, sanctum), include location-based effects. Make the abilities feel specifically tied to this card's identity, not generic effects."
            
            enhanced_prompt += f" The card costs {cmc} mana and should be {power_level}.{color_guidance}{type_specific_guidance}{creature_guidance}{creature_flying_guidance}{legendary_guidance}{name_inspiration}{x_guidance}"
        
        # Add CMC guidance that respects rarity limits (rarity limits take precedence)
        if card_data and 'creature' in card_data.get('type', '').lower():
            # CMC provides flavor guidance but CANNOT exceed rarity ability limits
            if cmc <= 1:
                enhanced_prompt += f" CMC GUIDANCE: 1 mana creatures prefer efficient, simple effects within your {rarity} rarity limit. Favor keywords like Haste, Deathtouch, Menace, Reach, or Lifelink. Avoid activated abilities on cheap creatures."
            elif cmc == 2:
                enhanced_prompt += f" CMC GUIDANCE: 2 mana creatures work well with keywords or simple triggers within your {rarity} rarity limit. Examples: single keywords or 'When this enters the battlefield' effects."
            elif cmc == 3:
                enhanced_prompt += f" CMC GUIDANCE: 3 mana creatures can support moderate complexity within your {rarity} rarity limit. Consider activated abilities like '{{T}}: Add mana' or utility effects."
            elif cmc <= 5:
                enhanced_prompt += f" CMC GUIDANCE: 4-5 mana creatures justify more abilities within your {rarity} rarity limit. Can support activated abilities and synergistic effects."
            else:
                enhanced_prompt += f" CMC GUIDANCE: High-cost creatures (6+ mana) should feel impactful within your {rarity} rarity limit. Focus on game-changing effects appropriate for the mana investment."
            
            enhanced_prompt += " REMEMBER: Activated abilities (costs like {T}:, {1}:) are the most complex. Keyword abilities (Haste, Trample, Deathtouch, Menace, Lifelink, Reach, etc.) and triggered abilities (When/Whenever) are simpler. Lower mana cost = fewer and simpler abilities. KEYWORD SELECTIVITY: Flying should only be given to creatures that logically can fly (dragons, angels, birds, spirits, etc.). Ground-based creatures like humans, elves, goblins, beasts, and warriors should use other keywords like trample, vigilance, first strike, deathtouch, reach, menace, or lifelink. VARIETY: Consider diverse keyword combinations beyond the overused 'Flying + Vigilance' pairing. CREATURE COHESION: All abilities must work together thematically. Good creature themes include: aggressive (Haste + Trample + attack benefits), defensive (Vigilance + blocking rewards), graveyard-focused (death triggers + graveyard recursion), token-maker (creates tokens + sacrifice outlets), tribal (creature type synergies), or utility (mana abilities + activated effects). AVOID mixing unrelated mechanics like 'Flying + Graveyard recursion + Mana production + Life gain' - pick 1-2 related themes. IMPORTANT: Multiple keywords should be comma-separated on one line (like 'Trample, menace'), not on separate lines."
        
        # Type-specific formatting instructions
        if 'planeswalker' in card_type:
            enhanced_prompt += " PLANESWALKER FORMATTING: Generate 2-4 loyalty abilities in the format '+X: [effect]', '0: [effect]', or '-X: [effect]'. List each ability on its own line. Include starting loyalty as the first line like 'Starting loyalty: 3'. Example format: 'Starting loyalty: 3\\n+1: Draw a card\\n-2: Deal 3 damage to any target\\n-7: You get an emblem with \"Creatures you control have flying\"'."
        elif 'instant' in card_type or 'sorcery' in card_type:
            enhanced_prompt += " INSTANT/SORCERY FORMATTING: Generate spell effects that happen when cast. Keep effects concise and focused. Example formats: 'Counter target spell', 'Destroy target creature', 'Draw three cards', 'Create two 1/1 creature tokens', 'Deal 4 damage to any target'. Use standard Magic spell language."
        elif 'land' in card_type:
            enhanced_prompt += " LAND FORMATTING: Focus on mana abilities and utility effects. Format activated abilities with proper costs. Example formats: '{T}: Add {W}', '{T}: Add one mana of any color', '{1}, {T}: Draw a card', '{T}: Target creature gets +1/+0 until end of turn'."
        else:
            enhanced_prompt += " FORMATTING: Keywords should be comma-separated on ONE line (like 'Trample, menace'), while other abilities use separate lines. For activated abilities use format '{cost}: {effect}'. For triggered abilities use 'When/Whenever/At' format. Example: 'Trample, menace\\n{T}: Add one mana of any color\\n{2}: Target creature gains first strike until end of turn'."
        
        enhanced_prompt += (" CREATIVITY AND UNIQUENESS REQUIREMENTS: 1) AVOID OVERUSED GENERIC ABILITIES: Never use these repetitive effects: 'Draw 3 cards', 'Draw a card', '{T}: Add one mana of any color', 'Tap: Create 1 mana', 'When this enters the battlefield, draw a card', 'Sacrifice this: Draw a card'. These are boring and overused. "
                           "2) FOCUS ON THE CARD'S IDENTITY: Use the card's name, type, and power level as inspiration. If the card is named 'Sword of Fire', create fire-themed combat abilities. If it's called 'Ancient Tome', focus on knowledge/library effects, not generic card draw. If it's a 'Dragon Engine', combine draconic and mechanical themes. "
                           "3) PRIORITIZE UNIQUE MECHANICS: Instead of generic effects, create interesting abilities like: temporary creature theft, conditional countering, combat phase manipulation, alternate win conditions, unique token creation, innovative triggered conditions, creative activated abilities that aren't just mana generation, spell copying with twists, unique protection effects, interesting sacrifice effects, creative pump effects, unique evasion beyond flying. "
                           "4) MAKE IT MEMORABLE: Every ability should feel distinctive and tied to the card's concept. VARIETY REQUIREMENT: Avoid overused effects like 'draw cards' and 'add mana' - instead prioritize diverse, creative effects that match the card's colors and type. Explore unique mechanics, interesting interactions, and varied effect types. "
                           "COHESION REQUIREMENT: All abilities on a single card must work together thematically and mechanically. Do NOT combine random unrelated abilities. Instead, create cards with unified themes such as: sacrifice synergies (sacrifice creatures → get benefits), +1/+1 counter themes (place counters → counter-based benefits), graveyard strategies (mill → graveyard value), tribal synergies (creature types matter), or mana ramp strategies (produce mana → expensive effects). Each ability should support or enhance the others. "
                           "Example of GOOD cohesion: 'When this enters, create two 1/1 tokens' + '{T}, Sacrifice a creature: Draw a card' (token generation supports sacrifice). Example of BAD cohesion: 'Flying' + '{T}: Add mana' + 'Whenever a creature dies, gain 2 life' + 'Discard a card: Deal 1 damage' (random unrelated abilities). "
                           "🚨🚨🚨 CRITICAL OUTPUT FORMAT - RULES TEXT ONLY 🚨🚨🚨: You MUST generate ONLY the rules text that goes INSIDE the text box. ABSOLUTELY DO NOT INCLUDE: \n"
                           "❌ Card name (like 'Pogo, Pokemon Master')\n"
                           "❌ Card type line (like 'Legendary Creature - Human Tamer')\n"
                           "❌ Mana cost (like '{3}{U}{U}')\n"
                           "❌ Power/Toughness (like '3/4')\n"
                           "❌ Flavor text\n"
                           "❌ Set symbols\n"
                           "❌ Any descriptive text about the card\n"
                           "✅ ONLY generate the abilities and rules text that would appear in the rules text box\n"
                           "If the card title appears in the rules text, you may use it there (like 'Pogo's power is equal to...')\n"
                           "Your ENTIRE response should be ONLY abilities like: 'Flying', 'Vigilance', 'When this enters the battlefield...', '{T}: Add {G}', etc.\n"
                           "MANDATORY QUOTE WRAPPING: Each distinct ability must be wrapped in double quotes to prevent parsing errors. This is CRITICAL for proper card rendering. Each complete ability (from start to end, including all sentences that belong together) should be enclosed in quotes. "
                           "IMPORTANT: Use {T} for tap symbol, never write 'Tap:'. Use standard Magic card formatting. NEVER include ability type labels like 'Triggered Ability:', 'Passive Ability:', 'Active Ability:', 'Keywords:', etc. Just write the abilities directly. "
                           "Example of CORRECT output: '\"Flying, trample\"' or '\"Flying, trample\" \"Whenever this creature attacks, gain 2 life\"' or '\"Flying, trample\" \"{T}: Add one mana of any color\" \"When this enters the battlefield, create a 1/1 token\"'. "
                           "Example of INCORRECT output: 'Pogo, Pokemon Master - Legendary Creature - Human Tamer. Sacrifice another creature: You gain control...' (includes name and type)\n"
                           "Example of INCORRECT output with labels: 'Keywords: Flying\\nTriggered Ability: When this enters, draw a card'. "
                           "Each ability must be in its own quoted section - this prevents multi-sentence abilities from being split incorrectly during parsing. Generate ONLY the quoted abilities as they would appear on an actual Magic card.")
        
        # Generate and validate rules text (retry if contaminated)
        max_attempts = 3
        card_text = ""
        
        for attempt in range(max_attempts):
            response = ollama.generate(
                model='mistral:latest',
                prompt=enhanced_prompt
            )
            
            # Clean up the response
            card_text = response['response'].strip()
            print(f"📜 Content model raw output: {repr(card_text)}")
            
            # Strip any non-rules text that might have been included
            card_text = strip_non_rules_text(card_text, card_data)
            print(f"🧹 After stripping non-rules text: {repr(card_text)}")
            
            # Validate the response doesn't contain type line elements
            if validate_rules_text(card_text, card_data):
                break
            else:
                print(f"❌ Rules text validation failed on attempt {attempt + 1}, regenerating...")
                if attempt < max_attempts - 1:
                    # Add additional constraint for retry
                    enhanced_prompt += f" CRITICAL: Do NOT include type line elements like '{card_data.get('type', '')}' or '{card_data.get('subtype', '')}' in the rules text. Generate ONLY the abilities text."
                else:
                    print(f"⚠️  Max validation attempts reached, using last generated text")
                    break
        
        # Remove surrounding quotes if present
        if (card_text.startswith('"') and card_text.endswith('"')) or \
           (card_text.startswith("'") and card_text.endswith("'")):
            card_text = card_text[1:-1].strip()
        
        # Fix common formatting issues
        # Replace literal \\n with actual newlines
        card_text = card_text.replace('\\n', '\n')
        
        # Replace "Tap:" with "{T}:" for tap symbols
        card_text = card_text.replace('Tap:', '{T}:')
        
        # Remove any ability type labels that might have slipped through
        ability_label_patterns = [
            r'Triggered Ability:\s*',
            r'Passive Ability:\s*',
            r'Active Ability:\s*',
            r'Keyword Ability:\s*',
            r'Keywords:\s*',
            r'Abilities:\s*',
            r'Static Ability:\s*',
            r'Activated Ability:\s*',
            r'Ability:\s*'
        ]
        
        for pattern in ability_label_patterns:
            card_text = re.sub(pattern, '', card_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up any extra whitespace or newlines created by label removal
        card_text = re.sub(r'\n\s*\n', '\n', card_text).strip()
        
        # Convert double newlines to single newlines
        card_text = card_text.replace('\n\n', '\n')
        
        # Replace ~ symbol with actual card name (simple global replacement)
        if card_data and card_data.get('name'):
            card_name = (card_data.get('name') or '').strip()
            if card_name:  # Only replace if we have a valid card name
                card_text = card_text.replace('~', card_name)
        
        # Ensure each ability ends with a period
        card_text = ensure_periods_on_abilities(card_text)
        
        # Add newlines before ability costs that come after periods (not in quotes)
        card_text = format_ability_newlines(card_text)
        
        # Limit to 3-4 sentences by splitting on periods and taking first 4
        sentences = [s.strip() for s in card_text.split('.') if s.strip()]
        print(f"🔍 Found {len(sentences)} sentences: {sentences}")
        
        # Remove card name elements (sentences that are just the card name with no abilities)
        filtered_sentences = []
        card_name = card_data.get('name', '') if card_data else ''
        print(f"   Card name: '{card_name}'")
        
        for i, sentence in enumerate(sentences):
            print(f"   Processing sentence {i}: '{sentence}'")
            
            # Skip if this sentence is just the card name (with possible quotes and prefixes)
            sentence_clean = sentence.strip().strip('\'"*').strip()
            print(f"   Cleaned version: '{sentence_clean}'")
            
            
            if sentence_clean == card_name:
                print(f"🗑️  Removed card name element: '{sentence}'")
                continue
            
            # Also remove card name from the beginning of sentences if it appears there
            if card_name and sentence_clean.startswith(card_name):
                print(f"   Found card name at start of sentence!")
                # Remove the card name from the beginning
                remaining_text = sentence_clean[len(card_name):].strip()
                print(f"   Remaining text after removing name: '{remaining_text}'")
                # Skip if nothing meaningful remains after removing the name
                if not remaining_text or remaining_text in ['"', "'", '"\n', "'\n"]:
                    print(f"🗑️  Removed card name prefix: '{sentence}'")
                    continue
                # Keep the sentence but without the card name prefix
                original_sentence = sentence
                sentence = sentence.replace(f'"{card_name}"', '').replace(f"'{card_name}'", '').replace(card_name, '').strip()
                if sentence.startswith('\n'):
                    sentence = sentence[1:].strip()
                print(f"🧹 Cleaned card name from sentence")
                print(f"   Before: '{original_sentence}'")
                print(f"   After: '{sentence}'")
            
            # Skip empty sentences or sentences that are just quotes/whitespace
            sentence_meaningful = sentence.strip().strip('\'"').strip()
            if not sentence_meaningful:
                print(f"🗑️  Removed empty/quote-only sentence: '{sentence}'")
                continue
                
            # Clean up leading quote fragments and newlines
            sentence = sentence.strip()
            if sentence.startswith("'\n") or sentence.startswith('"\n'):
                sentence = sentence[2:]  # Remove quote + newline
                print(f"🧹 Removed leading quote+newline")
            elif sentence.startswith('\n'):
                sentence = sentence[1:]  # Remove just newline
                print(f"🧹 Removed leading newline")
                
            filtered_sentences.append(sentence)
        
        sentences = filtered_sentences
        print(f"🔍 After filtering: {len(sentences)} sentences: {sentences}")
        
        if len(sentences) > 4:
            print(f"⚠️  Truncating from {len(sentences)} to 4 sentences")
            card_text = '. '.join(sentences[:4]) + '.'
        
        # UNIFIED SANITATION PIPELINE - Applied to ALL card types
        if card_data:
            card_type = card_data.get('type', '').lower()
            
            # Step 1: Type-specific ability limits
            if 'creature' in card_type:
                card_text = limit_creature_active_abilities(card_text)
            elif 'planeswalker' in card_type:
                # Planeswalkers should have proper loyalty ability format
                card_text = sanitize_planeswalker_abilities(card_text)
            elif 'instant' in card_type or 'sorcery' in card_type:
                # Spells should have single cohesive effects
                card_text = sanitize_spell_abilities(card_text)
            elif 'land' in card_type:
                # Lands should focus on mana abilities
                card_text = sanitize_land_abilities(card_text)
            elif 'artifact' in card_type or 'enchantment' in card_type:
                # Artifacts/enchantments get general permanent sanitation
                card_text = sanitize_permanent_abilities(card_text)
            
            # Step 2: Universal ability reordering (ALL card types benefit from proper ordering)
            print(f"🔍 Before reordering: {repr(card_text)}")
            card_text = reorder_abilities_properly(card_text, card_data)
            print(f"🔍 After reordering: {repr(card_text)}")
            
            # Step 3: Universal complexity limits (prevent overpowered cards)
            card_text = apply_universal_complexity_limits(card_text, card_data)
        
        # Final validation after all processing (check for post-processing issues)
        if not validate_rules_text(card_text, card_data):
            print(f"⚠️  Final validation failed after text processing - issues introduced during formatting")
            print(f"⚠️  Processed text: {repr(card_text)}")
            # For now, return the text anyway, but log the issue
            # TODO: Could implement full regeneration loop here if needed
        
        print(f"🧠 createCardContent returning:")
        print(f"   📝 Result: {repr(card_text)}")
        print(f"   📏 Length: {len(card_text) if card_text else 0}")
        return card_text
        
    except Exception as e:
        print(f"❌ Error in createCardContent: {e}")
        print("Make sure Mistral model is installed: 'ollama pull mistral:latest'")
        import traceback
        traceback.print_exc()
        return None