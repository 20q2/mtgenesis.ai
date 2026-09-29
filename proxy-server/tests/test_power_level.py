"""power_level: the creature stat curve, the budget by mana value and rarity, and the
rough mana-value estimate of rules text. Cases are real model outputs from the e2e runs.
Fast: no app import, no Ollama.
"""
import pytest

import power_level as P


def card(mv, rarity='common', type_='Creature', **extra):
    return {'name': 'Test', 'cmc': mv, 'rarity': rarity, 'type': type_, **extra}


@pytest.mark.parametrize('mv, rarity, stats', [
    (1, 'common', (1, 2)), (2, 'common', (2, 2)), (3, 'uncommon', (2, 3)),
    (4, 'rare', (3, 4)), (5, 'common', (4, 5)), (6, 'mythic', (5, 6)),
])
def test_stat_curve(mv, rarity, stats):
    assert P.creature_stats(card(mv, rarity)) == stats


def test_stat_curve_leans_by_creature_type():
    assert P.creature_stats(card(4, subtype='Goblin Warrior')) == (4, 3)
    assert P.creature_stats(card(4, subtype='Wall')) == (2, 5)


def test_budgets_grow_with_mana_value_and_rarity():
    assert P.budget(card(2, 'common')) == 0.75            # a 2/2 gets one small bonus
    assert P.budget(card(1, 'common', 'Instant')) == 1.25  # a 1-mana spell: about Shock
    assert P.budget(card(4, 'mythic', 'Planeswalker')) == 7.0
    assert P.budget(card(4, 'uncommon', 'Creature')) < P.budget(card(4, 'uncommon', 'Enchantment'))


@pytest.mark.parametrize('line, low, high', [
    ('Flying, trample', 1.3, 1.5),
    ('This spell deals 2 damage to any target.', 1.0, 1.4),
    ('Destroy target creature with power 3 or less.', 1.4, 1.6),
    ('When this creature enters, scry 2, then draw a card.', 1.0, 1.6),
    # repeating effects cost far more than one-shot ones
    ('At the beginning of your upkeep, create a 1/1 white Soldier creature token.', 2.0, 3.0),
    ('{T}: Create a 1/1 white Soldier creature token.', 2.0, 3.0),
    ('At the beginning of your end step, destroy all creatures with power 2 or less.', 8.0, 12.0),
    ('Whenever a creature you control attacks, create three 1/1 white Soldier creature tokens.', 6.0, 12.0),
])
def test_ability_value(line, low, high):
    assert low <= P.ability_value(line) <= high


def test_known_overpowered_cheap_cards_are_far_over_budget():
    # A 2-mana common that grew every spell and a 2-mana common token engine (both real outputs)
    for text in ["Whenever you cast a spell, put a +1/+1 counter on this creature and it can't be blocked by creatures with power 2 or less this turn.",
                 'When this creature attacks, create three 1/1 white Soldier creature tokens with first strike.']:
        value, allowed = P.assess(text, card(2))
        assert value - allowed > P.WARN_OVER, (text, value, allowed)


def test_fair_cards_are_within_budget():
    for text, c in [('Vigilance', card(2)),
                    ('When this creature dies, create a Treasure token.', card(3)),
                    ('This spell deals 2 damage to any target.', card(1, type_='Instant')),
                    ('Destroy target creature with power 3 or less.', card(3, type_='Sorcery'))]:
        value, allowed = P.assess(text, c)
        assert value - allowed <= P.WARN_OVER, (text, value, allowed)


def test_planeswalkers_count_their_best_ability_fully():
    text = '+1: Scry 2, then draw a card.\n−2: This planeswalker deals 3 damage to any target.\n−7: Draw three cards.'
    lines = text.split('\n')
    best = max(P.ability_value(l.split(':', 1)[1]) for l in lines)
    assert P.estimate(text, card(4, 'mythic', 'Planeswalker')) < sum(P.ability_value(l.split(':', 1)[1]) for l in lines)
    assert P.estimate(text, card(4, 'mythic', 'Planeswalker')) >= best


def test_trim_drops_the_most_valuable_ability_but_keeps_protected_lines():
    lines = ['Enchant creature', 'Enchanted creature gets +1/+1.',
             'At the beginning of your upkeep, destroy all creatures with power 2 or less.']
    trimmed = P.trim_to_budget(lines, card(2, 'uncommon', 'Enchantment', subtype='Aura'), keep_first=1)
    assert trimmed == ['Enchant creature', 'Enchanted creature gets +1/+1.']


def test_budget_line_names_concrete_sizes():
    assert 'almost nothing' not in P.describe_budget(card(2))  # creatures always get a small bonus
    assert 'small bonus' in P.describe_budget(card(2))
    assert 'land' in P.describe_budget(card(0, 'rare', 'Land'))
