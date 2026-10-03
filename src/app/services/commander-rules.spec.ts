import { Card, Rarity } from '../models/card.model';
import { setView } from '../testing/fixtures';
import {
  COMMANDER_CMCS, COMMANDER_RARITIES, ChecklistItem, autoStats, commanderChecklist, commanderCost, commanderPipValue,
  commanderStatsError, creatureTypeError, groupByCmc,
  statPoints
} from './commander-rules';

const WHOLE_NUMBERS = 'P/T must be whole numbers — X and * aren\'t allowed';

describe('commander rules', () => {
  it('has the three mana values and rarities', () => {
    expect(COMMANDER_CMCS).toEqual([3, 4, 5]);
    expect(COMMANDER_RARITIES).toEqual([Rarity.UNCOMMON, Rarity.RARE, Rarity.MYTHIC]);
  });

  it('statPoints is CMC + 1, plus 2 for a Vehicle', () => {
    expect(statPoints(3, 'creature')).toBe(4);
    expect(statPoints(5, 'creature')).toBe(6);
    expect(statPoints(3, 'vehicle')).toBe(6);
  });

  it('commanderPipValue counts colored pips; {2/W} is 2', () => {
    expect(commanderPipValue('{2}{W}{U}')).toBe(2);
    expect(commanderPipValue('{X}{2/W}{G}')).toBe(3);
    expect(commanderPipValue('')).toBe(0);
  });

  it('blank P/T is auto (no error)', () => {
    expect(commanderStatsError('', '', 3, 'creature')).toBeNull();
  });

  it('accepts P/T within the budget', () => {
    expect(commanderStatsError('3', '1', 3, 'creature')).toBeNull();
    expect(commanderStatsError('4', '2', 3, 'vehicle')).toBeNull();
    expect(commanderStatsError(' 2 ', '02', 3, 'creature')).toBeNull();
  });

  it('rejects P/T over the budget with the server message', () => {
    expect(commanderStatsError('3', '2', 3, 'creature'))
      .toBe('A 3-mana commander has 4 points; 3/2 uses 5');
  });

  it('rejects X, * and non-whole numbers', () => {
    expect(commanderStatsError('*', '*', 3, 'creature')).toBe(WHOLE_NUMBERS);
    expect(commanderStatsError('X', '2', 3, 'creature')).toBe(WHOLE_NUMBERS);
    expect(commanderStatsError('1.5', '1', 3, 'creature')).toBe(WHOLE_NUMBERS);
  });

  it('rejects toughness 0 and a missing half', () => {
    expect(commanderStatsError('2', '0', 3, 'creature')).not.toBeNull();
    expect(commanderStatsError('', '3', 3, 'creature')).not.toBeNull();
  });

  it('commanderCost keeps the pips and pads generic to the CMC, like the server', () => {
    expect(commanderCost('{W}{U}', 4)).toBe('{2}{W}{U}');
    expect(commanderCost('{X}{4}{B}{B}', 5)).toBe('{3}{B}{B}');
    expect(commanderCost('', 3)).toBe('{3}');
    expect(commanderCost('{W}{U}{B}', 3)).toBe('{W}{U}{B}');
    expect(commanderCost('{2/W}{G}', 3)).toBe('{2/W}{G}');
  });

  it('autoStats spends every point and leans by creature type, like the server', () => {
    expect(autoStats(3, 'creature', 'Human')).toEqual([2, 2]);
    expect(autoStats(4, 'creature', 'Human')).toEqual([2, 3]);
    expect(autoStats(5, 'creature', 'Goblin')).toEqual([4, 2]);
    expect(autoStats(5, 'vehicle', 'Vehicle')).toEqual([4, 4]);
    expect(autoStats(4, 'creature', 'Wall')).toEqual([1, 4]);
  });

  it('creatureTypeError accepts creature types and names a word that is not one', () => {
    expect(creatureTypeError('Human Wizard')).toBeNull();
    expect(creatureTypeError('')).toBeNull();
    expect(creatureTypeError('Equipment')).toBe('Equipment isn\'t a creature type');
    expect(creatureTypeError('Human Vehicle')).toBe('Vehicle isn\'t a creature type');
    expect(creatureTypeError('instant')).toBe('instant isn\'t a creature type');
  });

  describe('commanderChecklist', () => {
    const card = (overrides: Partial<Card> = {}): Card => ({
      name: '', manaCost: '{W}{U}', colors: ['W', 'U'], type: 'Creature', supertype: 'Legendary',
      subtype: 'Human', cmc: 4, rarity: Rarity.RARE, commanderKind: 'creature', power: '', toughness: '',
      ...overrides
    });
    const check = (name: string, c: Card | null, cmc = 4, taken: Partial<Record<Rarity, number>> = {}) =>
      Object.fromEntries(commanderChecklist(name, c, cmc, taken).map(i => [i.id, i])) as
        Record<ChecklistItem['id'], ChecklistItem>;

    it('passes a legal commander, with a line per rule', () => {
      const items = commanderChecklist('Zur', card(), 4, {});
      expect(items.map(i => i.id)).toEqual(['name', 'mana', 'type', 'body', 'rarity']);
      expect(items.every(i => i.ok)).toBeTrue();
      expect(items.find(i => i.id === 'mana')!.detail).toBe('{2}{W}{U}');
    });

    it('fails a blank or too-long name', () => {
      expect(check('  ', card()).name.ok).toBeFalse();
      expect(check('x'.repeat(41), card()).name.ok).toBeFalse();
    });

    it('fails pips worth more than the mana value', () => {
      const c = check('Zur', card({ manaCost: '{W}{W}{U}{U}{B}' }), 4);
      expect(c.mana.ok).toBeFalse();
      expect(c.mana.detail).toBe('Colored pips add up to 5 mana');
      expect(check('Zur', card({ manaCost: '{W}{W}{U}{U}{B}' }), 5).mana.ok).toBeTrue();
    });

    it('accepts only a Legendary Creature of creature types or a Legendary Artifact — Vehicle', () => {
      expect(check('Zur', card({ subtype: 'Equipment' })).type.ok).toBeFalse();
      expect(check('Zur', card({ subtype: 'Equipment' })).type.detail).toBe('Equipment isn\'t a creature type');
      expect(check('Zur', card({ type: 'Instant' })).type.ok).toBeFalse();
      expect(check('Zur', card({ supertype: '' })).type.ok).toBeFalse();
      const vehicle = card({ type: 'Artifact', subtype: 'Vehicle', commanderKind: 'vehicle' });
      expect(check('Zur', vehicle).type.ok).toBeTrue();
      expect(check('Zur', vehicle).type.detail).toBe('Legendary Artifact — Vehicle');
      expect(check('Zur', { ...vehicle, subtype: 'Construct Vehicle' }).type.ok).toBeFalse();
    });

    it('fails a body over the points, and says how many points there are', () => {
      expect(check('Zur', card({ power: '3', toughness: '3' })).body.ok).toBeFalse();
      expect(check('Zur', card({ power: '2', toughness: '3' })).body.ok).toBeTrue();
      expect(check('Zur', card()).body.label).toContain('5 points');
    });

    it('fails a rarity your other commander is locked in with, or a Common', () => {
      expect(check('Zur', card(), 4, { rare: 3 }).rarity.ok).toBeFalse();
      expect(check('Zur', card(), 4, { rare: 3 }).rarity.detail).toBe('Your 3 CMC commander is locked in as Rare');
      expect(check('Zur', card({ rarity: Rarity.COMMON })).rarity.ok).toBeFalse();
      expect(check('Zur', card({ rarity: '' as Rarity })).rarity.detail).toBe('Pick a rarity');
    });

    it('fails everything but the name when there is no design yet', () => {
      const c = check('Zur', null);
      expect(c.name.ok).toBeTrue();
      expect((['mana', 'type', 'body', 'rarity'] as const).every(id => !c[id].ok)).toBeTrue();
    });
  });

  it('groupByCmc orders 3, 4, 5 then Earlier sets, skipping empty groups', () => {
    const sets = [5, null, 3, 3].map((cmc, i) => setView({ id: `s-${i}`, cmc }));
    const groups = groupByCmc(sets);
    expect(groups.map(g => g.label)).toEqual(['3 CMC', '5 CMC', 'Earlier sets']);
    expect(groups.map(g => g.sets.length)).toEqual([2, 1, 1]);
    expect(groups.map(g => g.cmc)).toEqual([3, 5, null]);
  });
});
