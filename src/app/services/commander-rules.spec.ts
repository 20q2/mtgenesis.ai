import { Rarity } from '../models/card.model';
import { setView } from '../testing/fixtures';
import {
  COMMANDER_CMCS, COMMANDER_RARITIES, commanderPipValue, commanderStatsError, groupByCmc, statPoints
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

  it('groupByCmc orders 3, 4, 5 then Earlier sets, skipping empty groups', () => {
    const sets = [5, null, 3, 3].map((cmc, i) => setView({ id: `s-${i}`, cmc }));
    const groups = groupByCmc(sets);
    expect(groups.map(g => g.label)).toEqual(['3 CMC', '5 CMC', 'Earlier sets']);
    expect(groups.map(g => g.sets.length)).toEqual([2, 1, 1]);
    expect(groups.map(g => g.cmc)).toEqual([3, 5, null]);
  });
});
