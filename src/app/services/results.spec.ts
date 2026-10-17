import { poolEntry, setCard, setView } from '../testing/fixtures';
import { latestClosed, legalCommander, playerCommanders, poolWinners, sheetColumns } from './results';

describe('results', () => {
  it('picks the newest closed event or pool', () => {
    const items = [
      { id: 'open', status: 'open', createdAt: '2026-10-20T00:00:00Z' },
      { id: 'old', status: 'closed', createdAt: '2026-09-01T00:00:00Z' },
      { id: 'new', status: 'closed', createdAt: '2026-10-14T00:00:00Z' }
    ];
    expect(latestClosed(items)!.id).toBe('new');
    expect(latestClosed([items[0]])).toBeNull();
  });

  it('a winner is one version with its votes', () => {
    const lc = legalCommander(setView({ cards: [setCard({ id: 'a', slot: 1, votes: 1 }),
                                                setCard({ id: 'b', slot: 2, leader: true, votes: 6 })] }));
    expect(lc.outcome).toBe('winner');
    expect(lc.cards.map(c => c.id)).toEqual(['b']);
    expect(lc.label).toBe('Version 2 · 6 votes');
  });

  it('a tie keeps every tied version for the host to pick', () => {
    const lc = legalCommander(setView({ cards: [setCard({ id: 'a', slot: 1, tied: true, votes: 3 }),
                                                setCard({ id: 'b', slot: 2 }),
                                                setCard({ id: 'c', slot: 3, tied: true, votes: 3 })] }));
    expect(lc.outcome).toBe('tie');
    expect(lc.cards.map(c => c.id)).toEqual(['a', 'c']);
    expect(lc.label).toBe('Tied: Versions 1 & 3');
  });

  it('no votes keeps every finished version', () => {
    const lc = legalCommander(setView({ cards: [setCard({ id: 'b', slot: 2 }), setCard({ id: 'a', slot: 1 }),
                                                setCard({ id: 'x', slot: 3, status: 'failed' })] }));
    expect(lc.outcome).toBe('none');
    expect(lc.cards.map(c => c.id)).toEqual(['a', 'b']);
    expect(lc.label).toBe('No votes');
  });

  it('groups commanders by player, players by name and commanders by CMC', () => {
    const win = (id: string) => [setCard({ id, slot: 1, leader: true, votes: 2 })];
    const players = playerCommanders([
      setView({ id: 's1', userId: 'u-z', username: 'zed', cmc: 5, cards: win('z5') }),
      setView({ id: 's2', userId: 'u-a', username: 'Amy', cmc: 4, cards: win('a4') }),
      setView({ id: 's3', userId: 'u-z', username: 'zed', cmc: 3, cards: win('z3') }),
      setView({ id: 's4', userId: 'u-b', username: 'bo', cmc: null, cards: win('b-legacy') }),
      setView({ id: 's5', userId: 'u-b', username: 'bo', cmc: 4, cards: win('b4') })
    ]);
    expect(players.map(p => p.username)).toEqual(['Amy', 'bo', 'zed']);
    expect(players[1].commanders.map(c => c.cards[0].id)).toEqual(['b4', 'b-legacy']);
    expect(players[2].commanders.map(c => c.set.cmc)).toEqual([3, 5]);
  });

  it('the pool is the entries that are in', () => {
    const entries = [poolEntry({ id: '1', in: true }), poolEntry({ id: '2' }), poolEntry({ id: '3', in: true })];
    expect(poolWinners(entries).map(e => e.id)).toEqual(['1', '3']);
  });

  it('sheets put up to 5 cards in one row, then a wider-than-tall grid', () => {
    expect([1, 3, 5, 6, 16, 18].map(sheetColumns)).toEqual([1, 3, 5, 4, 6, 6]);
  });
});
