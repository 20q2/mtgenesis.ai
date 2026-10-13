import { setCard, setView } from '../testing/fixtures';
import { NIGHT_STEPS, ballot, legalCommanders, nextToVote, nightStep, voteProgress } from './commander-night';

describe('commander night', () => {
  it('has the four steps of the night in order', () => {
    expect(NIGHT_STEPS.map(s => s.id)).toEqual(['build', 'lock', 'vote', 'play']);
  });

  it('nightStep: build until something is generated, lock in until all 3 are, then vote, then play', () => {
    expect(nightStep(0, 0, 'open')).toBe('build');
    expect(nightStep(1, 0, 'open')).toBe('build');
    expect(nightStep(3, 1, 'open')).toBe('lock');
    expect(nightStep(3, 3, 'open')).toBe('vote');
    expect(nightStep(3, 3, 'closed')).toBe('play');
    expect(nightStep(0, 0, 'closed')).toBe('play');
    expect(nightStep(2, 0, null)).toBe('build');
  });

  describe('voteProgress', () => {
    const mine = setView({ id: 'm', userId: 'me', cmc: 3, myVoteCardId: 'm-c1' });
    const theirs = [
      setView({ id: 'a', userId: 'x', cmc: 3, myVoteCardId: 'a-c2' }),
      setView({ id: 'b', userId: 'y', cmc: 5, myVoteCardId: null })
    ];

    it('puts my commanders first and counts what I have voted on', () => {
      const p = voteProgress([...theirs, mine], 'me');
      expect(p.mine.map(s => s.id)).toEqual(['m']);
      expect(p.others.map(s => s.id)).toEqual(['a', 'b']);
      expect(p.voted).toBe(2);
      expect(p.total).toBe(3);
      expect(p.mineUnvoted).toBe(0);
    });

    it('counts my own commanders I have not voted on yet', () => {
      const p = voteProgress([{ ...mine, myVoteCardId: null }, ...theirs], 'me');
      expect(p.mineUnvoted).toBe(1);
    });

    it('with no user everything is "others"', () => {
      expect(voteProgress(theirs, null).mine).toEqual([]);
    });
  });

  describe('ballot', () => {
    const sets = [
      setView({ id: 'z5', userId: 'z', username: 'zed', cmc: 5, myVoteCardId: 'z5-c1' }),
      setView({ id: 'm4', userId: 'me', username: 'sam', cmc: 4 }),
      setView({ id: 'a3', userId: 'a', username: 'Alice', cmc: 3, myVoteCardId: 'a3-c2' }),
      setView({ id: 'z3', userId: 'z', username: 'zed', cmc: 3 }),
      setView({ id: 'b0', userId: 'b', username: 'bob', cmc: null })
    ];

    it('groups by player: mine first, then everyone else by name (ignoring case)', () => {
      const players = ballot(sets, 'me');
      expect(players.map(p => p.username)).toEqual(['sam', 'Alice', 'bob', 'zed']);
      expect(players.map(p => p.mine)).toEqual([true, false, false, false]);
    });

    it('keeps a slot for 3, 4 and 5 CMC (null where none is locked) and counts my votes', () => {
      const zed = ballot(sets, 'me')[3];
      expect(zed.slots.map(s => [s.cmc, s.set?.id ?? null])).toEqual([[3, 'z3'], [4, null], [5, 'z5']]);
      expect(zed.sets.map(s => s.id)).toEqual(['z3', 'z5']);
      expect(zed.voted).toBe(1);
    });

    it('sets from before the commander rules get one slot each and no CMC gaps', () => {
      const bob = ballot(sets, 'me')[2];
      expect(bob.slots.map(s => [s.cmc, s.set?.id])).toEqual([[null, 'b0']]);
    });

    it('nextToVote: the next player with a commander left to vote on, wrapping round', () => {
      const players = ballot(sets, 'me'); // sam 0/1, Alice 1/1, bob 0/1, zed 1/2
      expect(nextToVote(players)).toBe(0);
      expect(nextToVote(players, 0)).toBe(2);
      expect(nextToVote(players, 2)).toBe(3);
      expect(nextToVote(players, 3)).toBe(0);
      const done = players.map(p => ({ ...p, voted: p.sets.length }));
      expect(nextToVote(done, 1)).toBe(-1);
      expect(nextToVote([], 0)).toBe(-1);
    });
  });

  it('legalCommanders: my commanders and their winning (or tied) versions, by CMC', () => {
    const won = setView({ id: 'w', userId: 'me', cmc: 5, commanderName: 'Five',
      cards: [setCard({ id: 'w1', slot: 1 }), setCard({ id: 'w2', slot: 2, leader: true, votes: 3 }), setCard({ id: 'w3', slot: 3 })] });
    const tied = setView({ id: 't', userId: 'me', cmc: 3, commanderName: 'Three',
      cards: [setCard({ id: 't1', slot: 1, tied: true }), setCard({ id: 't2', slot: 2, tied: true }), setCard({ id: 't3', slot: 3 })] });
    const other = setView({ id: 'o', userId: 'x', cmc: 4 });
    const legal = legalCommanders([won, other, tied], 'me');
    expect(legal.map(l => [l.set.commanderName, l.kind, l.cards.map(c => c.id)])).toEqual([
      ['Three', 'tie', ['t1', 't2']],
      ['Five', 'winner', ['w2']]
    ]);
  });
});
