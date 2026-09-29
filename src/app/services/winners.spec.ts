import { setCard, setView } from '../testing/fixtures';
import { setOutcome, versionLabel } from './winners';

describe('winners', () => {
  it('reports the single leader as the winner', () => {
    const set = setView({ cards: [setCard({ slot: 1 }), setCard({ slot: 2, leader: true, votes: 2 }), setCard({ slot: 3 })] });
    const outcome = setOutcome(set);
    expect(outcome.kind).toBe('winner');
    expect(versionLabel(outcome.cards)).toBe('Version 2');
  });

  it('reports tied cards when there is no single leader', () => {
    const set = setView({ cards: [setCard({ slot: 3, tied: true }), setCard({ slot: 2 }), setCard({ slot: 1, tied: true })] });
    const outcome = setOutcome(set);
    expect(outcome.kind).toBe('tie');
    expect(versionLabel(outcome.cards)).toBe('Versions 1 & 3');
  });

  it('reports no votes when nothing leads or ties', () => {
    expect(setOutcome(setView()).kind).toBe('none');
  });
});
