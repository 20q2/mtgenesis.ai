import { SetCardView, SetView } from '../models/api.model';
import { setOutcome } from './winners';

/**
 * The shape of an AI Night, as the rule sheet runs it: build three commanders, lock each in
 * (posting its 3 versions), everyone votes, and the winning versions are legal tonight.
 */
export type NightStepId = 'build' | 'lock' | 'vote' | 'play';

export const NIGHT_STEPS: { id: NightStepId; title: string; text: string }[] = [
  { id: 'build', title: 'Build', text: 'Three commanders: one at 3, 4 and 5 mana, 3 versions each' },
  { id: 'lock', title: 'Lock in', text: 'Lock in each commander\'s 3 versions for the vote' },
  { id: 'vote', title: 'Vote', text: 'Pick the fairest version of every commander' },
  { id: 'play', title: 'Play', text: 'The version with the most votes is legal tonight' }
];

/** Where a player is: building until all 3 exist, locking in until all 3 are, then voting. */
export function nightStep(generated: number, locked: number, eventStatus: 'open' | 'closed' | null): NightStepId {
  if (eventStatus === 'closed') {
    return 'play';
  }
  if (locked >= 3) {
    return 'vote';
  }
  return generated >= 3 ? 'lock' : 'build';
}

export interface VoteProgress {
  /** My own commanders: rule 7 asks owners to vote on these first. */
  mine: SetView[];
  others: SetView[];
  voted: number;
  total: number;
  mineUnvoted: number;
}

export function voteProgress(sets: SetView[], userId: string | null): VoteProgress {
  const mine = userId ? sets.filter(s => s.userId === userId) : [];
  const others = sets.filter(s => !mine.includes(s));
  return {
    mine,
    others,
    voted: sets.filter(s => !!s.myVoteCardId).length,
    total: sets.length,
    mineUnvoted: mine.filter(s => !s.myVoteCardId).length
  };
}

export interface LegalCommander { set: SetView; kind: 'winner' | 'tie' | 'none'; cards: SetCardView[]; }

/** After the vote: each of my commanders and the version(s) that won it, by mana value. */
export function legalCommanders(sets: SetView[], userId: string | null): LegalCommander[] {
  return sets
    .filter(s => !!userId && s.userId === userId)
    .sort((a, b) => (a.cmc ?? 99) - (b.cmc ?? 99))
    .map(set => ({ set, ...setOutcome(set) }));
}
