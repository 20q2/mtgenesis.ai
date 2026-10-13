import { SetCardView, SetView } from '../models/api.model';
import { COMMANDER_CMCS } from './commander-rules';
import { setOutcome } from './winners';

/**
 * The shape of an AI Night, as the rule sheet runs it: build three commanders, lock each in
 * (posting its 3 versions), everyone votes, and the winning versions are legal on the event night.
 */
export type NightStepId = 'build' | 'lock' | 'vote' | 'play';

export const NIGHT_STEPS: { id: NightStepId; title: string; text: string }[] = [
  { id: 'build', title: 'Build', text: 'Three commanders: one at 3, 4 and 5 mana, 3 versions each' },
  { id: 'lock', title: 'Lock in', text: 'Lock in each commander\'s 3 versions for the vote' },
  { id: 'vote', title: 'Vote', text: 'Pick the fairest version of every commander' },
  { id: 'play', title: 'Play', text: 'The version with the most votes is legal on the event night' }
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

/** One place on a player's ballot: their commander at that mana value, or null if they locked none. */
export interface BallotSlot { cmc: number | null; set: SetView | null; }

/** One player on the vote page: their commanders and how many of them I've voted on. */
export interface BallotPlayer {
  userId: string;
  username: string;
  mine: boolean;
  /** 3, 4 and 5 CMC (gaps included), then sets from before the commander rules. */
  slots: BallotSlot[];
  /** The commanders themselves, in slot order. */
  sets: SetView[];
  voted: number;
}

/** The vote by player: mine first (owners vote first), then everyone else by name. */
export function ballot(sets: SetView[], userId: string | null): BallotPlayer[] {
  const byUser = new Map<string, SetView[]>();
  for (const set of sets) {
    byUser.set(set.userId, [...(byUser.get(set.userId) ?? []), set]);
  }
  const players = Array.from(byUser.entries()).map(([id, own]): BallotPlayer => {
    const ruled = own.some(s => COMMANDER_CMCS.includes(s.cmc ?? -1));
    const slots: BallotSlot[] = [
      ...(ruled ? COMMANDER_CMCS.map(cmc => ({ cmc, set: own.find(s => s.cmc === cmc) ?? null })) : []),
      ...own.filter(s => !COMMANDER_CMCS.includes(s.cmc ?? -1)).map(set => ({ cmc: null, set }))
    ];
    const ordered = slots.map(s => s.set).filter((s): s is SetView => !!s);
    return {
      userId: id, username: own[0].username, mine: !!userId && id === userId, slots, sets: ordered,
      voted: ordered.filter(s => !!s.myVoteCardId).length
    };
  });
  return players.sort((a, b) => Number(b.mine) - Number(a.mine)
    || a.username.localeCompare(b.username, undefined, { sensitivity: 'base' }));
}

/** The next player (after `from`, wrapping, `from` itself last) with a commander I haven't voted on; -1 if none. */
export function nextToVote(players: BallotPlayer[], from = -1): number {
  for (let step = 1; step <= players.length; step++) {
    const i = (from + step + players.length) % players.length;
    if (players[i].voted < players[i].sets.length) {
      return i;
    }
  }
  return -1;
}

export interface LegalCommander { set: SetView; kind: 'winner' | 'tie' | 'none'; cards: SetCardView[]; }

/** After the vote: each of my commanders and the version(s) that won it, by mana value. */
export function legalCommanders(sets: SetView[], userId: string | null): LegalCommander[] {
  return sets
    .filter(s => !!userId && s.userId === userId)
    .sort((a, b) => (a.cmc ?? 99) - (b.cmc ?? 99))
    .map(set => ({ set, ...setOutcome(set) }));
}
