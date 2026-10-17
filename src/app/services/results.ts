import { PoolEntryView, SetCardView, SetView } from '../models/api.model';
import { setOutcome, versionLabel } from './winners';

/** One commander's legal version: the winner, the tied versions (host picks), or every version without votes. */
export interface LegalCommander {
  set: SetView;
  outcome: 'winner' | 'tie' | 'none';
  cards: SetCardView[];
  /** "Version 2 · 6 votes", "Tied: Versions 1 & 3", "No votes". */
  label: string;
}

/** A player's legal commanders, lowest mana value first. */
export interface PlayerCommanders {
  userId: string;
  username: string;
  avatarUrl: string | null;
  commanders: LegalCommander[];
}

/** The newest closed item in a list (events and pools both come newest first, but don't rely on it). */
export function latestClosed<T extends { status: string; createdAt: string }>(items: T[]): T | null {
  const closed = items.filter(i => i.status === 'closed');
  closed.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return closed[0] ?? null;
}

export function legalCommander(set: SetView): LegalCommander {
  const outcome = setOutcome(set);
  if (outcome.kind === 'winner') {
    const votes = outcome.cards[0].votes;
    return { set, outcome: 'winner', cards: outcome.cards,
             label: `${versionLabel(outcome.cards)} · ${votes} ${votes === 1 ? 'vote' : 'votes'}` };
  }
  if (outcome.kind === 'tie') {
    return { set, outcome: 'tie', cards: outcome.cards, label: `Tied: ${versionLabel(outcome.cards)}` };
  }
  const cards = set.cards.filter(c => c.status === 'done').sort((a, b) => (a.slot ?? 0) - (b.slot ?? 0));
  return { set, outcome: 'none', cards, label: 'No votes' };
}

/** Every player with a commander locked in the event, by name; their commanders by CMC (legacy sets last). */
export function playerCommanders(sets: SetView[]): PlayerCommanders[] {
  const players = new Map<string, PlayerCommanders>();
  for (const set of sets) {
    let player = players.get(set.userId);
    if (!player) {
      player = { userId: set.userId, username: set.username, avatarUrl: set.avatarUrl, commanders: [] };
      players.set(set.userId, player);
    }
    player.commanders.push(legalCommander(set));
  }
  const byCmc = (a: LegalCommander, b: LegalCommander) =>
    (a.set.cmc ?? Infinity) - (b.set.cmc ?? Infinity);
  return Array.from(players.values())
    .map(p => ({ ...p, commanders: p.commanders.sort(byCmc) }))
    .sort((a, b) => a.username.localeCompare(b.username, undefined, { sensitivity: 'base' }));
}

/** The cards that made the Knowledge Pool, in rank order. */
export function poolWinners(entries: PoolEntryView[]): PoolEntryView[] {
  return entries.filter(e => e.in);
}

/** "9 points" / "1 point". */
export function pointsLabel(points: number): string {
  return `${points} ${points === 1 ? 'point' : 'points'}`;
}

/** Columns for an export sheet: one row up to 5 cards, then a roughly 3:2 grid. */
export function sheetColumns(count: number): number {
  if (count <= 5) {
    return Math.max(1, count);
  }
  return Math.ceil(Math.sqrt(count * 1.6));
}
