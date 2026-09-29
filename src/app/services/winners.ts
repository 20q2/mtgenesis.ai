import { SetCardView, SetView } from '../models/api.model';

export type SetOutcome =
  | { kind: 'winner'; cards: SetCardView[] }
  | { kind: 'tie'; cards: SetCardView[] }
  | { kind: 'none'; cards: [] };

/** The leader (one card), the tied cards (host decides), or no votes (spec §3 Leader). */
export function setOutcome(set: SetView): SetOutcome {
  const leader = set.cards.filter(c => c.leader);
  if (leader.length) {
    return { kind: 'winner', cards: leader };
  }
  const tied = set.cards.filter(c => c.tied);
  if (tied.length) {
    return { kind: 'tie', cards: tied };
  }
  return { kind: 'none', cards: [] };
}

/** "Version 2" / "Versions 1 & 3". */
export function versionLabel(cards: SetCardView[]): string {
  const slots = cards.map(c => c.slot ?? 0).sort((a, b) => a - b);
  if (slots.length === 1) {
    return `Version ${slots[0]}`;
  }
  return `Versions ${slots.slice(0, -1).join(', ')} & ${slots[slots.length - 1]}`;
}
