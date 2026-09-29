import { CardStatus, CardView } from '../models/api.model';

/** done or failed: the card will not change any more. */
export function isFinished(status: CardStatus): boolean {
  return status === 'done' || status === 'failed';
}

/** queued, generating or rendering (counts toward the 3-pending cap). */
export function isPending(status: CardStatus): boolean {
  return !isFinished(status);
}

/**
 * "Queued #4 · ~35s" while the card waits for the image worker (spec §7), or null when it
 * isn't waiting (painting now, art ready, or finished).
 */
export function queueLine(view: CardView): string | null {
  if (isFinished(view.status) || view.artReady || view.queuePosition == null || view.queuePosition <= 0) {
    return null;
  }
  const eta = view.etaSeconds != null ? ` · ~${Math.max(1, Math.ceil(view.etaSeconds))}s` : '';
  return `Queued #${view.queuePosition}${eta}`;
}

/** True while the image worker is painting this card's art. */
export function isPainting(view: CardView): boolean {
  return !isFinished(view.status) && !view.artReady && view.queuePosition === 0;
}
