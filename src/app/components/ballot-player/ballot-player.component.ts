import { Component, EventEmitter, Input, Output } from '@angular/core';
import { SetCardView, SetView } from '../../models/api.model';
import { BallotPlayer, BallotSlot } from '../../services/commander-night';
import { setOutcome, versionLabel } from '../../services/winners';

export type SlotState = 'busy' | 'voted' | 'todo' | 'winner' | 'tie' | 'none';

/**
 * One player on the vote page: their commanders at 3, 4 and 5 CMC as small fans of the
 * three versions, each marked voted or not (or, once voting closes, with its winner).
 * Clicking the tile, or one commander, opens the player's commanders to vote on.
 */
@Component({
  selector: 'app-ballot-player',
  templateUrl: './ballot-player.component.html',
  styleUrls: ['./ballot-player.component.scss']
})
export class BallotPlayerComponent {
  @Input() player!: BallotPlayer;
  @Input() closed = false;
  /** Sets with a vote in flight. */
  @Input() busy: ReadonlySet<string> = new Set();
  /** Open this player's commanders: at the clicked commander, or null for the tile itself. */
  @Output() open = new EventEmitter<SetView | null>();

  get done(): boolean {
    return this.player.voted === this.player.sets.length;
  }

  state(set: SetView): SlotState {
    if (this.closed) {
      return setOutcome(set).kind;
    }
    if (this.busy.has(set.id)) {
      return 'busy';
    }
    return set.myVoteCardId ? 'voted' : 'todo';
  }

  /** The version drawn in front: my pick while voting, the winner once it's closed. */
  frontId(set: SetView): string | null {
    if (this.closed) {
      const outcome = setOutcome(set);
      return outcome.kind === 'winner' ? outcome.cards[0].id : null;
    }
    return set.myVoteCardId;
  }

  stateText(set: SetView): string {
    switch (this.state(set)) {
      case 'busy': return 'Voting…';
      case 'voted': return this.pickLabel(set);
      case 'todo': return 'Vote';
      case 'winner': return versionLabel(setOutcome(set).cards);
      case 'tie': return 'Tied';
      default: return 'No votes';
    }
  }

  stateIcon(set: SetView): string {
    switch (this.state(set)) {
      case 'voted': return 'check_circle';
      case 'todo': return 'radio_button_unchecked';
      case 'winner': return 'workspace_premium';
      case 'tie': return 'balance';
      default: return 'remove';
    }
  }

  slotLabel(set: SetView): string {
    const what = `${set.commanderName}${set.cmc ? `, ${set.cmc} CMC` : ''}`;
    switch (this.state(set)) {
      case 'voted': return `${what}: you voted for ${this.pickLabel(set)}`;
      case 'todo': return `${what}: not voted yet`;
      case 'busy': return `${what}: saving your vote`;
      case 'winner': return `${what}: ${versionLabel(setOutcome(set).cards)} won`;
      case 'tie': return `${what}: tied`;
      default: return `${what}: no votes`;
    }
  }

  onSlot(event: Event, set: SetView): void {
    event.stopPropagation();
    this.open.emit(set);
  }

  trackSlot(index: number, slot: BallotSlot): string {
    return slot.set?.id ?? `gap-${slot.cmc ?? index}`;
  }

  trackCard(_index: number, card: SetCardView): string {
    return card.id;
  }

  private pickLabel(set: SetView): string {
    const pick = set.cards.find(c => c.id === set.myVoteCardId);
    return pick ? versionLabel([pick]) : 'Voted';
  }
}
