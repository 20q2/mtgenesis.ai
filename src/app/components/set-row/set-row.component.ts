import { Component, EventEmitter, Input, Output } from '@angular/core';
import { SetCardView, SetView } from '../../models/api.model';

/**
 * One locked set: commander name, "by <username>", and its 3 versions with vote
 * counts. Shared by the vote page and read-only event history (buttons hidden).
 */
@Component({
  selector: 'app-set-row',
  templateUrl: './set-row.component.html',
  styleUrls: ['./set-row.component.scss']
})
export class SetRowComponent {
  @Input() set!: SetView;
  @Input() showVoteButtons = true;
  /** Vote buttons disabled (event closed, or a vote in flight). */
  @Input() disabled = false;
  /** Final results: the leader is labelled "Winner" instead of "Leading". */
  @Input() final = false;
  @Output() vote = new EventEmitter<SetCardView>();

  isMine(card: SetCardView): boolean {
    return !!this.set.myVoteCardId && this.set.myVoteCardId === card.id;
  }

  onVote(card: SetCardView): void {
    if (!this.disabled) {
      this.vote.emit(card);
    }
  }

  trackCard(_index: number, card: SetCardView): string {
    return card.id;
  }
}
