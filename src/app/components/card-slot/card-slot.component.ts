import { Component, EventEmitter, HostListener, Input, Output } from '@angular/core';
import { CardView } from '../../models/api.model';
import { isPainting, queueLine } from '../../services/card-status';

/**
 * One card of a commander set (or a gallery tile): status line while pending
 * (spec §7), the finished card (tap to enlarge), or the failure reason, plus Reroll.
 */
@Component({
  selector: 'app-card-slot',
  templateUrl: './card-slot.component.html',
  styleUrls: ['./card-slot.component.scss']
})
export class CardSlotComponent {
  @Input() view: CardView | null = null;
  @Input() canReroll = false;
  @Input() showReroll = true;
  /** Shown when there is no card yet, e.g. "Version 2". */
  @Input() label = '';
  @Output() reroll = new EventEmitter<CardView>();

  enlarged = false;

  queueText(view: CardView): string | null {
    return queueLine(view);
  }

  painting(view: CardView): boolean {
    return isPainting(view);
  }

  onReroll(): void {
    if (this.view && this.canReroll) {
      this.reroll.emit(this.view);
    }
  }

  open(): void {
    this.enlarged = true;
  }

  close(): void {
    this.enlarged = false;
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    this.enlarged = false;
  }
}
