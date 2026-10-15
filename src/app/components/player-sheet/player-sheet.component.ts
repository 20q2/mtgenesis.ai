import {
  AfterViewChecked, AfterViewInit, Component, ElementRef, EventEmitter, Input, OnChanges, OnDestroy, Output,
  SimpleChanges, ViewChild
} from '@angular/core';
import { SetCardView, SetView } from '../../models/api.model';
import { RarityOptions } from '../../models/card.model';
import { BallotPlayer } from '../../services/commander-night';
import { versionLabel } from '../../services/winners';

export interface SheetVote { set: SetView; card: SetCardView; }
export interface SheetRerollRequest extends SheetVote { requested: boolean; }

/**
 * One player's commanders, all three versions of each, to vote on in one go. A native modal
 * <dialog>: it renders in the top layer (above every .panel's backdrop-filter), traps focus,
 * and gives focus back to the tile that opened it. Esc closes, ← and → move between players.
 */
@Component({
  selector: 'app-player-sheet',
  templateUrl: './player-sheet.component.html',
  styleUrls: ['./player-sheet.component.scss']
})
export class PlayerSheetComponent implements OnChanges, AfterViewInit, AfterViewChecked, OnDestroy {
  @Input() player!: BallotPlayer;
  @Input() closed = false;
  /** Sets with a vote in flight. */
  @Input() busy: ReadonlySet<string> = new Set();
  /** Versions with a reroll request or a reroll in flight. */
  @Input() cardBusy: ReadonlySet<string> = new Set();
  @Input() error: string | null = null;
  /** The commander to bring into view when the sheet opens or moves to another player. */
  @Input() focusSetId: string | null = null;
  /** Where this player is in the list ("2 of 7"). */
  @Input() position = 0;
  @Input() count = 0;
  /** The next other player with a commander I haven't voted on; null when there's none. */
  @Input() nextPlayer: BallotPlayer | null = null;

  @Output() vote = new EventEmitter<SheetVote>();
  /** Someone else's version: ask its owner to reroll it, or take the request back. */
  @Output() requestReroll = new EventEmitter<SheetRerollRequest>();
  /** My version, once enough players asked (or it failed). */
  @Output() reroll = new EventEmitter<SheetVote>();
  @Output() step = new EventEmitter<number>();
  @Output() goTo = new EventEmitter<BallotPlayer>();
  @Output() dismiss = new EventEmitter<void>();
  @Output() clearError = new EventEmitter<void>();

  @ViewChild('dialog', { static: true }) private dialog!: ElementRef<HTMLDialogElement>;
  @ViewChild('body', { static: true }) private body!: ElementRef<HTMLElement>;

  /** The version shown full size. */
  zoomed: SetCardView | null = null;
  /** Set when a vote lands, read out by the live region. */
  announcement = '';

  private scrollPending = true;

  ngOnChanges(changes: SimpleChanges): void {
    const before: BallotPlayer | undefined = changes['player']?.previousValue;
    if ((changes['player'] && before?.userId !== this.player.userId) || changes['focusSetId']) {
      this.scrollPending = true;
      this.zoomed = null;
    }
    if (changes['player'] && before?.userId === this.player.userId) {
      this.announceVote(before);
    }
  }

  ngAfterViewInit(): void {
    const dialog = this.dialog.nativeElement;
    if (!dialog.open) {
      if (typeof dialog.showModal === 'function') {
        dialog.showModal();
      } else {
        dialog.setAttribute('open', '');
      }
    }
  }

  ngAfterViewChecked(): void {
    if (this.scrollPending) {
      this.scrollPending = false;
      this.scrollToFocus();
    }
  }

  ngOnDestroy(): void {
    const dialog = this.dialog.nativeElement;
    if (dialog.open && typeof dialog.close === 'function') {
      dialog.close();
    }
  }

  get allVoted(): boolean {
    return this.player.voted === this.player.sets.length;
  }

  rarityLabel(set: SetView): string {
    return RarityOptions.find(r => r.value === set.rarity)?.label ?? '';
  }

  isPick(set: SetView, card: SetCardView): boolean {
    return !!set.myVoteCardId && set.myVoteCardId === card.id;
  }

  pickLabel(set: SetView): string {
    const pick = set.cards.find(c => this.isPick(set, c));
    return pick ? versionLabel([pick]) : '';
  }

  left(player: BallotPlayer): number {
    return player.sets.length - player.voted;
  }

  onVote(set: SetView, card: SetCardView): void {
    if (!this.closed && !this.busy.has(set.id)) {
      this.vote.emit({ set, card });
    }
  }

  /** While voting: the ask button on others' finished versions; for mine, the count or Reroll. */
  showRerollBar(card: SetCardView): boolean {
    if (this.closed) {
      return false;
    }
    return this.player.mine ? card.rerollRequests > 0 || card.canReroll : card.status === 'done';
  }

  /** What the owner sees about requests on their version (a failed one only needs the button). */
  ownerRerollText(set: SetView, card: SetCardView): string | null {
    if (card.status !== 'done' || !card.rerollRequests) {
      return null;
    }
    const n = card.rerollRequests;
    return card.canReroll ? `${n} ${n === 1 ? 'player says' : 'players say'} it doesn't work`
      : `${n} of ${set.rerollRequestsNeeded} asked for a reroll`;
  }

  notReadyText(card: SetCardView): string {
    return card.status === 'failed' ? `Version ${card.slot} failed` : `Rerolling version ${card.slot}…`;
  }

  onRequestReroll(set: SetView, card: SetCardView): void {
    if (!this.closed && !this.player.mine && !this.cardBusy.has(card.id)) {
      this.requestReroll.emit({ set, card, requested: !card.myRerollRequest });
    }
  }

  onReroll(set: SetView, card: SetCardView): void {
    if (!this.closed && card.canReroll && !this.cardBusy.has(card.id)) {
      this.reroll.emit({ set, card });
    }
  }

  zoom(card: SetCardView): void {
    if (card.cardImageUrl) {
      this.zoomed = card;
    }
  }

  /** Esc closes the enlarged card first, then the sheet. */
  onCancel(event: Event): void {
    event.preventDefault();
    if (this.zoomed) {
      this.zoomed = null;
    } else {
      this.dismiss.emit();
    }
  }

  /** A click on the dialog itself (not the sheet) is a click on the dimmed page around it. */
  onDialogClick(event: MouseEvent): void {
    if (event.target === this.dialog.nativeElement) {
      this.dismiss.emit();
    }
  }

  onKeydown(event: KeyboardEvent): void {
    if (this.zoomed || this.count < 2 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
      return;
    }
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      this.step.emit(event.key === 'ArrowLeft' ? -1 : 1);
    }
  }

  trackSet(_index: number, set: SetView): string {
    return set.id;
  }

  trackCard(_index: number, card: SetCardView): string {
    return card.id;
  }

  private announceVote(before: BallotPlayer): void {
    for (const set of this.player.sets) {
      const was = before.sets.find(s => s.id === set.id);
      if (was && set.myVoteCardId && was.myVoteCardId !== set.myVoteCardId) {
        this.announcement = `Voted for ${this.pickLabel(set)} of ${set.commanderName}`;
      }
    }
  }

  private scrollToFocus(): void {
    const body = this.body.nativeElement;
    const row = this.focusSetId
      ? body.querySelector<HTMLElement>(`[data-set-id="${CSS.escape(this.focusSetId)}"]`)
      : null;
    const first = body.querySelector<HTMLElement>('.commander-row');
    if (row && row !== first) {
      // The body is position: relative, so offsetTop is measured from its top.
      body.scrollTop = row.offsetTop;
    } else {
      body.scrollTop = 0;
    }
  }
}
