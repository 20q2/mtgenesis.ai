import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormControl } from '@angular/forms';
import { EMPTY, Subscription, catchError, finalize, interval, startWith, switchMap } from 'rxjs';
import { CardView, EventView, SetView } from '../../models/api.model';
import { Card } from '../../models/card.model';
import { apiErrorMessage } from '../../services/api.util';
import { isFinished, isPending } from '../../services/card-status';
import { EventService } from '../../services/event.service';
import { GenerationService } from '../../services/generation.service';

export const COMMANDER_NAME_MAX = 40;
export const EVENT_POLL_MS = 10000;
export const UNLOCK_CONFIRM = 'This clears votes on your set';

/**
 * /set: build a 3-card commander set, reroll slots, lock it into the open event (spec §7).
 * State comes from GET /me/sets/current, so a refresh loses nothing.
 */
@Component({
  selector: 'app-set-builder-page',
  templateUrl: './set-builder-page.component.html',
  styleUrls: ['./set-builder-page.component.scss']
})
export class SetBuilderPageComponent implements OnInit, OnDestroy {
  readonly maxNameLength = COMMANDER_NAME_MAX;
  readonly commanderName = new FormControl('', { nonNullable: true });

  /** Latest values from the card form. */
  formCard: Card | null = null;

  setId: string | null = null;
  setStatus: SetView['status'] | null = null;
  /** Current card per slot (index 0..2 = slot 1..3). */
  slots: (CardView | null)[] = [null, null, null];
  /** The current event (null when none is open). */
  event: EventView | null = null;

  loading = true;
  submitting = false;
  locking = false;
  unlocking = false;
  rerolling = [false, false, false];
  error: string | null = null;

  private watches: (Subscription | undefined)[] = [];
  private eventSub?: Subscription;
  private loadSub?: Subscription;

  constructor(private generation: GenerationService, private events: EventService) {}

  ngOnInit(): void {
    this.loadSub = this.events.mySet().subscribe({
      next: set => {
        this.applySet(set);
        this.loading = false;
      },
      error: err => {
        this.error = apiErrorMessage(err, 'Could not load your set.');
        this.loading = false;
      }
    });

    // Keep the event fresh so Lock in enables as soon as the host opens one.
    this.eventSub = interval(EVENT_POLL_MS).pipe(
      startWith(0),
      switchMap(() => this.events.current().pipe(catchError(() => EMPTY)))
    ).subscribe(event => (this.event = event));
  }

  ngOnDestroy(): void {
    this.loadSub?.unsubscribe();
    this.eventSub?.unsubscribe();
    this.watches.forEach(w => w?.unsubscribe());
  }

  get isLocked(): boolean {
    return this.setStatus === 'locked';
  }

  get eventOpen(): boolean {
    return !!this.event && this.event.status === 'open';
  }

  hasPending(): boolean {
    return this.slots.some(s => !!s && isPending(s.status));
  }

  canGenerate(): boolean {
    return !this.submitting && !this.hasPending() && !this.isLocked && !this.loading;
  }

  generateHint(): string | null {
    if (this.isLocked) {
      return 'Your set is locked in. Unlock it to start a new one.';
    }
    if (this.hasPending()) {
      return 'Wait for all 3 cards to finish before starting a new set.';
    }
    return null;
  }

  onCardChange(card: Card): void {
    this.formCard = card;
  }

  generate(): void {
    if (!this.canGenerate()) {
      return;
    }
    const name = this.commanderName.value.trim();
    if (!name) {
      this.error = 'Enter a commander name.';
      return;
    }
    if (name.length > COMMANDER_NAME_MAX) {
      this.error = `Commander names can be at most ${COMMANDER_NAME_MAX} characters.`;
      return;
    }
    if (this.setStatus === 'draft' && this.slots.some(s => !!s)
        && !window.confirm('Start a new set? Your current draft will be replaced (its cards stay in your Gallery).')) {
      return;
    }

    const card: Card = this.formCard ?? { name, manaCost: '', type: '', colors: [], cmc: 0, rarity: 'common' as Card['rarity'] };
    const prompt = this.generation.promptFor({ ...card, name });

    this.error = null;
    this.submitting = true;
    this.generation.submit({
      prompt,
      cardData: { ...this.generation.cardParams(card), name },
      count: 3,
      commanderName: name
    }).pipe(
      finalize(() => (this.submitting = false))
    ).subscribe({
      next: response => {
        this.stopWatches();
        this.setId = response.setId;
        this.setStatus = 'draft';
        this.slots = [null, null, null];
        for (const view of response.cards) {
          this.placeCard(view);
        }
      },
      error: err => (this.error = apiErrorMessage(err, 'Could not start the set.'))
    });
  }

  canRerollSlot(view: CardView | null): boolean {
    return !!view
      && this.setStatus === 'draft'
      && isFinished(view.status)
      && !this.rerolling[this.slotIndex(view)];
  }

  onReroll(view: CardView): void {
    if (!this.canRerollSlot(view)) {
      return;
    }
    const index = this.slotIndex(view);
    this.error = null;
    this.rerolling[index] = true;
    this.generation.reroll(view.id).pipe(
      finalize(() => (this.rerolling[index] = false))
    ).subscribe({
      next: fresh => this.placeCard(fresh, index),
      error: err => (this.error = apiErrorMessage(err, 'Could not reroll that card.'))
    });
  }

  /** Why Lock in is disabled, or null when it can be pressed. */
  lockDisabledReason(): string | null {
    if (!this.setId) {
      return 'Generate a set first';
    }
    if (!this.eventOpen) {
      return 'No event open — ask the host';
    }
    if (this.slots.some(s => !s || s.status !== 'done')) {
      return 'Waiting for all 3 cards';
    }
    if (!this.commanderName.value.trim()) {
      return 'Enter a commander name';
    }
    return null;
  }

  lock(): void {
    if (this.locking || this.isLocked || this.lockDisabledReason() || !this.setId) {
      return;
    }
    this.error = null;
    this.locking = true;
    this.events.lock(this.setId, this.commanderName.value.trim()).pipe(
      finalize(() => (this.locking = false))
    ).subscribe({
      next: set => this.applySet(set),
      error: err => (this.error = apiErrorMessage(err, 'Could not lock in your set.'))
    });
  }

  unlock(): void {
    if (this.unlocking || !this.isLocked || !this.setId) {
      return;
    }
    if (!window.confirm(UNLOCK_CONFIRM)) {
      return;
    }
    this.error = null;
    this.unlocking = true;
    this.events.unlock(this.setId).pipe(
      finalize(() => (this.unlocking = false))
    ).subscribe({
      next: set => this.applySet(set),
      error: err => (this.error = apiErrorMessage(err, 'Could not unlock your set.'))
    });
  }

  clearError(): void {
    this.error = null;
  }

  trackSlot(index: number): number {
    return index;
  }

  private applySet(set: SetView | null): void {
    this.stopWatches();
    if (!set || set.status === 'abandoned') {
      this.setId = null;
      this.setStatus = null;
      this.slots = [null, null, null];
      return;
    }
    this.setId = set.id;
    this.setStatus = set.status;
    this.commanderName.setValue(set.commanderName ?? '');
    this.slots = [null, null, null];
    for (const view of set.cards ?? []) {
      this.placeCard(view);
    }
  }

  /** Puts a card in its slot and watches it if it is still being made. */
  private placeCard(view: CardView, index = this.slotIndex(view)): void {
    this.slots[index] = view;
    this.watches[index]?.unsubscribe();
    this.watches[index] = undefined;
    if (isPending(view.status)) {
      this.watches[index] = this.generation.watch(view.id).subscribe({
        next: update => {
          if (this.slots[index]?.id === update.id) {
            this.slots[index] = update;
          }
        },
        error: err => (this.error = apiErrorMessage(err, 'Lost track of a card. Refresh the page.'))
      });
    }
  }

  private slotIndex(view: CardView): number {
    const slot = view.slot ?? 1;
    return Math.min(Math.max(slot, 1), 3) - 1;
  }

  private stopWatches(): void {
    this.watches.forEach(w => w?.unsubscribe());
    this.watches = [];
  }
}
