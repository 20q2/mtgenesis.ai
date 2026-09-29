import { Component, OnDestroy, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import {
  EMPTY, Observable, Subject, Subscription, catchError, finalize, of, startWith, switchMap
} from 'rxjs';
import {
  CardView, PoolColorRule, PoolEntryView, PoolSlotView, PoolSummary, PoolView, PowerCheck
} from '../../models/api.model';
import { apiErrorMessage, safeFileName } from '../../services/api.util';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { PoolService, cardFitsSlot } from '../../services/pool.service';

export const POOL_POLL_MS = 10000;

/** A slot's final result: its winner, or the tied cards the host picks from. */
export interface PoolResult {
  slot: PoolSlotView;
  entries: PoolEntryView[];
  tied: boolean;
}

/**
 * /pool: the open Knowledge Pool. Players submit finished cards into slots, then vote
 * once per slot (never for their own card); the top card of each slot becomes legal.
 * /pool/:id shows a pool read-only (past pools). Polls every 10s while visible.
 */
@Component({
  selector: 'app-pool-page',
  templateUrl: './pool-page.component.html',
  styleUrls: ['./pool-page.component.scss']
})
export class PoolPageComponent implements OnInit, OnDestroy {
  /** undefined while the first load is pending; null when no pool is open. */
  pool: PoolView | null | undefined = undefined;
  pastPools: PoolSummary[] | null = null;
  loadError: string | null = null;
  error: string | null = null;
  /** Slots with a request in flight (guards double clicks). */
  readonly busy = new Set<string>();

  /** The slot whose card picker is open. */
  pickerSlotId: string | null = null;
  myCards: CardView[] | null = null;
  myCardsError: string | null = null;

  private readonly refresh$ = new Subject<void>();
  private routeId: string | null = null;
  private lastPoolId: string | null = null;
  private sub?: Subscription;
  private routeSub?: Subscription;

  constructor(private pools: PoolService, private generation: GenerationService,
              private media: MediaService, private visibility: PageVisibilityService,
              private route: ActivatedRoute) {}

  ngOnInit(): void {
    this.loadPastPools();
    this.routeSub = this.route.paramMap.subscribe(params => {
      this.routeId = params.get('id');
      this.pool = undefined;
      this.pickerSlotId = null;
      this.refresh();
    });
    // refresh() restarts the poll timer with an immediate fetch.
    this.sub = this.refresh$.pipe(
      startWith(undefined),
      switchMap(() => this.visibility.poll(POOL_POLL_MS)),
      switchMap(() => this.load())
    ).subscribe(pool => this.apply(pool));
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
    this.routeSub?.unsubscribe();
  }

  get isOpen(): boolean {
    return this.pool?.status === 'open';
  }

  get isPast(): boolean {
    return !!this.routeId;
  }

  get submissionsLeft(): number {
    return this.pool ? Math.max(0, this.pool.maxEntriesPerUser - this.pool.myEntryCount) : 0;
  }

  get filledSlots(): number {
    return this.pool?.slots.filter(s => s.entries.some(e => e.leader)).length ?? 0;
  }

  get votesLeft(): number {
    return this.pool?.slots.filter(s => this.canVoteIn(s) && !s.myVoteEntryId).length ?? 0;
  }

  /** Winners (and tied groups) in slot order; slots nobody voted in are left out. */
  get results(): PoolResult[] {
    return (this.pool?.slots ?? []).flatMap(slot => {
      const winners = slot.entries.filter(e => e.leader || e.tied);
      return winners.length ? [{ slot, entries: winners, tied: winners.some(e => e.tied) }] : [];
    });
  }

  refresh(): void {
    this.refresh$.next();
  }

  // ----- slot state -----
  /** Someone else's card is in the slot, so there's something to vote on. */
  canVoteIn(slot: PoolSlotView): boolean {
    return slot.entries.some(e => !e.mine);
  }

  canSubmitTo(slot: PoolSlotView): boolean {
    return this.isOpen && !slot.myEntryId && this.submissionsLeft > 0;
  }

  slotState(slot: PoolSlotView): 'empty' | 'leader' | 'tied' | 'open' {
    if (!slot.entries.length) {
      return 'empty';
    }
    if (slot.entries.some(e => e.leader)) {
      return 'leader';
    }
    return slot.entries.some(e => e.tied) ? 'tied' : 'open';
  }

  leaderOf(slot: PoolSlotView): PoolEntryView | undefined {
    return slot.entries.find(e => e.leader);
  }

  isMyPick(slot: PoolSlotView, entry: PoolEntryView): boolean {
    return slot.myVoteEntryId === entry.id;
  }

  scrollTo(slot: PoolSlotView): void {
    document.getElementById(`pool-slot-${slot.position}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ----- actions -----
  vote(slot: PoolSlotView, entry: PoolEntryView): void {
    if (this.isMyPick(slot, entry)) {
      this.run(slot, this.pools.clearVote(slot.id), 'Could not clear your vote.');
    } else {
      this.run(slot, this.pools.vote(slot.id, entry.id), 'Your vote did not go through. Please try again.');
    }
  }

  withdraw(slot: PoolSlotView, entry: PoolEntryView): void {
    const name = entry.card.card?.name || 'this card';
    if (!window.confirm(`Withdraw "${name}" from ${slot.label}? Its votes are cleared.`)) {
      return;
    }
    this.run(slot, this.pools.withdraw(entry.id), 'Could not withdraw the card.');
  }

  openPicker(slot: PoolSlotView): void {
    if (this.pickerSlotId === slot.id) {
      this.pickerSlotId = null;
      return;
    }
    this.pickerSlotId = slot.id;
    this.loadMyCards();
  }

  closePicker(): void {
    this.pickerSlotId = null;
  }

  submit(slot: PoolSlotView, card: CardView): void {
    this.run(slot, this.pools.submit(slot.id, card.id), 'Could not submit the card.', () => {
      this.pickerSlotId = null;
    });
  }

  /** My finished cards that fit the slot and aren't in the pool yet. */
  eligibleCards(slot: PoolSlotView): CardView[] {
    const inPool = new Set(this.pool?.slots.flatMap(s => s.entries.map(e => e.cardId)) ?? []);
    return (this.myCards ?? []).filter(c =>
      c.status === 'done' && !inPool.has(c.id) && cardFitsSlot(c.card, slot.colorRule, slot.typeRule));
  }

  download(entry: PoolEntryView): void {
    this.media.download(entry.card.cardImageUrl, `${safeFileName(entry.card.card?.name)}_card.png`);
  }

  clearError(): void {
    this.error = null;
  }

  // ----- display helpers -----
  pipClass(rule: PoolColorRule): string | null {
    return ['W', 'U', 'B', 'R', 'G'].includes(rule) ? `ms ms-${rule.toLowerCase()} ms-cost` : null;
  }

  powerLabel(power: PowerCheck): string {
    return power.verdict === 'over' ? 'Over the curve' : power.verdict === 'pushed' ? 'Pushed' : 'Fair';
  }

  powerTitle(power: PowerCheck): string {
    return `Rough power check: its rules text is worth about ${power.estimate} mana; ` +
      `a card with this cost and rarity usually gets about ${power.budget}.`;
  }

  trackSlot(_index: number, slot: PoolSlotView): string {
    return slot.id;
  }

  trackEntry(_index: number, entry: PoolEntryView): string {
    return entry.id;
  }

  trackCard(_index: number, card: CardView): string {
    return card.id;
  }

  // ----- loading -----
  private run(slot: PoolSlotView, request: Observable<PoolView>, fallback: string, done?: () => void): void {
    if (!this.isOpen || this.busy.has(slot.id)) {
      return;
    }
    this.error = null;
    this.busy.add(slot.id);
    request.pipe(finalize(() => this.busy.delete(slot.id))).subscribe({
      next: pool => {
        this.pool = pool;
        done?.();
      },
      error: err => {
        this.error = apiErrorMessage(err, fallback);
        this.refresh();
      }
    });
  }

  private loadMyCards(): void {
    this.myCardsError = null;
    this.generation.myCards().subscribe({
      next: cards => (this.myCards = cards),
      error: err => (this.myCardsError = apiErrorMessage(err, 'Could not load your cards.'))
    });
  }

  private load(): Observable<PoolView | null> {
    const request: Observable<PoolView | null> = this.routeId
      ? this.pools.get(this.routeId)
      : this.pools.current().pipe(
        switchMap(current => {
          if (current || !this.lastPoolId) {
            return of(current);
          }
          // The pool we were showing is no longer current: it was closed. Show its results.
          return this.pools.get(this.lastPoolId).pipe(catchError(() => of(null)));
        })
      );
    return request.pipe(
      catchError(err => {
        this.loadError = apiErrorMessage(err, 'Could not load the Knowledge Pool.');
        return EMPTY;
      })
    );
  }

  private apply(pool: PoolView | null): void {
    this.pool = pool;
    this.loadError = null;
    if (pool) {
      this.lastPoolId = pool.id;
    }
  }

  private loadPastPools(): void {
    this.pools.list().subscribe({ next: list => (this.pastPools = list), error: () => (this.pastPools = []) });
  }
}
