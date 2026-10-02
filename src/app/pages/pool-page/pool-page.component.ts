import { Component, OnDestroy, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import {
  EMPTY, Observable, Subject, Subscription, catchError, finalize, of, startWith, switchMap
} from 'rxjs';
import { Medal, PoolEntryView, PoolSummary, PoolView, PowerCheck } from '../../models/api.model';
import { apiErrorMessage, safeFileName } from '../../services/api.util';
import { MediaService } from '../../services/media.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { PoolService } from '../../services/pool.service';

export const POOL_POLL_MS = 10000;

export const MEDALS: { medal: Medal; label: string; points: number }[] = [
  { medal: 'gold', label: 'Gold', points: 3 },
  { medal: 'silver', label: 'Silver', points: 2 },
  { medal: 'bronze', label: 'Bronze', points: 1 }
];

/**
 * /pool: the open Knowledge Pool (spec docs/superpowers/specs/2026-09-29-knowledge-pool-design.md).
 * Every entry in one ranked list; each player gives one gold, silver and bronze (3/2/1, never
 * their own card) and the top floor(players / 2) make the pool. Cards are submitted from the
 * create screen and the gallery. /pool/:id shows a pool read-only. Polls every 10s while visible.
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
  /** True while a medal or withdraw request is in flight (guards double clicks). */
  busy = false;
  readonly medals = MEDALS;

  private readonly refresh$ = new Subject<void>();
  private routeId: string | null = null;
  private lastPoolId: string | null = null;
  private sub?: Subscription;
  private routeSub?: Subscription;

  constructor(private pools: PoolService, private media: MediaService,
              private visibility: PageVisibilityService, private route: ActivatedRoute) {}

  ngOnInit(): void {
    this.loadPastPools();
    this.routeSub = this.route.paramMap.subscribe(params => {
      this.routeId = params.get('id');
      this.pool = undefined;
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

  /** The cards that make the pool (so far, while it's open). */
  get inEntries(): PoolEntryView[] {
    return this.pool?.entries.filter(e => e.in) ?? [];
  }

  /** Index of the last entry that is in; the pool line is drawn after it (-1: no line). */
  get lineIndex(): number {
    const entries = this.pool?.entries ?? [];
    for (let i = entries.length - 1; i >= 0; i--) {
      if (entries[i].in) {
        return i;
      }
    }
    return -1;
  }

  /** Medals I've given, for the header chips. */
  medalGiven(medal: Medal): boolean {
    return !!this.pool?.myMedals[medal];
  }

  canVote(entry: PoolEntryView): boolean {
    return this.isOpen && !entry.mine;
  }

  refresh(): void {
    this.refresh$.next();
  }

  // ----- actions -----
  /** Give the card this medal (moving it off another card), or take it back if it's already there. */
  award(entry: PoolEntryView, medal: Medal): void {
    if (entry.myMedal === medal) {
      this.run(this.pools.clearMedal(entry.id), 'Could not take your medal back.');
    } else {
      this.run(this.pools.medal(entry.id, medal), 'Your medal did not go through. Please try again.');
    }
  }

  withdraw(entry: PoolEntryView): void {
    const name = entry.card.card?.name || 'this card';
    if (!window.confirm(`Withdraw "${name}" from the pool? Its medals are cleared.`)) {
      return;
    }
    this.run(this.pools.withdraw(entry.id), 'Could not withdraw the card.');
  }

  medalTitle(entry: PoolEntryView, medal: { medal: Medal; points: number }): string {
    if (entry.myMedal === medal.medal) {
      return `Take your ${medal.medal} back`;
    }
    return this.pool?.myMedals[medal.medal]
      ? `Move your ${medal.medal} here (${medal.points} pts)`
      : `Give ${medal.medal} (${medal.points} pts)`;
  }

  download(entry: PoolEntryView): void {
    this.media.download(entry.card.cardImageUrl, `${safeFileName(entry.card.card?.name)}_card.png`);
  }

  clearError(): void {
    this.error = null;
  }

  // ----- display helpers -----
  powerLabel(power: PowerCheck): string {
    return power.verdict === 'over' ? 'Over the curve' : power.verdict === 'pushed' ? 'Pushed' : 'Fair';
  }

  powerTitle(power: PowerCheck): string {
    return `Rough power check: its rules text is worth about ${power.estimate} mana; ` +
      `a card with this cost and rarity usually gets about ${power.budget}.`;
  }

  trackEntry(_index: number, entry: PoolEntryView): string {
    return entry.id;
  }

  // ----- loading -----
  private run(request: Observable<PoolView>, fallback: string): void {
    if (!this.isOpen || this.busy) {
      return;
    }
    this.error = null;
    this.busy = true;
    request.pipe(finalize(() => (this.busy = false))).subscribe({
      next: pool => (this.pool = pool),
      error: err => {
        this.error = apiErrorMessage(err, fallback);
        this.refresh();
      }
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
