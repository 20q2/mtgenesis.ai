import { Component, OnDestroy, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { EMPTY, Observable, Subscription, catchError, exhaustMap, filter, finalize, timeout } from 'rxjs';
import { CardView, PoolView, SharedCardView } from '../../models/api.model';
import { apiErrorMessage, safeFileName } from '../../services/api.util';
import { isPending } from '../../services/card-status';
import { GenerationService, isTransientError } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { PoolService } from '../../services/pool.service';

export const GALLERY_POLL_MS = 5000;
export const GALLERY_REQUEST_TIMEOUT_MS = 15000;

export type GalleryTab = 'mine' | 'community';

/**
 * /gallery: every card I've generated (spec §7), newest first, with downloads and a
 * Share toggle; ?tab=community shows everyone's shared cards with their maker.
 */
@Component({
  selector: 'app-gallery-page',
  templateUrl: './gallery-page.component.html',
  styleUrls: ['./gallery-page.component.scss']
})
export class GalleryPageComponent implements OnInit, OnDestroy {
  tab: GalleryTab = 'mine';
  cards: CardView[] | null = null;
  /** Loaded the first time the Community tab opens. */
  communityCards: SharedCardView[] | null = null;
  /** Card ids with a share/unshare request in flight. */
  sharing = new Set<string>();
  /** The open Knowledge Pool (null when none), for the Submit to pool buttons. */
  pool: PoolView | null = null;
  error: string | null = null;

  private loadSub?: Subscription;
  private pollSub?: Subscription;
  private communitySub?: Subscription;

  constructor(private generation: GenerationService, private media: MediaService,
              private visibility: PageVisibilityService, private route: ActivatedRoute,
              private router: Router, private pools: PoolService) {}

  ngOnInit(): void {
    this.pools.current().subscribe({ next: pool => (this.pool = pool), error: () => (this.pool = null) });
    if (this.route.snapshot.queryParamMap.get('tab') === 'community') {
      this.selectTab('community');
    }
    this.loadSub = this.fetch().subscribe(cards => this.apply(cards));
    // Refresh while any card is still being made, so tiles finish on their own. Paused
    // while the page is hidden; a poll unanswered after 15s is dropped and retried.
    this.pollSub = this.visibility.poll(GALLERY_POLL_MS, GALLERY_POLL_MS).pipe(
      filter(() => this.hasPending()),
      exhaustMap(() => this.generation.myCards().pipe(
        timeout(GALLERY_REQUEST_TIMEOUT_MS),
        catchError(err => {
          if (!isTransientError(err)) {
            this.error = apiErrorMessage(err, 'Could not load your cards.');
          }
          return EMPTY;
        })
      ))
    ).subscribe(cards => this.apply(cards));
  }

  ngOnDestroy(): void {
    this.loadSub?.unsubscribe();
    this.pollSub?.unsubscribe();
    this.communitySub?.unsubscribe();
  }

  selectTab(tab: GalleryTab): void {
    if (tab === this.tab && (tab === 'mine' || this.communityCards)) {
      return;
    }
    this.tab = tab;
    this.router.navigate([], {
      relativeTo: this.route, replaceUrl: true,
      queryParams: { tab: tab === 'community' ? 'community' : null }
    });
    if (tab === 'community' && !this.communityCards) {
      this.loadCommunity();
    }
  }

  toggleShare(view: CardView): void {
    if (this.sharing.has(view.id)) {
      return;
    }
    this.sharing.add(view.id);
    this.generation.share(view.id, !view.shared).pipe(
      finalize(() => this.sharing.delete(view.id))
    ).subscribe({
      next: updated => {
        this.cards = (this.cards ?? []).map(c => (c.id === updated.id ? updated : c));
        this.communityCards = null; // reloaded next time the tab opens
      },
      error: err => (this.error = apiErrorMessage(err, 'Could not change sharing for that card.'))
    });
  }

  hasPending(): boolean {
    return !!this.cards?.some(c => isPending(c.status));
  }

  label(view: CardView): string {
    return view.setId ? `Commander set · Version ${view.slot ?? '?'}` : 'Free play';
  }

  download(view: CardView): void {
    this.media.download(view.cardImageUrl, `${safeFileName(view.card?.name)}_card.png`);
  }

  downloadArt(view: CardView): void {
    this.media.download(view.artImageUrl, `${safeFileName(view.card?.name)}_artwork.png`);
  }

  trackCard(_index: number, view: CardView): string {
    return view.id;
  }

  /** A card went into the pool: keep the pool and that card's entry in step. */
  onPoolSubmitted(view: CardView, pool: PoolView): void {
    this.pool = pool;
    const entry = pool.entries.find(e => e.cardId === view.id);
    if (entry) {
      this.cards = (this.cards ?? []).map(c => (c.id === view.id ? { ...c, poolEntryId: entry.id } : c));
    }
  }

  private loadCommunity(): void {
    this.communitySub?.unsubscribe();
    this.communitySub = this.generation.sharedCards().subscribe({
      next: cards => (this.communityCards = cards),
      error: err => (this.error = apiErrorMessage(err, 'Could not load shared cards.'))
    });
  }

  private fetch(): Observable<CardView[]> {
    return this.generation.myCards().pipe(
      catchError(err => {
        this.error = apiErrorMessage(err, 'Could not load your cards.');
        return EMPTY;
      })
    );
  }

  private apply(cards: CardView[]): void {
    this.error = null;
    // Newest first (the server already sorts; this keeps the page correct regardless).
    this.cards = [...cards].sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
  }
}
