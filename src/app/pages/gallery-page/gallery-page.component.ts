import { Component, OnDestroy, OnInit } from '@angular/core';
import { EMPTY, Observable, Subscription, catchError, exhaustMap, filter, interval } from 'rxjs';
import { CardView } from '../../models/api.model';
import { apiErrorMessage, safeFileName } from '../../services/api.util';
import { isPending } from '../../services/card-status';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';

export const GALLERY_POLL_MS = 5000;

/** /gallery: every card I've generated (spec §7), newest first, with downloads. */
@Component({
  selector: 'app-gallery-page',
  templateUrl: './gallery-page.component.html',
  styleUrls: ['./gallery-page.component.scss']
})
export class GalleryPageComponent implements OnInit, OnDestroy {
  cards: CardView[] | null = null;
  error: string | null = null;

  private loadSub?: Subscription;
  private pollSub?: Subscription;

  constructor(private generation: GenerationService, private media: MediaService) {}

  ngOnInit(): void {
    this.loadSub = this.fetch().subscribe(cards => this.apply(cards));
    // Refresh while any card is still being made, so tiles finish on their own.
    this.pollSub = interval(GALLERY_POLL_MS).pipe(
      filter(() => this.hasPending()),
      exhaustMap(() => this.fetch())
    ).subscribe(cards => this.apply(cards));
  }

  ngOnDestroy(): void {
    this.loadSub?.unsubscribe();
    this.pollSub?.unsubscribe();
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
