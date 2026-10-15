import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { SharedCardView } from '../../models/api.model';
import { apiErrorMessage, safeFileName } from '../../services/api.util';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';

/** /community: the Community Gallery, every card players have shared, with its maker. */
@Component({
  selector: 'app-community-page',
  templateUrl: './community-page.component.html',
  styleUrls: ['../gallery-page/gallery-page.component.scss']
})
export class CommunityPageComponent implements OnInit, OnDestroy {
  cards: SharedCardView[] | null = null;
  error: string | null = null;

  private sub?: Subscription;

  constructor(private generation: GenerationService, private media: MediaService) {}

  ngOnInit(): void {
    this.sub = this.generation.sharedCards().subscribe({
      next: cards => (this.cards = cards),
      error: err => (this.error = apiErrorMessage(err, 'Could not load shared cards.'))
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  download(view: SharedCardView): void {
    this.media.download(view.cardImageUrl, `${safeFileName(view.card?.name)}_card.png`);
  }

  downloadArt(view: SharedCardView): void {
    this.media.download(view.artImageUrl, `${safeFileName(view.card?.name)}_artwork.png`);
  }

  trackCard(_index: number, view: SharedCardView): string {
    return view.id;
  }
}
