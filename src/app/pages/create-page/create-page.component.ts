import { Component, ViewChild, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Subscription, finalize, switchMap, throwError } from 'rxjs';
import { CardView, PoolView } from '../../models/api.model';
import { Card, Rarity } from '../../models/card.model';
import { CardFormComponent } from '../../components/card-form/card-form.component';
import { apiErrorMessage, safeFileName } from '../../services/api.util';
import { isPainting, queueLine } from '../../services/card-status';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { PoolService } from '../../services/pool.service';
import { QueueService } from '../../services/queue.service';

/**
 * Free-play generation (/create): today's page, now submitting a `count: 1` job and
 * polling it. The result is saved server-side and appears in the Gallery.
 */
@Component({
  selector: 'app-create-page',
  templateUrl: './create-page.component.html',
  styleUrls: ['./create-page.component.scss']
})
export class CreatePageComponent implements OnInit, OnDestroy {
  @ViewChild(CardFormComponent) cardFormComponent!: CardFormComponent;

  currentCard: Card = {
    name: 'New Card',
    manaCost: '{1}',
    type: 'Creature',
    colors: ['C'],
    cmc: 1,
    rarity: Rarity.COMMON,
    artPrompt: 'Fantasy art of new card',
    description: 'Your custom Magic: The Gathering card will appear here.'
  };

  // Loading state
  isGenerating = false;
  /** Latest view of the running job (queue position, ETA, checklist). */
  jobView: CardView | null = null;
  sharing = false;
  /** The open Knowledge Pool (null when none), for the Submit to pool button. */
  pool: PoolView | null = null;

  // Message rotation properties
  currentMessage = { title: 'Generating Your Magic Card', subtitle: 'Creating artwork and card text with AI...' };
  generationMessages: any[] = [];
  messageInterval: any = null;
  messageVisible = true;

  // Error state
  error: string | null = null;

  // Success state
  successMessage: string | null = null;

  // Card transition state
  cardExiting: boolean = false;

  // Server status (driven by QueueService.online$)
  isCheckingHealth = true;
  healthError: string | null = null;
  modelsReady = false;
  showHealthBanner = false;

  private jobSub?: Subscription;
  private onlineSub?: Subscription;
  private bannerTimer: any = null;
  private successTimer: any = null;

  constructor(
    private generation: GenerationService,
    private queue: QueueService,
    private media: MediaService,
    private http: HttpClient,
    private pools: PoolService
  ) {}

  ngOnInit(): void {
    // Add a delay before showing the banner to prevent flash
    this.bannerTimer = setTimeout(() => {
      this.showHealthBanner = true;
    }, 500);

    this.loadGenerationMessages();
    this.pools.current().subscribe({ next: pool => (this.pool = pool), error: () => (this.pool = null) });
    this.onlineSub = this.queue.online$.subscribe(online => {
      this.isCheckingHealth = false;
      this.modelsReady = online;
      this.healthError = online ? null : 'Cannot reach the generation server. Retrying every few seconds…';
    });
  }

  ngOnDestroy(): void {
    this.jobSub?.unsubscribe();
    this.onlineSub?.unsubscribe();
    this.stopMessageRotation();
    clearTimeout(this.bannerTimer);
    clearTimeout(this.successTimer);
  }

  updateCard(card: Card): void {
    // If we had a complete card and form changed, trigger exit animation
    if (this.currentCard.cardImageUrl && !card.cardImageUrl) {
      this.cardExiting = true;

      // After animation completes, update the card
      setTimeout(() => {
        this.currentCard = { ...card };
        this.cardExiting = false;
      }, 600); // Match animation duration
    } else {
      // No animation needed, update immediately
      this.currentCard = { ...card };
    }

    // Clear any previous errors or success messages when card is updated
    this.error = null;
    this.successMessage = null;
  }

  generateCard(card: Card): void {
    if (this.isGenerating) {
      return;
    }
    if (!this.modelsReady) {
      this.error = 'The server is offline. Please wait for it to come back.';
      this.cardFormComponent?.setGenerating(false);
      return;
    }

    this.error = null;
    this.successMessage = null;
    this.isGenerating = true;
    this.jobView = null;
    this.startMessageRotation();

    const base: Card = { ...card, artPrompt: this.generation.promptFor(card, this.currentCard) };

    this.jobSub = this.generation.submit({
      prompt: base.artPrompt!,
      cardData: this.generation.cardParams(card),
      count: 1
    }).pipe(
      switchMap(response => {
        const job = response.cards[0];
        if (!job) {
          return throwError(() => new Error('The server did not return a card.'));
        }
        this.jobView = job;
        return this.generation.watch(job.id);
      }),
      finalize(() => {
        this.isGenerating = false;
        this.stopMessageRotation();
        // Reset the form's loading state
        this.cardFormComponent?.setGenerating(false);
      })
    ).subscribe({
      next: view => {
        this.jobView = view;
        if (view.status === 'done') {
          this.currentCard = this.generation.toCard(view, base);
          this.showSuccess('Card generated successfully! It is saved in your Gallery.');
        } else if (view.status === 'failed') {
          this.error = `Failed to generate card: ${view.error || 'unknown error'}`;
        }
      },
      error: err => {
        this.error = `Failed to generate card: ${apiErrorMessage(err)}`;
      }
    });
  }

  /** Shares the finished card to the gallery's Community tab, or takes it back. */
  toggleShare(): void {
    const job = this.jobView;
    if (!job || job.status !== 'done' || this.sharing) {
      return;
    }
    this.sharing = true;
    this.generation.share(job.id, !job.shared).pipe(
      finalize(() => (this.sharing = false))
    ).subscribe({
      next: view => {
        this.jobView = view;
        if (view.shared) {
          this.showSuccess('Shared! Everyone can see it on the Community tab of the Gallery.');
        }
      },
      error: err => (this.error = apiErrorMessage(err, 'Could not share the card.'))
    });
  }

  /** The finished card went into the pool: keep the pool and the card's entry in step. */
  onPoolSubmitted(pool: PoolView): void {
    this.pool = pool;
    const job = this.jobView;
    const entry = job && pool.entries.find(e => e.cardId === job.id);
    if (job && entry) {
      this.jobView = { ...job, poolEntryId: entry.id };
    }
  }

  /** Queue position / ETA line for the running job. */
  jobStatusLine(job: CardView): string {
    const queued = queueLine(job);
    if (queued) {
      return queued;
    }
    if (isPainting(job)) {
      return 'Painting your artwork…';
    }
    switch (job.status) {
      case 'queued': return 'Queued…';
      case 'generating': return job.artReady ? 'Writing rules text…' : 'Generating…';
      case 'rendering': return 'Rendering…';
      default: return '';
    }
  }

  // Clear error messages
  clearError(): void {
    this.error = null;
  }

  // Clear success message
  clearSuccessMessage(): void {
    this.successMessage = null;
  }

  // Download artwork image
  downloadCardImage(): void {
    if (!this.currentCard.imageUrl) {
      return;
    }
    this.media.download(this.currentCard.imageUrl, `${safeFileName(this.currentCard.name)}_artwork.png`);
    this.showSuccess('Artwork downloaded successfully!');
  }

  // Download complete card image
  downloadCompleteCard(): void {
    if (!this.currentCard.cardImageUrl) {
      return;
    }
    this.media.download(this.currentCard.cardImageUrl, `${safeFileName(this.currentCard.name)}_complete_card.png`);
    this.showSuccess('Complete card downloaded successfully!');
  }

  loadGenerationMessages(): void {
    this.http.get<{messages: any[]}>('assets/generation-messages.json')
      .subscribe({
        next: (data) => {
          this.generationMessages = data.messages;
        },
        error: (error) => {
          console.error('Failed to load generation messages:', error);
          // Fallback messages if file fails to load
          this.generationMessages = [
            { title: 'Generating Your Magic Card', subtitle: 'Creating artwork and card text with AI...' },
            { title: 'Shuffling the Digital Deck', subtitle: 'No mulligans needed here!' },
            { title: 'Tapping Mana Sources', subtitle: 'Converting coffee into card magic...' }
          ];
        }
      });
  }

  startMessageRotation(): void {
    this.stopMessageRotation();
    // Reset to first message
    this.currentMessage = this.generationMessages.length > 0
      ? this.generationMessages[0]
      : { title: 'Generating Your Magic Card', subtitle: 'Creating artwork and card text with AI...' };
    this.messageVisible = true;

    if (this.generationMessages.length <= 1) return;

    let messageIndex = 0;
    this.messageInterval = setInterval(() => {
      // Fade out current message
      this.messageVisible = false;

      // After fade out completes, change message and fade back in
      setTimeout(() => {
        messageIndex = (messageIndex + 1) % this.generationMessages.length;
        this.currentMessage = this.generationMessages[messageIndex];
        this.messageVisible = true;
      }, 300); // 300ms to match CSS transition

    }, 5000); // Change message every 5 seconds
  }

  stopMessageRotation(): void {
    if (this.messageInterval) {
      clearInterval(this.messageInterval);
      this.messageInterval = null;
    }
  }

  private showSuccess(message: string): void {
    this.successMessage = message;
    clearTimeout(this.successTimer);
    // Clear success message after 20 seconds
    this.successTimer = setTimeout(() => {
      this.successMessage = null;
    }, 20000);
  }
}
