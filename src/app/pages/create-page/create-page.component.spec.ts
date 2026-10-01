import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { BehaviorSubject, Subject, of, throwError } from 'rxjs';
import { environment } from '../../../environments/environment';
import { CardView, GenerationResponse } from '../../models/api.model';
import { Card, Rarity } from '../../models/card.model';
import { MediaPipe } from '../../pipes/media.pipe';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { QueueService } from '../../services/queue.service';
import { cardView, doneCard } from '../../testing/fixtures';
import { CreatePageComponent } from './create-page.component';

describe('CreatePageComponent (jobs)', () => {
  let fixture: ComponentFixture<CreatePageComponent>;
  let component: CreatePageComponent;
  let gen: jasmine.SpyObj<GenerationService>;
  let watch$: Subject<CardView>;
  let online$: BehaviorSubject<boolean>;

  const card: Card = {
    name: 'Ember Queen', manaCost: '{2}{R}', type: 'Creature', colors: ['R'], cmc: 3,
    rarity: Rarity.MYTHIC, artPrompt: 'Fantasy art of Ember Queen', description: 'Haste'
  };

  beforeEach(() => {
    watch$ = new Subject<CardView>();
    online$ = new BehaviorSubject<boolean>(true);
    gen = jasmine.createSpyObj<GenerationService>('GenerationService',
      ['submit', 'watch', 'toCard', 'cardParams', 'promptFor', 'share']);
    gen.watch.and.returnValue(watch$);
    gen.cardParams.and.callFake(GenerationService.prototype.cardParams);
    gen.promptFor.and.callFake(GenerationService.prototype.promptFor);
    gen.toCard.and.callFake((v: CardView, b: Card) => ({
      ...b, cardImageUrl: `${environment.apiUrl}${v.cardImageUrl}`, imageUrl: `${environment.apiUrl}${v.artImageUrl}`
    }));
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src', 'download']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));

    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      declarations: [CreatePageComponent, MediaPipe],
      providers: [
        { provide: GenerationService, useValue: gen },
        { provide: QueueService, useValue: { online$, status$: of(null) } },
        { provide: MediaService, useValue: media }
      ],
      schemas: [NO_ERRORS_SCHEMA]
    });
    fixture = TestBed.createComponent(CreatePageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  afterEach(() => fixture.destroy());

  it('models-ready follows queue online$', () => {
    expect(component.modelsReady).toBeTrue();
    online$.next(false);
    expect(component.modelsReady).toBeFalse();
    expect(component.healthError).toBeTruthy();
  });

  it('submits count 1 with the card params, then watches the job to done', () => {
    const response: GenerationResponse = { setId: null, cards: [cardView({ id: 'j-1', queuePosition: 3, etaSeconds: 30 })] };
    gen.submit.and.returnValue(of(response));

    component.generateCard(card);

    const req = gen.submit.calls.mostRecent().args[0];
    expect(req.count).toBe(1);
    expect(req.prompt).toBe('Fantasy art of Ember Queen');
    expect(req.cardData.name).toBe('Ember Queen');
    expect(req.cardData.rarity).toBe('mythic');
    expect((req.cardData as any).artPrompt).toBeUndefined();
    expect(gen.watch).toHaveBeenCalledWith('j-1');
    expect(component.isGenerating).toBeTrue();

    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Queued #3 · ~30s');

    watch$.next(cardView({ id: 'j-1', status: 'generating', queuePosition: 0, etaSeconds: 5 }));
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Painting your artwork');

    watch$.next(doneCard({ id: 'j-1' }));
    watch$.complete();
    fixture.detectChanges();

    expect(component.isGenerating).toBeFalse();
    expect(component.currentCard.cardImageUrl).toBe(`${environment.apiUrl}/api/v1/media/cards/j-1.png`);
    expect(component.successMessage).toContain('generated');
  });

  it('offers Share once the card is done and toggles it', () => {
    gen.submit.and.returnValue(of({ setId: null, cards: [cardView({ id: 'j-1' })] }));
    component.generateCard(card);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('button.share-card')).toBeNull();

    watch$.next(doneCard({ id: 'j-1' }));
    watch$.complete();
    fixture.detectChanges();
    const button = () => fixture.nativeElement.querySelector('button.share-card') as HTMLButtonElement;
    expect(button().textContent).toContain('Share to gallery');

    gen.share.and.returnValue(of(doneCard({ id: 'j-1', shared: true })));
    button().click();
    fixture.detectChanges();
    expect(gen.share).toHaveBeenCalledWith('j-1', true);
    expect(button().textContent).toContain('Shared');

    gen.share.and.returnValue(of(doneCard({ id: 'j-1', shared: false })));
    button().click();
    expect(gen.share).toHaveBeenCalledWith('j-1', false);
  });

  it('shows the failure reason when the job fails', () => {
    gen.submit.and.returnValue(of({ setId: null, cards: [cardView({ id: 'j-2' })] }));
    component.generateCard(card);
    watch$.next(cardView({ id: 'j-2', status: 'failed', error: 'Text generation failed: Ollama unreachable' }));
    watch$.complete();
    expect(component.isGenerating).toBeFalse();
    expect(component.error).toContain('Ollama unreachable');
  });

  it('shows the server text on a 429', () => {
    gen.submit.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 429, error: { error: 'You already have 3 cards in progress' }
    })));
    component.generateCard(card);
    expect(component.error).toContain('You already have 3 cards in progress');
    expect(component.isGenerating).toBeFalse();
  });

  it('has no text-only or art-only generation paths any more', () => {
    expect((component as any).generateCardText).toBeUndefined();
    expect((component as any).generateCardArt).toBeUndefined();
  });
});
