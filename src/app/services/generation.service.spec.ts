import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { environment } from '../../environments/environment';
import { CardView, GenerationRequest } from '../models/api.model';
import { Card, Rarity } from '../models/card.model';
import { fakeVisibility } from '../testing/fake-visibility';
import { cardView, doneCard } from '../testing/fixtures';
import { GenerationService, WATCH_REQUEST_TIMEOUT_MS } from './generation.service';
import { PageVisibilityService } from './page-visibility.service';

describe('GenerationService', () => {
  let service: GenerationService;
  let http: HttpTestingController;
  let visibility: ReturnType<typeof fakeVisibility>;
  const base = `${environment.apiUrl}/api/v1`;

  beforeEach(() => {
    visibility = fakeVisibility();
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [{ provide: PageVisibilityService, useValue: visibility }]
    });
    service = TestBed.inject(GenerationService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('submit POSTs the request to /generations', () => {
    const req: GenerationRequest = {
      prompt: 'Fantasy art of a dragon',
      cardData: { name: 'Dragon', manaCost: '{4}{R}', colors: ['R'], type: 'Creature', rarity: 'rare', cmc: 5 },
      count: 1
    };
    let setId: string | null | undefined;
    service.submit(req).subscribe(r => (setId = r.setId));
    const call = http.expectOne(`${base}/generations`);
    expect(call.request.method).toBe('POST');
    expect(call.request.body).toEqual(req);
    call.flush({ setId: null, cards: [cardView()] });
    expect(setId).toBeNull();
  });

  it('reroll POSTs to /cards/<id>/reroll and returns the new card', () => {
    let fresh: CardView | undefined;
    service.reroll('c-1').subscribe(v => (fresh = v));
    const call = http.expectOne(`${base}/cards/c-1/reroll`);
    expect(call.request.method).toBe('POST');
    call.flush(cardView({ id: 'c-2' }));
    expect(fresh?.id).toBe('c-2');
  });

  it('watch polls every 2s, emits queued -> generating -> done, then completes', fakeAsync(() => {
    const statuses: string[] = [];
    let completed = false;
    service.watch('c-1').subscribe({ next: v => statuses.push(v.status), complete: () => (completed = true) });

    tick(2000);
    http.expectOne(`${base}/cards/c-1`).flush(cardView({ status: 'queued' }));
    tick(2000);
    http.expectOne(`${base}/cards/c-1`).flush(cardView({ status: 'generating', queuePosition: 0 }));
    tick(2000);
    http.expectOne(`${base}/cards/c-1`).flush(doneCard());

    expect(statuses).toEqual(['queued', 'generating', 'done']);
    expect(completed).toBeTrue();

    tick(10000);
    http.expectNone(`${base}/cards/c-1`);
  }));

  it('watch completes after failed', fakeAsync(() => {
    let last: CardView | undefined;
    let completed = false;
    service.watch('c-1').subscribe({ next: v => (last = v), complete: () => (completed = true) });
    tick(2000);
    http.expectOne(`${base}/cards/c-1`).flush(cardView({ status: 'failed', error: 'Ollama is down' }));
    expect(last?.error).toBe('Ollama is down');
    expect(completed).toBeTrue();
  }));

  it('watch keeps polling through a network blip', fakeAsync(() => {
    const statuses: string[] = [];
    service.watch('c-1').subscribe(v => statuses.push(v.status));
    tick(2000);
    http.expectOne(`${base}/cards/c-1`).error(new ProgressEvent('error'), { status: 0 });
    tick(2000);
    http.expectOne(`${base}/cards/c-1`).flush(doneCard());
    expect(statuses).toEqual(['done']);
  }));

  it('watch errors on a 404 (card gone)', fakeAsync(() => {
    let error: any;
    service.watch('c-1').subscribe({ error: e => (error = e) });
    tick(2000);
    http.expectOne(`${base}/cards/c-1`).flush({ error: 'Not found' }, { status: 404, statusText: 'Not Found' });
    expect(error?.status).toBe(404);
    tick(4000);
    http.expectNone(`${base}/cards/c-1`);
  }));

  it('toCard maps the view onto the display card and prefixes media URLs', () => {
    const base: Card = {
      name: 'Old', manaCost: '{1}', type: 'Creature', colors: ['C'], cmc: 1,
      rarity: Rarity.COMMON, artPrompt: 'Fantasy art of x'
    };
    const view = doneCard({
      id: 'x',
      card: {
        name: 'Ember Queen', manaCost: '{2}{R}', colors: ['R'], type: 'Creature', subtype: 'Elemental',
        rarity: 'mythic', cmc: 3, description: 'Haste', flavorText: 'Burn bright.', power: '3', toughness: '2'
      }
    });

    const card = service.toCard(view, base);

    expect(card.cardImageUrl).toBe(`${environment.apiUrl}/api/v1/media/cards/x.png`);
    expect(card.imageUrl).toBe(`${environment.apiUrl}/api/v1/media/art/x.png`);
    expect(card.name).toBe('Ember Queen');
    expect(card.description).toBe('Haste');
    expect(card.flavorText).toBe('Burn bright.');
    expect(card.rarity).toBe(Rarity.MYTHIC);
    expect(card.power).toBe('3');
    expect(card.artPrompt).toBe('Fantasy art of x');
  });

  it('toCard keeps the base when the view has no card or images yet', () => {
    const base: Card = { name: 'Keep', manaCost: '{1}', type: 'Creature', colors: [], cmc: 1, rarity: Rarity.RARE };
    const card = service.toCard(cardView({ card: null }), base);
    expect(card.name).toBe('Keep');
    expect(card.cardImageUrl).toBeUndefined();
    expect(card.imageUrl).toBeUndefined();
  });

  it('watch drops a request with no answer after 15s and keeps polling (phone slept, tunnel stalled)', fakeAsync(() => {
    expect(WATCH_REQUEST_TIMEOUT_MS).toBe(15000);
    const statuses: string[] = [];
    let error: unknown;
    service.watch('c-1').subscribe({ next: v => statuses.push(v.status), error: e => (error = e) });

    tick(2000);
    const hung = http.expectOne(`${base}/cards/c-1`);
    tick(14999);
    expect(hung.cancelled).toBeFalse();
    tick(1);
    expect(hung.cancelled).toBeTrue();   // timed out: treated like a network blip
    expect(error).toBeUndefined();

    tick(1000);                           // the next 2s tick (t=18s) polls again
    http.expectOne(`${base}/cards/c-1`).flush(doneCard());
    expect(statuses).toEqual(['done']);
  }));

  it('watch pauses while the page is hidden and polls at once when it is shown again', fakeAsync(() => {
    const statuses: string[] = [];
    const sub = service.watch('c-1').subscribe(v => statuses.push(v.status));
    tick(2000);
    http.expectOne(`${base}/cards/c-1`).flush(cardView({ status: 'queued' }));

    visibility.visibleSubject.next(false);
    tick(60000);
    http.expectNone(`${base}/cards/c-1`);

    visibility.visibleSubject.next(true);
    http.expectOne(`${base}/cards/c-1`).flush(doneCard());
    expect(statuses).toEqual(['queued', 'done']);
    sub.unsubscribe();
  }));
});
