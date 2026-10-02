import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { of } from 'rxjs';
import { environment } from '../../../environments/environment';
import { CardSlotComponent } from '../../components/card-slot/card-slot.component';
import { PoolSubmitComponent } from '../../components/pool-submit/pool-submit.component';
import { MediaPipe } from '../../pipes/media.pipe';
import { MediaService } from '../../services/media.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { fakeVisibility } from '../../testing/fake-visibility';
import { cardView, doneCard, poolEntry, poolView } from '../../testing/fixtures';
import { GALLERY_POLL_MS, GALLERY_REQUEST_TIMEOUT_MS, GalleryPageComponent } from './gallery-page.component';

describe('GalleryPageComponent', () => {
  let fixture: ComponentFixture<GalleryPageComponent>;
  let http: HttpTestingController;
  let media: jasmine.SpyObj<MediaService>;
  const url = `${environment.apiUrl}/api/v1/me/cards`;

  beforeEach(() => {
    media = jasmine.createSpyObj<MediaService>('MediaService', ['src', 'download']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule, RouterTestingModule],
      declarations: [GalleryPageComponent, CardSlotComponent, PoolSubmitComponent, MediaPipe],
      providers: [
        { provide: MediaService, useValue: media },
        { provide: PageVisibilityService, useValue: fakeVisibility() }
      ]
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(GalleryPageComponent);
    fixture.detectChanges();
    http.expectOne(`${environment.apiUrl}/api/v1/pools/current`).flush(poolView({ myEntryCount: 1 }));
  });

  afterEach(() => fixture.destroy());

  const el = () => fixture.nativeElement as HTMLElement;

  it('renders one tile per card, newest first', () => {
    http.expectOne(url).flush([
      doneCard({ id: 'old', createdAt: '2026-09-28T18:00:00+00:00' }),
      cardView({ id: 'newest', status: 'queued', createdAt: '2026-09-28T21:00:00+00:00' }),
      doneCard({ id: 'mid', createdAt: '2026-09-28T20:00:00+00:00', replaced: true, setId: 's-1', slot: 2 })
    ]);
    fixture.detectChanges();

    const tiles = Array.from(el().querySelectorAll('.tile')) as HTMLElement[];
    expect(tiles.map(t => t.dataset['cardId'])).toEqual(['newest', 'mid', 'old']);
  });

  it('done cards show their image and a download link; pending cards show their status', () => {
    http.expectOne(url).flush([
      doneCard({ id: 'd1', createdAt: '2026-09-28T20:00:00+00:00' }),
      cardView({ id: 'p1', status: 'queued', queuePosition: 2, etaSeconds: 20, createdAt: '2026-09-28T19:00:00+00:00' })
    ]);
    fixture.detectChanges();

    const done = el().querySelector('[data-card-id="d1"]') as HTMLElement;
    const pending = el().querySelector('[data-card-id="p1"]') as HTMLElement;
    expect(done.querySelector('img.slot-image')).not.toBeNull();
    expect(pending.textContent).toContain('Queued #2');
    expect(pending.querySelector('.download')).toBeNull();

    (done.querySelector('.download') as HTMLElement).click();
    expect(media.download).toHaveBeenCalledWith('/api/v1/media/cards/d1.png', jasmine.stringMatching(/\.png$/));
  });

  it('labels rerolled and set cards', () => {
    http.expectOne(url).flush([
      doneCard({ id: 'r', replaced: true, setId: 's-1', slot: 3 }),
      doneCard({ id: 'f', setId: null })
    ]);
    fixture.detectChanges();
    expect(el().querySelector('[data-card-id="r"]')!.textContent).toContain('Rerolled');
    expect(el().querySelector('[data-card-id="r"]')!.textContent).toContain('Version 3');
    expect(el().querySelector('[data-card-id="f"]')!.textContent).toContain('Free play');
  });

  it('shares a finished card from My cards, and unshares it', () => {
    http.expectOne(url).flush([doneCard({ id: 'd1' })]);
    fixture.detectChanges();
    const tile = () => el().querySelector('[data-card-id="d1"]') as HTMLElement;
    const shareButton = () => tile().querySelector('button.share') as HTMLButtonElement;
    expect(shareButton().textContent).toContain('Share');

    shareButton().click();
    const req = http.expectOne(`${environment.apiUrl}/api/v1/cards/d1/share`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ shared: true });
    req.flush(doneCard({ id: 'd1', shared: true }));
    fixture.detectChanges();
    expect(tile().textContent).toContain('Shared');
    expect(shareButton().textContent).toContain('Unshare');

    shareButton().click();
    const undo = http.expectOne(`${environment.apiUrl}/api/v1/cards/d1/share`);
    expect(undo.request.body).toEqual({ shared: false });
    undo.flush(doneCard({ id: 'd1', shared: false }));
  });

  it('the Community tab lists shared cards with their maker', () => {
    http.expectOne(url).flush([]);
    fixture.detectChanges();
    (el().querySelector('button.tab-community') as HTMLElement).click();
    fixture.detectChanges();
    http.expectOne(`${environment.apiUrl}/api/v1/cards/shared`).flush([
      { ...doneCard({ id: 'x1', shared: true }), username: 'Beth' }
    ]);
    fixture.detectChanges();

    const tile = el().querySelector('[data-card-id="x1"]') as HTMLElement;
    expect(tile.textContent).toContain('by Beth');
    expect(tile.querySelector('button.share')).toBeNull();
    expect(tile.querySelector('button.pool-submit')).toBeNull();
    expect(tile.querySelector('.download')).not.toBeNull();
    expect(el().querySelector('button.tab-community')!.getAttribute('aria-selected')).toBe('true');

    // switching back doesn't refetch My cards; switching again doesn't refetch Community
    (el().querySelector('button.tab-mine') as HTMLElement).click();
    (el().querySelector('button.tab-community') as HTMLElement).click();
    http.expectNone(`${environment.apiUrl}/api/v1/cards/shared`);
  });

  it('done My cards tiles offer Submit to pool; pending tiles do not', () => {
    http.expectOne(url).flush([doneCard({ id: 'd1' }), cardView({ id: 'p1', status: 'queued' })]);
    fixture.detectChanges();
    const submit = el().querySelector('[data-card-id="d1"] button.pool-submit') as HTMLButtonElement;
    expect(submit.textContent).toContain('Submit to pool (1/3)');
    expect(el().querySelector('[data-card-id="p1"] button.pool-submit')).toBeNull();

    submit.click();
    const req = http.expectOne(`${environment.apiUrl}/api/v1/pools/entries`);
    expect(req.request.body).toEqual({ cardId: 'd1' });
    req.flush(poolView({
      myEntryCount: 2,
      entries: [poolEntry({ id: 'e9', cardId: 'd1', mine: true })]
    }));
    fixture.detectChanges();
    expect(el().querySelector('[data-card-id="d1"]')!.textContent).toContain('In the pool ✓');
  });

  it('shows an empty state when there are no cards', () => {
    http.expectOne(url).flush([]);
    fixture.detectChanges();
    expect(el().textContent).toContain('No cards yet');
  });
});

describe('GalleryPageComponent polling', () => {
  let fixture: ComponentFixture<GalleryPageComponent>;
  let http: HttpTestingController;
  let visibility: ReturnType<typeof fakeVisibility>;
  const url = `${environment.apiUrl}/api/v1/me/cards`;

  beforeEach(() => {
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src', 'download']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    visibility = fakeVisibility();
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule, RouterTestingModule],
      declarations: [GalleryPageComponent, CardSlotComponent, MediaPipe],
      providers: [
        { provide: MediaService, useValue: media },
        { provide: PageVisibilityService, useValue: visibility }
      ]
    });
    http = TestBed.inject(HttpTestingController);
  });

  function start(): void {
    fixture = TestBed.createComponent(GalleryPageComponent);
    fixture.detectChanges();
    http.expectOne(url).flush([cardView({ id: 'p1', status: 'queued' })]);
  }

  it('drops a poll with no answer after 15s and keeps polling', fakeAsync(() => {
    expect(GALLERY_REQUEST_TIMEOUT_MS).toBe(15000);
    start();
    tick(GALLERY_POLL_MS);
    const hung = http.expectOne(url);
    tick(GALLERY_REQUEST_TIMEOUT_MS);
    expect(hung.cancelled).toBeTrue();
    expect(fixture.componentInstance.error).toBeNull();

    tick(GALLERY_POLL_MS);   // the next poll after the timeout
    http.expectOne(url).flush([doneCard({ id: 'p1' })]);
    expect(fixture.componentInstance.hasPending()).toBeFalse();
    fixture.destroy();
  }));

  it('pauses while the page is hidden and polls at once when it is shown', fakeAsync(() => {
    start();
    visibility.visibleSubject.next(false);
    tick(60000);
    http.expectNone(url);

    visibility.visibleSubject.next(true);
    http.expectOne(url).flush([doneCard({ id: 'p1' })]);
    expect(fixture.componentInstance.hasPending()).toBeFalse();
    fixture.destroy();
  }));
});
