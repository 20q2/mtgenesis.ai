import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { of } from 'rxjs';
import { environment } from '../../../environments/environment';
import { CardSlotComponent } from '../../components/card-slot/card-slot.component';
import { MediaPipe } from '../../pipes/media.pipe';
import { MediaService } from '../../services/media.service';
import { cardView, doneCard } from '../../testing/fixtures';
import { GalleryPageComponent } from './gallery-page.component';

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
      declarations: [GalleryPageComponent, CardSlotComponent, MediaPipe],
      providers: [{ provide: MediaService, useValue: media }]
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(GalleryPageComponent);
    fixture.detectChanges();
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

  it('shows an empty state when there are no cards', () => {
    http.expectOne(url).flush([]);
    fixture.detectChanges();
    expect(el().textContent).toContain('No cards yet');
  });
});
