import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap } from '@angular/router';
import { RouterTestingModule } from '@angular/router/testing';
import { BehaviorSubject, of } from 'rxjs';
import { CardSlotComponent } from '../../components/card-slot/card-slot.component';
import { SetRowComponent } from '../../components/set-row/set-row.component';
import { WinnersBannerComponent } from '../../components/winners-banner/winners-banner.component';
import { CmcGroupsPipe } from '../../pipes/cmc-groups.pipe';
import { MediaPipe } from '../../pipes/media.pipe';
import { EventService } from '../../services/event.service';
import { MediaService } from '../../services/media.service';
import { eventView, setCard, setView } from '../../testing/fixtures';
import { EventHistoryPageComponent } from './event-history-page.component';

describe('EventHistoryPageComponent', () => {
  let fixture: ComponentFixture<EventHistoryPageComponent>;
  let events: jasmine.SpyObj<EventService>;
  let params: BehaviorSubject<any>;

  function setup(id: string | null) {
    params = new BehaviorSubject(convertToParamMap(id ? { id } : {}));
    events = jasmine.createSpyObj<EventService>('EventService', ['get', 'list']);
    events.list.and.returnValue(of([
      { id: 'e-2', name: 'AI Night #2', status: 'open', createdAt: '2026-09-28T19:00:00+00:00', closedAt: null },
      { id: 'e-1', name: 'AI Night #1', status: 'closed', createdAt: '2026-09-21T19:00:00+00:00', closedAt: '2026-09-21T23:00:00+00:00' }
    ]));
    events.get.and.returnValue(of(eventView({
      id: 'e-1', name: 'AI Night #1', status: 'closed',
      sets: [setView({ commanderName: 'Grimbold', cmc: 5, cards: [setCard({ id: 'g1', slot: 1, leader: true, votes: 4 }), setCard({ id: 'g2', slot: 2 }), setCard({ id: 'g3', slot: 3 })] }),
             setView({ id: 's-2', commanderName: 'Ashling', cmc: 3 })]
    })));
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      declarations: [EventHistoryPageComponent, SetRowComponent, WinnersBannerComponent, CardSlotComponent, MediaPipe,
                     CmcGroupsPipe],
      providers: [
        { provide: EventService, useValue: events },
        { provide: MediaService, useValue: media },
        { provide: ActivatedRoute, useValue: { paramMap: params } }
      ]
    });
    fixture = TestBed.createComponent(EventHistoryPageComponent);
    fixture.detectChanges();
  }

  const el = () => fixture.nativeElement as HTMLElement;

  it('/events lists every event with a link to its results', () => {
    setup(null);
    expect(events.list).toHaveBeenCalled();
    const links = Array.from(el().querySelectorAll('.event-list a')).map(a => a.getAttribute('href'));
    expect(links).toEqual(['/events/e-2', '/events/e-1']);
  });

  it('/events/:id shows the read-only results: winners and rows without vote buttons', () => {
    setup('e-1');
    expect(events.get).toHaveBeenCalledWith('e-1');
    expect(el().querySelector('.winners-banner')!.textContent).toContain('Grimbold');
    expect(el().querySelectorAll('app-set-row').length).toBe(2);
    expect(el().querySelector('button.vote-button')).toBeNull();
    expect(el().querySelector('[data-card-id="g1"]')!.classList).toContain('leader');
    expect(el().querySelector('[data-card-id="g1"]')!.textContent).toContain('Winner');
  });

  it('/events/:id groups the commanders by CMC', () => {
    setup('e-1');
    const headings = Array.from(el().querySelectorAll('.set-rows .cmc-heading')).map(h => h.textContent!.trim());
    expect(headings).toEqual(['3 CMC', '5 CMC']);
  });
});
