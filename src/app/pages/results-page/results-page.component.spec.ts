import { ComponentFixture, TestBed } from '@angular/core/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { of, throwError } from 'rxjs';
import { AvatarComponent } from '../../components/avatar/avatar.component';
import { CardSlotComponent } from '../../components/card-slot/card-slot.component';
import { MediaPipe } from '../../pipes/media.pipe';
import { EventService } from '../../services/event.service';
import { MediaService } from '../../services/media.service';
import { PoolService } from '../../services/pool.service';
import { eventView, poolEntry, poolView, setCard, setView } from '../../testing/fixtures';
import { ResultsPageComponent } from './results-page.component';

describe('ResultsPageComponent', () => {
  let fixture: ComponentFixture<ResultsPageComponent>;
  let events: jasmine.SpyObj<EventService>;
  let pools: jasmine.SpyObj<PoolService>;

  function setup(opts: { poolFails?: boolean; noneClosed?: boolean } = {}) {
    events = jasmine.createSpyObj<EventService>('EventService', ['list', 'get']);
    events.list.and.returnValue(of(opts.noneClosed ? [] : [
      { id: 'e-2', name: 'Next night', status: 'open', createdAt: '2026-10-18T00:00:00Z', closedAt: null },
      { id: 'e-1', name: 'AI Night Oct 2026', status: 'closed', createdAt: '2026-10-14T00:00:00Z', closedAt: '2026-10-17T21:04:00Z' }
    ]));
    events.get.and.returnValue(of(eventView({
      id: 'e-1', name: 'AI Night Oct 2026', status: 'closed', closedAt: '2026-10-17T21:04:00Z',
      sets: [
        setView({ id: 's-b5', userId: 'u-b', username: 'Bo', commanderName: 'Big Bo', cmc: 5,
                  cards: [setCard({ id: 'b5-1', slot: 1, leader: true, votes: 7 }), setCard({ id: 'b5-2', slot: 2, votes: 1 })] }),
        setView({ id: 's-a3', userId: 'u-a', username: 'amy', commanderName: 'Little Amy', cmc: 3,
                  cards: [setCard({ id: 'a3-1', slot: 1, tied: true, votes: 3 }), setCard({ id: 'a3-2', slot: 2, tied: true, votes: 3 }),
                          setCard({ id: 'a3-3', slot: 3 })] }),
        setView({ id: 's-b3', userId: 'u-b', username: 'Bo', commanderName: 'Small Bo', cmc: 3,
                  cards: [setCard({ id: 'b3-2', slot: 2, leader: true, votes: 5 })] })
      ]
    })));
    pools = jasmine.createSpyObj<PoolService>('PoolService', ['list', 'get']);
    pools.list.and.returnValue(opts.noneClosed ? of([]) : of([
      { id: 'p-1', name: 'AI Night Oct 2026', status: 'closed', maxEntriesPerUser: 4,
        createdAt: '2026-10-14T00:00:00Z', closedAt: '2026-10-17T21:04:00Z' }
    ]));
    pools.get.and.returnValue(opts.poolFails ? throwError(() => new Error('down')) : of(poolView({
      id: 'p-1', status: 'closed', entries: [
        poolEntry({ id: 'pe-1', username: 'Bo', rank: 1, points: 12, in: true }),
        poolEntry({ id: 'pe-2', username: 'amy', rank: 2, points: 1, in: true, tiedAtCutoff: true }),
        poolEntry({ id: 'pe-3', username: 'amy', rank: 3, points: 0 })
      ]
    })));
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      declarations: [ResultsPageComponent, CardSlotComponent, AvatarComponent, MediaPipe],
      providers: [
        { provide: EventService, useValue: events },
        { provide: PoolService, useValue: pools },
        { provide: MediaService, useValue: media }
      ]
    });
    fixture = TestBed.createComponent(ResultsPageComponent);
    fixture.detectChanges();
  }

  const el = () => fixture.nativeElement as HTMLElement;
  const ids = (selector: string, attr: string) =>
    Array.from(el().querySelectorAll(selector)).map(n => n.getAttribute(attr));

  it('loads the newest closed event and pool', () => {
    setup();
    expect(events.get).toHaveBeenCalledWith('e-1');
    expect(pools.get).toHaveBeenCalledWith('p-1');
    expect(el().querySelector('.results-header')!.textContent).toContain('AI Night Oct 2026');
  });

  it('shows only the cards that made the pool, with their makers and points', () => {
    setup();
    expect(ids('.pool-group .result-card', 'data-entry-id')).toEqual(['pe-1', 'pe-2']);
    const text = el().querySelector('.pool-group')!.textContent!;
    expect(text).toContain('#1 · 12 points');
    expect(text).toContain('tied at the line');
  });

  it("shows each player's legal commanders by CMC, with tied versions for the host", () => {
    setup();
    const groups = Array.from(el().querySelectorAll('.player-group'));
    expect(groups.map(g => g.querySelector('.group-head .section-title')!.textContent!.trim())).toEqual(['amy', 'Bo']);
    expect(ids('[data-sheet="player:u-a"] .result-card', 'data-card-id')).toEqual(['a3-1', 'a3-2']);
    expect(el().querySelector('[data-sheet="player:u-a"]')!.textContent).toContain('host picks');
    expect(ids('[data-sheet="player:u-b"] .result-card', 'data-card-id')).toEqual(['b3-2', 'b5-1']);
    expect(el().querySelector('[data-sheet="player:u-b"]')!.textContent).toContain('5 CMC · Version 1 · 7 votes');
  });

  it('every group has a PNG export', () => {
    setup();
    expect(el().querySelectorAll('.group .export-png').length).toBe(3);
  });

  it('still shows the commanders when the pool fails to load', () => {
    setup({ poolFails: true });
    expect(el().querySelector('.pool-group')).toBeNull();
    expect(el().querySelectorAll('.player-group').length).toBe(2);
  });

  it('says there are no results before anything closes', () => {
    setup({ noneClosed: true });
    expect(el().querySelector('.empty-state')!.textContent).toContain('No results yet');
  });

  it('PNG draws the sheet and saves it under the group name', async () => {
    setup();
    const clicked: string[] = [];
    spyOn(HTMLAnchorElement.prototype, 'click').and.callFake(function (this: HTMLAnchorElement) {
      clicked.push(this.download);
    });
    (el().querySelector('[data-sheet="player:u-b"] .export-png') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(el().querySelector('[data-sheet="player:u-b"] .drawing')).not.toBeNull();
    await fixture.whenStable();
    await new Promise(r => setTimeout(r, 0));
    await waitFor(() => clicked.length > 0);
    expect(clicked).toEqual(['ai_night_oct_2026_bo_commanders.png']);
  });

  async function waitFor(done: () => boolean, ms = 5000): Promise<void> {
    const start = Date.now();
    while (!done() && Date.now() - start < ms) {
      await new Promise(r => setTimeout(r, 20));
    }
  }
});
