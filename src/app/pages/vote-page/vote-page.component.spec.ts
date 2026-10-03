import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { RouterTestingModule } from '@angular/router/testing';
import { Subject, of, throwError } from 'rxjs';
import { EventView, SetView } from '../../models/api.model';
import { CardSlotComponent } from '../../components/card-slot/card-slot.component';
import { SetRowComponent } from '../../components/set-row/set-row.component';
import { WinnersBannerComponent } from '../../components/winners-banner/winners-banner.component';
import { CmcGroupsPipe } from '../../pipes/cmc-groups.pipe';
import { MediaPipe } from '../../pipes/media.pipe';
import { EventService } from '../../services/event.service';
import { MediaService } from '../../services/media.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { fakeVisibility } from '../../testing/fake-visibility';
import { eventView, setCard, setView } from '../../testing/fixtures';
import { VOTE_POLL_MS, VotePageComponent } from './vote-page.component';

describe('VotePageComponent', () => {
  let fixture: ComponentFixture<VotePageComponent>;
  let events: jasmine.SpyObj<EventService>;
  let visibility: ReturnType<typeof fakeVisibility>;

  function setA(overrides: Partial<SetView> = {}): SetView {
    return setView({
      id: 'sA', username: 'Alice', commanderName: 'Zur\'ka, Élan of Ash',
      cards: [
        setCard({ id: 'a1', slot: 1, setId: 'sA', votes: 3, leader: true }),
        setCard({ id: 'a2', slot: 2, setId: 'sA', votes: 1 }),
        setCard({ id: 'a3', slot: 3, setId: 'sA', votes: 0 })
      ],
      ...overrides
    });
  }

  function setB(overrides: Partial<SetView> = {}): SetView {
    return setView({
      id: 'sB', username: 'Bob', commanderName: 'Grimbold the Unbowed',
      cards: [
        setCard({ id: 'b1', slot: 1, setId: 'sB', votes: 2, tied: true }),
        setCard({ id: 'b2', slot: 2, setId: 'sB', votes: 0 }),
        setCard({ id: 'b3', slot: 3, setId: 'sB', votes: 2, tied: true })
      ],
      ...overrides
    });
  }

  function setup(current: EventView | null) {
    events = jasmine.createSpyObj<EventService>('EventService', ['current', 'vote', 'get']);
    events.current.and.returnValue(of(current));
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    visibility = fakeVisibility();
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      declarations: [VotePageComponent, SetRowComponent, WinnersBannerComponent, CardSlotComponent, MediaPipe,
                     CmcGroupsPipe],
      providers: [
        { provide: EventService, useValue: events },
        { provide: MediaService, useValue: media },
        { provide: PageVisibilityService, useValue: visibility }
      ]
    });
    fixture = TestBed.createComponent(VotePageComponent);
    fixture.detectChanges();
  }

  afterEach(() => fixture.destroy());

  const el = () => fixture.nativeElement as HTMLElement;
  const voteButtons = () => Array.from(el().querySelectorAll('button.vote-button')) as HTMLButtonElement[];
  const cardEl = (id: string) => el().querySelector(`[data-card-id="${id}"]`) as HTMLElement;

  it('renders a row per set with the commander name and "by <username>"', () => {
    setup(eventView({ sets: [setA(), setB()] }));
    const rows = el().querySelectorAll('app-set-row');
    expect(rows.length).toBe(2);
    expect(rows[0].textContent).toContain('Zur\'ka, Élan of Ash');
    expect(rows[0].textContent).toContain('by Alice');
    expect(voteButtons().length).toBe(6);
  });

  it('groups commanders under 3, 4 and 5 CMC headings, then Earlier sets', () => {
    const legacy = setView({ id: 'sL', username: 'Cy', commanderName: 'Old Set', cmc: null });
    setup(eventView({ sets: [setA({ cmc: 5 }), setB({ cmc: 3 }), legacy] }));
    const groups = Array.from(el().querySelectorAll('.cmc-group')) as HTMLElement[];
    expect(groups.map(g => g.querySelector('.cmc-heading')!.textContent!.trim()))
      .toEqual(['3 CMC', '5 CMC', 'Earlier sets']);
    expect(groups[0].textContent).toContain('Grimbold the Unbowed');
    expect(groups[1].textContent).toContain("Zur'ka, Élan of Ash");
    expect(groups[2].textContent).toContain('Old Set');
  });

  it("explains the vote rules, including the owner's double vote", () => {
    setup(eventView({ sets: [setA()] }));
    expect(el().querySelector('.vote-header')!.textContent).toContain(
      'Vote for one version of each commander. A vote on your own commander counts as two — owners, vote first.');
  });

  it("marks the owner's pick as owner ×2", () => {
    const owned = setA();
    owned.cards[1] = { ...owned.cards[1], ownerVote: true, votes: 2 };
    setup(eventView({ sets: [owned] }));
    expect(cardEl('a2').textContent).toContain('owner ×2');
    expect(cardEl('a1').textContent).not.toContain('owner ×2');
  });

  it('groups the winners by CMC once the event is closed', () => {
    setup(eventView({ status: 'closed', closedAt: '2026-09-28T23:00:00+00:00',
                      sets: [setA({ cmc: 4 }), setB({ cmc: 3 })] }));
    const titles = Array.from(el().querySelectorAll('.winners-banner .cmc-heading'))
      .map(h => h.textContent!.trim());
    expect(titles).toEqual(['3 CMC', '4 CMC']);
  });

  it('a vote click calls EventService.vote and then refreshes', () => {
    setup(eventView({ sets: [setA()] }));
    const updated = setA({ myVoteCardId: 'a2' });
    events.vote.and.returnValue(of(updated));
    const before = events.current.calls.count();

    voteButtons()[1].click();

    expect(events.vote).toHaveBeenCalledOnceWith('sA', 'a2');
    expect(events.current.calls.count()).toBe(before + 1);
  });

  it('outlines my pick', () => {
    setup(eventView({ sets: [setA({ myVoteCardId: 'a2' })] }));
    expect(cardEl('a2').classList).toContain('mine');
    expect(cardEl('a1').classList).not.toContain('mine');
  });

  it('gives the leader .leader and a Leading crown; tied cards get .tied and no .leader', () => {
    setup(eventView({ sets: [setA(), setB()] }));
    expect(cardEl('a1').classList).toContain('leader');
    expect(cardEl('a1').textContent).toContain('Leading');
    expect(cardEl('a2').classList).not.toContain('leader');

    for (const id of ['b1', 'b3']) {
      expect(cardEl(id).classList).toContain('tied');
      expect(cardEl(id).classList).not.toContain('leader');
      expect(cardEl(id).textContent).toContain('Tied');
    }
    expect(cardEl('b2').classList).not.toContain('tied');
  });

  it('shows the live vote counts', () => {
    setup(eventView({ sets: [setA()] }));
    expect(cardEl('a1').textContent).toContain('3 votes');
    expect(cardEl('a2').textContent).toContain('1 vote');
  });

  it('a closed event disables all vote buttons and renders the winner names', () => {
    setup(eventView({ status: 'closed', closedAt: '2026-09-28T23:00:00+00:00', sets: [setA(), setB()] }));
    expect(voteButtons().length).toBe(6);
    expect(voteButtons().every(b => b.disabled)).toBeTrue();

    const banner = el().querySelector('.winners-banner') as HTMLElement;
    expect(banner).not.toBeNull();
    expect(banner.textContent).toContain('Zur\'ka, Élan of Ash');
    expect(banner.textContent).toContain('Version 1');
    expect(banner.textContent).toContain('Grimbold the Unbowed');
    expect(banner.textContent).toContain('Tied');
  });

  it('does not vote on a closed event', () => {
    setup(eventView({ status: 'closed', sets: [setA()] }));
    fixture.componentInstance.vote(setA(), setA().cards[1]);
    expect(events.vote).not.toHaveBeenCalled();
  });

  it('a 409 shows "Voting is closed" and refreshes immediately', () => {
    setup(eventView({ sets: [setA()] }));
    events.vote.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 409, error: { error: 'Voting is closed' }
    })));
    const before = events.current.calls.count();

    voteButtons()[0].click();
    fixture.detectChanges();

    expect(el().textContent).toContain('Voting is closed');
    expect(events.current.calls.count()).toBe(before + 1);
  });

  it('a 409 without a body still says "Voting is closed"', () => {
    setup(eventView({ sets: [setA()] }));
    events.vote.and.returnValue(throwError(() => new HttpErrorResponse({ status: 409 })));
    voteButtons()[0].click();
    fixture.detectChanges();
    expect(el().textContent).toContain('Voting is closed');
  });

  it('ignores a second click while the first vote is in flight', () => {
    setup(eventView({ sets: [setA()] }));
    events.vote.and.returnValue(new Subject<SetView>());
    voteButtons()[1].click();
    fixture.detectChanges();
    expect(voteButtons().every(b => b.disabled)).toBeTrue();
    voteButtons()[2].click();
    fixture.componentInstance.vote(setA(), setA().cards[2]);
    expect(events.vote).toHaveBeenCalledTimes(1);
  });

  it('with no event says "No event open" and links to past events', () => {
    setup(null);
    expect(el().textContent).toContain('No event open');
    const link = el().querySelector('a[href="/events"]');
    expect(link).not.toBeNull();
  });

  it('polls every 10s and shows the winners when the event it was showing gets closed', fakeAsync(() => {
    expect(VOTE_POLL_MS).toBe(10000);
    setup(eventView({ sets: [setA()] }));
    const closed = eventView({ status: 'closed', sets: [setA()] });
    events.current.and.returnValue(of(null));
    events.get.and.returnValue(of(closed));

    tick(9999);
    expect(events.get).not.toHaveBeenCalled();
    tick(1);
    fixture.detectChanges();

    expect(events.get).toHaveBeenCalledWith('e-1');
    expect(el().querySelector('.winners-banner')).not.toBeNull();
    expect(voteButtons().every(b => b.disabled)).toBeTrue();
    fixture.destroy();
  }));

  it('stops polling while the page is hidden and refreshes at once when it is shown', fakeAsync(() => {
    setup(eventView({ sets: [setA()] }));
    expect(events.current).toHaveBeenCalledTimes(1);

    visibility.visibleSubject.next(false);
    tick(60000);
    expect(events.current).toHaveBeenCalledTimes(1);

    events.current.and.returnValue(of(eventView({ sets: [setA(), setB()] })));
    visibility.visibleSubject.next(true);
    expect(events.current).toHaveBeenCalledTimes(2);
    fixture.detectChanges();
    expect(el().querySelectorAll('app-set-row').length).toBe(2);

    tick(10000);
    expect(events.current).toHaveBeenCalledTimes(3);
    fixture.destroy();
  }));
});
