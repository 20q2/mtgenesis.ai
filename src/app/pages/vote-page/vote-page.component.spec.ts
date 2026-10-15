import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { AvatarComponent } from '../../components/avatar/avatar.component';
import { HttpErrorResponse } from '@angular/common/http';
import { RouterTestingModule } from '@angular/router/testing';
import { Subject, of, throwError } from 'rxjs';
import { EventView, SetView } from '../../models/api.model';
import { BallotPlayerComponent } from '../../components/ballot-player/ballot-player.component';
import { PlayerSheetComponent } from '../../components/player-sheet/player-sheet.component';
import { WinnersBannerComponent } from '../../components/winners-banner/winners-banner.component';
import { TiltDirective } from '../../directives/tilt.directive';
import { CmcGroupsPipe } from '../../pipes/cmc-groups.pipe';
import { MediaPipe } from '../../pipes/media.pipe';
import { EventService } from '../../services/event.service';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { UserService } from '../../services/user.service';
import { NightStepsComponent } from '../../components/night-steps/night-steps.component';
import { fakeVisibility } from '../../testing/fake-visibility';
import { eventView, setCard, setView } from '../../testing/fixtures';
import { VOTE_POLL_MS, VotePageComponent } from './vote-page.component';

describe('VotePageComponent', () => {
  let fixture: ComponentFixture<VotePageComponent>;
  let events: jasmine.SpyObj<EventService>;
  let generation: jasmine.SpyObj<GenerationService>;
  let visibility: ReturnType<typeof fakeVisibility>;

  function setA(overrides: Partial<SetView> = {}): SetView {
    return setView({
      id: 'sA', userId: 'u-a', username: 'Alice', commanderName: 'Zur\'ka, Élan of Ash', cmc: 3,
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
      id: 'sB', userId: 'u-b', username: 'Bob', commanderName: 'Grimbold the Unbowed', cmc: 4,
      cards: [
        setCard({ id: 'b1', slot: 1, setId: 'sB', votes: 2, tied: true }),
        setCard({ id: 'b2', slot: 2, setId: 'sB', votes: 0 }),
        setCard({ id: 'b3', slot: 3, setId: 'sB', votes: 2, tied: true })
      ],
      ...overrides
    });
  }

  function setup(current: EventView | null, me = 'u-me', latest: EventView | null = null) {
    events = jasmine.createSpyObj<EventService>('EventService', ['current', 'vote', 'get', 'list', 'requestReroll']);
    generation = jasmine.createSpyObj<GenerationService>('GenerationService', ['reroll']);
    events.list.and.returnValue(of([]));
    events.current.and.returnValue(of(current));
    if (latest) {
      events.list.and.returnValue(of([{ id: latest.id, name: latest.name, status: latest.status,
                                        createdAt: latest.createdAt, closedAt: latest.closedAt }]));
      events.get.and.returnValue(of(latest));
    }
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    visibility = fakeVisibility();
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      declarations: [VotePageComponent, BallotPlayerComponent, PlayerSheetComponent, AvatarComponent, WinnersBannerComponent,
                     MediaPipe, CmcGroupsPipe, NightStepsComponent, TiltDirective],
      providers: [
        { provide: EventService, useValue: events },
        { provide: GenerationService, useValue: generation },
        { provide: MediaService, useValue: media },
        { provide: PageVisibilityService, useValue: visibility },
        { provide: UserService, useValue: { currentUser: () => ({ id: me, username: 'Me' }) } }
      ]
    });
    fixture = TestBed.createComponent(VotePageComponent);
    fixture.detectChanges();
  }

  afterEach(() => fixture.destroy());

  const el = () => fixture.nativeElement as HTMLElement;
  const tiles = () => Array.from(el().querySelectorAll('app-ballot-player')) as HTMLElement[];
  const tile = (userId: string) => el().querySelector(`.ballot-player[data-user-id="${userId}"]`) as HTMLElement;
  const slot = (setId: string) => el().querySelector(`.ballot-slot[data-set-id="${setId}"]`) as HTMLElement;
  const sheet = () => el().querySelector('app-player-sheet dialog') as HTMLDialogElement | null;
  const voteButtons = () => Array.from(el().querySelectorAll('button.vote-button')) as HTMLButtonElement[];
  const cardEl = (id: string) => el().querySelector(`.version[data-card-id="${id}"]`) as HTMLElement;
  const text = (node: Element | null) => (node?.textContent ?? '').replace(/\s+/g, ' ');

  function open(userId: string): void {
    tile(userId).click();
    fixture.detectChanges();
  }

  it('shows one tile per player, with their commanders and their name', () => {
    setup(eventView({ sets: [setA(), setB()] }));
    expect(tiles().length).toBe(2);
    expect(text(tile('u-a'))).toContain('Alice');
    expect(text(tile('u-a'))).toContain('Zur\'ka, Élan of Ash');
    expect(text(tile('u-b'))).toContain('Grimbold the Unbowed');
    expect(voteButtons().length).toBe(0);
  });

  it("keeps a slot for each of a player's 3, 4 and 5 CMC commanders, marking any not locked in", () => {
    setup(eventView({ sets: [setA({ cmc: 5 }), setA({ id: 'sA3', cmc: 3, commanderName: 'Small One' })] }));
    const slots = Array.from(tile('u-a').querySelectorAll('.ballot-slot')) as HTMLElement[];
    expect(slots.map(s => text(s.querySelector('.slot-name')).trim()))
      .toEqual(['Small One', 'Not locked in', "Zur'ka, Élan of Ash"]);
    expect(slots[0].querySelector('.fan-pip i.ms-3')).not.toBeNull();
    expect(slots[1].classList).toContain('gap');
  });

  it('marks each commander voted (with my version) or still to vote on, and counts per player', () => {
    setup(eventView({ sets: [setA({ myVoteCardId: 'a2' }), setA({ id: 'sA5', cmc: 5 }), setB()] }));
    expect(slot('sA').classList).toContain('state-voted');
    expect(text(slot('sA'))).toContain('Version 2');
    expect(slot('sA').getAttribute('aria-label')).toContain('you voted for Version 2');
    expect(slot('sA').querySelector('.fan-card.front')).not.toBeNull();
    expect(slot('sA5').classList).toContain('state-todo');
    expect(slot('sA5').getAttribute('aria-label')).toContain('not voted yet');
    expect(text(tile('u-a').querySelector('.player-progress'))).toContain('1/2');
  });

  it('explains how to vote: one per commander, in good faith, owner x2, owners first', () => {
    setup(eventView({ sets: [setA()] }));
    const rules = text(el().querySelector('.how-to-vote'));
    expect(rules).toContain('Pick one favorite version of each commander');
    expect(rules).toContain('good faith');
    expect(rules).toContain('balanced');
    expect(rules).toContain('counts as two');
    expect(rules).toContain('vote on your own first');
    expect(el().querySelector('app-night-steps')).not.toBeNull();
  });

  it('puts my own commanders first and nudges me to vote on them', () => {
    setup(eventView({ sets: [setB(), setA({ cmc: 4, userId: 'u-me', username: 'Me' })] }));
    expect(tiles()[0].textContent).toContain("Zur'ka");
    expect(tiles()[0].textContent).toContain('your vote counts double');
    expect(tiles()[1].textContent).toContain('Grimbold');
    expect(text(el().querySelector('.mine-nudge'))).toContain('Vote on yours first');
    expect(text(el().querySelector('.mine-nudge'))).toContain('counts double');
    expect(text(el().querySelector('.next-to-vote'))).toContain('Vote on yours');
  });

  it("counts the commanders I've voted on", () => {
    setup(eventView({ sets: [setA({ myVoteCardId: 'a1' }), setB()] }));
    expect(text(el().querySelector('.vote-progress'))).toContain("You've voted on 1 of 2 commanders");
    expect(el().querySelector('[role="progressbar"]')!.getAttribute('aria-valuenow')).toBe('1');
    expect(text(el().querySelector('.next-to-vote'))).toContain('Vote on Bob’s');
  });

  it('says so once every commander has my vote', () => {
    setup(eventView({ sets: [setA({ myVoteCardId: 'a1' })] }));
    expect(el().querySelector('.next-to-vote')).toBeNull();
    expect(text(el().querySelector('.all-voted'))).toContain('All voted');
  });

  it('a tile opens that player in a sheet with every version of each of their commanders', () => {
    setup(eventView({ sets: [setA(), setA({ id: 'sA5', cmc: 5, commanderName: 'Big One' }), setB()] }));
    expect(sheet()).toBeNull();
    open('u-a');
    expect(sheet()!.open).toBeTrue();
    expect(text(el().querySelector('.sheet-title'))).toContain('Alice');
    expect(el().querySelectorAll('.commander-row').length).toBe(2);
    expect(voteButtons().length).toBe(6);
    expect(text(el().querySelector('.fair-play'))).toContain('good faith');
  });

  it('Vote on yours opens my commanders', () => {
    setup(eventView({ sets: [setB(), setA({ userId: 'u-me', username: 'Me' })] }));
    (el().querySelector('.next-to-vote') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(text(el().querySelector('.sheet-title'))).toContain('Your commanders');
    expect(fixture.componentInstance.sheetFocusSetId).toBe('sA');
  });

  it('a commander in a tile opens the sheet at that commander', () => {
    setup(eventView({ sets: [setA(), setA({ id: 'sA5', cmc: 5 })] }));
    slot('sA5').click();
    fixture.detectChanges();
    expect(fixture.componentInstance.sheetUserId).toBe('u-a');
    expect(fixture.componentInstance.sheetFocusSetId).toBe('sA5');
  });

  it('the sheet moves on to the next player with commanders left, and steps between players', () => {
    const carol = setView({ id: 'sC', userId: 'u-c', username: 'Carol', myVoteCardId: 'sC-c1' });
    setup(eventView({ sets: [setA(), setB(), carol] }));
    open('u-a');
    expect(text(el().querySelector('.next-player'))).toContain('Next: Bob');
    (el().querySelector('.next-player') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(text(el().querySelector('.sheet-title'))).toContain('Bob');
    expect(text(el().querySelector('.player-count'))).toContain('2 / 3');

    (el().querySelector('[aria-label="Next player"]') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(text(el().querySelector('.sheet-title'))).toContain('Carol');

    sheet()!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    fixture.detectChanges();
    expect(text(el().querySelector('.sheet-title'))).toContain('Alice');
  });

  it('Esc, the close button and a click outside the sheet all close it', () => {
    setup(eventView({ sets: [setA()] }));
    open('u-a');
    sheet()!.dispatchEvent(new Event('cancel', { cancelable: true }));
    fixture.detectChanges();
    expect(sheet()).toBeNull();

    open('u-a');
    (el().querySelector('.close-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(sheet()).toBeNull();

    open('u-a');
    sheet()!.click();
    fixture.detectChanges();
    expect(sheet()).toBeNull();
  });

  it('after the vote, lists my legal commanders', () => {
    setup(eventView({ status: 'closed', closedAt: '2026-09-28T23:00:00+00:00',
                      sets: [setA({ userId: 'u-me', cmc: 3 }), setB({ cmc: 4 })] }));
    const legal = el().querySelector('.my-legal') as HTMLElement;
    expect(legal.textContent).toContain('Your legal commanders');
    expect(legal.textContent).toContain("Zur'ka");
    expect(legal.textContent).toContain('Version 1');
    expect(legal.textContent).not.toContain('Grimbold');
  });

  it("marks the owner's pick as owner ×2", () => {
    const owned = setA();
    owned.cards[1] = { ...owned.cards[1], ownerVote: true, votes: 2 };
    setup(eventView({ sets: [owned] }));
    open('u-a');
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

  it('once closed, each tile shows the winning version (or the tie)', () => {
    setup(eventView({ status: 'closed', closedAt: '2026-09-28T23:00:00+00:00', sets: [setA(), setB()] }));
    expect(slot('sA').classList).toContain('state-winner');
    expect(text(slot('sA'))).toContain('Version 1');
    expect(slot('sA').querySelector('.fan-card.front')!.getAttribute('src')).toContain('a1');
    expect(slot('sB').classList).toContain('state-tie');
    expect(el().querySelector('.player-progress')).toBeNull();
  });

  it('a vote in the sheet calls EventService.vote and then refreshes', () => {
    setup(eventView({ sets: [setA()] }));
    open('u-a');
    const updated = setA({ myVoteCardId: 'a2' });
    events.vote.and.returnValue(of(updated));
    const before = events.current.calls.count();

    voteButtons()[1].click();
    fixture.detectChanges();

    expect(events.vote).toHaveBeenCalledOnceWith('sA', 'a2');
    expect(events.current.calls.count()).toBe(before + 1);
  });

  it('outlines my pick in the sheet and marks it on the tile', () => {
    setup(eventView({ sets: [setA({ myVoteCardId: 'a2' })] }));
    open('u-a');
    expect(cardEl('a2').classList).toContain('mine');
    expect(cardEl('a1').classList).not.toContain('mine');
    expect(cardEl('a2').querySelector('.vote-button')!.getAttribute('aria-pressed')).toBe('true');
    expect(text(el().querySelector('.row-state'))).toContain('You picked Version 2');
  });

  it('gives the leader .leader and a Leading crown; tied cards get .tied and no .leader', () => {
    setup(eventView({ sets: [setA(), setB()] }));
    open('u-a');
    expect(cardEl('a1').classList).toContain('leader');
    expect(cardEl('a1').textContent).toContain('Leading');
    expect(cardEl('a2').classList).not.toContain('leader');

    fixture.componentInstance.closeSheet();
    fixture.detectChanges();
    open('u-b');
    for (const id of ['b1', 'b3']) {
      expect(cardEl(id).classList).toContain('tied');
      expect(cardEl(id).classList).not.toContain('leader');
      expect(cardEl(id).textContent).toContain('Tied');
    }
    expect(cardEl('b2').classList).not.toContain('tied');
  });

  it('shows the live vote counts', () => {
    setup(eventView({ sets: [setA()] }));
    open('u-a');
    expect(cardEl('a1').textContent).toContain('3 votes');
    expect(cardEl('a2').textContent).toContain('1 vote');
  });

  it('shows each commander in the sheet with its mana value and rarity', () => {
    setup(eventView({ sets: [setA({ cmc: 4, rarity: 'rare' })] }));
    open('u-a');
    const row = el().querySelector('.commander-row') as HTMLElement;
    expect(row.querySelector('.row-pip i.ms-4')).not.toBeNull();
    expect(row.textContent).toContain('4 CMC');
    expect(row.textContent).toContain('Rare');
  });

  it('a closed event hides the vote buttons and renders the winner names', () => {
    setup(eventView({ status: 'closed', closedAt: '2026-09-28T23:00:00+00:00', sets: [setA(), setB()] }));
    open('u-a');
    expect(voteButtons().length).toBe(0);
    expect(cardEl('a1').textContent).toContain('Winner');

    const banner = el().querySelector('.winners-banner') as HTMLElement;
    expect(banner).not.toBeNull();
    expect(banner.textContent).toContain('Zur\'ka, Élan of Ash');
    expect(banner.textContent).toContain('Version 1');
    expect(banner.textContent).toContain('Grimbold the Unbowed');
    expect(banner.textContent).toContain('Tied');
    expect(banner.textContent).toContain('Ask the host to pick');
    expect(banner.textContent).toContain("Legal commanders on the event night");
    expect(banner.textContent).toContain('One winning version per commander');
  });

  it('does not vote on a closed event', () => {
    setup(eventView({ status: 'closed', sets: [setA()] }));
    fixture.componentInstance.vote(setA(), setA().cards[1]);
    expect(events.vote).not.toHaveBeenCalled();
  });

  it('a 409 shows "Voting is closed" in the sheet and refreshes immediately', () => {
    setup(eventView({ sets: [setA()] }));
    open('u-a');
    events.vote.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 409, error: { error: 'Voting is closed' }
    })));
    const before = events.current.calls.count();

    voteButtons()[0].click();
    fixture.detectChanges();

    expect(text(el().querySelector('.sheet-foot .alert'))).toContain('Voting is closed');
    expect(events.current.calls.count()).toBe(before + 1);
  });

  it('a 409 without a body still says "Voting is closed"', () => {
    setup(eventView({ sets: [setA()] }));
    open('u-a');
    events.vote.and.returnValue(throwError(() => new HttpErrorResponse({ status: 409 })));
    voteButtons()[0].click();
    fixture.detectChanges();
    expect(el().textContent).toContain('Voting is closed');
  });

  it('ignores a second click while the first vote is in flight', () => {
    setup(eventView({ sets: [setA()] }));
    open('u-a');
    events.vote.and.returnValue(new Subject<SetView>());
    voteButtons()[1].click();
    fixture.detectChanges();
    expect(voteButtons().every(b => b.disabled)).toBeTrue();
    expect(slot('sA').classList).toContain('state-busy');
    voteButtons()[2].click();
    fixture.componentInstance.vote(setA(), setA().cards[2]);
    expect(events.vote).toHaveBeenCalledTimes(1);
  });

  describe('reroll requests', () => {
    const askButtons = () => Array.from(el().querySelectorAll('button.reroll-ask')) as HTMLButtonElement[];
    const mine = (overrides: Partial<SetView> = {}) => setA({ userId: 'u-me', username: 'Me', ...overrides });

    it('explains them in how voting works', () => {
      setup(eventView({ sets: [setA()] }));
      const rules = text(el().querySelector('.how-to-vote'));
      expect(rules).toContain('Ask its owner for a reroll');
      expect(rules).toContain('Once 3 players ask');
    });

    it("lets me ask for a reroll of someone else's version, and take it back", () => {
      setup(eventView({ sets: [setA()] }));
      open('u-a');
      expect(askButtons().length).toBe(3);
      expect(text(askButtons()[0])).toContain('Broken? Ask for a reroll');

      const asked = setA();
      asked.cards[0] = { ...asked.cards[0], rerollRequests: 1, myRerollRequest: true };
      events.requestReroll.and.returnValue(of(asked));
      askButtons()[0].click();
      fixture.detectChanges();
      expect(events.requestReroll).toHaveBeenCalledOnceWith('a1', true);
      expect(askButtons()[0].getAttribute('aria-pressed')).toBe('true');
      expect(text(askButtons()[0])).toContain('You asked for a reroll');
      expect(text(cardEl('a1').querySelector('.reroll-count'))).toContain('1 of 3 asked');

      events.requestReroll.and.returnValue(of(setA()));
      askButtons()[0].click();
      expect(events.requestReroll).toHaveBeenCalledWith('a1', false);
    });

    it('shows a failed request in the sheet', () => {
      setup(eventView({ sets: [setA()] }));
      open('u-a');
      events.requestReroll.and.returnValue(throwError(() => new HttpErrorResponse({
        status: 409, error: { error: "That version isn't ready yet" }
      })));
      askButtons()[1].click();
      fixture.detectChanges();
      expect(text(el().querySelector('.sheet-foot .alert'))).toContain("That version isn't ready yet");
    });

    it('shows the owner the count, then a Reroll button once enough players asked', () => {
      const set = mine();
      set.cards[0] = { ...set.cards[0], rerollRequests: 2 };
      set.cards[1] = { ...set.cards[1], rerollRequests: 3, canReroll: true };
      setup(eventView({ sets: [set] }));
      open('u-me');
      expect(askButtons().length).toBe(0);
      expect(text(cardEl('a1'))).toContain('2 of 3 asked for a reroll');
      expect(cardEl('a1').querySelector('.reroll-now')).toBeNull();
      expect(text(cardEl('a2'))).toContain("3 players say it doesn't work");
      expect(cardEl('a3').querySelector('.reroll-bar')).toBeNull();

      const confirm = spyOn(window, 'confirm').and.returnValue(true);
      generation.reroll.and.returnValue(of(setCard({ id: 'a2-new', slot: 2, status: 'queued' })));
      const before = events.current.calls.count();
      (cardEl('a2').querySelector('.reroll-now') as HTMLButtonElement).click();
      expect(confirm.calls.mostRecent().args[0]).toContain('votes on it are cleared');
      expect(generation.reroll).toHaveBeenCalledOnceWith('a2');
      expect(events.current.calls.count()).toBe(before + 1);
    });

    it("doesn't reroll when the owner cancels", () => {
      const set = mine();
      set.cards[1] = { ...set.cards[1], rerollRequests: 3, canReroll: true };
      setup(eventView({ sets: [set] }));
      open('u-me');
      spyOn(window, 'confirm').and.returnValue(false);
      (cardEl('a2').querySelector('.reroll-now') as HTMLButtonElement).click();
      expect(generation.reroll).not.toHaveBeenCalled();
    });

    it('tells the owner when a version can be rerolled, and Review opens it', () => {
      const set = mine({ cmc: 4 });
      set.cards[1] = { ...set.cards[1], rerollRequests: 3, canReroll: true };
      setup(eventView({ sets: [setB(), set] }));
      expect(text(el().querySelector('.reroll-notice'))).toContain("Players say version 2 of Zur'ka, Élan of Ash doesn't work");
      (el().querySelector('.reroll-notice button') as HTMLButtonElement).click();
      fixture.detectChanges();
      expect(fixture.componentInstance.sheetUserId).toBe('u-me');
      expect(fixture.componentInstance.sheetFocusSetId).toBe('sA');
    });

    it('shows a version being rerolled, with no vote or ask button, and a failed one to the owner', () => {
      const theirs = setA();
      theirs.cards[0] = setCard({ id: 'a1', slot: 1, setId: 'sA', status: 'generating', cardImageUrl: null });
      setup(eventView({ sets: [theirs] }));
      expect(slot('sA').querySelectorAll('img.fan-card').length).toBe(2);
      expect(slot('sA').querySelector('.fan-blank')).not.toBeNull();
      open('u-a');
      expect(text(cardEl('a1'))).toContain('Rerolling version 1…');
      expect(cardEl('a1').querySelector('.vote-button')).toBeNull();
      expect(cardEl('a1').querySelector('.reroll-ask')).toBeNull();
      expect(voteButtons().length).toBe(2);

      fixture.destroy();
      TestBed.resetTestingModule();
      const failed = mine();
      failed.cards[0] = setCard({ id: 'a1', slot: 1, setId: 'sA', status: 'failed', cardImageUrl: null, canReroll: true });
      setup(eventView({ sets: [failed] }));
      expect(text(el().querySelector('.reroll-notice'))).toContain('Version 1 of');
      open('u-me');
      expect(text(cardEl('a1'))).toContain('Version 1 failed');
      expect(cardEl('a1').querySelector('.reroll-now')).not.toBeNull();
    });

    it('shows no reroll controls once voting is closed', () => {
      const set = mine();
      set.cards[1] = { ...set.cards[1], rerollRequests: 3, canReroll: true };
      setup(eventView({ status: 'closed', closedAt: '2026-09-28T23:00:00+00:00', sets: [set, setB()] }));
      expect(el().querySelector('.reroll-notice')).toBeNull();
      open('u-me');
      expect(el().querySelector('.reroll-bar')).toBeNull();
      fixture.componentInstance.reroll(set, set.cards[1]);
      fixture.componentInstance.requestReroll(setB().cards[0], true);
      expect(generation.reroll).not.toHaveBeenCalled();
      expect(events.requestReroll).not.toHaveBeenCalled();
    });
  });

  it("after a reload once voting has closed, shows the latest event's results", () => {
    const closed = eventView({ id: 'e-9', name: 'AI Night 9', status: 'closed', closedAt: '2026-10-03T23:00:00+00:00',
                               sets: [setA({ userId: 'u-me', cmc: 3 })] });
    setup(null, 'u-me', closed);
    expect(events.get).toHaveBeenCalledWith('e-9');
    expect(el().textContent).toContain('AI Night 9');
    expect(el().querySelector('.my-legal')).not.toBeNull();
    expect(el().textContent).not.toContain('No event open');
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
    expect(el().querySelector('.next-to-vote')).toBeNull();
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
    expect(tiles().length).toBe(2);

    tick(10000);
    expect(events.current).toHaveBeenCalledTimes(3);
    fixture.destroy();
  }));
});
