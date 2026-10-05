import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { of } from 'rxjs';
import { EventView, SetView } from '../../models/api.model';
import { Rarity } from '../../models/card.model';
import { EventService } from '../../services/event.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { fakeVisibility } from '../../testing/fake-visibility';
import { eventView, setView } from '../../testing/fixtures';
import { EVENT_POLL_MS, SetBuilderPageComponent } from './set-builder-page.component';

describe('SetBuilderPageComponent', () => {
  let fixture: ComponentFixture<SetBuilderPageComponent>;
  let component: SetBuilderPageComponent;
  let events: jasmine.SpyObj<EventService>;
  let visibility: ReturnType<typeof fakeVisibility>;

  const openEvent = eventView({ sets: [] });

  function setup(mySets: SetView[], current: EventView | null) {
    events = jasmine.createSpyObj<EventService>('EventService', ['mySets', 'current']);
    events.mySets.and.returnValue(of(mySets));
    events.current.and.returnValue(of(current));
    visibility = fakeVisibility();

    TestBed.configureTestingModule({
      declarations: [SetBuilderPageComponent],
      providers: [
        { provide: EventService, useValue: events },
        { provide: PageVisibilityService, useValue: visibility }
      ],
      schemas: [NO_ERRORS_SCHEMA]
    });
    fixture = TestBed.createComponent(SetBuilderPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  afterEach(() => fixture.destroy());

  const three = () => setView({ id: 's-3', status: 'draft', cmc: 3, rarity: 'uncommon' });
  const five = () => setView({ id: 's-5', status: 'locked', cmc: 5, rarity: 'mythic' });

  it('loads mySets into one panel per CMC', () => {
    setup([three(), five()], openEvent);
    expect(component.sets[3]!.id).toBe('s-3');
    expect(component.sets[4]).toBeNull();
    expect(component.sets[5]!.id).toBe('s-5');
    expect(fixture.nativeElement.querySelectorAll('app-commander-panel').length).toBe(3);
  });

  it("a panel's taken rarities are the other CMCs' locked rarities (drafts may share one)", () => {
    setup([three(), five()], openEvent);
    expect(component.takenRaritiesFor(4)).toEqual({ mythic: 5 });
    expect(component.takenRaritiesFor(3)).toEqual({ mythic: 5 });
    expect(component.takenRaritiesFor(5)).toEqual({});
  });

  it('the summary strip shows each commander\'s rarity and status', () => {
    setup([three(), five()], openEvent);
    expect(component.summary(3)).toBe('Uncommon · Draft');
    expect(component.summary(4)).toBe('Not started');
    expect(component.summary(5)).toBe('Mythic · Locked');
    const lineup = fixture.nativeElement.querySelector('.lineup').textContent;
    expect(lineup).toContain('Mythic · Locked');
    expect(lineup).toContain('Zur');  // the commander's name
    expect(lineup).toContain('Not started');
  });

  it('shows where the player is in the night', () => {
    setup([three(), five()], openEvent);
    expect(component.nightStep).toBe('build');
    expect(fixture.nativeElement.querySelector('app-night-steps')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('.next-step')).toBeNull();
  });

  it('once all three are locked in, says the next step is voting', () => {
    const locked = (id: string, cmc: number, rarity: string) => setView({ id, status: 'locked', cmc, rarity });
    setup([locked('a', 3, 'uncommon'), locked('b', 4, 'rare'), locked('c', 5, 'mythic')], openEvent);
    fixture.detectChanges();
    expect(component.nightStep).toBe('vote');
    const next = fixture.nativeElement.querySelector('.next-step') as HTMLElement;
    expect(next.textContent).toContain('All three commanders are locked in');
    expect(next.querySelector('a')!.getAttribute('routerLink') ?? next.querySelector('a')!.getAttribute('ng-reflect-router-link'))
      .toBe('/vote');
  });

  it('counts the commanders locked in', () => {
    setup([three(), five()], openEvent);
    expect(component.lockedCount).toBe(1);
    expect(fixture.nativeElement.querySelector('.progress').textContent.replace(/\s+/g, ' ')).toContain('1 of 3 locked in');
  });

  it('lays out the rules: mana values, type, body, rarity, versions and the double self-vote', () => {
    setup([], openEvent);
    const rules = (fixture.nativeElement.querySelector('.ledger') as HTMLElement).textContent!.replace(/\s+/g, ' ');
    expect(rules).toContain('One at each mana value');
    expect(rules).toContain('Legendary Creature or Vehicle');
    expect(rules).toContain('Multicolor is fine');
    expect(rules).toContain('CMC + 1 points');
    expect(rules).toContain('no X or *');
    expect(rules).toContain('One Uncommon, one Rare, one Mythic');
    expect(rules).toContain('No tweaking');
  });

  it('folds the rules away until opened', () => {
    setup([], openEvent);
    const details = fixture.nativeElement.querySelector('details.rules') as HTMLDetailsElement;
    expect(details.open).toBeFalse();
    expect(details.querySelector('summary')!.textContent).toContain('The rules');
    expect(details.querySelector('.ledger')).not.toBeNull();
  });

  it('tracks which commander holds each rarity', () => {
    setup([three(), five()], openEvent);
    expect(component.rarityHolders('uncommon')).toEqual([{ cmc: 3, locked: false }]);
    expect(component.rarityHolders('mythic')).toEqual([{ cmc: 5, locked: true }]);
    expect(component.rarityHolders('rare')).toEqual([]);
    const tracker = fixture.nativeElement.querySelector('.rarity-track').textContent.replace(/\s+/g, ' ');
    expect(tracker).toContain('Uncommon 3 CMC');
    expect(tracker).toContain('Rare open');
  });

  it('a panel state change updates the summary and the others\' taken rarities', () => {
    setup([three()], openEvent);
    component.onStateChange(4, { status: 'draft', rarity: Rarity.RARE, name: 'Mother' });
    expect(component.summary(4)).toBe('Rare · Draft');
    expect(component.takenRaritiesFor(5)).toEqual({});
    component.onStateChange(4, { status: 'locked', rarity: Rarity.RARE, name: 'Mother' });
    expect(component.taken[5]).toEqual({ rare: 4 });
    component.onStateChange(4, null);
    expect(component.taken[5]).toEqual({});
  });

  it('tabs switch the visible commander', () => {
    setup([], openEvent);
    expect(component.activeCmc).toBe(3);
    component.selectCmc(5);
    fixture.detectChanges();
    const panels = Array.from<HTMLElement>(fixture.nativeElement.querySelectorAll('app-commander-panel'));
    expect(panels.map(p => p.hidden)).toEqual([true, true, false]);
  });

  describe('event polling', () => {
    it('checks for an open event every 30s', fakeAsync(() => {
      expect(EVENT_POLL_MS).toBe(30000);
      setup([], null);
      expect(events.current).toHaveBeenCalledTimes(1);
      tick(29999);
      expect(events.current).toHaveBeenCalledTimes(1);
      tick(1);
      expect(events.current).toHaveBeenCalledTimes(2);
      fixture.destroy();
    }));

    it('pauses while the page is hidden and checks at once when it is shown', fakeAsync(() => {
      setup([], null);
      visibility.visibleSubject.next(false);
      tick(120000);
      expect(events.current).toHaveBeenCalledTimes(1);

      events.current.and.returnValue(of(openEvent));
      visibility.visibleSubject.next(true);
      expect(events.current).toHaveBeenCalledTimes(2);
      expect(component.event?.status).toBe('open');
      fixture.destroy();
    }));
  });
});
