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

  it("a panel's taken rarities are the other CMCs' rarities", () => {
    setup([three(), five()], openEvent);
    expect(component.takenRaritiesFor(4)).toEqual({ uncommon: 3, mythic: 5 });
    expect(component.takenRaritiesFor(3)).toEqual({ mythic: 5 });
  });

  it('the summary strip shows each commander\'s rarity and status', () => {
    setup([three(), five()], openEvent);
    expect(component.summary(3)).toBe('Uncommon · Draft');
    expect(component.summary(4)).toBe('Not started');
    expect(component.summary(5)).toBe('Mythic · Locked');
    const strip = fixture.nativeElement.querySelector('.summary-strip').textContent;
    expect(strip).toContain('3 CMC');
    expect(strip).toContain('Mythic · Locked');
  });

  it('a panel state change updates the summary and the others\' taken rarities', () => {
    setup([three()], openEvent);
    component.onStateChange(4, { status: 'draft', rarity: Rarity.RARE });
    expect(component.summary(4)).toBe('Rare · Draft');
    expect(component.takenRaritiesFor(5)).toEqual({ uncommon: 3, rare: 4 });
    component.onStateChange(3, null);
    expect(component.takenRaritiesFor(5)).toEqual({ rare: 4 });
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
