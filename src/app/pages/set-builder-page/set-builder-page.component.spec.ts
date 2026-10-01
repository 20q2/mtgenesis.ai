import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { ReactiveFormsModule } from '@angular/forms';
import { Subject, of, throwError } from 'rxjs';
import { CardView, EventView, GenerationResponse, SetView } from '../../models/api.model';
import { Rarity } from '../../models/card.model';
import { EventService } from '../../services/event.service';
import { GenerationService } from '../../services/generation.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { fakeVisibility } from '../../testing/fake-visibility';
import { cardView, doneCard, eventView, setCard, setView } from '../../testing/fixtures';
import { EVENT_POLL_MS, SetBuilderPageComponent, commanderPipValue } from './set-builder-page.component';

describe('SetBuilderPageComponent', () => {
  let fixture: ComponentFixture<SetBuilderPageComponent>;
  let component: SetBuilderPageComponent;
  let gen: jasmine.SpyObj<GenerationService>;
  let events: jasmine.SpyObj<EventService>;
  let watches: Record<string, Subject<CardView>>;
  let visibility: ReturnType<typeof fakeVisibility>;

  const openEvent = eventView({ sets: [] });

  function draft(cards = [1, 2, 3].map(slot => setCard({ id: `c${slot}`, slot, setId: 's-1' }))): SetView {
    return setView({ id: 's-1', status: 'draft', eventId: null, lockedAt: null, commanderName: 'Zur\'ka', cards });
  }

  function setup(mySet: SetView | null, current: EventView | null) {
    watches = {};
    gen = jasmine.createSpyObj<GenerationService>('GenerationService',
      ['submit', 'reroll', 'watch', 'cardParams', 'promptFor']);
    gen.watch.and.callFake((id: string) => (watches[id] = new Subject<CardView>()));
    gen.cardParams.and.callFake(GenerationService.prototype.cardParams);
    gen.promptFor.and.callFake(GenerationService.prototype.promptFor);
    events = jasmine.createSpyObj<EventService>('EventService', ['mySet', 'current', 'lock', 'unlock']);
    events.mySet.and.returnValue(of(mySet));
    events.current.and.returnValue(of(current));
    visibility = fakeVisibility();

    TestBed.configureTestingModule({
      imports: [ReactiveFormsModule],
      declarations: [SetBuilderPageComponent],
      providers: [
        { provide: GenerationService, useValue: gen },
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

  function generateButton(): HTMLButtonElement {
    return fixture.nativeElement.querySelector('button.generate-set');
  }

  function nameInput(): HTMLInputElement {
    return fixture.nativeElement.querySelector('#commanderName');
  }

  function changeNameButton(): HTMLButtonElement | null {
    return fixture.nativeElement.querySelector('button.change-name');
  }

  describe('lockDisabledReason()', () => {
    it('says no event is open when there is no current event', () => {
      setup(draft(), null);
      expect(component.lockDisabledReason()).toBe('No event open — ask the host');
    });

    it('says it is waiting when any card is not done', () => {
      const cards = draft().cards;
      cards[1] = { ...cards[1], status: 'generating', artReady: false, queuePosition: 0 };
      setup(draft(cards), openEvent);
      expect(component.lockDisabledReason()).toBe('Waiting for all 3 cards');
    });

    it('is null when all 3 cards are done and an event is open', () => {
      setup(draft(), openEvent);
      expect(component.lockDisabledReason()).toBeNull();
      fixture.detectChanges();
      expect((fixture.nativeElement.querySelector('button.lock') as HTMLButtonElement).disabled).toBeFalse();
    });

    it('treats a closed event as no event', () => {
      setup(draft(), eventView({ status: 'closed', sets: [] }));
      expect(component.lockDisabledReason()).toBe('No event open — ask the host');
    });
  });

  it('resumes watching unfinished cards on load', () => {
    const cards = draft().cards;
    cards[0] = { ...cards[0], status: 'queued', queuePosition: 2, artReady: false, textReady: false };
    setup(draft(cards), openEvent);
    expect(gen.watch).toHaveBeenCalledOnceWith('c1');

    watches['c1'].next(doneCard({ id: 'c1', slot: 1, setId: 's-1' }));
    expect(component.slots[0]!.status).toBe('done');
  });

  it('a reroll replaces the slot view with the returned CardView and watches it', () => {
    setup(draft(), openEvent);
    const fresh = cardView({ id: 'c2-new', slot: 2, setId: 's-1', status: 'queued', queuePosition: 1 });
    gen.reroll.and.returnValue(of(fresh));

    component.onReroll(component.slots[1]!);

    expect(gen.reroll).toHaveBeenCalledWith('c2');
    expect(component.slots[1]!.id).toBe('c2-new');
    expect(gen.watch).toHaveBeenCalledWith('c2-new');

    watches['c2-new'].next(doneCard({ id: 'c2-new', slot: 2, setId: 's-1' }));
    expect(component.slots[1]!.status).toBe('done');
  });

  it('a 409 on reroll shows the server error text', () => {
    setup(draft(), openEvent);
    gen.reroll.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 409, error: { error: 'Card is still generating' }
    })));
    component.onReroll(component.slots[0]!);
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Card is still generating');
    expect(component.slots[0]!.id).toBe('c1');
  });

  it('disables the generate button while a submit is in flight (no double submit)', () => {
    setup(null, openEvent);
    const response$ = new Subject<GenerationResponse>();
    gen.submit.and.returnValue(response$);
    component.commanderName.setValue('  Zur\'ka, Élan of Ash  ');
    component.onCardChange({
      name: 'ignored', manaCost: '{2}{R}', type: 'Creature', colors: ['R'], cmc: 3,
      rarity: Rarity.MYTHIC, artPrompt: 'Fantasy art of a fire queen', supertype: 'Legendary'
    });
    fixture.detectChanges();
    expect(generateButton().disabled).toBeFalse();

    generateButton().click();
    fixture.detectChanges();
    expect(generateButton().disabled).toBeTrue();
    generateButton().click();
    component.generate();
    expect(gen.submit).toHaveBeenCalledTimes(1);

    const req = gen.submit.calls.mostRecent().args[0];
    expect(req.count).toBe(3);
    expect(req.commanderName).toBe('Zur\'ka, Élan of Ash');
    expect(req.cardData.name).toBe('Zur\'ka, Élan of Ash');
    expect(req.prompt).toBe('Fantasy art of a fire queen');

    response$.next({
      setId: 's-9',
      cards: [1, 2, 3].map(slot => cardView({ id: `n${slot}`, slot, setId: 's-9' }))
    });
    response$.complete();
    fixture.detectChanges();

    expect(component.slots.map(s => s!.id)).toEqual(['n1', 'n2', 'n3']);
    expect(gen.watch.calls.allArgs().map(a => a[0])).toEqual(['n1', 'n2', 'n3']);
    // Still disabled: the new set has pending cards.
    expect(generateButton().disabled).toBeTrue();
  });

  describe('mana value per version', () => {
    it('labels the versions 3, 4 and 5 mana', () => {
      setup(null, openEvent);
      expect(component.slotLabel(0)).toBe('Version 1 · 3 mana');
      expect(component.slotLabel(2)).toBe('Version 3 · 5 mana');
    });

    it('counts colored pips, not generic or X', () => {
      expect(commanderPipValue('{X}{4}{W}{U}')).toBe(2);
      expect(commanderPipValue('{2/W}{G/P}')).toBe(3);
      expect(commanderPipValue('')).toBe(0);
    });

    it('disables Generate with a reason when the pips cost more than 3', () => {
      setup(null, openEvent);
      component.commanderName.setValue('Zur');
      component.onCardChange({
        name: '', manaCost: '{W}{W}{U}{U}', type: 'Creature', colors: ['W', 'U'], cmc: 4,
        rarity: Rarity.RARE
      });
      fixture.detectChanges();
      expect(component.canGenerate()).toBeFalse();
      expect(generateButton().disabled).toBeTrue();
      expect(component.generateHint()).toContain('at most 3');
      component.generate();
      expect(gen.submit).not.toHaveBeenCalled();
    });
  });

  it('requires a commander name', () => {
    setup(null, openEvent);
    component.commanderName.setValue('   ');
    component.generate();
    expect(gen.submit).not.toHaveBeenCalled();
    expect(component.error).toBe('Enter a commander name.');
  });

  it('a 429 on generate shows the server error text', () => {
    setup(null, openEvent);
    gen.submit.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 429, error: { error: 'You already have 3 cards in progress' }
    })));
    component.commanderName.setValue('Zur\'ka');
    component.generate();
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('You already have 3 cards in progress');
    expect(component.submitting).toBeFalse();
  });

  it('locks in with the commander name and shows the Locked badge', () => {
    setup(draft(), openEvent);
    events.lock.and.returnValue(of(setView({ id: 's-1', status: 'locked', commanderName: 'Zur\'ka' })));
    component.lock();
    fixture.detectChanges();
    expect(events.lock).toHaveBeenCalledWith('s-1', 'Zur\'ka');
    expect(fixture.nativeElement.querySelector('.locked-badge')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('button.unlock')).not.toBeNull();
    expect(component.canRerollSlot(component.slots[0])).toBeFalse();
  });

  it('asks for confirmation before unlocking', () => {
    setup(setView({ id: 's-1', status: 'locked' }), openEvent);
    events.unlock.and.returnValue(of(draft()));
    const confirmSpy = spyOn(window, 'confirm').and.returnValue(false);

    component.unlock();
    expect(confirmSpy).toHaveBeenCalledWith('This clears votes on your set');
    expect(events.unlock).not.toHaveBeenCalled();

    confirmSpy.and.returnValue(true);
    component.unlock();
    expect(events.unlock).toHaveBeenCalledWith('s-1');
    expect(component.setStatus).toBe('draft');
  });

  describe('commander name (the vote-page title must match the rendered cards)', () => {
    it('is editable before any set exists', () => {
      setup(null, openEvent);
      expect(nameInput().readOnly).toBeFalse();
      expect(changeNameButton()).toBeNull();
    });

    it('is read-only once a draft exists', () => {
      setup(draft(), openEvent);
      expect(nameInput().readOnly).toBeTrue();
      expect(nameInput().value).toBe("Zur'ka");
    });

    it('is read-only once the set is locked, with no Change name button', () => {
      setup(setView({ id: 's-1', status: 'locked', commanderName: "Zur'ka" }), openEvent);
      expect(nameInput().readOnly).toBeTrue();
      expect(changeNameButton()).toBeNull();
    });

    it('becomes read-only after Generate returns the new set', () => {
      setup(null, openEvent);
      gen.submit.and.returnValue(of({
        setId: 's-9', cards: [1, 2, 3].map(slot => cardView({ id: `n${slot}`, slot, setId: 's-9' }))
      }));
      component.commanderName.setValue('Grimbold');
      component.generate();
      fixture.detectChanges();
      expect(nameInput().readOnly).toBeTrue();
    });

    it("lock() sends the set's stored name, not whatever the field holds", () => {
      setup(draft(), openEvent);
      events.lock.and.returnValue(of(setView({ id: 's-1', status: 'locked', commanderName: "Zur'ka" })));
      component.commanderName.setValue('Something Else');
      component.lock();
      expect(events.lock).toHaveBeenCalledWith('s-1', "Zur'ka");
    });

    it('Change name unlocks the field for a new set; Lock in waits until that set is generated', () => {
      setup(draft(), openEvent);
      expect(component.lockDisabledReason()).toBeNull();

      changeNameButton()!.click();
      fixture.detectChanges();
      expect(nameInput().readOnly).toBeFalse();

      component.commanderName.setValue('Grimbold');
      expect(component.lockDisabledReason()).toBe('Generate a new set to use the new name');
      component.lock();
      expect(events.lock).not.toHaveBeenCalled();

      // Cancel puts the set's name back and Lock in works again.
      changeNameButton()!.click();
      fixture.detectChanges();
      expect(nameInput().readOnly).toBeTrue();
      expect(component.commanderName.value).toBe("Zur'ka");
      expect(component.lockDisabledReason()).toBeNull();
    });
  });

  describe('event polling', () => {
    it('checks for an open event every 30s', fakeAsync(() => {
      expect(EVENT_POLL_MS).toBe(30000);
      setup(draft(), null);
      expect(events.current).toHaveBeenCalledTimes(1);
      tick(29999);
      expect(events.current).toHaveBeenCalledTimes(1);
      tick(1);
      expect(events.current).toHaveBeenCalledTimes(2);
      fixture.destroy();
    }));

    it('pauses while the page is hidden and checks at once when it is shown', fakeAsync(() => {
      setup(draft(), null);
      visibility.visibleSubject.next(false);
      tick(120000);
      expect(events.current).toHaveBeenCalledTimes(1);

      events.current.and.returnValue(of(openEvent));
      visibility.visibleSubject.next(true);
      expect(events.current).toHaveBeenCalledTimes(2);
      expect(component.eventOpen).toBeTrue();
      fixture.destroy();
    }));
  });
});
