import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { ReactiveFormsModule } from '@angular/forms';
import { Subject, of, throwError } from 'rxjs';
import { CardView, EventView, GenerationResponse, SetView } from '../../models/api.model';
import { Card, Rarity } from '../../models/card.model';
import { EventService } from '../../services/event.service';
import { GenerationService } from '../../services/generation.service';
import { cardView, doneCard, eventView, setCard, setView } from '../../testing/fixtures';
import { CommanderPanelComponent, CommanderState, REROLL_CONFIRM } from './commander-panel.component';

describe('CommanderPanelComponent', () => {
  let fixture: ComponentFixture<CommanderPanelComponent>;
  let component: CommanderPanelComponent;
  let gen: jasmine.SpyObj<GenerationService>;
  let events: jasmine.SpyObj<EventService>;
  let watches: Record<string, Subject<CardView>>;
  let emitted: (CommanderState | null)[];

  const openEvent = eventView({ sets: [] });

  function draft(cards = [1, 2, 3].map(slot => setCard({ id: `c${slot}`, slot, setId: 's-1' }))): SetView {
    return setView({ id: 's-1', status: 'draft', eventId: null, lockedAt: null, commanderName: 'Zur\'ka',
                     cmc: 4, rarity: 'rare', cards });
  }

  function formCard(overrides: Partial<Card> = {}): Card {
    return { name: 'ignored', manaCost: '{2}{R}', type: 'Creature', colors: ['R'], cmc: 0,
             rarity: Rarity.MYTHIC, artPrompt: 'Fantasy art of a fire queen', supertype: 'Legendary',
             commanderKind: 'creature', power: '', toughness: '', ...overrides };
  }

  function setup(set: SetView | null, current: EventView | null,
                 taken: Partial<Record<Rarity, number>> = {}, cmc = 4) {
    watches = {};
    emitted = [];
    gen = jasmine.createSpyObj<GenerationService>('GenerationService',
      ['submit', 'reroll', 'watch', 'cardParams', 'promptFor']);
    gen.watch.and.callFake((id: string) => (watches[id] = new Subject<CardView>()));
    gen.cardParams.and.callFake(GenerationService.prototype.cardParams);
    gen.promptFor.and.callFake(GenerationService.prototype.promptFor);
    events = jasmine.createSpyObj<EventService>('EventService', ['lock', 'unlock']);

    TestBed.configureTestingModule({
      imports: [ReactiveFormsModule],
      declarations: [CommanderPanelComponent],
      providers: [
        { provide: GenerationService, useValue: gen },
        { provide: EventService, useValue: events }
      ],
      schemas: [NO_ERRORS_SCHEMA]
    });
    fixture = TestBed.createComponent(CommanderPanelComponent);
    component = fixture.componentInstance;
    component.stateChange.subscribe(s => emitted.push(s));
    fixture.componentRef.setInput('cmc', cmc);
    fixture.componentRef.setInput('set', set);
    fixture.componentRef.setInput('event', current);
    fixture.componentRef.setInput('takenRarities', taken);
    fixture.detectChanges();
  }

  afterEach(() => fixture.destroy());

  function generateButton(): HTMLButtonElement {
    return fixture.nativeElement.querySelector('button.generate-set');
  }

  function nameInput(): HTMLInputElement {
    return fixture.nativeElement.querySelector('input.commander-name');
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
      expect(component.lockDisabledReason()).toBe('Waiting for all 3 versions');
    });

    it('is null when all 3 cards are done and an event is open', () => {
      setup(draft(), openEvent);
      expect(component.lockDisabledReason()).toBeNull();
      fixture.detectChanges();
      expect((fixture.nativeElement.querySelector('button.lock') as HTMLButtonElement).disabled).toBeFalse();
    });

    it("explains a draft whose rarity another locked commander already has", () => {
      setup(draft(), openEvent, { rare: 3 });
      expect(component.lockDisabledReason())
        .toBe('Your 3 CMC commander is locked in as Rare — generate this one with another rarity');
    });

    it('treats a closed event as no event', () => {
      setup(draft(), eventView({ status: 'closed', sets: [] }));
      expect(component.lockDisabledReason()).toBe('No event open — ask the host');
    });
  });

  it('labels the slots Version 1..3', () => {
    setup(null, openEvent);
    expect([0, 1, 2].map(i => component.slotLabel(i))).toEqual(['Version 1', 'Version 2', 'Version 3']);
  });

  it('resumes watching unfinished cards on load', () => {
    const cards = draft().cards;
    cards[0] = { ...cards[0], status: 'queued', queuePosition: 2, artReady: false, textReady: false };
    setup(draft(cards), openEvent);
    expect(gen.watch).toHaveBeenCalledOnceWith('c1');

    watches['c1'].next(doneCard({ id: 'c1', slot: 1, setId: 's-1' }));
    expect(component.slots[0]!.status).toBe('done');
  });

  describe('reroll', () => {
    it('asks the rule-6 question and does nothing on cancel', () => {
      setup(draft(), openEvent);
      const confirmSpy = spyOn(window, 'confirm').and.returnValue(false);
      component.onReroll(component.slots[1]!);
      expect(confirmSpy).toHaveBeenCalledWith(REROLL_CONFIRM);
      expect(REROLL_CONFIRM).toBe(
        'Rerolls are only for a version that doesn\'t function under any circumstance. Reroll it?');
      expect(gen.reroll).not.toHaveBeenCalled();
    });

    it('replaces the slot with the returned card and watches it on OK', () => {
      setup(draft(), openEvent);
      spyOn(window, 'confirm').and.returnValue(true);
      const fresh = cardView({ id: 'c2-new', slot: 2, setId: 's-1', status: 'queued', queuePosition: 1 });
      gen.reroll.and.returnValue(of(fresh));

      component.onReroll(component.slots[1]!);

      expect(gen.reroll).toHaveBeenCalledWith('c2');
      expect(component.slots[1]!.id).toBe('c2-new');
      watches['c2-new'].next(doneCard({ id: 'c2-new', slot: 2, setId: 's-1' }));
      expect(component.slots[1]!.status).toBe('done');
    });

    it('a 409 shows the server error text', () => {
      setup(draft(), openEvent);
      spyOn(window, 'confirm').and.returnValue(true);
      gen.reroll.and.returnValue(throwError(() => new HttpErrorResponse({
        status: 409, error: { error: 'Card is still generating' }
      })));
      component.onReroll(component.slots[0]!);
      fixture.detectChanges();
      expect(fixture.nativeElement.textContent).toContain('Card is still generating');
      expect(component.slots[0]!.id).toBe('c1');
    });
  });

  it('generate sends count 3 and the cmc, disables while in flight, and reports a draft', () => {
    setup(null, openEvent, {}, 5);
    const response$ = new Subject<GenerationResponse>();
    gen.submit.and.returnValue(response$);
    component.commanderName.setValue('  Zur\'ka, Élan of Ash  ');
    component.onCardChange(formCard({ power: '3', toughness: '3' }));
    fixture.detectChanges();
    expect(generateButton().disabled).toBeFalse();

    generateButton().click();
    fixture.detectChanges();
    expect(generateButton().disabled).toBeTrue();
    component.generate();
    expect(gen.submit).toHaveBeenCalledTimes(1);

    const req = gen.submit.calls.mostRecent().args[0];
    expect(req.count).toBe(3);
    expect(req.cmc).toBe(5);
    expect(req.commanderName).toBe('Zur\'ka, Élan of Ash');
    expect(req.cardData.name).toBe('Zur\'ka, Élan of Ash');
    expect(req.cardData.power).toBe('3');
    expect(req.cardData.commanderKind).toBe('creature');
    expect(req.prompt).toBe('Fantasy art of a fire queen');

    response$.next({
      setId: 's-9',
      cards: [1, 2, 3].map(slot => cardView({ id: `n${slot}`, slot, setId: 's-9' }))
    });
    response$.complete();
    fixture.detectChanges();

    expect(component.slots.map(s => s!.id)).toEqual(['n1', 'n2', 'n3']);
    expect(emitted).toEqual([{ status: 'draft', rarity: Rarity.MYTHIC, name: "Zur'ka, Élan of Ash" }]);
    expect(generateButton().disabled).toBeTrue(); // the new versions are pending
  });

  it('disables Generate with a reason when the pips cost more than 3', () => {
    setup(null, openEvent);
    component.commanderName.setValue('Zur');
    component.onCardChange(formCard({ manaCost: '{W}{W}{U}{U}' }));
    fixture.detectChanges();
    expect(component.canGenerate()).toBeFalse();
    expect(generateButton().disabled).toBeTrue();
    expect(component.generateHint()).toContain('at most 3');
    component.generate();
    expect(gen.submit).not.toHaveBeenCalled();
  });

  it('disables Generate with a hint when the rarity is taken', () => {
    setup(null, openEvent, { rare: 3 });
    component.commanderName.setValue('Zur');
    component.onCardChange(formCard({ rarity: Rarity.RARE }));
    expect(component.canGenerate()).toBeFalse();
    expect(component.generateHint()).toContain('3 CMC');
  });

  it('disables Generate with the P/T error when stats are invalid', () => {
    setup(null, openEvent);
    component.commanderName.setValue('Zur');
    component.onCardChange(formCard({ power: '*', toughness: '*' }));
    expect(component.canGenerate()).toBeFalse();
    expect(component.generateHint()).toBe('P/T must be whole numbers — X and * aren\'t allowed');
  });

  it('requires a commander name', () => {
    setup(null, openEvent);
    component.commanderName.setValue('   ');
    component.generate();
    expect(gen.submit).not.toHaveBeenCalled();
    expect(component.error).toBe('Enter a commander name.');
  });

  it('a 409 on generate shows the server error text', () => {
    setup(null, openEvent);
    gen.submit.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 409, error: { error: 'You already have a Rare commander (3 CMC)' }
    })));
    component.commanderName.setValue('Zur\'ka');
    component.onCardChange(formCard());
    component.generate();
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('You already have a Rare commander (3 CMC)');
    expect(component.submitting).toBeFalse();
    expect(emitted).toEqual([]);
  });

  it('locks in with the commander name, shows the Locked badge and reports it', () => {
    setup(draft(), openEvent);
    events.lock.and.returnValue(of(setView({ id: 's-1', status: 'locked', commanderName: 'Zur\'ka',
                                             cmc: 4, rarity: 'rare' })));
    component.lock();
    fixture.detectChanges();
    expect(events.lock).toHaveBeenCalledWith('s-1', 'Zur\'ka');
    expect(fixture.nativeElement.querySelector('.locked-badge')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('button.unlock')).not.toBeNull();
    expect(component.canRerollSlot(component.slots[0])).toBeFalse();
    expect(emitted).toEqual([{ status: 'locked', rarity: Rarity.RARE, name: "Zur'ka" }]);
  });

  it('asks for confirmation before unlocking', () => {
    setup(setView({ id: 's-1', status: 'locked', cmc: 4 }), openEvent);
    events.unlock.and.returnValue(of(draft()));
    const confirmSpy = spyOn(window, 'confirm').and.returnValue(false);

    component.unlock();
    expect(confirmSpy).toHaveBeenCalledWith('This clears votes on this commander');
    expect(events.unlock).not.toHaveBeenCalled();

    confirmSpy.and.returnValue(true);
    component.unlock();
    expect(events.unlock).toHaveBeenCalledWith('s-1');
    expect(component.setStatus).toBe('draft');
    expect(emitted).toEqual([{ status: 'draft', rarity: Rarity.RARE, name: "Zur'ka" }]);
  });

  describe('locked commander', () => {
    const lockedCard = { name: "Zur'ka", manaCost: '{1}{W}{U}', colors: ['W', 'U'], type: 'Creature',
                         supertype: 'Legendary', subtype: 'Human Cleric', rarity: 'rare', cmc: 3,
                         power: '2', toughness: '2' };

    it('shows a read-only summary instead of the designer', () => {
      const cards = [1, 2, 3].map(slot => setCard({ id: `c${slot}`, slot, setId: 's-1', card: lockedCard }));
      setup(setView({ id: 's-1', status: 'locked', commanderName: "Zur'ka", cmc: 3, rarity: 'rare', cards }),
            openEvent, {}, 3);
      const summary = fixture.nativeElement.querySelector('.locked-summary') as HTMLElement;
      expect(summary).not.toBeNull();
      expect(summary.textContent).toContain('Legendary Creature — Human Cleric');
      expect(summary.textContent).toContain('2/2');
      expect(summary.textContent).toContain('Rare');
      expect(fixture.nativeElement.querySelector('app-commander-form')).toBeNull();
      expect(component.costSymbols(lockedCard.manaCost)).toEqual(['{1}', '{W}', '{U}']);
      expect(component.symbolClass('{W}')).toBe('ms-w');
      expect(summary.querySelector('.ls-cost i.ms-1')).not.toBeNull();
    });
  });

  it("hands a draft's saved choices to the designer", () => {
    const saved = { name: "Zur'ka", manaCost: '{2}{B}', colors: ['B'], type: 'Creature', rarity: 'rare', cmc: 4,
                    power: '3', toughness: '2' };
    const cards = [1, 2, 3].map(slot => setCard({ id: `c${slot}`, slot, setId: 's-1', card: saved }));
    setup(draft(cards), openEvent);
    expect(component.loadedCard).toEqual(saved);
    expect(fixture.nativeElement.querySelector('app-commander-form')).not.toBeNull();
  });

  it('labels Reroll as only for broken versions', () => {
    setup(draft(), openEvent);
    expect(component.rerollLabel).toBe('Reroll (only if broken)');
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
      expect(component.lockDisabledReason()).toBe('Generate again to use the new name');
      component.lock();
      expect(events.lock).not.toHaveBeenCalled();

      changeNameButton()!.click();
      fixture.detectChanges();
      expect(nameInput().readOnly).toBeTrue();
      expect(component.commanderName.value).toBe("Zur'ka");
      expect(component.lockDisabledReason()).toBeNull();
    });
  });
});
