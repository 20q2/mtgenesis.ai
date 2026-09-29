import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { ReactiveFormsModule } from '@angular/forms';
import { Subject, of, throwError } from 'rxjs';
import { CardView, EventView, GenerationResponse, SetView } from '../../models/api.model';
import { Rarity } from '../../models/card.model';
import { EventService } from '../../services/event.service';
import { GenerationService } from '../../services/generation.service';
import { cardView, doneCard, eventView, setCard, setView } from '../../testing/fixtures';
import { SetBuilderPageComponent } from './set-builder-page.component';

describe('SetBuilderPageComponent', () => {
  let fixture: ComponentFixture<SetBuilderPageComponent>;
  let component: SetBuilderPageComponent;
  let gen: jasmine.SpyObj<GenerationService>;
  let events: jasmine.SpyObj<EventService>;
  let watches: Record<string, Subject<CardView>>;

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

    TestBed.configureTestingModule({
      imports: [ReactiveFormsModule],
      declarations: [SetBuilderPageComponent],
      providers: [
        { provide: GenerationService, useValue: gen },
        { provide: EventService, useValue: events }
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
});
