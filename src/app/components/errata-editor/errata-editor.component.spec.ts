import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ReactiveFormsModule } from '@angular/forms';
import { of } from 'rxjs';
import { ErrataItem } from '../../models/api.model';
import { MediaPipe } from '../../pipes/media.pipe';
import { EventService } from '../../services/event.service';
import { MediaService } from '../../services/media.service';
import { doneCard } from '../../testing/fixtures';
import { ErrataEditorComponent } from './errata-editor.component';

describe('ErrataEditorComponent', () => {
  let fixture: ComponentFixture<ErrataEditorComponent>;
  let events: jasmine.SpyObj<EventService>;

  const creature = { name: 'Zur', manaCost: '{2}{R}', colors: ['R'], type: 'Creature', rarity: 'rare',
                     cmc: 3, description: 'Flying', power: '3', toughness: '3' };
  function item(card: object = creature, errata = false): ErrataItem {
    return { card: doneCard({ id: 'c1', card: card as never, errata }), original: errata ? creature as never : null,
             label: 'Zur', username: 'alice', detail: 'Winner' };
  }

  function create(it: ErrataItem = item(), pin = '9999'): ErrataEditorComponent {
    events = jasmine.createSpyObj<EventService>('EventService', ['previewErrata', 'saveErrata', 'revertErrata']);
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [ReactiveFormsModule],
      declarations: [ErrataEditorComponent, MediaPipe],
      providers: [{ provide: EventService, useValue: events }, { provide: MediaService, useValue: media }]
    });
    fixture = TestBed.createComponent(ErrataEditorComponent);
    fixture.componentInstance.item = it;
    fixture.componentInstance.pin = pin;
    fixture.detectChanges();
    return fixture.componentInstance;
  }

  afterEach(() => fixture.destroy());

  const el = () => fixture.nativeElement as HTMLElement;
  const button = (label: string) => Array.from(el().querySelectorAll('button'))
    .find(b => b.textContent!.includes(label)) as HTMLButtonElement;

  it('opens as a modal with the card text, and P/T for a creature', () => {
    const c = create();
    expect((el().querySelector('dialog') as HTMLDialogElement).open).toBeTrue();
    expect(c.description.value).toBe('Flying');
    expect(el().querySelectorAll('.pt-row input').length).toBe(2);
    expect(el().textContent).toContain('Name, mana cost, type and rarity stay as printed');
  });

  it('has no P/T for a non-creature and sends none', () => {
    const c = create(item({ name: 'Bolt', type: 'Instant', description: 'Deal 3.' }));
    expect(el().querySelector('.pt-row')).toBeNull();
    events.saveErrata.and.returnValue(of(item()));
    c.description.setValue('Deal 2.');
    c.save();
    expect(events.saveErrata).toHaveBeenCalledWith('c1', { description: 'Deal 2.' }, '9999');
  });

  it('Save stays off until something changes', () => {
    const c = create();
    expect(button('Save errata').disabled).toBeTrue();
    c.toughness.setValue('5');
    fixture.detectChanges();
    expect(button('Save errata').disabled).toBeFalse();
  });

  it('Preview draws the change without saving, and goes stale when edited again', () => {
    const c = create();
    events.previewErrata.and.returnValue(of({ image: 'data:image/png;base64,AAAA',
      power: { estimate: 4, budget: 3, verdict: 'pushed' } }));
    c.description.setValue('Flying, lifelink');
    button('Preview').click();
    fixture.detectChanges();
    expect(events.previewErrata).toHaveBeenCalledWith('c1', { description: 'Flying, lifelink', power: '3', toughness: '3' }, '9999');
    expect(el().querySelector('.errata-image')!.getAttribute('src')).toBe('data:image/png;base64,AAAA');
    expect(el().querySelector('.power')!.textContent).toContain('Pushed');
    expect(events.saveErrata).not.toHaveBeenCalled();

    c.description.setValue('Flying, lifelink, haste');
    fixture.detectChanges();
    expect(el().querySelector('.caption')!.textContent).toContain('out of date');
  });

  it('Save emits the updated card', () => {
    const c = create();
    const updated = item(creature, true);
    events.saveErrata.and.returnValue(of(updated));
    const saved = jasmine.createSpy('saved');
    c.saved.subscribe(saved);
    c.power.setValue('2');
    el().querySelector('form')!.dispatchEvent(new Event('submit'));
    expect(events.saveErrata).toHaveBeenCalledWith('c1', { description: 'Flying', power: '2', toughness: '3' }, '9999');
    expect(saved).toHaveBeenCalledWith(updated);
  });

  it('a card with errata shows the printed text and can be reverted', () => {
    const c = create(item({ ...creature, description: 'Flying, lifelink' }, true));
    expect(el().querySelector('.printed-text')!.textContent).toContain('Flying');
    spyOn(window, 'confirm').and.returnValue(true);
    events.revertErrata.and.returnValue(of(item()));
    const saved = jasmine.createSpy('saved');
    c.saved.subscribe(saved);
    button('Revert to printed').click();
    expect(events.revertErrata).toHaveBeenCalledWith('c1', '9999');
    expect(saved).toHaveBeenCalled();
  });

  it('asks for the host PIN instead of calling the server without one', () => {
    const c = create(item(), '');
    c.description.setValue('Vigilance');
    c.save();
    fixture.detectChanges();
    expect(events.saveErrata).not.toHaveBeenCalled();
    expect(el().querySelector('.alert')!.textContent).toContain('host PIN');
  });
});
