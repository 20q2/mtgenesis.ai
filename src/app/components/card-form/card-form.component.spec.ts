import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ReactiveFormsModule } from '@angular/forms';

import { CardFormComponent } from './card-form.component';

describe('CardFormComponent', () => {
  let component: CardFormComponent;
  let fixture: ComponentFixture<CardFormComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ReactiveFormsModule],
      declarations: [CardFormComponent],
      schemas: [NO_ERRORS_SCHEMA]
    });
    fixture = TestBed.createComponent(CardFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('offers every type and rarity, and P/T only for creatures and Vehicles', () => {
    expect(fixture.nativeElement.querySelector('#type')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('#supertype')).not.toBeNull();
    expect(fixture.nativeElement.querySelectorAll('.rarity-option').length).toBe(4);
    component.cardForm.patchValue({ type: 'Creature' });
    expect(component.showPowerToughness).toBeTrue();
    component.cardForm.patchValue({ type: 'Instant' });
    expect(component.showPowerToughness).toBeFalse();
  });

  it('never emits a commander kind', () => {
    const sent: any[] = [];
    component.cardChange.subscribe(card => sent.push(card));
    component.generateCard.subscribe(card => sent.push(card));
    component.cardForm.patchValue({ name: 'Bolt', type: 'Instant' });
    component.onSubmit();
    expect(sent.length).toBeGreaterThan(1);
    expect(sent.every(card => !('commanderKind' in card))).toBeTrue();
  });

  describe('filling the blanks', () => {
    const generated = { name: 'Stormcaller', manaCost: '{2}{U}', type: 'Creature', supertype: 'Legendary',
                        subtype: 'Bird Wizard', colors: ['U'], cmc: 3, rarity: 'rare', power: '2', toughness: '3' };

    it('fills only the fields the player left empty, never the supertype', () => {
      component.cardForm.patchValue({ name: 'My Name', manaCost: '', type: '', subtype: '', supertype: '' });
      component.fillBlanks(generated);
      const v = component.cardForm.value;
      expect(v.name).toBe('My Name');
      expect(v.manaCost).toBe('{2}{U}');
      expect(v.type).toBe('Creature');
      expect(v.subtype).toBe('Bird Wizard');
      expect(v.supertype).toBe('');
      expect(v.powerToughness).toBe('2/3');
      expect(v.colors).toEqual(['U']);
    });

    it('fills silently, so the finished card on the page is not replaced by the form', () => {
      const emitted: unknown[] = [];
      component.cardChange.subscribe(c => emitted.push(c));
      component.cardForm.patchValue({ name: '', manaCost: '', type: '', subtype: '' }, { emitEvent: false });
      component.fillBlanks(generated);
      expect(emitted).toEqual([]);
      expect(component.cardForm.value.cmc).toBe(3);
    });

    it('leaves a typed body alone', () => {
      component.cardForm.patchValue({ type: 'Creature', powerToughness: '4/4', power: '4', toughness: '4' });
      component.fillBlanks(generated);
      expect(component.cardForm.value.powerToughness).toBe('4/4');
    });

    it('tells the player blank fields are filled in by the AI', () => {
      expect(fixture.nativeElement.querySelector('.blank-hint').textContent)
        .toContain('Leave a field blank and the AI fills it in');
    });
  });
});
