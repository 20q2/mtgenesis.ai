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

  describe('commanderMode', () => {
    beforeEach(() => {
      fixture = TestBed.createComponent(CardFormComponent);
      component = fixture.componentInstance;
      component.commanderMode = true;
      fixture.detectChanges();
    });

    it('fixes the type line to Legendary Creature and hides those fields', () => {
      expect(component.cardForm.value.type).toBe('Creature');
      expect(component.cardForm.value.supertype).toBe('Legendary');
      expect(fixture.nativeElement.querySelector('#type')).toBeNull();
      expect(fixture.nativeElement.querySelector('#supertype')).toBeNull();
      expect(component.filteredSubtypes.length).toBeGreaterThan(0);
    });

    it('keeps Legendary Creature after Clear form', () => {
      component.cardForm.patchValue({ commanderKind: 'vehicle' });
      component.clearForm();
      expect(component.cardForm.value.type).toBe('Creature');
      expect(component.cardForm.value.supertype).toBe('Legendary');
      expect(component.cardForm.value.commanderKind).toBe('creature');
    });

    it('offers Uncommon, Rare and Mythic only', () => {
      const labels = Array.from<HTMLElement>(fixture.nativeElement.querySelectorAll('.rarity-option'))
        .map(el => el.textContent!.trim());
      expect(labels.length).toBe(3);
      expect(labels[0]).toContain('Uncommon');
      expect(labels[1]).toContain('Rare');
      expect(labels[2]).toContain('Mythic');
      expect(component.cardForm.value.rarity).toBe('uncommon');
    });

    it('disables a rarity taken by another commander and says where', () => {
      component.takenRarities = { rare: 4 };
      component.ngOnChanges();
      fixture.detectChanges();
      const rare = fixture.nativeElement.querySelector('.rarity-rare') as HTMLElement;
      expect((rare.querySelector('input') as HTMLInputElement).disabled).toBeTrue();
      expect(rare.textContent).toContain('used at 4 CMC');
    });

    it('moves off a rarity that becomes taken', () => {
      component.cardForm.patchValue({ rarity: 'rare' });
      component.takenRarities = { rare: 4, uncommon: 3 };
      component.ngOnChanges();
      expect(component.cardForm.value.rarity).toBe('mythic');
    });

    it('the Vehicle toggle makes a Legendary Artifact — Vehicle', () => {
      let last: any;
      component.cardChange.subscribe(card => (last = card));
      component.cardForm.patchValue({ subtype: 'Construct' });
      component.setCommanderKind('vehicle');
      expect(last.type).toBe('Artifact');
      expect(last.supertype).toBe('Legendary');
      expect(last.subtype).toBe('Construct Vehicle');
      expect(last.commanderKind).toBe('vehicle');
    });

    it('the Creature toggle puts back Legendary Creature', () => {
      let last: any;
      component.cardChange.subscribe(card => (last = card));
      component.setCommanderKind('vehicle');
      component.setCommanderKind('creature');
      expect(last.type).toBe('Creature');
      expect(last.subtype ?? '').not.toContain('Vehicle');
      expect(last.commanderKind).toBe('creature');
    });

    it('shows the point budget for its CMC', () => {
      fixture = TestBed.createComponent(CardFormComponent);
      component = fixture.componentInstance;
      component.commanderMode = true;
      component.commanderCmc = 3;
      fixture.detectChanges();
      const hint = () => (fixture.nativeElement.querySelector('.pt-hint') as HTMLElement).textContent;
      expect(component.statsHint).toBe('4 points · e.g. 2/2, 3/1, 1/3 · leave blank for auto');
      expect(hint()).toContain('4 points');
      component.setCommanderKind('vehicle');
      fixture.detectChanges();
      expect(component.statsHint).toBe('6 points (Vehicle +2) · leave blank for auto');
    });

    it('flags P/T over the budget', () => {
      component.commanderCmc = 3;
      component.cardForm.patchValue({ powerToughness: '3/2' });
      fixture.detectChanges();
      expect(component.statsError).toBe('A 3-mana commander has 4 points; 3/2 uses 5');
      expect(fixture.nativeElement.querySelector('.pt-error').textContent)
        .toContain('A 3-mana commander has 4 points');
    });

    it('emits typed P/T, and empty P/T with no error when blank', () => {
      let last: any;
      component.cardChange.subscribe(card => (last = card));
      component.cardForm.patchValue({ powerToughness: '1/3' });
      expect([last.power, last.toughness]).toEqual(['1', '3']);
      component.cardForm.patchValue({ powerToughness: '' });
      expect([last.power, last.toughness]).toEqual(['', '']);
      expect(component.statsError).toBeNull();
    });

    it('sizes the art prompt by its CMC', () => {
      component.cardForm.patchValue({ name: 'Zur' });
      component.commanderCmc = 3;
      expect(component.generateArtPromptText()).toContain('medium scale');
      component.commanderCmc = 5;
      expect(component.generateArtPromptText()).toContain('large and imposing');
    });
  });

  it('normal mode never sends a commander kind (Generate or Regenerate text)', () => {
    const sent: any[] = [];
    component.generateCard.subscribe(card => sent.push(card));
    component.regenerateText.subscribe(card => sent.push(card));
    component.cardForm.patchValue({ name: 'Bolt', type: 'Instant' });
    component.onSubmit();
    component.isGenerating = false;
    component.onRegenerateText();
    expect(sent.length).toBe(2);
    expect(sent.every(card => !('commanderKind' in card))).toBeTrue();
  });

  it('normal mode keeps all four rarities and its P/T rules', () => {
    const labels = fixture.nativeElement.querySelectorAll('.rarity-option');
    expect(labels.length).toBe(4);
    component.cardForm.patchValue({ type: 'Creature' });
    component.updatePowerToughnessVisibility();
    expect(component.showPowerToughness).toBeTrue();
    component.cardForm.patchValue({ type: 'Instant' });
    component.updatePowerToughnessVisibility();
    expect(component.showPowerToughness).toBeFalse();
  });
});
