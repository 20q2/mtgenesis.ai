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
});
