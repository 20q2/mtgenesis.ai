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

    it('hides power/toughness and keeps it out of the emitted card', () => {
      let last: any;
      component.cardChange.subscribe(card => (last = card));
      component.cardForm.patchValue({ power: '9', toughness: '9' });
      expect(component.showPowerToughness).toBeFalse();
      expect(fixture.nativeElement.querySelector('#powerToughness')).toBeNull();
      expect(last.power).toBeUndefined();
    });

    it('keeps Legendary Creature after Clear form', () => {
      component.clearForm();
      expect(component.cardForm.value.type).toBe('Creature');
      expect(component.cardForm.value.supertype).toBe('Legendary');
    });
  });
});
