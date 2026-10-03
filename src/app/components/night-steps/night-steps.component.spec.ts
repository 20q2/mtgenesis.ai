import { ComponentFixture, TestBed } from '@angular/core/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { NightStepsComponent } from './night-steps.component';

describe('NightStepsComponent', () => {
  let fixture: ComponentFixture<NightStepsComponent>;

  function setup(current: 'build' | 'lock' | 'vote' | 'play') {
    TestBed.configureTestingModule({ imports: [RouterTestingModule], declarations: [NightStepsComponent] });
    fixture = TestBed.createComponent(NightStepsComponent);
    fixture.componentRef.setInput('current', current);
    fixture.detectChanges();
  }

  const steps = () => Array.from<HTMLElement>(fixture.nativeElement.querySelectorAll('.night-step'));

  it('shows the four steps of the night', () => {
    setup('build');
    expect(steps().map(s => s.querySelector('.step-title')!.textContent!.trim()))
      .toEqual(['Build', 'Lock in', 'Vote', 'Play']);
  });

  it('marks the current step, and the ones before it as done', () => {
    setup('vote');
    expect(steps().map(s => s.getAttribute('data-state'))).toEqual(['done', 'done', 'current', 'next']);
    expect(steps()[2].getAttribute('aria-current')).toBe('step');
  });
});
