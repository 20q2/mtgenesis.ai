import { Component, Input } from '@angular/core';
import { NIGHT_STEPS, NightStepId } from '../../services/commander-night';

/** The four steps of an AI Night (build, lock in, vote, play), with where the player is now. */
@Component({
  selector: 'app-night-steps',
  templateUrl: './night-steps.component.html',
  styleUrls: ['./night-steps.component.scss']
})
export class NightStepsComponent {
  @Input() current: NightStepId = 'build';

  readonly steps = NIGHT_STEPS;

  stateOf(index: number): 'done' | 'current' | 'next' {
    const at = this.steps.findIndex(s => s.id === this.current);
    return index < at ? 'done' : index === at ? 'current' : 'next';
  }
}
