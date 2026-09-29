import { Component, Input } from '@angular/core';
import { SetView } from '../../models/api.model';
import { SetOutcome, setOutcome, versionLabel } from '../../services/winners';

/** After close: each set's winning version, or its tied versions (host decides at the table). */
@Component({
  selector: 'app-winners-banner',
  templateUrl: './winners-banner.component.html',
  styleUrls: ['./winners-banner.component.scss']
})
export class WinnersBannerComponent {
  @Input() sets: SetView[] = [];
  @Input() title = 'Winners';

  outcome(set: SetView): SetOutcome {
    return setOutcome(set);
  }

  label(outcome: SetOutcome): string {
    switch (outcome.kind) {
      case 'winner': return versionLabel(outcome.cards);
      case 'tie': return `Tied: ${versionLabel(outcome.cards)} — host decides`;
      default: return 'No votes';
    }
  }

  trackSet(_index: number, set: SetView): string {
    return set.id;
  }
}
