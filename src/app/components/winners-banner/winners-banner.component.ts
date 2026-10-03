import { Component, Input } from '@angular/core';
import { SetView } from '../../models/api.model';
import { CmcGroup } from '../../services/commander-rules';
import { SetOutcome, setOutcome, versionLabel } from '../../services/winners';

/** After close: each commander's winning version, or its tied versions (host decides at the table),
 *  grouped by CMC. */
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
      case 'tie': return `Tied: ${versionLabel(outcome.cards)}. Ask the host to pick`;
      default: return 'No votes';
    }
  }

  trackSet(_index: number, set: SetView): string {
    return set.id;
  }

  trackGroup(_index: number, group: CmcGroup): string {
    return group.label;
  }
}
