import { Component, EventEmitter, Input, Output } from '@angular/core';
import { finalize } from 'rxjs';
import { CardView, PoolView } from '../../models/api.model';
import { apiErrorMessage } from '../../services/api.util';
import { PoolService, poolColorOk } from '../../services/pool.service';

/**
 * "Submit to pool (k/cap)" for one finished card, on the create screen and gallery tiles.
 * Disabled with a visible reason when the card can't go in; the server has the final say.
 * The host page owns `pool` (PoolService.current(), loaded once) and updates it from
 * `submitted`, along with the card's poolEntryId.
 */
@Component({
  selector: 'app-pool-submit',
  templateUrl: './pool-submit.component.html',
  styleUrls: ['./pool-submit.component.scss']
})
export class PoolSubmitComponent {
  @Input() card!: CardView;
  /** The open pool, or null when none is open. */
  @Input() pool: PoolView | null = null;
  @Output() submitted = new EventEmitter<PoolView>();

  sending = false;
  error: string | null = null;

  constructor(private pools: PoolService) {}

  get entered(): boolean {
    return !!this.card.poolEntryId;
  }

  /** Why the card can't be submitted right now, or null when it can. */
  disabledReason(): string | null {
    if (!this.pool || this.pool.status !== 'open') {
      return 'No pool is open';
    }
    if (!poolColorOk(this.card.card)) {
      return "Multicolor cards can't enter the pool";
    }
    if (this.pool.myEntryCount >= this.pool.maxEntriesPerUser) {
      return `You've used all ${this.pool.maxEntriesPerUser} entries`;
    }
    return null;
  }

  label(): string {
    if (this.entered) {
      return 'In the pool ✓';
    }
    return this.pool
      ? `Submit to pool (${this.pool.myEntryCount}/${this.pool.maxEntriesPerUser})`
      : 'Submit to pool';
  }

  submit(): void {
    if (this.sending || this.entered || this.disabledReason()) {
      return;
    }
    this.error = null;
    this.sending = true;
    this.pools.submit(this.card.id).pipe(finalize(() => (this.sending = false))).subscribe({
      next: pool => this.submitted.emit(pool),
      error: err => (this.error = apiErrorMessage(err, 'Could not submit the card.'))
    });
  }
}
