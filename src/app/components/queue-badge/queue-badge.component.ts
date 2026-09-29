import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { QueueStatus } from '../../models/api.model';
import { QueueService } from '../../services/queue.service';

/** "Server idle" / "Server busy · N cards ahead · ~M min" / "Server offline" (spec §7). */
@Component({
  selector: 'app-queue-badge',
  templateUrl: './queue-badge.component.html',
  styleUrls: ['./queue-badge.component.scss']
})
export class QueueBadgeComponent implements OnInit, OnDestroy {
  /** undefined until the first answer, null when offline. */
  status: QueueStatus | null | undefined = undefined;
  private sub?: Subscription;

  constructor(private queue: QueueService) {}

  ngOnInit(): void {
    this.sub = this.queue.status$.subscribe(status => (this.status = status));
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  text(status: QueueStatus | null | undefined): string {
    if (status === undefined) {
      return 'Checking server…';
    }
    if (status === null) {
      return 'Server offline';
    }
    if (!status.busy) {
      return 'Server idle';
    }
    return `Server busy · ${status.cardsAhead} cards ahead · ~${Math.ceil(status.etaSeconds / 60)} min`;
  }

  stateClass(status: QueueStatus | null | undefined): string {
    if (status === undefined) {
      return 'checking';
    }
    if (status === null) {
      return 'offline';
    }
    return status.busy ? 'busy' : 'idle';
  }
}
