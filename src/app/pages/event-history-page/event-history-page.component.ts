import { Component, OnDestroy, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { EMPTY, Subscription, catchError, switchMap, tap } from 'rxjs';
import { EventSummary, EventView, SetView } from '../../models/api.model';
import { CmcGroup } from '../../services/commander-rules';
import { apiErrorMessage } from '../../services/api.util';
import { EventService } from '../../services/event.service';

/** /events (history list) and /events/:id (read-only results). */
@Component({
  selector: 'app-event-history-page',
  templateUrl: './event-history-page.component.html',
  styleUrls: ['./event-history-page.component.scss']
})
export class EventHistoryPageComponent implements OnInit, OnDestroy {
  mode: 'list' | 'detail' = 'list';
  events: EventSummary[] | null = null;
  event: EventView | null = null;
  loading = true;
  error: string | null = null;

  private sub?: Subscription;

  constructor(private route: ActivatedRoute, private eventService: EventService) {}

  ngOnInit(): void {
    this.sub = this.route.paramMap.pipe(
      tap(() => {
        this.loading = true;
        this.error = null;
      }),
      switchMap(params => {
        const id = params.get('id');
        if (id) {
          this.mode = 'detail';
          return this.eventService.get(id).pipe(
            tap(event => (this.event = event)),
            catchError(err => this.fail(err, 'Could not load that event.'))
          );
        }
        this.mode = 'list';
        return this.eventService.list().pipe(
          tap(list => (this.events = list)),
          catchError(err => this.fail(err, 'Could not load past events.'))
        );
      })
    ).subscribe(() => (this.loading = false));
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  trackSet(_index: number, set: SetView): string {
    return set.id;
  }

  trackGroup(_index: number, group: CmcGroup): string {
    return group.label;
  }

  private fail(err: unknown, fallback: string) {
    this.error = apiErrorMessage(err, fallback);
    this.loading = false;
    return EMPTY;
  }
}
