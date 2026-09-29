import { Component, OnDestroy, OnInit } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import {
  EMPTY, Observable, Subject, Subscription, catchError, finalize, of, startWith, switchMap
} from 'rxjs';
import { EventView, SetCardView, SetView } from '../../models/api.model';
import { apiErrorMessage } from '../../services/api.util';
import { EventService } from '../../services/event.service';
import { PageVisibilityService } from '../../services/page-visibility.service';

export const VOTE_POLL_MS = 10000;

/**
 * /vote: every locked set in the current event, one vote per set, live counts (spec §7).
 * Polls every 10s while the page is visible (paused while hidden, refreshed at once
 * when shown again). If the event being shown gets closed, it keeps showing that event
 * (now with the winners banner) instead of dropping to "No event open".
 */
@Component({
  selector: 'app-vote-page',
  templateUrl: './vote-page.component.html',
  styleUrls: ['./vote-page.component.scss']
})
export class VotePageComponent implements OnInit, OnDestroy {
  /** undefined while the first load is pending; null when no event is open. */
  event: EventView | null | undefined = undefined;
  error: string | null = null;
  loadError: string | null = null;
  /** Sets with a vote request in flight (guards double clicks). */
  readonly voting = new Set<string>();

  private readonly refresh$ = new Subject<void>();
  private lastEventId: string | null = null;
  private sub?: Subscription;

  constructor(private events: EventService, private visibility: PageVisibilityService) {}

  ngOnInit(): void {
    // refresh() restarts the poll timer with an immediate fetch.
    this.sub = this.refresh$.pipe(
      startWith(undefined),
      switchMap(() => this.visibility.poll(VOTE_POLL_MS)),
      switchMap(() => this.load())
    ).subscribe(event => {
      this.event = event;
      this.loadError = null;
      if (event) {
        this.lastEventId = event.id;
      }
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  get isClosed(): boolean {
    return this.event?.status === 'closed';
  }

  refresh(): void {
    this.refresh$.next();
  }

  vote(set: SetView, card: SetCardView): void {
    if (!this.event || this.isClosed || this.voting.has(set.id)) {
      return;
    }
    this.error = null;
    this.voting.add(set.id);
    this.events.vote(set.id, card.id).pipe(
      finalize(() => this.voting.delete(set.id))
    ).subscribe({
      next: updated => {
        this.replaceSet(updated);
        this.refresh();
      },
      error: (err: unknown) => {
        this.error = err instanceof HttpErrorResponse && err.status === 409
          ? apiErrorMessage(err, 'Voting is closed')
          : apiErrorMessage(err, 'Your vote did not go through. Please try again.');
        this.refresh();
      }
    });
  }

  clearError(): void {
    this.error = null;
  }

  trackSet(_index: number, set: SetView): string {
    return set.id;
  }

  private load(): Observable<EventView | null> {
    return this.events.current().pipe(
      switchMap(current => {
        if (current || !this.lastEventId) {
          return of(current);
        }
        // The event we were showing is no longer current: it was closed. Show its results.
        return this.events.get(this.lastEventId).pipe(catchError(() => of(null)));
      }),
      catchError(err => {
        this.loadError = apiErrorMessage(err, 'Could not load the event.');
        return EMPTY;
      })
    );
  }

  private replaceSet(updated: SetView): void {
    if (!this.event) {
      return;
    }
    this.event = {
      ...this.event,
      sets: this.event.sets.map(s => (s.id === updated.id ? updated : s))
    };
  }
}
