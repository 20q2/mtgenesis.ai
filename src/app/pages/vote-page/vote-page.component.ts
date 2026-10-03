import { Component, OnDestroy, OnInit } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import {
  EMPTY, Observable, Subject, Subscription, catchError, finalize, of, startWith, switchMap
} from 'rxjs';
import { EventView, SetCardView, SetView } from '../../models/api.model';
import { LegalCommander, VoteProgress, legalCommanders, voteProgress } from '../../services/commander-night';
import { CmcGroup } from '../../services/commander-rules';
import { apiErrorMessage } from '../../services/api.util';
import { EventService } from '../../services/event.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { UserService } from '../../services/user.service';
import { versionLabel } from '../../services/winners';

export const VOTE_POLL_MS = 10000;

/**
 * /vote: every commander locked in the current event, one vote per commander, live counts
 * (spec §7). The player's own commanders come first: the rules ask owners to vote first, and
 * their own vote counts ×2. After the vote it lists the player's legal commanders.
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

  constructor(private events: EventService, private visibility: PageVisibilityService,
              private users: UserService) {}

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

  /** My commanders (vote on these first), everyone else's, and how many I've voted on. */
  get progress(): VoteProgress {
    return voteProgress(this.event?.sets ?? [], this.users.currentUser()?.id ?? null);
  }

  /** After the vote: my commanders and the versions that won them. */
  get legal(): LegalCommander[] {
    return legalCommanders(this.event?.sets ?? [], this.users.currentUser()?.id ?? null);
  }

  /** Rule 7: owners vote first. */
  get mineNudge(): string {
    const { mine, mineUnvoted } = this.progress;
    if (mineUnvoted < mine.length) {
      return `${mineUnvoted} of yours still to vote on`;
    }
    return mine.length === 1 ? 'Vote on yours first' : `Vote on your ${mine.length} first`;
  }

  legalLabel(l: LegalCommander): string {
    switch (l.kind) {
      case 'winner': return `${l.set.cmc} CMC · ${versionLabel(l.cards)} is legal`;
      case 'tie': return `${l.set.cmc} CMC · ${versionLabel(l.cards)} tied: ask the host to pick`;
      default: return `${l.set.cmc} CMC · no votes`;
    }
  }

  trackLegal(_index: number, l: LegalCommander): string {
    return l.set.id;
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

  trackGroup(_index: number, group: CmcGroup): string {
    return group.label;
  }

  private load(): Observable<EventView | null> {
    return this.events.current().pipe(
      switchMap(current => {
        if (current) {
          return of(current);
        }
        if (this.lastEventId) {
          // The event we were showing is no longer current: it was closed. Show its results.
          return this.events.get(this.lastEventId).pipe(catchError(() => of(null)));
        }
        // Nothing open (e.g. a reload after the vote): show the latest event's results, so
        // every player can still see their legal commanders.
        return this.events.list().pipe(
          switchMap(list => (list[0]?.status === 'closed' ? this.events.get(list[0].id) : of(null))),
          catchError(() => of(null)));
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
