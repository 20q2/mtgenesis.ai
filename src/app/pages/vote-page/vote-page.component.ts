import { Component, OnDestroy, OnInit } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import {
  EMPTY, Observable, Subject, Subscription, catchError, finalize, of, startWith, switchMap
} from 'rxjs';
import { EventView, SetCardView, SetView } from '../../models/api.model';
import {
  BallotPlayer, LegalCommander, VoteProgress, ballot, legalCommanders, nextToVote, voteProgress
} from '../../services/commander-night';
import { apiErrorMessage } from '../../services/api.util';
import { EventService } from '../../services/event.service';
import { GenerationService } from '../../services/generation.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { UserService } from '../../services/user.service';
import { versionLabel } from '../../services/winners';

export const VOTE_POLL_MS = 10000;

export function rerollConfirm(set: SetView, card: SetCardView): string {
  return `Reroll version ${card.slot} of ${set.commanderName}? It's made again from scratch, `
    + 'and the votes on it are cleared.';
}

/**
 * /vote: every commander locked in the current event, one vote per commander, live counts
 * (spec §7). The overview is a grid of players, each with small fans of their commanders
 * marked voted or not; a player opens in a sheet with all their commanders' versions to vote
 * on. The player's own commanders come first: the rules ask owners to vote first, and their
 * own vote counts ×2. After the vote it lists the player's legal commanders.
 * Players can ask an owner to reroll a version that doesn't work; once enough ask, the owner
 * rerolls it from here (docs/superpowers/specs/2026-10-14-reroll-requests-design.md).
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
  /** Versions with a reroll request or a reroll in flight. */
  readonly cardBusy = new Set<string>();
  /** The vote by player, mine first. */
  players: BallotPlayer[] = [];
  /** The player open in the sheet, and the commander to bring into view there. */
  sheetUserId: string | null = null;
  sheetFocusSetId: string | null = null;

  private readonly refresh$ = new Subject<void>();
  private lastEventId: string | null = null;
  private sub?: Subscription;

  constructor(private events: EventService, private visibility: PageVisibilityService,
              private users: UserService, private generation: GenerationService) {}

  ngOnInit(): void {
    // refresh() restarts the poll timer with an immediate fetch.
    this.sub = this.refresh$.pipe(
      startWith(undefined),
      switchMap(() => this.visibility.poll(VOTE_POLL_MS)),
      switchMap(() => this.load())
    ).subscribe(event => {
      this.setEvent(event);
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

  /** How many players must ask before an owner can reroll a version. */
  get rerollsNeeded(): number {
    return this.event?.sets[0]?.rerollRequestsNeeded ?? 3;
  }

  /** While voting: my versions I can reroll now (enough players asked, or it failed). */
  get myRerolls(): { set: SetView; card: SetCardView }[] {
    if (this.isClosed) {
      return [];
    }
    return this.progress.mine.flatMap(set => set.cards.filter(c => c.canReroll).map(card => ({ set, card })));
  }

  get rerollNotice(): string | null {
    const ready = this.myRerolls;
    if (ready.length !== 1) {
      return ready.length ? `${ready.length} of your versions can be rerolled.` : null;
    }
    const { set, card } = ready[0];
    return card.status === 'failed'
      ? `Version ${card.slot} of ${set.commanderName} failed. You can reroll it.`
      : `Players say version ${card.slot} of ${set.commanderName} doesn't work. You can reroll it.`;
  }

  /** Opens my commanders at the first one with a version to reroll. */
  reviewRerolls(): void {
    const first = this.myRerolls[0];
    const me = this.players.find(p => p.mine);
    if (first && me) {
      this.openPlayer(me, first.set);
    }
  }

  /** The first player with something left to vote on (mine come first); null when I'm done. */
  get firstToVote(): BallotPlayer | null {
    return this.players[nextToVote(this.players)] ?? null;
  }

  get sheetIndex(): number {
    return this.players.findIndex(p => p.userId === this.sheetUserId);
  }

  get sheetPlayer(): BallotPlayer | null {
    return this.players[this.sheetIndex] ?? null;
  }

  /** The next player after the open one with something left to vote on, never the open one. */
  get nextPlayer(): BallotPlayer | null {
    const next = this.players[nextToVote(this.players, this.sheetIndex)];
    return next && next.userId !== this.sheetUserId ? next : null;
  }

  get votedPercent(): number {
    const { voted, total } = this.progress;
    return total ? Math.round((voted / total) * 100) : 0;
  }

  /** Opens a player's commanders, at `set` or else (while voting) the first one I haven't voted on. */
  openPlayer(player: BallotPlayer, set: SetView | null = null): void {
    this.sheetUserId = player.userId;
    const target = set ?? (this.isClosed ? null : player.sets.find(s => !s.myVoteCardId));
    this.sheetFocusSetId = target?.id ?? null;
  }

  stepPlayer(delta: number): void {
    const count = this.players.length;
    if (count) {
      this.openPlayer(this.players[(this.sheetIndex + delta + count) % count]);
    }
  }

  voteNext(): void {
    if (this.firstToVote) {
      this.openPlayer(this.firstToVote);
    }
  }

  closeSheet(): void {
    this.sheetUserId = null;
    this.sheetFocusSetId = null;
  }

  legalLabel(l: LegalCommander): string {
    switch (l.kind) {
      case 'winner': return `${l.set.cmc} CMC · ${versionLabel(l.cards)} is legal`;
      case 'tie': return `${l.set.cmc} CMC · ${versionLabel(l.cards)} tied: ask the host to pick`;
      default: return `${l.set.cmc} CMC · no votes`;
    }
  }

  hasErrata(l: LegalCommander): boolean {
    return l.cards.some(c => c.errata);
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

  /** Asks the owner to reroll a version that doesn't work, or takes the request back. */
  requestReroll(card: SetCardView, requested: boolean): void {
    if (!this.event || this.isClosed || this.cardBusy.has(card.id)) {
      return;
    }
    this.error = null;
    this.cardBusy.add(card.id);
    this.events.requestReroll(card.id, requested).pipe(
      finalize(() => this.cardBusy.delete(card.id))
    ).subscribe({
      next: updated => this.replaceSet(updated),
      error: (err: unknown) => {
        this.error = apiErrorMessage(err, 'Your request did not go through. Please try again.');
        this.refresh();
      }
    });
  }

  /** My version, once enough players asked (or it failed): made again, its votes cleared. */
  reroll(set: SetView, card: SetCardView): void {
    if (!this.event || this.isClosed || !card.canReroll || this.cardBusy.has(card.id)
        || !window.confirm(rerollConfirm(set, card))) {
      return;
    }
    this.error = null;
    this.cardBusy.add(card.id);
    this.generation.reroll(card.id).pipe(
      finalize(() => this.cardBusy.delete(card.id))
    ).subscribe({
      next: () => this.refresh(),
      error: (err: unknown) => {
        this.error = apiErrorMessage(err, 'Could not reroll this version. Please try again.');
        this.refresh();
      }
    });
  }

  clearError(): void {
    this.error = null;
  }

  trackPlayer(_index: number, player: BallotPlayer): string {
    return player.userId;
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
    this.setEvent({
      ...this.event,
      sets: this.event.sets.map(s => (s.id === updated.id ? updated : s))
    });
  }

  private setEvent(event: EventView | null): void {
    this.event = event;
    this.players = ballot(event?.sets ?? [], this.users.currentUser()?.id ?? null);
    if (this.sheetUserId && !this.sheetPlayer) {
      this.closeSheet();
    }
  }
}
