import { Component, OnDestroy, OnInit } from '@angular/core';
import { EMPTY, Subscription, catchError, switchMap } from 'rxjs';
import { EventView, SetView } from '../../models/api.model';
import { Rarity, RarityOptions } from '../../models/card.model';
import { CommanderState, rarityLabel } from '../../components/commander-panel/commander-panel.component';
import { apiErrorMessage } from '../../services/api.util';
import { COMMANDER_CMCS, COMMANDER_RARITIES } from '../../services/commander-rules';
import { EventService } from '../../services/event.service';
import { PageVisibilityService } from '../../services/page-visibility.service';

export const EVENT_POLL_MS = 30000;
const TAB_KEY = 'mtgenesis.setBuilder.tab';

/**
 * /set: a player's three commanders, one per CMC, each with 3 versions to reroll and lock in
 * (docs/superpowers/specs/2026-10-03-commander-rules-design.md §5). State comes from
 * GET /me/sets/current, so a refresh loses nothing; each panel owns its commander after load.
 */
@Component({
  selector: 'app-set-builder-page',
  templateUrl: './set-builder-page.component.html',
  styleUrls: ['./set-builder-page.component.scss']
})
export class SetBuilderPageComponent implements OnInit, OnDestroy {
  readonly cmcs = COMMANDER_CMCS;
  readonly rarities = RarityOptions.filter(r => COMMANDER_RARITIES.includes(r.value as Rarity));

  /** Each commander as loaded (null: not started), handed to its panel once. */
  sets: Record<number, SetView | null> = { 3: null, 4: null, 5: null };
  /** Each commander's current status and rarity, kept up to date by its panel. */
  states: Record<number, CommanderState | null> = { 3: null, 4: null, 5: null };
  /** Per CMC, the rarities its two siblings use (kept as objects so inputs change only on purpose). */
  taken: Record<number, Partial<Record<Rarity, number>>> = { 3: {}, 4: {}, 5: {} };
  /** The current event (null when none is open). */
  event: EventView | null = null;
  activeCmc = COMMANDER_CMCS[0];

  loading = true;
  error: string | null = null;

  private eventSub?: Subscription;
  private loadSub?: Subscription;

  constructor(private events: EventService, private visibility: PageVisibilityService) {}

  ngOnInit(): void {
    this.activeCmc = readTab() ?? this.activeCmc;
    this.loadSub = this.events.mySets().subscribe({
      next: sets => {
        for (const set of sets) {
          if (set.cmc !== null && this.cmcs.includes(set.cmc)) {
            this.sets[set.cmc] = set;
            this.states[set.cmc] = stateOf(set);
          }
        }
        this.updateTaken();
        this.loading = false;
      },
      error: err => {
        this.error = apiErrorMessage(err, 'Could not load your commanders.');
        this.loading = false;
      }
    });

    // Keep the event fresh so Lock in enables soon after the host opens one
    // (paused while the page is hidden, checked at once when it is shown).
    this.eventSub = this.visibility.poll(EVENT_POLL_MS).pipe(
      switchMap(() => this.events.current().pipe(catchError(() => EMPTY)))
    ).subscribe(event => (this.event = event));
  }

  ngOnDestroy(): void {
    this.loadSub?.unsubscribe();
    this.eventSub?.unsubscribe();
  }

  /** Rarities the other two commanders are locked in with, with their CMC. Drafts may share a
   *  rarity while the player reassigns them; Lock in enforces one of each. */
  takenRaritiesFor(cmc: number): Partial<Record<Rarity, number>> {
    const taken: Partial<Record<Rarity, number>> = {};
    for (const other of this.cmcs) {
      const state = this.states[other];
      if (other !== cmc && state?.status === 'locked') {
        taken[state.rarity] = other;
      }
    }
    return taken;
  }

  get lockedCount(): number {
    return this.cmcs.filter(cmc => this.states[cmc]?.status === 'locked').length;
  }

  /** The commanders (by CMC) using this rarity: drafts may share one until one is locked. */
  rarityHolders(rarity: string): { cmc: number; locked: boolean }[] {
    return this.cmcs
      .filter(cmc => this.states[cmc]?.rarity === rarity)
      .map(cmc => ({ cmc, locked: this.states[cmc]!.status === 'locked' }));
  }

  /** 'Rare · Locked', 'Uncommon · Draft' or 'Not started'. */
  summary(cmc: number): string {
    const state = this.states[cmc];
    if (!state) {
      return 'Not started';
    }
    return `${rarityLabel(state.rarity)} · ${state.status === 'locked' ? 'Locked' : 'Draft'}`;
  }

  onStateChange(cmc: number, state: CommanderState | null): void {
    this.states = { ...this.states, [cmc]: state };
    this.updateTaken();
  }

  private updateTaken(): void {
    this.taken = Object.fromEntries(this.cmcs.map(cmc => [cmc, this.takenRaritiesFor(cmc)]));
  }

  selectCmc(cmc: number): void {
    this.activeCmc = cmc;
    try {
      localStorage.setItem(TAB_KEY, String(cmc));
    } catch {
      // storage unavailable: the tab just isn't remembered
    }
  }

  /** Arrow keys, Home and End move between the commander tabs (the ARIA tabs pattern). */
  onTabKey(event: KeyboardEvent): void {
    const index = this.cmcs.indexOf(this.activeCmc);
    const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: this.cmcs.length - 1 }[event.key];
    if (next === undefined) {
      return;
    }
    event.preventDefault();
    const cmc = this.cmcs[(next + this.cmcs.length) % this.cmcs.length];
    this.selectCmc(cmc);
    (document.getElementById(`commander-tab-${cmc}`) as HTMLElement | null)?.focus();
  }

  clearError(): void {
    this.error = null;
  }
}

function stateOf(set: SetView): CommanderState | null {
  return (set.status === 'draft' || set.status === 'locked') && set.rarity
    ? { status: set.status, rarity: set.rarity as Rarity, name: set.commanderName } : null;
}

function readTab(): number | null {
  try {
    const cmc = Number(localStorage.getItem(TAB_KEY));
    return COMMANDER_CMCS.includes(cmc) ? cmc : null;
  } catch {
    return null;
  }
}
