import { Component, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges } from '@angular/core';
import { FormControl } from '@angular/forms';
import { Subscription, finalize } from 'rxjs';
import { CardView, EventView, GeneratedCardData, SetView } from '../../models/api.model';
import { Card, Rarity, RarityOptions } from '../../models/card.model';
import { apiErrorMessage } from '../../services/api.util';
import { isFinished, isPending } from '../../services/card-status';
import { MAX_PIP_VALUE, commanderPipValue, commanderStatsError } from '../../services/commander-rules';
import { EventService } from '../../services/event.service';
import { GenerationService } from '../../services/generation.service';
import { ManaService } from '../../services/mana.service';

export const COMMANDER_NAME_MAX = 40;
export const UNLOCK_CONFIRM = 'This clears votes on this commander';
/** Rule 6: a reroll is only for a version that doesn't work at all. */
export const REROLL_CONFIRM =
  'Rerolls are only for a version that doesn\'t function under any circumstance. Reroll it?';

/** What the page needs to know about a commander: its status and rarity, or null for none. */
export interface CommanderState { status: 'draft' | 'locked'; rarity: Rarity; name: string; }

/**
 * One of a player's three commanders (docs/superpowers/specs/2026-10-03-commander-rules-design.md
 * §5): name it, design it at `cmc` mana, generate its 3 versions, reroll a broken one, lock it in.
 *
 * Once a commander exists the name is read-only and Lock in sends the stored name, so the name on
 * /vote always matches the name rendered on the three cards. "Change name" frees the field for
 * the next Generate (which starts this commander over).
 */
@Component({
  selector: 'app-commander-panel',
  templateUrl: './commander-panel.component.html',
  styleUrls: ['./commander-panel.component.scss']
})
export class CommanderPanelComponent implements OnChanges, OnDestroy {
  @Input() cmc = 3;
  /** The commander as loaded with the page (null: not started). The panel owns it from then on. */
  @Input() set: SetView | null = null;
  @Input() event: EventView | null = null;
  /** Rarities the player's other commanders use, with their CMC. */
  @Input() takenRarities: Partial<Record<Rarity, number>> = {};
  /** After generate, lock and unlock: this commander's status and rarity for the page. */
  @Output() stateChange = new EventEmitter<CommanderState | null>();

  readonly maxNameLength = COMMANDER_NAME_MAX;
  readonly rerollLabel = 'Reroll (only if broken)';
  readonly commanderName = new FormControl('', { nonNullable: true });

  /** Latest values from the card form. */
  formCard: Card | null = null;

  setId: string | null = null;
  setStatus: SetView['status'] | null = null;
  /** The commander's name as stored on the server (what its cards show). */
  setName: string | null = null;
  setRarity: Rarity | null = null;
  /** True after "Change name": the field is editable for the next Generate. */
  renaming = false;
  /** The saved commander's first version, to fill the designer in with on load. */
  loadedCard: GeneratedCardData | null = null;
  /** Current card per slot (index 0..2 = version 1..3). */
  slots: (CardView | null)[] = [null, null, null];

  submitting = false;
  locking = false;
  unlocking = false;
  rerolling = [false, false, false];
  error: string | null = null;

  private watches: (Subscription | undefined)[] = [];

  constructor(private generation: GenerationService, private events: EventService,
              private mana: ManaService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['set']) {
      this.applySet(this.set);
    }
  }

  ngOnDestroy(): void {
    this.stopWatches();
  }

  get isLocked(): boolean {
    return this.setStatus === 'locked';
  }

  /** The name field is read-only while a commander exists, unless "Change name" was pressed. */
  get nameReadOnly(): boolean {
    return !!this.setId && !this.renaming;
  }

  /** "Change name" is offered for a draft (a locked commander must be unlocked first). */
  get canChangeName(): boolean {
    return !!this.setId && !this.isLocked;
  }

  /** Toggles "Change name"; cancelling puts the stored name back. */
  toggleRename(): void {
    if (!this.canChangeName) {
      return;
    }
    this.renaming = !this.renaming;
    if (!this.renaming) {
      this.commanderName.setValue(this.setName ?? '');
    }
  }

  get eventOpen(): boolean {
    return !!this.event && this.event.status === 'open';
  }

  hasPending(): boolean {
    return this.slots.some(s => !!s && isPending(s.status));
  }

  /** True when the chosen pips are worth more than the cheapest commander's mana value. */
  get tooManyPips(): boolean {
    return commanderPipValue(this.formCard?.manaCost ?? '') > MAX_PIP_VALUE;
  }

  /** The CMC of the other commander that already uses the chosen rarity, if any. */
  get rarityTakenAt(): number | undefined {
    return this.formCard ? this.takenRarities[this.formCard.rarity] : undefined;
  }

  get statsError(): string | null {
    const card = this.formCard;
    return card ? commanderStatsError(card.power ?? '', card.toughness ?? '', this.cmc,
                                      card.commanderKind ?? 'creature') : null;
  }

  /** What the versions print (the first version's card): shown read-only once locked in. */
  get lockedCard(): GeneratedCardData | null {
    return this.isLocked ? this.slots[0]?.card ?? null : null;
  }

  typeLineOf(card: GeneratedCardData): string {
    const main = [card.supertype, card.type].filter(Boolean).join(' ');
    return card.subtype ? `${main} — ${card.subtype}` : main;
  }

  rarityName(rarity: string | null | undefined): string {
    return rarity ? rarityLabel(rarity) : '';
  }

  costSymbols(cost: string | null | undefined): string[] {
    return this.mana.extractManaSymbols(cost ?? '');
  }

  /** The mana-font class for a symbol: ms-w, ms-2, ms-wu. */
  symbolClass(symbol: string): string {
    return `ms-${this.mana.getSymbolClass(symbol)}`;
  }

  slotLabel(index: number): string {
    return `Version ${index + 1}`;
  }

  canGenerate(): boolean {
    return !this.submitting && !this.hasPending() && !this.isLocked && !this.tooManyPips
      && this.rarityTakenAt === undefined && !this.statsError;
  }

  generateHint(): string | null {
    if (this.isLocked) {
      return 'This commander is locked in. Unlock it to start over.';
    }
    if (this.tooManyPips) {
      return `Colored pips can add up to at most ${MAX_PIP_VALUE} mana.`;
    }
    if (this.rarityTakenAt !== undefined) {
      return `Your ${this.rarityTakenAt} CMC commander is locked in as ${rarityLabel(this.formCard!.rarity)}.`;
    }
    if (this.statsError) {
      return this.statsError;
    }
    if (this.hasPending()) {
      return 'Wait for all 3 versions to finish before starting over.';
    }
    return null;
  }

  onCardChange(card: Card): void {
    this.formCard = card;
  }

  generate(): void {
    if (!this.canGenerate()) {
      return;
    }
    const name = this.commanderName.value.trim();
    if (!name) {
      this.error = 'Enter a commander name.';
      return;
    }
    if (name.length > COMMANDER_NAME_MAX) {
      this.error = `Commander names can be at most ${COMMANDER_NAME_MAX} characters.`;
      return;
    }
    if (this.setStatus === 'draft' && this.slots.some(s => !!s)
        && !window.confirm('Start this commander over? Its current versions stay in your Gallery.')) {
      return;
    }

    const card: Card = this.formCard
      ?? { name, manaCost: '', type: 'Creature', colors: [], cmc: this.cmc, rarity: Rarity.UNCOMMON };
    const prompt = this.generation.promptFor({ ...card, name });

    this.error = null;
    this.submitting = true;
    this.generation.submit({
      prompt,
      cardData: { ...this.generation.cardParams(card), name },
      count: 3,
      commanderName: name,
      cmc: this.cmc
    }).pipe(
      finalize(() => (this.submitting = false))
    ).subscribe({
      next: response => {
        this.stopWatches();
        this.setId = response.setId;
        this.setStatus = 'draft';
        this.setName = name;
        this.setRarity = card.rarity;
        this.renaming = false;
        this.commanderName.setValue(name);
        this.slots = [null, null, null];
        for (const view of response.cards) {
          this.placeCard(view);
        }
        this.emitState();
      },
      error: err => (this.error = apiErrorMessage(err, 'Could not start this commander.'))
    });
  }

  canRerollSlot(view: CardView | null): boolean {
    return !!view
      && this.setStatus === 'draft'
      && isFinished(view.status)
      && !this.rerolling[this.slotIndex(view)];
  }

  onReroll(view: CardView): void {
    if (!this.canRerollSlot(view) || !window.confirm(REROLL_CONFIRM)) {
      return;
    }
    const index = this.slotIndex(view);
    this.error = null;
    this.rerolling[index] = true;
    this.generation.reroll(view.id).pipe(
      finalize(() => (this.rerolling[index] = false))
    ).subscribe({
      next: fresh => this.placeCard(fresh, index),
      error: err => (this.error = apiErrorMessage(err, 'Could not reroll that version.'))
    });
  }

  /** Why Lock in is disabled, or null when it can be pressed. */
  lockDisabledReason(): string | null {
    if (!this.setId) {
      return 'Generate this commander first';
    }
    if (!this.eventOpen) {
      return 'No event open — ask the host';
    }
    if (this.slots.some(s => !s || s.status !== 'done')) {
      return 'Waiting for all 3 versions';
    }
    const clash = this.setRarity ? this.takenRarities[this.setRarity] : undefined;
    if (clash !== undefined) {
      return `Your ${clash} CMC commander is locked in as ${rarityLabel(this.setRarity!)} — `
        + 'generate this one with another rarity';
    }
    if (!this.setName) {
      return 'Enter a commander name';
    }
    if (this.renaming && this.commanderName.value.trim() !== this.setName) {
      return 'Generate again to use the new name';
    }
    return null;
  }

  lock(): void {
    if (this.locking || this.isLocked || this.lockDisabledReason() || !this.setId || !this.setName) {
      return;
    }
    this.error = null;
    this.locking = true;
    // The stored name, never the field: the cards were rendered with it.
    this.events.lock(this.setId, this.setName).pipe(
      finalize(() => (this.locking = false))
    ).subscribe({
      next: set => {
        this.applySet(set);
        this.emitState();
      },
      error: err => (this.error = apiErrorMessage(err, 'Could not lock in this commander.'))
    });
  }

  unlock(): void {
    if (this.unlocking || !this.isLocked || !this.setId) {
      return;
    }
    if (!window.confirm(UNLOCK_CONFIRM)) {
      return;
    }
    this.error = null;
    this.unlocking = true;
    this.events.unlock(this.setId).pipe(
      finalize(() => (this.unlocking = false))
    ).subscribe({
      next: set => {
        this.applySet(set);
        this.emitState();
      },
      error: err => (this.error = apiErrorMessage(err, 'Could not unlock this commander.'))
    });
  }

  clearError(): void {
    this.error = null;
  }

  trackSlot(index: number): number {
    return index;
  }

  private emitState(): void {
    const live = this.setStatus === 'draft' || this.setStatus === 'locked';
    this.stateChange.emit(live && this.setRarity
      ? { status: this.setStatus as CommanderState['status'], rarity: this.setRarity, name: this.setName ?? '' }
      : null);
  }

  private applySet(set: SetView | null): void {
    this.stopWatches();
    this.renaming = false;
    if (!set || set.status === 'abandoned') {
      this.loadedCard = null;
      this.setId = null;
      this.setStatus = null;
      this.setName = null;
      this.setRarity = null;
      this.slots = [null, null, null];
      return;
    }
    this.setId = set.id;
    this.setStatus = set.status;
    this.setName = set.commanderName ?? '';
    this.setRarity = (set.rarity as Rarity | null) ?? this.setRarity;
    this.commanderName.setValue(this.setName);
    this.loadedCard = set.cards?.[0]?.card ?? this.loadedCard;
    this.slots = [null, null, null];
    for (const view of set.cards ?? []) {
      this.placeCard(view);
    }
  }

  /** Puts a card in its slot and watches it if it is still being made. */
  private placeCard(view: CardView, index = this.slotIndex(view)): void {
    this.slots[index] = view;
    this.watches[index]?.unsubscribe();
    this.watches[index] = undefined;
    if (isPending(view.status)) {
      this.watches[index] = this.generation.watch(view.id).subscribe({
        next: update => {
          if (this.slots[index]?.id === update.id) {
            this.slots[index] = update;
          }
        },
        error: err => (this.error = apiErrorMessage(err, 'Lost track of a card. Refresh the page.'))
      });
    }
  }

  private slotIndex(view: CardView): number {
    const slot = view.slot ?? 1;
    return Math.min(Math.max(slot, 1), 3) - 1;
  }

  private stopWatches(): void {
    this.watches.forEach(w => w?.unsubscribe());
    this.watches = [];
  }
}

/** 'Rare' for 'rare'. */
export function rarityLabel(rarity: string): string {
  return RarityOptions.find(r => r.value === rarity)?.label ?? rarity;
}
