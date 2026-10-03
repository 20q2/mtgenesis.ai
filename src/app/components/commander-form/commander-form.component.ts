import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { GeneratedCardData } from '../../models/api.model';
import { Card, CommanderKind, CommonSubtypes, Rarity, RarityOptions } from '../../models/card.model';
import {
  COMMANDER_RARITIES, MAX_PIP_VALUE, autoStats, commanderCost, commanderPipValue, statPoints
} from '../../services/commander-rules';
import { ManaService } from '../../services/mana.service';

/** The pips a commander can add: the five colors and colorless. Generic is added for you. */
const PIP_BUTTONS = [
  { color: 'W', label: 'White' }, { color: 'U', label: 'Blue' }, { color: 'B', label: 'Black' },
  { color: 'R', label: 'Red' }, { color: 'G', label: 'Green' }, { color: 'C', label: 'Colorless' }
];

/** Creature types offered as one-tap chips (any type can be typed). */
const CREATURE_TYPES = CommonSubtypes['Creature'].slice(0, 12);

/**
 * The commander designer (docs/superpowers/specs/2026-10-03-commander-rules-design.md §5):
 * colors, Creature or Vehicle, a point-buy body that can't leave the rules, and Uncommon /
 * Rare / Mythic. Emits the card params the panel sends; the server validates them again.
 */
@Component({
  selector: 'app-commander-form',
  templateUrl: './commander-form.component.html',
  styleUrls: ['./commander-form.component.scss']
})
export class CommanderFormComponent implements OnChanges {
  @Input() cmc = 3;
  /** Rarities the player's other commanders are locked in with, and at which CMC. */
  @Input() takenRarities: Partial<Record<Rarity, number>> = {};
  /** A saved commander to start from (its first version's card). */
  @Input() initialCard: Partial<GeneratedCardData> | null = null;
  @Output() cardChange = new EventEmitter<Card>();

  readonly pipButtons = PIP_BUTTONS;
  readonly creatureTypes = CREATURE_TYPES;
  readonly rarities = RarityOptions.filter(r => COMMANDER_RARITIES.includes(r.value as Rarity));

  pips: string[] = [];
  kind: CommanderKind = 'creature';
  subtype = '';
  rarity: Rarity = Rarity.UNCOMMON;
  /** Auto: the server spends every point (the split shown is the one it will pick). */
  auto = true;
  power = 0;
  toughness = 1;

  constructor(private mana: ManaService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['initialCard'] && this.initialCard) {
      this.hydrate(this.initialCard);
    }
    this.ensureFreeRarity();
    this.clampBody();
    this.emit();
  }

  get points(): number {
    return statPoints(this.cmc, this.kind);
  }

  get pipValue(): number {
    return commanderPipValue(this.pips.join(''));
  }

  /** The cost every version prints: pips plus generic up to the CMC. */
  get fullCost(): string {
    return commanderCost(this.pips.join(''), this.cmc);
  }

  /** The generic part of the cost ({2}), added for you; null when the pips fill the CMC. */
  get genericSymbol(): string | null {
    const generic = this.cmc - this.pipValue;
    return generic > 0 ? `{${generic}}` : null;
  }

  get costSymbols(): string[] {
    return this.mana.extractManaSymbols(this.fullCost);
  }

  /** The body shown and sent: the auto split, or the typed one. */
  get shownStats(): [number, number] {
    return this.auto ? autoStats(this.cmc, this.kind, this.subtypeOut) : [this.power, this.toughness];
  }

  get pointsUsed(): number {
    const [p, t] = this.shownStats;
    return p + t;
  }

  /** One entry per point: whether it is spent on power, on toughness, or unspent. */
  get pointCells(): ('power' | 'toughness' | 'free')[] {
    const [p, t] = this.shownStats;
    return Array.from({ length: this.points }, (_, i) => (i < p ? 'power' : i < p + t ? 'toughness' : 'free'));
  }

  get typeLine(): string {
    const sub = this.subtypeOut;
    const main = this.kind === 'vehicle' ? 'Legendary Artifact' : 'Legendary Creature';
    return sub ? `${main} — ${sub}` : main;
  }

  get rarityLabel(): string {
    return this.rarities.find(r => r.value === this.rarity)?.label ?? this.rarity;
  }

  canAdd(color: string): boolean {
    return this.pipValue + 1 <= MAX_PIP_VALUE && color.length > 0;
  }

  addPip(color: string): void {
    if (!this.canAdd(color)) {
      return;
    }
    this.pips = this.mana.extractManaSymbols(this.mana.reorderManaSymbols([...this.pips, `{${color}}`].join('')));
    this.emit();
  }

  removePip(index: number): void {
    this.pips = this.pips.filter((_, i) => i !== index);
    this.emit();
  }

  clearPips(): void {
    this.pips = [];
    this.emit();
  }

  setKind(kind: CommanderKind): void {
    this.kind = kind;
    this.clampBody();
    this.emit();
  }

  pickType(type: string): void {
    this.subtype = this.subtype === type ? '' : type;
    this.emit();
  }

  onSubtypeInput(value: string): void {
    this.subtype = value;
    this.emit();
  }

  canStep(stat: 'power' | 'toughness', delta: 1 | -1): boolean {
    const [p, t] = this.shownStats;
    if (delta > 0) {
      return p + t < this.points;
    }
    return stat === 'power' ? p > 0 : t > 1;
  }

  step(stat: 'power' | 'toughness', delta: 1 | -1): void {
    if (!this.canStep(stat, delta)) {
      return;
    }
    [this.power, this.toughness] = this.shownStats;
    this.auto = false;
    if (stat === 'power') {
      this.power += delta;
    } else {
      this.toughness += delta;
    }
    this.emit();
  }

  setAuto(): void {
    this.auto = true;
    this.emit();
  }

  /** The CMC of the other commander locked in with this rarity, if any. */
  takenAt(rarity: string): number | undefined {
    return this.takenRarities[rarity as Rarity];
  }

  pickRarity(rarity: string): void {
    if (this.takenAt(rarity) !== undefined) {
      return;
    }
    this.rarity = rarity as Rarity;
    this.emit();
  }

  /** The mana-font class for a symbol: ms-w, ms-2, ms-wu. */
  symbolClass(symbol: string): string {
    return `ms-${this.mana.getSymbolClass(symbol)}`;
  }

  emit(): void {
    const [power, toughness] = this.shownStats;
    this.cardChange.emit({
      name: '',
      manaCost: this.pips.join(''),
      supertype: 'Legendary',
      type: this.kind === 'vehicle' ? 'Artifact' : 'Creature',
      subtype: this.subtypeOut,
      colors: this.mana.extractColorsFromManaCost(this.pips.join('')),
      cmc: this.cmc,
      rarity: this.rarity,
      commanderKind: this.kind,
      power: this.auto ? '' : String(power),
      toughness: this.auto ? '' : String(toughness),
      artPrompt: this.artPrompt()
    });
  }

  trackIndex(index: number): number {
    return index;
  }

  /** The subtype as sent: a Vehicle always says Vehicle. */
  private get subtypeOut(): string {
    const sub = this.subtype.replace(/\s*\bVehicle\b/i, '').trim();
    return this.kind === 'vehicle' ? `${sub} Vehicle`.trim() : sub;
  }

  /** "a legendary dragon, large and imposing": the image model reads the type, sized by CMC. */
  private artPrompt(): string {
    if (this.kind === 'vehicle') {
      return 'a legendary vehicle artifact';
    }
    const noun = (this.subtypeOut || 'creature').toLowerCase();
    return `a legendary ${noun}, ${this.cmc <= 4 ? 'medium scale' : 'large and imposing'}`;
  }

  private hydrate(card: Partial<GeneratedCardData>): void {
    this.pips = this.mana.extractManaSymbols(card.manaCost ?? '')
      .filter(s => !/^\{\d+\}$/.test(s) && !/^\{[XYZ]\}$/i.test(s));
    this.kind = /vehicle/i.test(`${card.type ?? ''} ${card.subtype ?? ''}`) && !/creature/i.test(card.type ?? '')
      ? 'vehicle' : 'creature';
    this.subtype = (card.subtype ?? '').replace(/\s*\bVehicle\b/i, '').trim();
    if (COMMANDER_RARITIES.includes(card.rarity as Rarity)) {
      this.rarity = card.rarity as Rarity;
    }
    const p = Number(card.power);
    const t = Number(card.toughness);
    if (card.power !== undefined && card.power !== '' && Number.isInteger(p) && Number.isInteger(t)) {
      this.auto = false;
      this.power = p;
      this.toughness = t;
    } else {
      this.auto = true;
    }
  }

  /** Moves off a rarity another commander is locked in with. */
  private ensureFreeRarity(): void {
    if (this.takenAt(this.rarity) === undefined) {
      return;
    }
    const free = COMMANDER_RARITIES.find(r => this.takenAt(r) === undefined);
    if (free) {
      this.rarity = free;
    }
  }

  /** Keeps a typed body inside the points (e.g. after Vehicle → Creature). */
  private clampBody(): void {
    if (this.auto) {
      return;
    }
    this.toughness = Math.min(Math.max(this.toughness, 1), this.points);
    this.power = Math.min(Math.max(this.power, 0), this.points - this.toughness);
  }
}
