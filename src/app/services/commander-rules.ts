import { SetView } from '../models/api.model';
import { Card, CommanderKind, Rarity, RarityOptions } from '../models/card.model';

/**
 * AI Night commander rules, mirrored from proxy-server/commander_rules.py for instant feedback
 * (docs/superpowers/specs/2026-10-03-commander-rules-design.md). The server has the final say.
 */
export const COMMANDER_CMCS = [3, 4, 5];
export const COMMANDER_RARITIES: Rarity[] = [Rarity.UNCOMMON, Rarity.RARE, Rarity.MYTHIC];
export const VEHICLE_BONUS = 2;
export const COMMANDER_NAME_MAX = 40;
/** Words that can't be in a creature commander's type line (mirrors NOT_CREATURE_TYPES). */
const NOT_CREATURE_TYPES = new Set(`
  vehicle equipment fortification aura saga curse shrine class room cartouche background role
  treasure food clue blood gold map powerstone incubator attraction contraption
  plains island swamp mountain forest desert gate lair locus mine tower cave sphere
  creature artifact enchantment instant sorcery land planeswalker battle kindred tribal
  legendary basic snow world ongoing elite host
`.split(/\s+/).filter(Boolean));

/** Why this isn't a creature type line ("Equipment isn't a creature type"), or null. */
export function creatureTypeError(subtype: string): string | null {
  const word = (subtype || '').split(/\s+/).find(w => NOT_CREATURE_TYPES.has(w.toLowerCase()));
  return word ? `${word} isn't a creature type` : null;
}

/** Mana the colored pips of a cost are worth (generic and X are dropped; {2/W} counts 2). */
export function commanderPipValue(manaCost: string): number {
  const symbols = (manaCost || '').match(/\{[^}]+\}/g) || [];
  return symbols
    .map(s => s.slice(1, -1))
    .filter(s => !/^\d+$/.test(s) && !/^[XYZ]$/i.test(s))
    .reduce((sum, s) => sum + (s.startsWith('2/') ? 2 : 1), 0);
}

/** The cost the server prints: the colored pips, padded with generic mana up to the CMC. */
export function commanderCost(manaCost: string, cmc: number): string {
  const pips = ((manaCost || '').match(/\{[^}]+\}/g) || [])
    .filter(s => !/^\{\d+\}$/.test(s) && !/^\{[XYZ]\}$/i.test(s));
  const generic = cmc - commanderPipValue(pips.join(''));
  return (generic > 0 ? `{${generic}}` : '') + pips.join('');
}

/** The split the server picks for a blank P/T: every point spent, leaning by creature type
 *  (mirrors power_level.lean). */
export function autoStats(cmc: number, kind: CommanderKind, subtype: string): [number, number] {
  const points = statPoints(cmc, kind);
  let power = Math.floor(points / 2);
  let toughness = points - power;
  const sub = (subtype || '').toLowerCase();
  if (/\b(wall|treefolk|golem|construct|turtle)\b/.test(sub) && power > 1) {
    power--; toughness++;
  } else if (/\b(goblin|berserker|warrior|dragon|demon|cat|rogue)\b/.test(sub) && toughness > 1) {
    power++; toughness--;
  }
  return [power, toughness];
}

/** Total power + toughness a commander may have: CMC + 1, plus 2 for a Vehicle. */
export function statPoints(cmc: number, kind: CommanderKind): number {
  return cmc + 1 + (kind === 'vehicle' ? VEHICLE_BONUS : 0);
}

/** Why this P/T breaks the point buy, or null (both blank = the server picks a split). */
export function commanderStatsError(power: string, toughness: string, cmc: number,
                                    kind: CommanderKind): string | null {
  const p = (power ?? '').trim();
  const t = (toughness ?? '').trim();
  if (!p && !t) {
    return null;
  }
  if (!p || !t) {
    return 'Give both power and toughness, or leave both blank';
  }
  if (!/^\d+$/.test(p) || !/^\d+$/.test(t)) {
    return 'P/T must be whole numbers — X and * aren\'t allowed';
  }
  const pn = Number(p);
  const tn = Number(t);
  if (tn < 1) {
    return 'Toughness must be at least 1';
  }
  const points = statPoints(cmc, kind);
  if (pn + tn > points) {
    return `A ${cmc}-mana commander has ${points} points; ${pn}/${tn} uses ${pn + tn}`;
  }
  return null;
}

export interface CmcGroup { label: string; cmc: number | null; sets: SetView[]; }

/** Sets under 3, 4 and 5 CMC headings, then sets made before the rules; empty groups dropped. */
export function groupByCmc(sets: SetView[]): CmcGroup[] {
  const groups: CmcGroup[] = [
    ...COMMANDER_CMCS.map(cmc => ({ label: `${cmc} CMC`, cmc, sets: sets.filter(s => s.cmc === cmc) })),
    { label: 'Earlier sets', cmc: null, sets: sets.filter(s => !COMMANDER_CMCS.includes(s.cmc ?? -1)) }
  ];
  return groups.filter(g => g.sets.length > 0);
}

export interface ChecklistItem {
  id: 'name' | 'mana' | 'type' | 'body' | 'rarity';
  label: string;
  ok: boolean;
  /** What the commander has now, or what is wrong. */
  detail: string;
}

/**
 * The rules a commander must meet before it can be generated, each checked against the current
 * design (null: not designed yet). Mirrors the server's checks, which have the final say.
 */
export function commanderChecklist(name: string, card: Card | null, cmc: number,
                                   takenRarities: Partial<Record<Rarity, number>>): ChecklistItem[] {
  const trimmed = (name || '').trim();
  const kind: CommanderKind = card?.commanderKind ?? 'creature';
  const points = statPoints(cmc, kind);
  const items: ChecklistItem[] = [{
    id: 'name', label: 'Named',
    ok: trimmed.length > 0 && trimmed.length <= COMMANDER_NAME_MAX,
    detail: !trimmed ? 'Give your commander a name'
      : trimmed.length > COMMANDER_NAME_MAX ? `At most ${COMMANDER_NAME_MAX} characters` : trimmed
  }];
  if (!card) {
    return [...items,
      { id: 'mana', label: `Costs exactly ${cmc} mana`, ok: false, detail: 'Pick its colors' },
      { id: 'type', label: 'Legendary Creature or Vehicle', ok: false, detail: 'Pick a type' },
      { id: 'body', label: `Power + toughness within ${points} points`, ok: false, detail: 'Set its body' },
      { id: 'rarity', label: 'Uncommon, Rare or Mythic, one of each', ok: false, detail: 'Pick a rarity' }];
  }

  const pipValue = commanderPipValue(card.manaCost ?? '');
  const typeError = !/legendary/i.test(card.supertype ?? '') ? 'It must be Legendary'
    : kind === 'vehicle'
      ? (card.type === 'Artifact' && (card.subtype ?? '') === 'Vehicle' ? null : 'A Vehicle is exactly Artifact — Vehicle')
      : (card.type === 'Creature' ? creatureTypeError(card.subtype ?? '') : 'It must be a Creature or a Vehicle');
  const statsError = commanderStatsError(card.power ?? '', card.toughness ?? '', cmc, kind);
  const takenAt = takenRarities[card.rarity];
  const rarityOk = COMMANDER_RARITIES.includes(card.rarity) && takenAt === undefined;
  const rarityLabel = RarityOptions.find(r => r.value === card.rarity)?.label ?? card.rarity;
  const typeLine = kind === 'vehicle' ? 'Legendary Artifact — Vehicle'
    : `Legendary Creature${card.subtype ? ` — ${card.subtype}` : ''}`;
  const [p, t] = card.power && card.toughness ? [card.power, card.toughness] : autoStats(cmc, kind, card.subtype ?? '');

  return [...items,
    { id: 'mana', label: `Costs exactly ${cmc} mana`, ok: pipValue <= cmc,
      detail: pipValue <= cmc ? commanderCost(card.manaCost ?? '', cmc) : `Colored pips add up to ${pipValue} mana` },
    { id: 'type', label: 'Legendary Creature or Vehicle', ok: !typeError,
      detail: typeError ?? typeLine },
    { id: 'body', label: `Power + toughness within ${points} points`, ok: !statsError,
      detail: statsError ?? `${p}/${t}${card.power ? '' : ' (auto)'}` },
    { id: 'rarity', label: 'Uncommon, Rare or Mythic, one of each', ok: rarityOk,
      detail: takenAt !== undefined ? `Your ${takenAt} CMC commander is locked in as ${rarityLabel}`
        : rarityOk ? rarityLabel : !card.rarity ? 'Pick a rarity' : 'Commanders are Uncommon, Rare or Mythic' }];
}
