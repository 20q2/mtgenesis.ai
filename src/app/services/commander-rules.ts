import { SetView } from '../models/api.model';
import { CommanderKind, Rarity } from '../models/card.model';

/**
 * AI Night commander rules, mirrored from proxy-server/commander_rules.py for instant feedback
 * (docs/superpowers/specs/2026-10-03-commander-rules-design.md). The server has the final say.
 */
export const COMMANDER_CMCS = [3, 4, 5];
export const COMMANDER_RARITIES: Rarity[] = [Rarity.UNCOMMON, Rarity.RARE, Rarity.MYTHIC];
export const VEHICLE_BONUS = 2;
/** Colored pips are capped by the cheapest commander's mana value. */
export const MAX_PIP_VALUE = COMMANDER_CMCS[0];

/** Mana the colored pips of a cost are worth (generic and X are dropped; {2/W} counts 2). */
export function commanderPipValue(manaCost: string): number {
  const symbols = (manaCost || '').match(/\{[^}]+\}/g) || [];
  return symbols
    .map(s => s.slice(1, -1))
    .filter(s => !/^\d+$/.test(s) && !/^[XYZ]$/i.test(s))
    .reduce((sum, s) => sum + (s.startsWith('2/') ? 2 : 1), 0);
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
