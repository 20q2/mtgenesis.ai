import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  CardParams, Medal, PoolColorRule, PoolSlotSpec, PoolSummary, PoolTypeRule, PoolView
} from '../models/api.model';
import { api } from './api.util';

/** Knowledge Pool: the pool, submissions, votes and host actions. */
@Injectable({ providedIn: 'root' })
export class PoolService {
  constructor(private http: HttpClient) {}

  /** The open pool, or null when none is open. */
  current(): Observable<PoolView | null> {
    return this.http.get<PoolView | null>(api('/pools/current'));
  }

  get(id: string): Observable<PoolView> {
    return this.http.get<PoolView>(api(`/pools/${encodeURIComponent(id)}`));
  }

  /** All pools, newest first. */
  list(): Observable<PoolSummary[]> {
    return this.http.get<PoolSummary[]>(api('/pools'));
  }

  submit(slotId: string, cardId: string): Observable<PoolView> {
    return this.http.post<PoolView>(api('/pools/entries'), { slotId, cardId });
  }

  withdraw(entryId: string): Observable<PoolView> {
    return this.http.post<PoolView>(api(`/pools/entries/${encodeURIComponent(entryId)}/withdraw`), {});
  }

  /** Gold/silver/bronze a card (moves that medal off any other card in the slot). Not your own card. */
  medal(entryId: string, medal: Medal): Observable<PoolView> {
    return this.http.post<PoolView>(api('/pools/medals'), { entryId, medal });
  }

  clearMedal(entryId: string): Observable<PoolView> {
    return this.http.post<PoolView>(api('/pools/medals/clear'), { entryId });
  }

  /** Spend one of your bans on a card; banThreshold bans disqualify it at close. */
  ban(entryId: string): Observable<PoolView> {
    return this.http.post<PoolView>(api('/pools/bans'), { entryId });
  }

  unban(entryId: string): Observable<PoolView> {
    return this.http.post<PoolView>(api('/pools/bans/clear'), { entryId });
  }

  createPool(name: string, maxEntriesPerUser: number, slots: PoolSlotSpec[], pin: string): Observable<PoolView> {
    return this.http.post<PoolView>(api('/admin/pools'), { name, maxEntriesPerUser, slots },
      { headers: this.adminHeaders(pin) });
  }

  closePool(id: string, pin: string): Observable<PoolView> {
    return this.http.post<PoolView>(api(`/admin/pools/${encodeURIComponent(id)}/close`), {},
      { headers: this.adminHeaders(pin) });
  }

  private adminHeaders(pin: string): HttpHeaders {
    return new HttpHeaders({ 'X-Admin-Pin': pin });
  }
}

const MONO_COLORS = ['W', 'U', 'B', 'R', 'G'];

export const POOL_COLOR_RULES: { value: PoolColorRule; label: string }[] = [
  { value: 'W', label: 'White' },
  { value: 'U', label: 'Blue' },
  { value: 'B', label: 'Black' },
  { value: 'R', label: 'Red' },
  { value: 'G', label: 'Green' },
  { value: 'multicolor', label: 'Multicolor' },
  { value: 'colorless', label: 'Colorless' },
  { value: 'any', label: 'Any color' }
];

export const POOL_TYPE_RULES: { value: PoolTypeRule; label: string }[] = [
  { value: 'creature', label: 'Creature' },
  { value: 'noncreature', label: 'Noncreature' },
  { value: 'land', label: 'Land' },
  { value: 'any', label: 'Any type' }
];

/**
 * 16 slots for 8 players: a creature and a noncreature per color, 2 multicolor,
 * 2 colorless, a land and a wild card.
 */
export function defaultPoolSlots(): PoolSlotSpec[] {
  const colors: [PoolColorRule, string][] = [['W', 'White'], ['U', 'Blue'], ['B', 'Black'], ['R', 'Red'], ['G', 'Green']];
  return [
    ...colors.flatMap(([rule, name]): PoolSlotSpec[] => [
      { label: `${name} creature`, colorRule: rule, typeRule: 'creature' },
      { label: `${name} noncreature`, colorRule: rule, typeRule: 'noncreature' }
    ]),
    { label: 'Multicolor I', colorRule: 'multicolor', typeRule: 'any' },
    { label: 'Multicolor II', colorRule: 'multicolor', typeRule: 'any' },
    { label: 'Colorless I', colorRule: 'colorless', typeRule: 'any' },
    { label: 'Colorless II', colorRule: 'colorless', typeRule: 'any' },
    { label: 'Land', colorRule: 'any', typeRule: 'land' },
    { label: 'Wild card', colorRule: 'any', typeRule: 'any' }
  ];
}

/** The card's colors among WUBRG: its colors list, else the symbols in its mana cost. */
export function cardColors(card: Partial<CardParams> | null | undefined): string[] {
  const listed = (card?.colors ?? []).filter(c => MONO_COLORS.includes(c));
  if (listed.length) {
    return Array.from(new Set(listed));
  }
  return Array.from(new Set((card?.manaCost ?? '').toUpperCase().match(/[WUBRG]/g) ?? []));
}

/** Mirrors storage.card_fits_slot so the picker only offers cards the server accepts. */
export function cardFitsSlot(card: Partial<CardParams> | null | undefined,
                             colorRule: PoolColorRule, typeRule: PoolTypeRule): boolean {
  const colors = cardColors(card);
  let colorOk = true;
  if (MONO_COLORS.includes(colorRule)) {
    colorOk = colors.length === 1 && colors[0] === colorRule;
  } else if (colorRule === 'multicolor') {
    colorOk = colors.length >= 2;
  } else if (colorRule === 'colorless') {
    colorOk = colors.length === 0;
  }
  const type = `${card?.supertype ?? ''} ${card?.type ?? ''}`.toLowerCase();
  const isCreature = type.includes('creature');
  const isLand = type.includes('land');
  const typeOk = typeRule === 'creature' ? isCreature
    : typeRule === 'land' ? isLand
      : typeRule === 'noncreature' ? !isCreature && !isLand
        : true;
  return colorOk && typeOk;
}
