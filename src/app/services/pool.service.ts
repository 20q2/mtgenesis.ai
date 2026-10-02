import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { CardParams, Medal, PoolSummary, PoolView } from '../models/api.model';
import { api } from './api.util';

/** Knowledge Pool: the pool, submissions, medals and host actions. */
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

  /** Enter one of my finished, colorless or mono-colored cards in the open pool. */
  submit(cardId: string): Observable<PoolView> {
    return this.http.post<PoolView>(api('/pools/entries'), { cardId });
  }

  withdraw(entryId: string): Observable<PoolView> {
    return this.http.post<PoolView>(api(`/pools/entries/${encodeURIComponent(entryId)}/withdraw`), {});
  }

  /** Gold/silver/bronze a card (moves that medal off any other card in the pool). Not your own card. */
  medal(entryId: string, medal: Medal): Observable<PoolView> {
    return this.http.post<PoolView>(api('/pools/medals'), { entryId, medal });
  }

  clearMedal(entryId: string): Observable<PoolView> {
    return this.http.post<PoolView>(api('/pools/medals/clear'), { entryId });
  }

  createPool(name: string, maxEntriesPerUser: number, pin: string): Observable<PoolView> {
    return this.http.post<PoolView>(api('/admin/pools'), { name, maxEntriesPerUser },
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

/** Entries per player a new pool gets unless the host changes it. */
export const POOL_DEFAULT_ENTRIES = 3;
export const POOL_ENTRY_CAP_MAX = 10;

/** The card's colors among WUBRG: its colors list, else the symbols in its mana cost. */
export function cardColors(card: Partial<CardParams> | null | undefined): string[] {
  const listed = (card?.colors ?? []).filter(c => MONO_COLORS.includes(c));
  if (listed.length) {
    return Array.from(new Set(listed));
  }
  return Array.from(new Set((card?.manaCost ?? '').toUpperCase().match(/[WUBRG]/g) ?? []));
}

/**
 * Only colorless or mono-colored cards can enter the pool. Mirrors storage.pool_card_eligible
 * (and card_colors) so the Submit button only offers what the server accepts.
 */
export function poolColorOk(card: Partial<CardParams> | null | undefined): boolean {
  return cardColors(card).length <= 1;
}
