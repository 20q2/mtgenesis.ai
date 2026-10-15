import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import {
  ErrataFields, ErrataGroup, ErrataItem, ErrataPreview, EventSummary, EventView, SetView
} from '../models/api.model';
import { api } from './api.util';

/** Events, sets, votes and host (admin) actions (spec §4). */
@Injectable({ providedIn: 'root' })
export class EventService {
  constructor(private http: HttpClient) {}

  /** The open event, or null when none is open. */
  current(): Observable<EventView | null> {
    return this.http.get<EventView | null>(api('/events/current'));
  }

  get(id: string): Observable<EventView> {
    return this.http.get<EventView>(api(`/events/${encodeURIComponent(id)}`));
  }

  /** All events, newest first. */
  list(): Observable<EventSummary[]> {
    return this.http.get<EventSummary[]>(api('/events'));
  }

  /** My commanders, one per CMC: each a draft, or else locked in the open event. */
  mySets(): Observable<SetView[]> {
    return this.http.get<SetView[]>(api('/me/sets/current'));
  }
  lock(setId: string, commanderName: string): Observable<SetView> {
    return this.http.post<SetView>(api(`/sets/${encodeURIComponent(setId)}/lock`), { commanderName });
  }

  unlock(setId: string): Observable<SetView> {
    return this.http.post<SetView>(api(`/sets/${encodeURIComponent(setId)}/unlock`), {});
  }

  /** One vote per voter per set; voting again moves the vote. Returns the updated set. */
  vote(setId: string, cardId: string): Observable<SetView> {
    return this.http.post<SetView>(api('/votes'), { setId, cardId });
  }

  /** Asks the owner to reroll a broken version, or takes the request back. Returns the updated set. */
  requestReroll(cardId: string, requested: boolean): Observable<SetView> {
    return this.http.post<SetView>(api(`/cards/${encodeURIComponent(cardId)}/reroll-request`), { requested });
  }

  createEvent(name: string, pin: string): Observable<EventView> {
    return this.http.post<EventView>(api('/admin/events'), { name }, { headers: this.adminHeaders(pin) });
  }

  /** Closes the event; the response carries the final leader/tied flags (the winners). */
  closeEvent(id: string, pin: string): Observable<EventView> {
    return this.http.post<EventView>(
      api(`/admin/events/${encodeURIComponent(id)}/close`), {}, { headers: this.adminHeaders(pin) });
  }

  /** Host errata: the cards in the vote, the last closed event's legal commanders and the pool. */
  errataList(pin: string): Observable<ErrataGroup[]> {
    return this.http.get<{ groups: ErrataGroup[] }>(api('/admin/errata'), { headers: this.adminHeaders(pin) })
      .pipe(map(body => body.groups));
  }

  /** The card re-rendered with these changes, not saved. */
  previewErrata(cardId: string, fields: ErrataFields, pin: string): Observable<ErrataPreview> {
    return this.http.post<ErrataPreview>(api(`/admin/cards/${encodeURIComponent(cardId)}/errata`),
      { ...fields, preview: true }, { headers: this.adminHeaders(pin) });
  }

  saveErrata(cardId: string, fields: ErrataFields, pin: string): Observable<ErrataItem> {
    return this.http.post<ErrataItem>(api(`/admin/cards/${encodeURIComponent(cardId)}/errata`),
      fields, { headers: this.adminHeaders(pin) });
  }

  revertErrata(cardId: string, pin: string): Observable<ErrataItem> {
    return this.http.post<ErrataItem>(api(`/admin/cards/${encodeURIComponent(cardId)}/errata/revert`),
      {}, { headers: this.adminHeaders(pin) });
  }

  private adminHeaders(pin: string): HttpHeaders {
    return new HttpHeaders({ 'X-Admin-Pin': pin });
  }
}
