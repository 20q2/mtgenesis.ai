import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { EventSummary, EventView, SetView } from '../models/api.model';
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

  /** My draft, or else my set locked in the open event, or null. */
  mySet(): Observable<SetView | null> {
    return this.http.get<SetView | null>(api('/me/sets/current'));
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

  createEvent(name: string, pin: string): Observable<EventView> {
    return this.http.post<EventView>(api('/admin/events'), { name }, { headers: this.adminHeaders(pin) });
  }

  /** Closes the event; the response carries the final leader/tied flags (the winners). */
  closeEvent(id: string, pin: string): Observable<EventView> {
    return this.http.post<EventView>(
      api(`/admin/events/${encodeURIComponent(id)}/close`), {}, { headers: this.adminHeaders(pin) });
  }

  private adminHeaders(pin: string): HttpHeaders {
    return new HttpHeaders({ 'X-Admin-Pin': pin });
  }
}
