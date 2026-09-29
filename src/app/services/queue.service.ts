import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, distinctUntilChanged, map, of, shareReplay, switchMap, timer } from 'rxjs';
import { QueueStatus } from '../models/api.model';
import { api } from './api.util';

export const QUEUE_POLL_MS = 5000;

/** Polls GET /queue_status every 5s (replaces the old /health polling). */
@Injectable({ providedIn: 'root' })
export class QueueService {
  /** Latest queue status; null when the server can't be reached. Shared between subscribers. */
  readonly status$: Observable<QueueStatus | null>;
  /** True while /queue_status answers. */
  readonly online$: Observable<boolean>;

  constructor(private http: HttpClient) {
    this.status$ = timer(0, QUEUE_POLL_MS).pipe(
      switchMap(() => this.http.get<QueueStatus>(api('/queue_status')).pipe(
        catchError(() => of(null))
      )),
      shareReplay({ bufferSize: 1, refCount: true })
    );
    this.online$ = this.status$.pipe(
      map(status => status !== null),
      distinctUntilChanged()
    );
  }
}
