import { Inject, Injectable } from '@angular/core';
import { DOCUMENT } from '@angular/common';
import {
  EMPTY, Observable, distinctUntilChanged, fromEvent, interval, map, shareReplay, startWith, switchMap,
  timer
} from 'rxjs';

/**
 * Pauses polling while the tab is hidden (phone asleep, other app, background tab) so
 * idle phones don't keep hitting the tunnel, and resumes immediately when it is shown.
 */
export function pollWhileVisible(visible$: Observable<boolean>, periodMs: number,
                                 firstDelayMs = 0): Observable<number> {
  return visible$.pipe(
    distinctUntilChanged(),
    // The first delay only applies when polling starts visible; a resume polls at once.
    switchMap((visible, index) => visible ? ticks(periodMs, index === 0 ? firstDelayMs : 0) : EMPTY)
  );
}

/** timer(firstDelayMs, periodMs), except that a zero first delay ticks synchronously. */
function ticks(periodMs: number, firstDelayMs: number): Observable<number> {
  return firstDelayMs > 0
    ? timer(firstDelayMs, periodMs)
    : interval(periodMs).pipe(map(n => n + 1), startWith(0));
}

@Injectable({ providedIn: 'root' })
export class PageVisibilityService {
  /** True while the page is visible; emits the current state on subscribe. */
  readonly visible$: Observable<boolean>;

  constructor(@Inject(DOCUMENT) doc: Document) {
    this.visible$ = fromEvent(doc, 'visibilitychange').pipe(
      startWith(null),
      map(() => doc.visibilityState !== 'hidden'),
      distinctUntilChanged(),
      shareReplay({ bufferSize: 1, refCount: true })
    );
  }

  /**
   * Like timer(firstDelayMs, periodMs), but silent while the page is hidden and
   * ticking again immediately when it becomes visible.
   */
  poll(periodMs: number, firstDelayMs = 0): Observable<number> {
    return pollWhileVisible(this.visible$, periodMs, firstDelayMs);
  }
}
