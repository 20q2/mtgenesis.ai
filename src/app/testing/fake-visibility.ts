import { BehaviorSubject } from 'rxjs';
import { PageVisibilityService } from '../services/page-visibility.service';

/**
 * Test-only PageVisibilityService whose visibility the spec controls:
 * `const vis = fakeVisibility(); ... vis.visibleSubject.next(false)` hides the page.
 * Its poll() is the real one, so timers behave as in the app.
 */
export function fakeVisibility(visible = true): PageVisibilityService & { visibleSubject: BehaviorSubject<boolean> } {
  const visibleSubject = new BehaviorSubject<boolean>(visible);
  const fake = Object.create(PageVisibilityService.prototype);
  fake.visible$ = visibleSubject.asObservable();
  fake.visibleSubject = visibleSubject;
  return fake;
}
