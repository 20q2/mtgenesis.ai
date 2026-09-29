import { fakeAsync, tick } from '@angular/core/testing';
import { PageVisibilityService } from './page-visibility.service';

/** A stand-in document: an EventTarget with a settable visibilityState. */
class FakeDocument extends EventTarget {
  visibilityState: DocumentVisibilityState = 'visible';

  set(state: DocumentVisibilityState): void {
    this.visibilityState = state;
    this.dispatchEvent(new Event('visibilitychange'));
  }
}

describe('PageVisibilityService', () => {
  let doc: FakeDocument;
  let service: PageVisibilityService;

  beforeEach(() => {
    doc = new FakeDocument();
    service = new PageVisibilityService(doc as unknown as Document);
  });

  it('visible$ follows document.visibilityState', () => {
    const seen: boolean[] = [];
    const sub = service.visible$.subscribe(v => seen.push(v));
    doc.set('hidden');
    doc.set('hidden');
    doc.set('visible');
    sub.unsubscribe();
    expect(seen).toEqual([true, false, true]);
  });

  it('poll ticks immediately, then every period, while visible', fakeAsync(() => {
    let ticks = 0;
    const sub = service.poll(1000).subscribe(() => ticks++);
    expect(ticks).toBe(1);
    tick(3000);
    expect(ticks).toBe(4);
    sub.unsubscribe();
  }));

  it('poll pauses while hidden and resumes immediately on becoming visible', fakeAsync(() => {
    let ticks = 0;
    const sub = service.poll(1000, 1000).subscribe(() => ticks++);
    expect(ticks).toBe(0);         // honours the first delay on start
    tick(1000);
    expect(ticks).toBe(1);
    doc.set('hidden');
    tick(60000);
    expect(ticks).toBe(1);         // nothing while hidden
    doc.set('visible');
    expect(ticks).toBe(2);         // immediate on resume, not after the first delay
    tick(1000);
    expect(ticks).toBe(3);
    sub.unsubscribe();
  }));

  it('poll started while hidden waits for the page to become visible', fakeAsync(() => {
    doc.visibilityState = 'hidden';
    let ticks = 0;
    const sub = service.poll(1000).subscribe(() => ticks++);
    tick(5000);
    expect(ticks).toBe(0);
    doc.set('visible');
    expect(ticks).toBe(1);
    sub.unsubscribe();
  }));
});
