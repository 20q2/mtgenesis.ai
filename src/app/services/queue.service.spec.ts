import { TestBed, discardPeriodicTasks, fakeAsync, tick } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { environment } from '../../environments/environment';
import { QueueStatus } from '../models/api.model';
import { fakeVisibility } from '../testing/fake-visibility';
import { PageVisibilityService } from './page-visibility.service';
import { QueueService } from './queue.service';

describe('QueueService', () => {
  let service: QueueService;
  let http: HttpTestingController;
  let visibility: ReturnType<typeof fakeVisibility>;
  const url = `${environment.apiUrl}/api/v1/queue_status`;
  const idle: QueueStatus = { busy: false, cardsAhead: 0, generatingNow: 0, avgImageSeconds: 10, etaSeconds: 10 };

  beforeEach(() => {
    visibility = fakeVisibility();
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [{ provide: PageVisibilityService, useValue: visibility }]
    });
    service = TestBed.inject(QueueService);
    http = TestBed.inject(HttpTestingController);
  });

  it('polls /queue_status immediately and every 5s, emitting null on error', fakeAsync(() => {
    const seen: (QueueStatus | null)[] = [];
    const online: boolean[] = [];
    const sub = service.status$.subscribe(s => seen.push(s));
    const sub2 = service.online$.subscribe(o => online.push(o));

    tick(0);
    http.expectOne(url).flush(idle);
    tick(5000);
    http.expectOne(url).error(new ProgressEvent('error'), { status: 0 });
    tick(5000);
    http.expectOne(url).flush({ ...idle, busy: true });

    expect(seen).toEqual([idle, null, { ...idle, busy: true }]);
    expect(online).toEqual([true, false, true]);

    sub.unsubscribe();
    sub2.unsubscribe();
    discardPeriodicTasks();
    http.verify();
  }));

  it('shares one poll between subscribers', fakeAsync(() => {
    const a = service.status$.subscribe();
    const b = service.status$.subscribe();
    tick(0);
    http.expectOne(url).flush(idle);
    a.unsubscribe();
    b.unsubscribe();
    discardPeriodicTasks();
    http.verify();
  }));

  it('stops polling while the page is hidden and polls at once when it is shown', fakeAsync(() => {
    const seen: (QueueStatus | null)[] = [];
    const sub = service.status$.subscribe(s => seen.push(s));
    tick(0);
    http.expectOne(url).flush(idle);

    visibility.visibleSubject.next(false);
    tick(60000);
    http.expectNone(url);

    visibility.visibleSubject.next(true);
    http.expectOne(url).flush({ ...idle, busy: true });
    expect(seen).toEqual([idle, { ...idle, busy: true }]);

    sub.unsubscribe();
    discardPeriodicTasks();
    http.verify();
  }));
});
