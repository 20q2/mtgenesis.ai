import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { environment } from '../../environments/environment';
import { EventView, SetView } from '../models/api.model';
import { eventView, setView } from '../testing/fixtures';
import { EventService } from './event.service';

describe('EventService', () => {
  let service: EventService;
  let http: HttpTestingController;
  const base = `${environment.apiUrl}/api/v1`;

  beforeEach(() => {
    TestBed.configureTestingModule({ imports: [HttpClientTestingModule] });
    service = TestBed.inject(EventService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('current() GETs /events/current and passes null through', () => {
    let result: EventView | null | undefined;
    service.current().subscribe(e => (result = e));
    const req = http.expectOne(`${base}/events/current`);
    expect(req.request.method).toBe('GET');
    req.flush(null);
    expect(result).toBeNull();
  });

  it('get(id) and list() hit /events/<id> and /events', () => {
    service.get('e-1').subscribe();
    http.expectOne(`${base}/events/e-1`).flush(eventView());
    service.list().subscribe();
    http.expectOne(`${base}/events`).flush([]);
  });

  it('mySet() GETs /me/sets/current', () => {
    let result: SetView | null | undefined;
    service.mySet().subscribe(s => (result = s));
    http.expectOne(`${base}/me/sets/current`).flush(setView({ status: 'draft' }));
    expect(result?.status).toBe('draft');
  });

  it('lock() and unlock() POST to /sets/<id>/lock|unlock', () => {
    service.lock('s-1', 'Zur\'ka').subscribe();
    const lock = http.expectOne(`${base}/sets/s-1/lock`);
    expect(lock.request.method).toBe('POST');
    expect(lock.request.body).toEqual({ commanderName: 'Zur\'ka' });
    lock.flush(setView());

    service.unlock('s-1').subscribe();
    const unlock = http.expectOne(`${base}/sets/s-1/unlock`);
    expect(unlock.request.method).toBe('POST');
    unlock.flush(setView({ status: 'draft' }));
  });

  it('vote() POSTs {setId, cardId} to /votes', () => {
    service.vote('s-1', 'c-2').subscribe();
    const req = http.expectOne(`${base}/votes`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ setId: 's-1', cardId: 'c-2' });
    req.flush(setView());
  });

  it('createEvent() and closeEvent() send X-Admin-Pin', () => {
    service.createEvent('AI Night #2', '4242').subscribe();
    const create = http.expectOne(`${base}/admin/events`);
    expect(create.request.method).toBe('POST');
    expect(create.request.body).toEqual({ name: 'AI Night #2' });
    expect(create.request.headers.get('X-Admin-Pin')).toBe('4242');
    create.flush(eventView());

    service.closeEvent('e-1', '4242').subscribe();
    const close = http.expectOne(`${base}/admin/events/e-1/close`);
    expect(close.request.method).toBe('POST');
    expect(close.request.headers.get('X-Admin-Pin')).toBe('4242');
    close.flush(eventView({ status: 'closed' }));
  });
});
