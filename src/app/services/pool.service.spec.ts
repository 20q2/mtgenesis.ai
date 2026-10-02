import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { environment } from '../../environments/environment';
import { PoolService, cardColors, poolColorOk } from './pool.service';

describe('PoolService', () => {
  let service: PoolService;
  let http: HttpTestingController;
  const base = `${environment.apiUrl}/api/v1`;

  beforeEach(() => {
    TestBed.configureTestingModule({ imports: [HttpClientTestingModule] });
    service = TestBed.inject(PoolService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('submits by card id, withdraws, gives and takes back medals', () => {
    service.submit('c1').subscribe();
    let req = http.expectOne(`${base}/pools/entries`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ cardId: 'c1' });
    req.flush({});

    service.withdraw('e1').subscribe();
    http.expectOne(`${base}/pools/entries/e1/withdraw`).flush({});

    service.medal('e2', 'silver').subscribe();
    req = http.expectOne(`${base}/pools/medals`);
    expect(req.request.body).toEqual({ entryId: 'e2', medal: 'silver' });
    req.flush({});

    service.clearMedal('e2').subscribe();
    req = http.expectOne(`${base}/pools/medals/clear`);
    expect(req.request.body).toEqual({ entryId: 'e2' });
    req.flush({});
  });

  it('sends X-Admin-Pin for host actions, with just a name and cap', () => {
    service.createPool('KP', 3, '4321').subscribe();
    const req = http.expectOne(`${base}/admin/pools`);
    expect(req.request.headers.get('X-Admin-Pin')).toBe('4321');
    expect(req.request.body).toEqual({ name: 'KP', maxEntriesPerUser: 3 });
    req.flush({});

    service.closePool('p1', '4321').subscribe();
    const close = http.expectOne(`${base}/admin/pools/p1/close`);
    expect(close.request.headers.get('X-Admin-Pin')).toBe('4321');
    close.flush({});
  });
});

describe('poolColorOk', () => {
  it('allows mono-colored and colorless cards', () => {
    expect(poolColorOk({ colors: ['R'], manaCost: '{1}{R}' })).toBeTrue();
    expect(poolColorOk({ colors: [], manaCost: '{4}' })).toBeTrue();
    expect(poolColorOk({ colors: ['C'], manaCost: '{2}' })).toBeTrue();
    expect(poolColorOk({ colors: [], manaCost: '{G}{G}' })).toBeTrue();
    expect(poolColorOk(null)).toBeTrue();
  });

  it('rejects multicolor cards, including hybrid costs', () => {
    expect(poolColorOk({ colors: ['W', 'U'], manaCost: '{W}{U}' })).toBeFalse();
    expect(poolColorOk({ colors: [], manaCost: '{W/U}' })).toBeFalse();
  });

  it('reads colors from the list first, then the mana cost', () => {
    expect(cardColors({ colors: ['B'], manaCost: '{R}' })).toEqual(['B']);
    expect(cardColors({ colors: [], manaCost: '{2}{U}{U}' })).toEqual(['U']);
  });
});
