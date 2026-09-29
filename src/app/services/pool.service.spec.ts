import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { environment } from '../../environments/environment';
import { PoolService, cardFitsSlot, defaultPoolSlots } from './pool.service';

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

  it('submits, withdraws, gives and takes back medals, bans and unbans', () => {
    service.submit('s1', 'c1').subscribe();
    let req = http.expectOne(`${base}/pools/entries`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ slotId: 's1', cardId: 'c1' });
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

    service.ban('e3').subscribe();
    req = http.expectOne(`${base}/pools/bans`);
    expect(req.request.body).toEqual({ entryId: 'e3' });
    req.flush({});

    service.unban('e3').subscribe();
    http.expectOne(`${base}/pools/bans/clear`).flush({});
  });

  it('sends X-Admin-Pin for host actions', () => {
    service.createPool('KP', 3, defaultPoolSlots(), '4321').subscribe();
    const req = http.expectOne(`${base}/admin/pools`);
    expect(req.request.headers.get('X-Admin-Pin')).toBe('4321');
    expect(req.request.body.maxEntriesPerUser).toBe(3);
    expect(req.request.body.slots.length).toBe(16);
    req.flush({});

    service.closePool('p1', '4321').subscribe();
    const close = http.expectOne(`${base}/admin/pools/p1/close`);
    expect(close.request.headers.get('X-Admin-Pin')).toBe('4321');
    close.flush({});
  });
});

describe('cardFitsSlot', () => {
  const red = { colors: ['R'], type: 'Creature', manaCost: '{1}{R}' };
  const potOfGreen = { colors: [], type: 'Artifact', manaCost: '{0}' };

  it('matches mono colors exactly', () => {
    expect(cardFitsSlot(red, 'R', 'creature')).toBeTrue();
    expect(cardFitsSlot(red, 'G', 'any')).toBeFalse();
    expect(cardFitsSlot({ colors: ['R', 'G'], type: 'Creature' }, 'R', 'any')).toBeFalse();
  });

  it('handles multicolor, colorless and mana-cost fallback', () => {
    expect(cardFitsSlot({ colors: ['R', 'G'], type: 'Creature' }, 'multicolor', 'any')).toBeTrue();
    expect(cardFitsSlot(potOfGreen, 'colorless', 'noncreature')).toBeTrue();
    expect(cardFitsSlot({ colors: ['C'], type: 'Land' }, 'colorless', 'land')).toBeTrue();
    expect(cardFitsSlot({ colors: [], type: 'Instant', manaCost: '{2}{U}' }, 'U', 'noncreature')).toBeTrue();
  });

  it('applies type rules', () => {
    expect(cardFitsSlot(red, 'R', 'noncreature')).toBeFalse();
    expect(cardFitsSlot({ colors: ['W'], type: 'Artifact Creature' }, 'W', 'creature')).toBeTrue();
    expect(cardFitsSlot({ colors: [], type: 'Land' }, 'any', 'noncreature')).toBeFalse();
    expect(cardFitsSlot(null, 'any', 'any')).toBeTrue();
  });
});

describe('defaultPoolSlots', () => {
  it('has 16 slots: 10 mono-color, 2 multicolor, 2 colorless, a land and a wild card', () => {
    const slots = defaultPoolSlots();
    expect(slots.length).toBe(16);
    expect(slots.filter(s => ['W', 'U', 'B', 'R', 'G'].includes(s.colorRule)).length).toBe(10);
    expect(slots.filter(s => s.colorRule === 'multicolor').length).toBe(2);
    expect(slots.filter(s => s.colorRule === 'colorless').length).toBe(2);
    expect(slots.filter(s => s.typeRule === 'land').length).toBe(1);
  });
});
