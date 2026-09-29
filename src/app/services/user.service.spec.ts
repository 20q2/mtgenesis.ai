import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { environment } from '../../environments/environment';
import { User } from '../models/api.model';
import { USER_STORAGE_KEY, UserService } from './user.service';

describe('UserService', () => {
  let service: UserService;
  let http: HttpTestingController;
  const alice: User = { id: 'u-1', username: 'Alice' };

  beforeEach(() => {
    localStorage.removeItem(USER_STORAGE_KEY);
    TestBed.configureTestingModule({ imports: [HttpClientTestingModule] });
    service = TestBed.inject(UserService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  afterAll(() => localStorage.removeItem(USER_STORAGE_KEY));

  it('uses the mtgenesis.user storage key', () => {
    expect(USER_STORAGE_KEY).toBe('mtgenesis.user');
  });

  it('login POSTs the username and stores the returned user', () => {
    let result: User | undefined;
    service.login('Alice').subscribe(u => (result = u));

    const req = http.expectOne(`${environment.apiUrl}/api/v1/users/login`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ username: 'Alice' });
    req.flush(alice);

    expect(result).toEqual(alice);
    expect(service.currentUser()).toEqual(alice);
    expect(JSON.parse(localStorage.getItem(USER_STORAGE_KEY)!)).toEqual(alice);
  });

  it('login trims the username before sending', () => {
    service.login('  Alice  ').subscribe();
    const req = http.expectOne(`${environment.apiUrl}/api/v1/users/login`);
    expect(req.request.body).toEqual({ username: 'Alice' });
    req.flush(alice);
  });

  it('currentUser returns null when nothing is stored', () => {
    expect(service.currentUser()).toBeNull();
  });

  it('currentUser returns null for corrupt storage instead of throwing', () => {
    localStorage.setItem(USER_STORAGE_KEY, '{not json');
    expect(service.currentUser()).toBeNull();
  });

  it('logout clears the stored user', () => {
    service.login('Alice').subscribe();
    http.expectOne(`${environment.apiUrl}/api/v1/users/login`).flush(alice);

    service.logout();

    expect(service.currentUser()).toBeNull();
    expect(localStorage.getItem(USER_STORAGE_KEY)).toBeNull();
  });

  it('user$ emits on login and logout', () => {
    const seen: (User | null)[] = [];
    service.user$.subscribe(u => seen.push(u));
    service.login('Alice').subscribe();
    http.expectOne(`${environment.apiUrl}/api/v1/users/login`).flush(alice);
    service.logout();
    expect(seen).toEqual([null, alice, null]);
  });

  it('survives localStorage throwing', () => {
    spyOn(Storage.prototype, 'getItem').and.throwError('SecurityError');
    spyOn(Storage.prototype, 'setItem').and.throwError('SecurityError');
    spyOn(Storage.prototype, 'removeItem').and.throwError('SecurityError');

    expect(service.currentUser()).toBeNull();
    service.login('Alice').subscribe();
    http.expectOne(`${environment.apiUrl}/api/v1/users/login`).flush(alice);
    // Falls back to the in-memory copy for this session.
    expect(service.currentUser()).toEqual(alice);
    expect(() => service.logout()).not.toThrow();
    expect(service.currentUser()).toBeNull();
  });
});
