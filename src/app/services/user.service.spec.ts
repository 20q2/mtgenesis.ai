import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { environment } from '../../environments/environment';
import { User } from '../models/api.model';
import { AVATAR_PROMPT_KEY_PREFIX, USER_STORAGE_KEY, UserService } from './user.service';

describe('UserService', () => {
  let service: UserService;
  let http: HttpTestingController;
  const alice: User = { id: 'u-1', username: 'Alice', avatarUrl: null };
  const me = `${environment.apiUrl}/api/v1/me`;

  beforeEach(() => {
    localStorage.removeItem(USER_STORAGE_KEY);
    localStorage.removeItem(AVATAR_PROMPT_KEY_PREFIX + 'u-1');
    TestBed.configureTestingModule({ imports: [HttpClientTestingModule] });
    service = TestBed.inject(UserService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  afterAll(() => {
    localStorage.removeItem(USER_STORAGE_KEY);
    localStorage.removeItem(AVATAR_PROMPT_KEY_PREFIX + 'u-1');
  });

  function loginAlice(): void {
    service.login('Alice').subscribe();
    http.expectOne(`${environment.apiUrl}/api/v1/users/login`).flush(alice);
  }

  it('reads a user stored before profile pictures as having none', () => {
    localStorage.setItem(USER_STORAGE_KEY, JSON.stringify({ id: 'u-1', username: 'Alice' }));
    expect(service.currentUser()).toEqual(alice);
  });

  it('loggedIn$ emits on each login', () => {
    const seen: User[] = [];
    service.loggedIn$.subscribe(u => seen.push(u));
    loginAlice();
    expect(seen).toEqual([alice]);
  });

  it('refresh GETs /me and stores the picture', () => {
    loginAlice();
    const withPic = { ...alice, avatarUrl: '/api/v1/media/avatars/x.jpg' };
    service.refresh().subscribe();
    http.expectOne(me).flush(withPic);
    expect(service.currentUser()).toEqual(withPic);
  });

  it('refresh does not store a user after logging out', () => {
    loginAlice();
    service.refresh().subscribe();
    service.logout();
    http.expectOne(me).flush(alice);
    expect(service.currentUser()).toBeNull();
  });

  it('setAvatar PUTs the picture as the body and stores the new URL; removeAvatar DELETEs it', () => {
    loginAlice();
    const picture = new Blob(['jpeg'], { type: 'image/jpeg' });
    service.setAvatar(picture).subscribe();
    const put = http.expectOne(`${me}/avatar`);
    expect(put.request.method).toBe('PUT');
    expect(put.request.body).toBe(picture);
    expect(put.request.headers.get('Content-Type')).toBe('image/jpeg');
    put.flush({ ...alice, avatarUrl: '/api/v1/media/avatars/y.jpg' });
    expect(service.currentUser()?.avatarUrl).toBe('/api/v1/media/avatars/y.jpg');

    service.removeAvatar().subscribe();
    const del = http.expectOne(`${me}/avatar`);
    expect(del.request.method).toBe('DELETE');
    del.flush(alice);
    expect(service.currentUser()?.avatarUrl).toBeNull();
  });

  it('remembers per user that the picture prompt was shown', () => {
    expect(service.wasPrompted('u-1')).toBeFalse();
    service.markPrompted('u-1');
    expect(service.wasPrompted('u-1')).toBeTrue();
    expect(service.wasPrompted('u-2')).toBeFalse();
    expect(localStorage.getItem(AVATAR_PROMPT_KEY_PREFIX + 'u-1')).toBe('1');
  });

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
