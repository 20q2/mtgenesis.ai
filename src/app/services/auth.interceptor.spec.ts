import { TestBed } from '@angular/core/testing';
import { HTTP_INTERCEPTORS, HttpClient, HttpErrorResponse } from '@angular/common/http';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { Router } from '@angular/router';
import { environment } from '../../environments/environment';
import { AuthInterceptor } from './auth.interceptor';
import { UserService } from './user.service';

describe('AuthInterceptor', () => {
  let http: HttpClient;
  let backend: HttpTestingController;
  let users: jasmine.SpyObj<UserService>;
  let router: jasmine.SpyObj<Router>;
  const api = `${environment.apiUrl}/api/v1/me/cards`;

  beforeEach(() => {
    users = jasmine.createSpyObj<UserService>('UserService', ['currentUser', 'logout']);
    router = jasmine.createSpyObj<Router>('Router', ['navigate']);
    router.navigate.and.returnValue(Promise.resolve(true));
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [
        { provide: HTTP_INTERCEPTORS, useClass: AuthInterceptor, multi: true },
        { provide: UserService, useValue: users },
        { provide: Router, useValue: router }
      ]
    });
    http = TestBed.inject(HttpClient);
    backend = TestBed.inject(HttpTestingController);
  });

  afterEach(() => backend.verify());

  it('sets X-User-Id and the ngrok header on API requests', () => {
    users.currentUser.and.returnValue({ id: 'u-1', username: 'Alice' });
    http.get(api).subscribe();
    const req = backend.expectOne(api);
    expect(req.request.headers.get('X-User-Id')).toBe('u-1');
    expect(req.request.headers.get('ngrok-skip-browser-warning')).toBe('true');
    req.flush([]);
  });

  it('omits X-User-Id when nobody is logged in', () => {
    users.currentUser.and.returnValue(null);
    http.get(api).subscribe();
    const req = backend.expectOne(api);
    expect(req.request.headers.has('X-User-Id')).toBeFalse();
    expect(req.request.headers.get('ngrok-skip-browser-warning')).toBe('true');
    req.flush([]);
  });

  it('leaves non-API requests alone', () => {
    users.currentUser.and.returnValue({ id: 'u-1', username: 'Alice' });
    http.get('/assets/generation-messages.json').subscribe();
    const req = backend.expectOne('/assets/generation-messages.json');
    expect(req.request.headers.has('X-User-Id')).toBeFalse();
    expect(req.request.headers.has('ngrok-skip-browser-warning')).toBeFalse();
    req.flush({});
  });

  it('on a 401 clears the stored user and navigates to /login', () => {
    users.currentUser.and.returnValue({ id: 'stale', username: 'Ghost' });
    let error: HttpErrorResponse | undefined;
    http.get(api).subscribe({ error: e => (error = e) });

    backend.expectOne(api).flush({ error: 'Unknown user' }, { status: 401, statusText: 'Unauthorized' });

    expect(users.logout).toHaveBeenCalled();
    expect(router.navigate).toHaveBeenCalledWith(['/login']);
    expect(error?.status).toBe(401);
  });

  it('does not log out on other errors', () => {
    users.currentUser.and.returnValue({ id: 'u-1', username: 'Alice' });
    http.get(api).subscribe({ error: () => undefined });
    backend.expectOne(api).flush({ error: 'nope' }, { status: 409, statusText: 'Conflict' });
    expect(users.logout).not.toHaveBeenCalled();
    expect(router.navigate).not.toHaveBeenCalled();
  });
});
