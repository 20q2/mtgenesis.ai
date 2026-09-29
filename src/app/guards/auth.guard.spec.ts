import { TestBed } from '@angular/core/testing';
import { ActivatedRouteSnapshot, Router, RouterStateSnapshot, UrlTree } from '@angular/router';
import { RouterTestingModule } from '@angular/router/testing';
import { UserService } from '../services/user.service';
import { authGuard } from './auth.guard';

describe('authGuard', () => {
  let users: jasmine.SpyObj<UserService>;

  beforeEach(() => {
    users = jasmine.createSpyObj<UserService>('UserService', ['currentUser']);
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      providers: [{ provide: UserService, useValue: users }]
    });
  });

  function run(url: string) {
    return TestBed.runInInjectionContext(() =>
      authGuard({} as ActivatedRouteSnapshot, { url } as RouterStateSnapshot));
  }

  it('returns a UrlTree to /login when no user is stored', () => {
    users.currentUser.and.returnValue(null);
    const result = run('/vote');
    expect(result instanceof UrlTree).toBeTrue();
    const tree = result as UrlTree;
    expect(tree.root.children['primary'].segments.map(s => s.path)).toEqual(['login']);
    expect(tree.queryParams['returnUrl']).toBe('/vote');
    expect(TestBed.inject(Router).serializeUrl(tree)).toMatch(/^\/login/);
  });

  it('returns true when a user is stored', () => {
    users.currentUser.and.returnValue({ id: 'u-1', username: 'Alice' });
    expect(run('/vote')).toBeTrue();
  });
});
