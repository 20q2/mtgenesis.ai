import { routes } from './app-routing.module';
import { authGuard } from './guards/auth.guard';

describe('app routes', () => {
  const byPath = (path: string) => routes.find(r => r.path === path);

  it('has every spec §7 route plus the /events list', () => {
    for (const path of ['login', 'create', 'set', 'gallery', 'vote', 'events', 'events/:id', 'admin']) {
      expect(byPath(path)).withContext(path).toBeDefined();
    }
  });

  it("redirects '' to create", () => {
    expect(byPath('')).toEqual(jasmine.objectContaining({ redirectTo: 'create', pathMatch: 'full' }));
  });

  it('guards every page except /login', () => {
    expect(byPath('login')!.canActivate).toBeUndefined();
    for (const path of ['create', 'set', 'gallery', 'vote', 'events', 'events/:id', 'admin']) {
      expect(byPath(path)!.canActivate).withContext(path).toEqual([authGuard]);
    }
  });
});
