import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { UserService } from '../services/user.service';

/** Sends visitors without a stored user to /login, remembering where they were going. */
export const authGuard: CanActivateFn = (_route, state) => {
  if (inject(UserService).currentUser()) {
    return true;
  }
  return inject(Router).createUrlTree(['/login'], { queryParams: { returnUrl: state.url } });
};
