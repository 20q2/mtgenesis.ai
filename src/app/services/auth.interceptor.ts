import { Injectable } from '@angular/core';
import {
  HttpErrorResponse, HttpEvent, HttpHandler, HttpInterceptor, HttpRequest
} from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, catchError, throwError } from 'rxjs';
import { isApiRequest } from './api.util';
import { UserService } from './user.service';

/**
 * For requests to environment.apiUrl:
 * - adds `X-User-Id` (when logged in) and `ngrok-skip-browser-warning`;
 * - on a 401 (stale or unknown user, e.g. after the data dir was wiped between nights)
 *   clears the stored user and sends the browser back to /login.
 */
@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  constructor(private users: UserService, private router: Router) {}

  intercept(req: HttpRequest<unknown>, next: HttpHandler): Observable<HttpEvent<unknown>> {
    if (!isApiRequest(req.url)) {
      return next.handle(req);
    }

    const user = this.users.currentUser();
    const setHeaders: Record<string, string> = { 'ngrok-skip-browser-warning': 'true' };
    if (user) {
      setHeaders['X-User-Id'] = user.id;
    }

    return next.handle(req.clone({ setHeaders })).pipe(
      catchError((err: unknown) => {
        if (err instanceof HttpErrorResponse && err.status === 401) {
          this.users.logout();
          this.router.navigate(['/login']);
        }
        return throwError(() => err);
      })
    );
  }
}
