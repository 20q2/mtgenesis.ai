import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { Observable } from 'rxjs';
import { User } from './models/api.model';
import { UserService } from './services/user.service';

/** App shell: header, nav, username + Log out, and the routed page. */
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'MTGenesis.AI';
  readonly user$: Observable<User | null>;

  readonly navLinks = [
    { path: '/create', label: 'Create' },
    { path: '/set', label: 'Commander Set' },
    { path: '/gallery', label: 'Gallery' },
    { path: '/vote', label: 'Vote' }
  ];

  constructor(private users: UserService, private router: Router) {
    this.user$ = users.user$;
  }

  logout(): void {
    this.users.logout();
    this.router.navigate(['/login']);
  }
}
