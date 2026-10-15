import { Component, OnDestroy, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { Observable, Subscription } from 'rxjs';
import { User } from './models/api.model';
import { ADMIN_PIN_STORAGE_KEY } from './pages/admin-page/admin-page.component';
import { UserService } from './services/user.service';

/**
 * App shell: header, nav, avatar + username + Log out, and the routed page. It also hosts the
 * profile picture dialog: the avatar opens it, and it offers itself after a login (or once per
 * device to players already logged in) when there's no picture yet.
 */
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'MTGenesis.AI';
  readonly user$: Observable<User | null>;
  /** The profile picture dialog is open; prompted: the app asked, rather than the player. */
  pictureOpen = false;
  picturePrompted = false;
  private sub = new Subscription();

  readonly navLinks = [
    { path: '/create', label: 'Create', short: 'Create', icon: 'auto_awesome' },
    { path: '/set', label: 'Commanders', short: 'Commanders', icon: 'style' },
    { path: '/gallery', label: 'Gallery', short: 'Gallery', icon: 'collections' },
    { path: '/vote', label: 'Vote', short: 'Vote', icon: 'how_to_vote' },
    { path: '/pool', label: 'Knowledge Pool', short: 'Pool', icon: 'auto_stories' }
  ];

  constructor(private users: UserService, private router: Router) {
    this.user$ = users.user$;
  }

  ngOnInit(): void {
    this.sub.add(this.users.loggedIn$.subscribe(user => {
      if (!user.avatarUrl) {
        this.openPicture(true);
      }
    }));
    // Already logged in: pick up a picture set elsewhere, and nudge once if there's none.
    const current = this.users.currentUser();
    if (current) {
      this.sub.add(this.users.refresh().subscribe({
        next: user => {
          if (!user.avatarUrl && !this.users.wasPrompted(user.id) && !this.pictureOpen) {
            this.openPicture(true);
          }
        },
        error: () => undefined // offline or a stale login: the interceptor deals with 401s
      }));
    }
  }

  ngOnDestroy(): void {
    this.sub.unsubscribe();
  }

  openPicture(prompted = false): void {
    this.picturePrompted = prompted;
    this.pictureOpen = true;
  }

  closePicture(): void {
    const user = this.users.currentUser();
    if (user) {
      this.users.markPrompted(user.id);
    }
    this.pictureOpen = false;
  }

  /**
   * Host tools is only linked in a tab that already holds the host PIN (the host
   * opens /admin by URL once); attendees never see it.
   */
  get showHostTools(): boolean {
    try {
      return !!sessionStorage.getItem(ADMIN_PIN_STORAGE_KEY);
    } catch {
      return false;
    }
  }

  logout(): void {
    this.users.logout();
    this.router.navigate(['/login']);
  }
}
