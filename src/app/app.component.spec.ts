import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { Router } from '@angular/router';
import { RouterTestingModule } from '@angular/router/testing';
import { BehaviorSubject, Subject, of } from 'rxjs';
import { AppComponent } from './app.component';
import { User } from './models/api.model';
import { UserService } from './services/user.service';
import { ADMIN_PIN_STORAGE_KEY } from './pages/admin-page/admin-page.component';

describe('AppComponent (shell)', () => {
  let fixture: ComponentFixture<AppComponent>;
  let user$: BehaviorSubject<User | null>;
  let users: jasmine.SpyObj<UserService>;
  let loggedIn$: Subject<User>;
  const alice: User = { id: 'u-1', username: 'Alice', avatarUrl: null };

  function configure(): void {
    sessionStorage.removeItem(ADMIN_PIN_STORAGE_KEY);
    user$ = new BehaviorSubject<User | null>(alice);
    loggedIn$ = new Subject<User>();
    users = jasmine.createSpyObj<UserService>('UserService',
      ['logout', 'currentUser', 'refresh', 'wasPrompted', 'markPrompted'], { user$, loggedIn$ });
    users.logout.and.callFake(() => user$.next(null));
    users.currentUser.and.returnValue(null);
    users.refresh.and.returnValue(of(alice));
    users.wasPrompted.and.returnValue(false);
    TestBed.configureTestingModule({
      imports: [RouterTestingModule, HttpClientTestingModule],
      declarations: [AppComponent],
      providers: [{ provide: UserService, useValue: users }],
      schemas: [NO_ERRORS_SCHEMA]
    });
  }

  function create(): void {
    fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
  }

  beforeEach(() => {
    configure();
    create();
  });

  const picture = () => fixture.nativeElement.querySelector('app-profile-picture');

  it('offers a profile picture right after a login without one', () => {
    expect(picture()).toBeNull();
    loggedIn$.next(alice);
    fixture.detectChanges();
    expect(picture()).not.toBeNull();
    expect(fixture.componentInstance.picturePrompted).toBeTrue();
  });

  it('does not ask after a login that already has a picture', () => {
    loggedIn$.next({ ...alice, avatarUrl: '/api/v1/media/avatars/a.jpg' });
    fixture.detectChanges();
    expect(picture()).toBeNull();
  });

  it('closing the dialog remembers the ask for this player on this device', () => {
    users.currentUser.and.returnValue(alice);
    fixture.componentInstance.openPicture(true);
    fixture.componentInstance.closePicture();
    fixture.detectChanges();
    expect(users.markPrompted).toHaveBeenCalledWith('u-1');
    expect(picture()).toBeNull();
  });

  it('the avatar button opens the dialog (not as a prompt)', () => {
    (fixture.nativeElement.querySelector('.avatar-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(picture()).not.toBeNull();
    expect(fixture.componentInstance.picturePrompted).toBeFalse();
  });

  it('should create the app', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it(`should have as title 'MTGenesis.AI'`, () => {
    expect(fixture.componentInstance.title).toEqual('MTGenesis.AI');
  });

  it('shows the nav links and the username when logged in', () => {
    const el: HTMLElement = fixture.nativeElement;
    const links = Array.from(el.querySelectorAll('nav.nav-links a')).map(a => a.textContent!.trim());
    expect(links).toEqual(['Create', 'Commanders', 'Gallery', 'Vote', 'Knowledge Pool']);
    expect(el.querySelector('.username')!.textContent).toContain('Alice');
  });

  it('shows the queue badge in the nav', () => {
    expect(fixture.nativeElement.querySelector('.nav-bar app-queue-badge')).not.toBeNull();
  });

  it('hides the nav when logged out', () => {
    user$.next(null);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('nav.nav-links')).toBeNull();
    expect(fixture.nativeElement.querySelector('nav.footer-links')).toBeNull();
  });

  it('Log out clears the user and goes to /login', () => {
    const router = TestBed.inject(Router);
    spyOn(router, 'navigate').and.returnValue(Promise.resolve(true));
    (fixture.nativeElement.querySelector('.logout') as HTMLElement).click();
    expect(users.logout).toHaveBeenCalled();
    expect(router.navigate).toHaveBeenCalledWith(['/login']);
  });

  it('renders a router outlet', () => {
    expect(fixture.nativeElement.querySelector('router-outlet')).not.toBeNull();
  });

  afterEach(() => sessionStorage.removeItem(ADMIN_PIN_STORAGE_KEY));

  describe('already logged in at startup', () => {
    beforeEach(() => {
      fixture.destroy();
      TestBed.resetTestingModule();
      configure();
      users.currentUser.and.returnValue(alice);
    });

    it('refreshes me and nudges once when there is no picture', () => {
      create();
      expect(users.refresh).toHaveBeenCalled();
      expect(picture()).not.toBeNull();
    });

    it('does not nudge again once the player has seen it on this device', () => {
      users.wasPrompted.and.returnValue(true);
      create();
      expect(picture()).toBeNull();
    });

    it('does not nudge a player who has a picture', () => {
      users.refresh.and.returnValue(of({ ...alice, avatarUrl: '/api/v1/media/avatars/a.jpg' }));
      create();
      expect(picture()).toBeNull();
    });
  });

  it('does not show attendees the Host tools link (the /admin route still works by URL)', () => {
    const el: HTMLElement = fixture.nativeElement;
    const links = Array.from(el.querySelectorAll('nav.footer-links a')).map(a => a.textContent!.trim());
    expect(links).toEqual(['Past events']);
    expect(el.querySelector('a[href="/admin"]')).toBeNull();
  });

  it('shows Host tools in a tab that already holds the host PIN', () => {
    sessionStorage.setItem(ADMIN_PIN_STORAGE_KEY, 'k7#Qm2vX9p');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('nav.footer-links a[href="/admin"]')).not.toBeNull();
  });
});
