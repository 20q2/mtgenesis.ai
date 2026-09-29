import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { Router } from '@angular/router';
import { RouterTestingModule } from '@angular/router/testing';
import { BehaviorSubject } from 'rxjs';
import { AppComponent } from './app.component';
import { User } from './models/api.model';
import { UserService } from './services/user.service';
import { ADMIN_PIN_STORAGE_KEY } from './pages/admin-page/admin-page.component';

describe('AppComponent (shell)', () => {
  let fixture: ComponentFixture<AppComponent>;
  let user$: BehaviorSubject<User | null>;
  let users: jasmine.SpyObj<UserService>;

  beforeEach(() => {
    sessionStorage.removeItem(ADMIN_PIN_STORAGE_KEY);
    user$ = new BehaviorSubject<User | null>({ id: 'u-1', username: 'Alice' });
    users = jasmine.createSpyObj<UserService>('UserService', ['logout', 'currentUser'], { user$ });
    users.logout.and.callFake(() => user$.next(null));
    TestBed.configureTestingModule({
      imports: [RouterTestingModule, HttpClientTestingModule],
      declarations: [AppComponent],
      providers: [{ provide: UserService, useValue: users }],
      schemas: [NO_ERRORS_SCHEMA]
    });
    fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
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
    expect(links).toEqual(['Create', 'Commander Set', 'Gallery', 'Vote', 'Knowledge Pool']);
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
