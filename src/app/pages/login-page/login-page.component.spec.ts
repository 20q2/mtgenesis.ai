import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { ReactiveFormsModule } from '@angular/forms';
import { ActivatedRoute, Router, convertToParamMap } from '@angular/router';
import { of, throwError } from 'rxjs';
import { UserService } from '../../services/user.service';
import { LoginPageComponent } from './login-page.component';

describe('LoginPageComponent', () => {
  let fixture: ComponentFixture<LoginPageComponent>;
  let component: LoginPageComponent;
  let users: jasmine.SpyObj<UserService>;
  let router: jasmine.SpyObj<Router>;

  function setup(returnUrl: string | null = null) {
    users = jasmine.createSpyObj<UserService>('UserService', ['login', 'currentUser']);
    users.currentUser.and.returnValue(null);
    router = jasmine.createSpyObj<Router>('Router', ['navigateByUrl']);
    router.navigateByUrl.and.returnValue(Promise.resolve(true));
    TestBed.configureTestingModule({
      imports: [ReactiveFormsModule],
      declarations: [LoginPageComponent],
      providers: [
        { provide: UserService, useValue: users },
        { provide: Router, useValue: router },
        {
          provide: ActivatedRoute,
          useValue: { snapshot: { queryParamMap: convertToParamMap(returnUrl ? { returnUrl } : {}) } }
        }
      ]
    });
    fixture = TestBed.createComponent(LoginPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  it('logs in and goes to /create by default', () => {
    setup();
    users.login.and.returnValue(of({ id: 'u-1', username: 'Alice' }));
    component.username.setValue(' Alice ');
    component.submit();
    expect(users.login).toHaveBeenCalledWith('Alice');
    expect(router.navigateByUrl).toHaveBeenCalledWith('/create');
  });

  it('returns to the page the guard bounced from', () => {
    setup('/vote');
    users.login.and.returnValue(of({ id: 'u-1', username: 'Alice' }));
    component.username.setValue('Alice');
    component.submit();
    expect(router.navigateByUrl).toHaveBeenCalledWith('/vote');
  });

  it('rejects blank and invalid names without calling the server', () => {
    setup();
    component.username.setValue('   ');
    component.submit();
    component.username.setValue('bad!name');
    component.submit();
    component.username.setValue('x'.repeat(25));
    component.submit();
    expect(users.login).not.toHaveBeenCalled();
  });

  it('shows the server error text', () => {
    setup();
    users.login.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 400, error: { error: 'Username may only contain letters, digits, spaces, _ and -' }
    })));
    component.username.setValue('Alice');
    component.submit();
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Username may only contain');
    expect(component.submitting).toBeFalse();
  });
});
