import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { finalize } from 'rxjs';
import { apiErrorMessage } from '../../services/api.util';
import { UserService } from '../../services/user.service';

/** Same rule as the server (§3): trimmed, 1-24 chars of [A-Za-z0-9 _-]. */
export const USERNAME_PATTERN = /^[A-Za-z0-9 _-]{1,24}$/;

@Component({
  selector: 'app-login-page',
  templateUrl: './login-page.component.html',
  styleUrls: ['./login-page.component.scss']
})
export class LoginPageComponent implements OnInit {
  readonly username = new FormControl('', { nonNullable: true, validators: [Validators.required] });
  /** [formGroup] makes (ngSubmit) work and prevents a native page submit on Enter. */
  readonly form = new FormGroup({ username: this.username });
  submitting = false;
  error: string | null = null;

  constructor(
    private users: UserService,
    private router: Router,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    if (this.users.currentUser()) {
      this.router.navigateByUrl(this.returnUrl());
    }
  }

  submit(): void {
    if (this.submitting) {
      return;
    }
    const name = this.username.value.trim();
    if (!name) {
      this.error = 'Enter a username.';
      return;
    }
    if (!USERNAME_PATTERN.test(name)) {
      this.error = 'Use 1-24 letters, numbers, spaces, _ or -.';
      return;
    }

    this.error = null;
    this.submitting = true;
    this.users.login(name)
      .pipe(finalize(() => (this.submitting = false)))
      .subscribe({
        next: () => this.router.navigateByUrl(this.returnUrl()),
        error: err => (this.error = apiErrorMessage(err, 'Login failed. Please try again.'))
      });
  }

  private returnUrl(): string {
    const target = this.route.snapshot.queryParamMap.get('returnUrl');
    // Only allow in-app paths.
    return target && target.startsWith('/') && !target.startsWith('//') && !target.startsWith('/login')
      ? target
      : '/create';
  }
}
