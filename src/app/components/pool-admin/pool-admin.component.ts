import { Component, Input, OnInit } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { finalize } from 'rxjs';
import { PoolView } from '../../models/api.model';
import { apiErrorMessage } from '../../services/api.util';
import { POOL_DEFAULT_ENTRIES, POOL_ENTRY_CAP_MAX, PoolService } from '../../services/pool.service';

/** Host controls for the Knowledge Pool on /admin: open a pool (name and entry cap), close it. */
@Component({
  selector: 'app-pool-admin',
  templateUrl: './pool-admin.component.html',
  styleUrls: ['./pool-admin.component.scss']
})
export class PoolAdminComponent implements OnInit {
  /** The host PIN from the admin page. */
  @Input() pin = '';

  readonly capMax = POOL_ENTRY_CAP_MAX;
  readonly form = new FormGroup({
    name: new FormControl('', { nonNullable: true }),
    maxEntries: new FormControl(POOL_DEFAULT_ENTRIES, {
      nonNullable: true, validators: [Validators.min(1), Validators.max(POOL_ENTRY_CAP_MAX)]
    })
  });

  current: PoolView | null = null;
  closedResult: PoolView | null = null;
  loading = true;
  saving = false;
  error: string | null = null;
  success: string | null = null;

  constructor(private pools: PoolService) {}

  ngOnInit(): void {
    this.pools.current().pipe(finalize(() => (this.loading = false))).subscribe({
      next: pool => (this.current = pool),
      error: err => (this.error = apiErrorMessage(err, 'Could not load the Knowledge Pool.'))
    });
  }

  get entryCount(): number {
    return this.current?.entries.length ?? 0;
  }

  get madeItCount(): number {
    return this.closedResult?.entries.filter(e => e.in).length ?? 0;
  }

  createPool(): void {
    if (this.saving) {
      return;
    }
    const pin = this.pin.trim();
    const name = this.form.controls.name.value.trim();
    const maxEntries = Number(this.form.controls.maxEntries.value);
    this.clearMessages();
    if (!pin) {
      this.error = 'Enter the host PIN.';
      return;
    }
    if (!name) {
      this.error = 'Enter a name for the pool.';
      return;
    }
    if (!Number.isInteger(maxEntries) || maxEntries < 1 || maxEntries > POOL_ENTRY_CAP_MAX) {
      this.error = `Entries per player must be a whole number from 1 to ${POOL_ENTRY_CAP_MAX}.`;
      return;
    }
    this.saving = true;
    this.pools.createPool(name, maxEntries, pin).pipe(finalize(() => (this.saving = false))).subscribe({
      next: pool => {
        this.current = pool;
        this.closedResult = null;
        this.form.controls.name.setValue('');
        this.success = `"${pool.name}" is open. Players submit from Create or the Gallery and hand out medals on /pool.`;
      },
      error: err => this.handleError(err, 'Could not open the pool.')
    });
  }

  closePool(): void {
    const pool = this.current;
    if (!pool || this.saving) {
      return;
    }
    const pin = this.pin.trim();
    if (!pin) {
      this.error = 'Enter the host PIN.';
      return;
    }
    if (!window.confirm(`Close "${pool.name}"? Submissions and voting stop and the cards above the pool line become legal.`)) {
      return;
    }
    this.clearMessages();
    this.saving = true;
    this.pools.closePool(pool.id, pin).pipe(finalize(() => (this.saving = false))).subscribe({
      next: closed => {
        this.current = null;
        this.closedResult = closed;
        this.success = `"${closed.name}" is closed.`;
      },
      error: err => this.handleError(err, 'Could not close the pool.')
    });
  }

  clearMessages(): void {
    this.error = null;
    this.success = null;
  }

  private handleError(err: unknown, fallback: string): void {
    if (err instanceof HttpErrorResponse && err.status === 403) {
      this.error = 'Wrong PIN';
      return;
    }
    this.error = apiErrorMessage(err, fallback);
    if (err instanceof HttpErrorResponse && err.status === 409) {
      this.pools.current().subscribe({ next: pool => (this.current = pool), error: () => undefined });
    }
  }
}
