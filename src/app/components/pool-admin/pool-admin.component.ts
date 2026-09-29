import { Component, Input, OnInit } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { FormArray, FormControl, FormGroup, Validators } from '@angular/forms';
import { finalize } from 'rxjs';
import { PoolColorRule, PoolSlotSpec, PoolTypeRule, PoolView } from '../../models/api.model';
import { apiErrorMessage } from '../../services/api.util';
import {
  POOL_COLOR_RULES, POOL_TYPE_RULES, PoolService, defaultPoolSlots
} from '../../services/pool.service';

type SlotForm = FormGroup<{
  label: FormControl<string>;
  colorRule: FormControl<PoolColorRule>;
  typeRule: FormControl<PoolTypeRule>;
}>;

/** Host controls for the Knowledge Pool on /admin: open a pool with its slots, close it. */
@Component({
  selector: 'app-pool-admin',
  templateUrl: './pool-admin.component.html',
  styleUrls: ['./pool-admin.component.scss']
})
export class PoolAdminComponent implements OnInit {
  /** The host PIN from the admin page. */
  @Input() pin = '';

  readonly colorRules = POOL_COLOR_RULES;
  readonly typeRules = POOL_TYPE_RULES;
  readonly slots = new FormArray<SlotForm>([]);
  readonly form = new FormGroup({
    name: new FormControl('', { nonNullable: true }),
    maxEntries: new FormControl(4, { nonNullable: true, validators: [Validators.min(1), Validators.max(40)] }),
    slots: this.slots
  });

  current: PoolView | null = null;
  closedResult: PoolView | null = null;
  loading = true;
  saving = false;
  error: string | null = null;
  success: string | null = null;

  constructor(private pools: PoolService) {}

  ngOnInit(): void {
    this.resetSlots();
    this.pools.current().pipe(finalize(() => (this.loading = false))).subscribe({
      next: pool => (this.current = pool),
      error: err => (this.error = apiErrorMessage(err, 'Could not load the Knowledge Pool.'))
    });
  }

  get entryCount(): number {
    return this.current?.slots.reduce((n, s) => n + s.entries.length, 0) ?? 0;
  }

  get winnerCount(): number {
    return this.closedResult?.slots.filter(s => s.entries.some(e => e.leader || e.tied)).length ?? 0;
  }

  resetSlots(): void {
    this.slots.clear();
    defaultPoolSlots().forEach(slot => this.slots.push(this.slotForm(slot)));
  }

  addSlot(): void {
    this.slots.push(this.slotForm({ label: '', colorRule: 'any', typeRule: 'any' }));
  }

  removeSlot(index: number): void {
    this.slots.removeAt(index);
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
    if (!Number.isInteger(maxEntries) || maxEntries < 1 || maxEntries > 40) {
      this.error = 'Submissions per player must be a whole number from 1 to 40.';
      return;
    }
    if (!this.slots.length) {
      this.error = 'Add at least one slot.';
      return;
    }
    const slots: PoolSlotSpec[] = this.slots.controls.map(c => ({ ...c.getRawValue(), label: c.controls.label.value.trim() }));
    this.saving = true;
    this.pools.createPool(name, maxEntries, slots, pin).pipe(finalize(() => (this.saving = false))).subscribe({
      next: pool => {
        this.current = pool;
        this.closedResult = null;
        this.form.controls.name.setValue('');
        this.success = `"${pool.name}" is open with ${pool.slots.length} slots. Players can submit and hand out medals on /pool.`;
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
    if (!window.confirm(`Close "${pool.name}"? Submissions and voting stop and the winners become legal.`)) {
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

  private slotForm(slot: PoolSlotSpec): SlotForm {
    return new FormGroup({
      label: new FormControl(slot.label, { nonNullable: true }),
      colorRule: new FormControl(slot.colorRule, { nonNullable: true }),
      typeRule: new FormControl(slot.typeRule, { nonNullable: true })
    });
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
