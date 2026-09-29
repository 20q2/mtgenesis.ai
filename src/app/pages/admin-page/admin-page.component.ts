import { Component, OnDestroy, OnInit } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { FormControl, FormGroup } from '@angular/forms';
import { Subscription, finalize } from 'rxjs';
import { EventSummary, EventView } from '../../models/api.model';
import { apiErrorMessage } from '../../services/api.util';
import { EventService } from '../../services/event.service';

export const ADMIN_PIN_STORAGE_KEY = 'mtgenesis.adminPin';

/** /admin: host PIN (sessionStorage), create/close the event, history (spec §7). */
@Component({
  selector: 'app-admin-page',
  templateUrl: './admin-page.component.html',
  styleUrls: ['./admin-page.component.scss']
})
export class AdminPageComponent implements OnInit, OnDestroy {
  readonly pin = new FormControl('', { nonNullable: true });
  readonly eventName = new FormControl('', { nonNullable: true });
  /** [formGroup] makes (ngSubmit) work and prevents a native page submit on Enter. */
  readonly createForm = new FormGroup({ name: this.eventName });

  current: EventView | null = null;
  history: EventSummary[] = [];
  /** Set when GET /events fails, so the page never claims "No events yet." on an error. */
  historyError: string | null = null;
  /** The event just closed from this page, with its winners. */
  closedResult: EventView | null = null;

  loading = true;
  creating = false;
  closing = false;
  error: string | null = null;
  success: string | null = null;

  private pinSub?: Subscription;

  constructor(private events: EventService) {}

  ngOnInit(): void {
    this.pin.setValue(this.readPin());
    this.pinSub = this.pin.valueChanges.subscribe(value => this.writePin(value));
    this.loadCurrent();
    this.loadHistory();
  }

  ngOnDestroy(): void {
    this.pinSub?.unsubscribe();
  }

  createEvent(): void {
    if (this.creating) {
      return;
    }
    const pin = this.requirePin();
    const name = this.eventName.value.trim();
    if (!pin) {
      return;
    }
    if (!name) {
      this.error = 'Enter a name for the event.';
      return;
    }
    this.clearMessages();
    this.creating = true;
    this.events.createEvent(name, pin).pipe(
      finalize(() => (this.creating = false))
    ).subscribe({
      next: event => {
        this.current = event;
        this.closedResult = null;
        this.eventName.setValue('');
        this.success = `"${event.name}" is open. Players can lock in their sets and vote.`;
        this.loadHistory();
      },
      error: err => this.handleAdminError(err, 'Could not create the event.')
    });
  }

  closeCurrent(): void {
    const event = this.current;
    if (!event || this.closing) {
      return;
    }
    const pin = this.requirePin();
    if (!pin) {
      return;
    }
    if (!window.confirm(`Close "${event.name}"? Voting stops and the winners become final.`)) {
      return;
    }
    this.clearMessages();
    this.closing = true;
    this.events.closeEvent(event.id, pin).pipe(
      finalize(() => (this.closing = false))
    ).subscribe({
      next: closed => {
        this.current = null;
        this.closedResult = closed;
        this.success = `"${closed.name}" is closed.`;
        this.loadHistory();
      },
      error: err => this.handleAdminError(err, 'Could not close the event.')
    });
  }

  clearMessages(): void {
    this.error = null;
    this.success = null;
  }

  private loadCurrent(): void {
    this.events.current().pipe(
      finalize(() => (this.loading = false))
    ).subscribe({
      next: event => (this.current = event),
      error: err => (this.error = apiErrorMessage(err, 'Could not load the current event.'))
    });
  }

  private loadHistory(): void {
    this.events.list().subscribe({
      next: list => {
        this.history = list;
        this.historyError = null;
      },
      error: err => (this.historyError = apiErrorMessage(err, 'Could not load the event history.'))
    });
  }

  private requirePin(): string | null {
    const pin = this.pin.value.trim();
    if (!pin) {
      this.error = 'Enter the host PIN.';
      this.success = null;
      return null;
    }
    return pin;
  }

  private handleAdminError(err: unknown, fallback: string): void {
    if (err instanceof HttpErrorResponse && err.status === 403) {
      this.error = 'Wrong PIN';
      return;
    }
    this.error = apiErrorMessage(err, fallback);
    if (err instanceof HttpErrorResponse && err.status === 409) {
      // Someone else opened/closed an event: resync what this page shows.
      this.loadCurrent();
      this.loadHistory();
    }
  }

  private readPin(): string {
    try {
      return sessionStorage.getItem(ADMIN_PIN_STORAGE_KEY) ?? '';
    } catch {
      return '';
    }
  }

  private writePin(value: string): void {
    try {
      if (value) {
        sessionStorage.setItem(ADMIN_PIN_STORAGE_KEY, value);
      } else {
        sessionStorage.removeItem(ADMIN_PIN_STORAGE_KEY);
      }
    } catch {
      // storage unavailable: the PIN stays in the form for this page view
    }
  }
}
