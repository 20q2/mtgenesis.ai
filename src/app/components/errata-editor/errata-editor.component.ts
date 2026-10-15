import {
  AfterViewInit, Component, ElementRef, EventEmitter, Input, OnDestroy, OnInit, Output, ViewChild
} from '@angular/core';
import { FormControl, FormGroup } from '@angular/forms';
import { Subscription, finalize } from 'rxjs';
import { ErrataFields, ErrataItem, ErrataPreview, PowerCheck } from '../../models/api.model';
import { apiErrorMessage } from '../../services/api.util';
import { EventService } from '../../services/event.service';

/**
 * Edit one card's rules text and P/T (docs/superpowers/specs/2026-10-14-errata-design.md).
 * Preview renders it without saving; Save re-renders it for everyone; Revert goes back to the
 * card as printed. Name, cost, type and rarity can't change. A native modal <dialog>.
 */
@Component({
  selector: 'app-errata-editor',
  templateUrl: './errata-editor.component.html',
  styleUrls: ['./errata-editor.component.scss']
})
export class ErrataEditorComponent implements OnInit, AfterViewInit, OnDestroy {
  @Input() item!: ErrataItem;
  @Input() pin = '';
  /** The card after a save or revert. */
  @Output() saved = new EventEmitter<ErrataItem>();
  @Output() closed = new EventEmitter<void>();

  @ViewChild('dialog', { static: true }) private dialog!: ElementRef<HTMLDialogElement>;

  readonly description = new FormControl('', { nonNullable: true });
  readonly power = new FormControl('', { nonNullable: true });
  readonly toughness = new FormControl('', { nonNullable: true });
  readonly form = new FormGroup({ description: this.description, power: this.power, toughness: this.toughness });

  preview: ErrataPreview | null = null;
  /** The form changed since the preview was drawn. */
  previewStale = false;
  busy: 'preview' | 'save' | 'revert' | null = null;
  error: string | null = null;

  private subs = new Subscription();

  constructor(private events: EventService) {}

  ngOnInit(): void {
    const card = this.item.card.card;
    this.description.setValue(card?.description ?? '');
    this.power.setValue(card?.power ?? '');
    this.toughness.setValue(card?.toughness ?? '');
    this.subs.add(this.form.valueChanges.subscribe(() => (this.previewStale = !!this.preview)));
  }

  ngAfterViewInit(): void {
    const dialog = this.dialog.nativeElement;
    if (!dialog.open) {
      if (typeof dialog.showModal === 'function') {
        dialog.showModal();
      } else {
        dialog.setAttribute('open', '');
      }
    }
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    const dialog = this.dialog.nativeElement;
    if (dialog.open && typeof dialog.close === 'function') {
      dialog.close();
    }
  }

  /** Creatures and Vehicles print power/toughness. */
  get hasBody(): boolean {
    const card = this.item.card.card;
    return /creature/i.test(card?.type ?? '') || /vehicle/i.test(card?.subtype ?? '');
  }

  get dirty(): boolean {
    const card = this.item.card.card;
    return this.description.value.trim() !== (card?.description ?? '').trim()
      || (this.hasBody && (this.power.value.trim() !== (card?.power ?? '')
                           || this.toughness.value.trim() !== (card?.toughness ?? '')));
  }

  get caption(): string {
    if (this.preview) {
      return this.previewStale ? 'Preview · out of date, preview again' : 'Preview · not saved yet';
    }
    return this.item.card.errata ? 'Now · with errata' : 'As printed';
  }

  powerLabel(power: PowerCheck): string {
    return power.verdict === 'over' ? 'Over the curve' : power.verdict === 'pushed' ? 'Pushed' : 'Fair';
  }

  powerTitle(power: PowerCheck): string {
    return `Rough power check: its rules text is worth about ${power.estimate} mana; ` +
      `a card with this cost and rarity usually gets about ${power.budget}.`;
  }

  doPreview(): void {
    if (this.busy) {
      return;
    }
    if (!this.start('preview')) {
      return;
    }
    this.subs.add(this.events.previewErrata(this.item.card.id, this.fields(), this.pin)
      .pipe(finalize(() => (this.busy = null)))
      .subscribe({
        next: preview => {
          this.preview = preview;
          this.previewStale = false;
        },
        error: err => (this.error = apiErrorMessage(err, 'Could not draw the preview.'))
      }));
  }

  save(): void {
    if (this.busy || !this.dirty) {
      return;
    }
    if (!this.start('save')) {
      return;
    }
    this.subs.add(this.events.saveErrata(this.item.card.id, this.fields(), this.pin)
      .pipe(finalize(() => (this.busy = null)))
      .subscribe({
        next: item => this.saved.emit(item),
        error: err => (this.error = apiErrorMessage(err, 'The errata did not save.'))
      }));
  }

  revert(): void {
    if (this.busy || !window.confirm(`Put ${this.item.label} back the way it was printed?`)) {
      return;
    }
    if (!this.start('revert')) {
      return;
    }
    this.subs.add(this.events.revertErrata(this.item.card.id, this.pin)
      .pipe(finalize(() => (this.busy = null)))
      .subscribe({
        next: item => this.saved.emit(item),
        error: err => (this.error = apiErrorMessage(err, 'Could not revert the card.'))
      }));
  }

  close(): void {
    if (!this.busy) {
      this.closed.emit();
    }
  }

  onCancel(event: Event): void {
    event.preventDefault();
    this.close();
  }

  onDialogClick(event: MouseEvent): void {
    if (event.target === this.dialog.nativeElement) {
      this.close();
    }
  }

  /** False (with a message) when there's no host PIN to send. */
  private start(action: 'preview' | 'save' | 'revert'): boolean {
    if (!this.pin.trim()) {
      this.error = 'Enter the host PIN first.';
      return false;
    }
    this.error = null;
    this.busy = action;
    return true;
  }

  private fields(): ErrataFields {
    const fields: ErrataFields = { description: this.description.value };
    if (this.hasBody) {
      fields.power = this.power.value.trim();
      fields.toughness = this.toughness.value.trim();
    }
    return fields;
  }
}
