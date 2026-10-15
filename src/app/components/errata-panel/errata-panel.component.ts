import { Component, Input, OnChanges, OnDestroy } from '@angular/core';
import { Subject, Subscription, debounceTime, distinctUntilChanged, of, switchMap, catchError, tap } from 'rxjs';
import { ErrataGroup, ErrataItem } from '../../models/api.model';
import { apiErrorMessage } from '../../services/api.util';
import { EventService } from '../../services/event.service';

/**
 * Host errata on /admin (docs/superpowers/specs/2026-10-14-errata-design.md): the cards in the
 * commander vote, the last closed event's legal commanders and the Knowledge Pool, each a tile
 * that opens the errata editor. Loads once the host PIN is entered.
 */
@Component({
  selector: 'app-errata-panel',
  templateUrl: './errata-panel.component.html',
  styleUrls: ['./errata-panel.component.scss']
})
export class ErrataPanelComponent implements OnChanges, OnDestroy {
  /** The host PIN from the admin page. */
  @Input() pin = '';

  groups: ErrataGroup[] | null = null;
  activeId: ErrataGroup['id'] | null = null;
  filter = '';
  loading = false;
  error: string | null = null;
  success: string | null = null;
  /** The card open in the editor. */
  editing: ErrataItem | null = null;

  private readonly load$ = new Subject<{ pin: string; force: boolean }>();
  private readonly sub: Subscription;

  constructor(private events: EventService) {
    this.sub = this.load$.pipe(
      debounceTime(300),
      distinctUntilChanged((a, b) => a.pin === b.pin && !b.force),
      switchMap(({ pin }) => {
        if (!pin) {
          this.groups = null;
          return of(null);
        }
        this.loading = true;
        this.error = null;
        return this.events.errataList(pin).pipe(
          tap(() => (this.loading = false)),
          catchError(err => {
            this.loading = false;
            this.groups = null;
            this.error = apiErrorMessage(err, 'Could not load the cards.');
            return of(null);
          }));
      })
    ).subscribe(groups => {
      if (groups) {
        this.setGroups(groups);
      }
    });
  }

  ngOnChanges(): void {
    this.load$.next({ pin: this.pin.trim(), force: false });
  }

  ngOnDestroy(): void {
    this.sub.unsubscribe();
  }

  /** Reload (after closing an event, or by hand). */
  reload(): void {
    this.load$.next({ pin: this.pin.trim(), force: true });
  }

  get active(): ErrataGroup | null {
    return this.groups?.find(g => g.id === this.activeId) ?? null;
  }

  /** The active group's cards that match the filter (card or commander name, or player). */
  get shown(): ErrataItem[] {
    const items = this.active?.items ?? [];
    const words = this.filter.trim().toLowerCase();
    if (!words) {
      return items;
    }
    return items.filter(i => [i.label, i.username, i.card.card?.name ?? '', i.detail]
      .some(text => text.toLowerCase().includes(words)));
  }

  errataCount(group: ErrataGroup): number {
    return group.items.filter(i => i.card.errata).length;
  }

  tabLabel(group: ErrataGroup): string {
    return group.title.split(' · ')[0];
  }

  tabSub(group: ErrataGroup): string {
    return group.title.split(' · ').slice(1).join(' · ');
  }

  onFilter(event: Event): void {
    this.filter = (event.target as HTMLInputElement).value;
  }

  open(item: ErrataItem): void {
    this.success = null;
    this.editing = item;
  }

  /** A save or revert: the same card may sit in several groups (the vote and the pool). */
  onSaved(updated: ErrataItem): void {
    const name = this.editing?.label ?? updated.label;
    for (const group of this.groups ?? []) {
      group.items = group.items.map(i => (i.card.id === updated.card.id
        ? { ...i, card: updated.card, original: updated.original } : i));
    }
    this.editing = null;
    this.success = updated.card.errata ? `Errata saved: ${name} is re-rendered for everyone.`
      : `${name} is back the way it was printed.`;
  }

  trackItem(_index: number, item: ErrataItem): string {
    return item.card.id;
  }

  private setGroups(groups: ErrataGroup[]): void {
    this.groups = groups;
    // Keep the open tab; otherwise start where errata is most likely: the night's legal commanders.
    if (!groups.some(g => g.id === this.activeId)) {
      this.activeId = (groups.find(g => g.id === 'legal') ?? groups[0])?.id ?? null;
    }
  }
}
