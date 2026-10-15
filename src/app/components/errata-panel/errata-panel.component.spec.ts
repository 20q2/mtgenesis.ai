import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA } from '@angular/core';
import { of } from 'rxjs';
import { ErrataGroup, ErrataItem } from '../../models/api.model';
import { MediaPipe } from '../../pipes/media.pipe';
import { EventService } from '../../services/event.service';
import { MediaService } from '../../services/media.service';
import { doneCard } from '../../testing/fixtures';
import { ErrataPanelComponent } from './errata-panel.component';

describe('ErrataPanelComponent', () => {
  let fixture: ComponentFixture<ErrataPanelComponent>;
  let events: jasmine.SpyObj<EventService>;

  function it_(id: string, label: string, username = 'alice', detail = 'Version 1', errata = false): ErrataItem {
    return { card: doneCard({ id, errata }), original: null, label, username, detail };
  }
  function groups(): ErrataGroup[] {
    return [
      { id: 'vote', title: 'Commander vote · AI Night #4', items: [it_('v1', 'Gamma'), it_('p1', 'Pup', 'carl')] },
      { id: 'legal', title: 'Legal commanders · AI Night #3', items: [it_('w1', 'Alpha', 'alice', 'Winner'),
                                                                       it_('w2', 'Beta', 'bob', 'Tied', true)] },
      { id: 'pool', title: 'Knowledge Pool · Fall', items: [it_('p1', 'Pup', 'carl', 'In · 9 points')] }
    ];
  }

  beforeEach(() => {
    events = jasmine.createSpyObj<EventService>('EventService', ['errataList']);
    events.errataList.and.callFake(() => of(groups()));
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      declarations: [ErrataPanelComponent, MediaPipe],
      providers: [{ provide: EventService, useValue: events }, { provide: MediaService, useValue: media }],
      schemas: [NO_ERRORS_SCHEMA]
    });
    fixture = TestBed.createComponent(ErrataPanelComponent);
  });

  const el = () => fixture.nativeElement as HTMLElement;

  function setPin(pin: string): void {
    fixture.componentInstance.pin = pin;
    fixture.componentInstance.ngOnChanges();
    tick(300);
    fixture.detectChanges();
  }

  it('waits for the host PIN', fakeAsync(() => {
    setPin('');
    expect(events.errataList).not.toHaveBeenCalled();
    expect(el().textContent).toContain('Enter the host PIN');
  }));

  it('loads the three sources and opens on the legal commanders', fakeAsync(() => {
    setPin('9999');
    expect(events.errataList).toHaveBeenCalledWith('9999');
    const tabs = Array.from(el().querySelectorAll('.source'));
    expect(tabs.map(t => t.querySelector('.source-name')!.textContent!.trim()))
      .toEqual(['Commander vote', 'Legal commanders', 'Knowledge Pool']);
    expect(tabs[1].getAttribute('aria-selected')).toBe('true');
    expect(tabs[1].textContent).toContain('2 cards · 1 with errata');
    const tiles = el().querySelectorAll('.errata-tile');
    expect(tiles.length).toBe(2);
    expect(tiles[1].querySelector('.errata-tag')).not.toBeNull();
    expect(tiles[0].textContent).toContain('alice · Winner');
  }));

  it('filters by card or player', fakeAsync(() => {
    setPin('9999');
    fixture.componentInstance.filter = 'bob';
    fixture.detectChanges();
    expect(Array.from(el().querySelectorAll('.tile-label')).map(t => t.textContent!.trim())).toEqual(['Beta']);
  }));

  it('opens the editor and updates the card everywhere it appears after a save', fakeAsync(() => {
    setPin('9999');
    const c = fixture.componentInstance;
    c.activeId = 'pool';
    fixture.detectChanges();
    (el().querySelector('.errata-tile') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(el().querySelector('app-errata-editor')).not.toBeNull();

    const updated = { ...it_('p1', 'Pup'), card: doneCard({ id: 'p1', errata: true }) };
    c.onSaved(updated);
    fixture.detectChanges();
    expect(el().querySelector('app-errata-editor')).toBeNull();
    expect(c.groups![0].items[1].card.errata).toBeTrue();
    expect(c.groups![2].items[0].card.errata).toBeTrue();
    expect(c.groups![2].items[0].detail).toBe('In · 9 points');
    expect(el().querySelector('.alert-success')!.textContent).toContain('Errata saved');
  }));
});
