import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { RouterTestingModule } from '@angular/router/testing';
import { of, throwError } from 'rxjs';
import { PoolView } from '../../models/api.model';
import { CardSlotComponent } from '../../components/card-slot/card-slot.component';
import { MediaPipe } from '../../pipes/media.pipe';
import { MediaService } from '../../services/media.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { PoolService } from '../../services/pool.service';
import { fakeVisibility } from '../../testing/fake-visibility';
import { doneCard, poolEntry, poolView } from '../../testing/fixtures';
import { PoolPageComponent } from './pool-page.component';

describe('PoolPageComponent', () => {
  let fixture: ComponentFixture<PoolPageComponent>;
  let pools: jasmine.SpyObj<PoolService>;

  const card = (id: string, name = `Card ${id}`) => doneCard({
    id, card: { name, manaCost: '{1}{R}', colors: ['R'], type: 'Creature', rarity: 'common', cmc: 2 }
  });

  /** 4 players -> top 2; e1 and e2 tie at the line so both are in, e3 is out, e4 is mine. */
  function contested(overrides: Partial<PoolView> = {}): PoolView {
    return poolView({
      submitters: 4, cutoff: 2, myEntryCount: 1,
      myMedals: { gold: 'e1', silver: null, bronze: null },
      entries: [
        poolEntry({ id: 'e0', gold: 2, points: 6, rank: 1, in: true, card: card('c0'), username: 'Ann' }),
        poolEntry({ id: 'e1', gold: 1, points: 3, rank: 2, in: true, tiedAtCutoff: true, myMedal: 'gold',
                    card: card('c1'), username: 'Bob' }),
        poolEntry({ id: 'e2', gold: 1, points: 3, rank: 2, in: true, tiedAtCutoff: true, card: card('c2') }),
        poolEntry({ id: 'e3', bronze: 1, points: 1, rank: 4, card: card('c3'),
                    power: { estimate: 3.6, budget: 0.25, verdict: 'over' } }),
        poolEntry({ id: 'e4', rank: 5, mine: true, username: 'Me', card: card('c4') })
      ],
      ...overrides
    });
  }

  function setup(current: PoolView | null) {
    pools = jasmine.createSpyObj<PoolService>('PoolService',
      ['current', 'get', 'list', 'withdraw', 'medal', 'clearMedal']);
    pools.current.and.returnValue(of(current));
    pools.list.and.returnValue(of([]));
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src', 'download']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      declarations: [PoolPageComponent, CardSlotComponent, MediaPipe],
      providers: [
        { provide: PoolService, useValue: pools },
        { provide: MediaService, useValue: media },
        { provide: PageVisibilityService, useValue: fakeVisibility() }
      ]
    });
    fixture = TestBed.createComponent(PoolPageComponent);
    fixture.detectChanges();
  }

  afterEach(() => fixture.destroy());

  const el = () => fixture.nativeElement as HTMLElement;
  const tile = (id: string) => el().querySelector(`[data-entry-id="${id}"]`) as HTMLElement;
  const tileIds = () => Array.from(el().querySelectorAll('.entries [data-entry-id]'))
    .map(t => (t as HTMLElement).dataset['entryId']);

  it('explains the rules and how many cards make the pool', () => {
    setup(contested());
    const text = el().textContent!;
    expect(text).toContain('Colorless or mono-colored cards only');
    expect(text).toContain('4 players → top 2 make the pool');
    expect(text).toContain('Your entries 1 / 3');
  });

  it('lists entries in the server order with the pool line after the last card that is in', () => {
    setup(contested());
    expect(tileIds()).toEqual(['e0', 'e1', 'e2', 'e3', 'e4']);
    const children = Array.from(el().querySelector('.entries')!.children) as HTMLElement[];
    const line = children.findIndex(c => c.classList.contains('pool-line'));
    expect(el().querySelectorAll('.pool-line').length).toBe(1);
    expect(children[line - 1].dataset['entryId']).toBe('e2');
    expect(tile('e0').textContent).toContain('In');
    expect(tile('e1').textContent).toContain('Tied · in');
    expect(tile('e3').textContent).not.toContain('In ');
    expect(tile('e3').textContent).toContain('Over the curve');
    expect(tile('e1').textContent).toContain('by Bob');
  });

  it('gives, moves and takes back medals; your own card gets Withdraw instead', () => {
    setup(contested());
    pools.medal.and.returnValue(of(contested()));
    pools.clearMedal.and.returnValue(of(contested()));

    const gold = tile('e1').querySelector('.medal-button.gold') as HTMLButtonElement;
    expect(gold.getAttribute('aria-pressed')).toBe('true');
    gold.click();
    expect(pools.clearMedal).toHaveBeenCalledWith('e1');

    (tile('e3').querySelector('.medal-button.silver') as HTMLButtonElement).click();
    expect(pools.medal).toHaveBeenCalledWith('e3', 'silver');

    expect(tile('e4').querySelector('.medal-button')).toBeNull();
    expect(tile('e4').querySelector('.withdraw-button')).not.toBeNull();
  });

  it('shows the medals I have given in the header', () => {
    setup(contested());
    const chips = Array.from(el().querySelectorAll('.my-medal')) as HTMLElement[];
    expect(chips.map(c => c.classList.contains('given'))).toEqual([true, false, false]);
  });

  it('shows the server error when a medal fails', () => {
    setup(contested());
    pools.medal.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 409, error: { error: 'This Knowledge Pool is closed' }
    })));
    (tile('e3').querySelector('.medal-button.gold') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(el().textContent).toContain('This Knowledge Pool is closed');
  });

  it('a closed pool shows the cards that made it and no voting buttons', () => {
    setup(contested({ status: 'closed', closedAt: '2026-10-01T23:00:00+00:00' }));
    expect(el().querySelector('.medal-button')).toBeNull();
    expect(el().querySelector('.withdraw-button')).toBeNull();
    const made = Array.from(el().querySelectorAll('.results [data-result-id]'))
      .map(r => (r as HTMLElement).dataset['resultId']);
    expect(made).toEqual(['e0', 'e1', 'e2']);
  });

  it('an empty open pool points to the create page', () => {
    setup(poolView());
    expect(el().querySelector('.pool-line')).toBeNull();
    expect(el().querySelector('a[href="/create"]')).not.toBeNull();
  });

  it('shows a friendly state when no pool is open', () => {
    setup(null);
    expect(el().textContent).toContain('No Knowledge Pool open');
  });
});
