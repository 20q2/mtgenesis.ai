import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { RouterTestingModule } from '@angular/router/testing';
import { of, throwError } from 'rxjs';
import { PoolView } from '../../models/api.model';
import { CardSlotComponent } from '../../components/card-slot/card-slot.component';
import { MediaPipe } from '../../pipes/media.pipe';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { PageVisibilityService } from '../../services/page-visibility.service';
import { PoolService } from '../../services/pool.service';
import { fakeVisibility } from '../../testing/fake-visibility';
import { doneCard, poolEntry, poolSlot, poolView } from '../../testing/fixtures';
import { PoolPageComponent } from './pool-page.component';

describe('PoolPageComponent', () => {
  let fixture: ComponentFixture<PoolPageComponent>;
  let pools: jasmine.SpyObj<PoolService>;
  let generation: jasmine.SpyObj<GenerationService>;

  const redCard = (id: string) => doneCard({
    id, card: { name: `Card ${id}`, manaCost: '{1}{R}', colors: ['R'], type: 'Creature', rarity: 'common', cmc: 2 }
  });

  function contested(overrides: Partial<PoolView> = {}): PoolView {
    return poolView({
      slots: [
        poolSlot({
          id: 'red', position: 1,
          entries: [
            poolEntry({ id: 'e1', slotId: 'red', votes: 3, leader: true, card: redCard('c1') }),
            poolEntry({ id: 'e2', slotId: 'red', votes: 1, mine: true, username: 'Me', card: redCard('c2') })
          ],
          myEntryId: 'e2'
        }),
        poolSlot({
          id: 'colorless', position: 2, label: 'Colorless', colorRule: 'colorless', typeRule: 'any',
          ruleText: 'Colorless card',
          entries: [poolEntry({
            id: 'pot', slotId: 'colorless', power: { estimate: 3.6, budget: 0.25, verdict: 'over' },
            card: doneCard({ id: 'pot-card', card: { name: 'Pot of Green', manaCost: '{0}', colors: [], type: 'Artifact', rarity: 'common', cmc: 0 } })
          })]
        })
      ],
      myEntryCount: 1,
      ...overrides
    });
  }

  function setup(current: PoolView | null) {
    pools = jasmine.createSpyObj<PoolService>('PoolService',
      ['current', 'get', 'list', 'submit', 'withdraw', 'vote', 'clearVote']);
    pools.current.and.returnValue(of(current));
    pools.list.and.returnValue(of([]));
    generation = jasmine.createSpyObj<GenerationService>('GenerationService', ['myCards']);
    generation.myCards.and.returnValue(of([]));
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src', 'download']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      declarations: [PoolPageComponent, CardSlotComponent, MediaPipe],
      providers: [
        { provide: PoolService, useValue: pools },
        { provide: GenerationService, useValue: generation },
        { provide: MediaService, useValue: media },
        { provide: PageVisibilityService, useValue: fakeVisibility() }
      ]
    });
    fixture = TestBed.createComponent(PoolPageComponent);
    fixture.detectChanges();
  }

  afterEach(() => fixture.destroy());

  const el = () => fixture.nativeElement as HTMLElement;
  const entry = (id: string) => el().querySelector(`[data-entry-id="${id}"]`) as HTMLElement;
  const button = (root: HTMLElement, selector: string) => root.querySelector(selector) as HTMLButtonElement | null;

  it('shows the empty state when no pool is open', () => {
    setup(null);
    expect(el().textContent).toContain('No Knowledge Pool open');
  });

  it('renders the slots, the leader and my submission count', () => {
    setup(contested());
    expect(el().textContent).toContain('Knowledge Pool 2026');
    expect(el().textContent).toContain('1 / 2 submitted');
    expect(entry('e1').classList).toContain('leader');
    expect(entry('e1').textContent).toContain('Leading');
    expect(entry('e1').textContent).toContain('3 votes');
  });

  it('has no vote button on my own card, only Withdraw', () => {
    setup(contested());
    expect(button(entry('e2'), '.vote-button')).toBeNull();
    expect(button(entry('e2'), '.withdraw-button')).not.toBeNull();
    expect(entry('e2').textContent).toContain('Your card');
    expect(button(entry('e1'), '.vote-button')).not.toBeNull();
  });

  it('flags an over-the-curve card', () => {
    setup(contested());
    const power = entry('pot').querySelector('.power') as HTMLElement;
    expect(power.classList).toContain('power-over');
    expect(power.textContent).toContain('Over the curve');
  });

  it('votes, and a second click on my pick clears the vote', () => {
    setup(contested());
    const voted = contested();
    voted.slots[0].myVoteEntryId = 'e1';
    pools.vote.and.returnValue(of(voted));
    button(entry('e1'), '.vote-button')!.click();
    fixture.detectChanges();
    expect(pools.vote).toHaveBeenCalledOnceWith('red', 'e1');
    expect(entry('e1').classList).toContain('pick');

    pools.clearVote.and.returnValue(of(contested()));
    button(entry('e1'), '.vote-button')!.click();
    fixture.detectChanges();
    expect(pools.clearVote).toHaveBeenCalledOnceWith('red');
    expect(entry('e1').classList).not.toContain('pick');
  });

  it('shows the server error when a vote is rejected', () => {
    setup(contested());
    pools.vote.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 403, error: { error: "You can't vote for your own card" }
    })));
    button(entry('e1'), '.vote-button')!.click();
    fixture.detectChanges();
    expect(el().querySelector('.alert-error')!.textContent).toContain("You can't vote for your own card");
  });

  it('the picker offers only my finished cards that fit the slot and are not in the pool', () => {
    setup(contested());
    generation.myCards.and.returnValue(of([
      redCard('c2'),                                             // already in the pool
      doneCard({ id: 'artifact', card: { name: 'Gizmo', manaCost: '{2}', colors: [], type: 'Artifact', rarity: 'common', cmc: 2 } }),
      doneCard({ id: 'red-spell', card: { name: 'Zap', manaCost: '{R}', colors: ['R'], type: 'Instant', rarity: 'common', cmc: 1 } }),
      { ...redCard('pending'), status: 'queued' }
    ]));
    const tile = el().querySelectorAll('.submit-tile');
    expect(tile.length).toBe(1);                                // no tile in the slot I already filled
    (tile[0] as HTMLButtonElement).click();
    fixture.detectChanges();
    const names = Array.from(el().querySelectorAll('.picker-name')).map(n => n.textContent!.trim());
    expect(names).toEqual(['Gizmo']);

    pools.submit.and.returnValue(of(contested()));
    (el().querySelector('.picker-card') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(pools.submit).toHaveBeenCalledOnceWith('colorless', 'artifact');
    expect(el().querySelector('.picker')).toBeNull();
  });

  it('disables the submit tile once all submissions are used', () => {
    setup(contested({ myEntryCount: 2 }));
    const tile = el().querySelector('.submit-tile') as HTMLButtonElement;
    expect(tile.disabled).toBeTrue();
    expect(tile.textContent).toContain('No submissions left');
  });

  it('a closed pool lists the legal cards with their submitters and hides voting', () => {
    const closed = contested({ status: 'closed', closedAt: '2026-10-10T23:00:00+00:00' });
    closed.slots[0].entries[0].username = 'Alice';
    setup(closed);
    const results = el().querySelector('.results') as HTMLElement;
    expect(results.textContent).toContain('Legal cards');
    expect(results.textContent).toContain('Red creature');
    expect(results.textContent).toContain('by Alice');
    expect(el().querySelectorAll('.vote-button').length).toBe(0);
    expect(el().querySelector('.submit-tile')).toBeNull();
    expect(entry('e1').textContent).toContain('Legal');
  });
});
