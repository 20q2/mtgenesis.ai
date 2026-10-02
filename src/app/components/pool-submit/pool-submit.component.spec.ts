import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { of, throwError } from 'rxjs';
import { CardView, PoolView } from '../../models/api.model';
import { PoolService } from '../../services/pool.service';
import { cardView, doneCard, poolEntry, poolView } from '../../testing/fixtures';
import { PoolSubmitComponent } from './pool-submit.component';

describe('PoolSubmitComponent', () => {
  let fixture: ComponentFixture<PoolSubmitComponent>;
  let component: PoolSubmitComponent;
  let pools: jasmine.SpyObj<PoolService>;

  const red = doneCard({ id: 'c1' });
  const gold = doneCard({
    id: 'c2', card: { name: 'Envoy', manaCost: '{W}{U}', colors: ['W', 'U'], type: 'Creature', rarity: 'rare', cmc: 2 }
  });

  function setup(card: CardView, pool: PoolView | null) {
    pools = jasmine.createSpyObj<PoolService>('PoolService', ['submit']);
    TestBed.configureTestingModule({
      declarations: [PoolSubmitComponent],
      providers: [{ provide: PoolService, useValue: pools }]
    });
    fixture = TestBed.createComponent(PoolSubmitComponent);
    component = fixture.componentInstance;
    component.card = card;
    component.pool = pool;
    fixture.detectChanges();
  }

  afterEach(() => fixture.destroy());

  const button = () => fixture.nativeElement.querySelector('button.pool-submit') as HTMLButtonElement | null;
  const text = () => (fixture.nativeElement as HTMLElement).textContent ?? '';

  it('offers Submit with my entry count', () => {
    setup(red, poolView({ myEntryCount: 1 }));
    expect(button()!.disabled).toBeFalse();
    expect(button()!.textContent).toContain('Submit to pool (1/3)');
    expect(component.disabledReason()).toBeNull();
  });

  it('explains why it is disabled', () => {
    setup(red, null);
    expect(button()!.disabled).toBeTrue();
    expect(text()).toContain('No pool is open');

    component.pool = poolView();
    component.card = gold;
    fixture.detectChanges();
    expect(text()).toContain("Multicolor cards can't enter the pool");

    component.card = red;
    component.pool = poolView({ myEntryCount: 3 });
    fixture.detectChanges();
    expect(text()).toContain("You've used all 3 entries");
  });

  it('is hidden until the card is done', () => {
    setup(cardView({ id: 'c3', status: 'generating' }), poolView());
    expect(button()).toBeNull();
  });

  it('shows "In the pool" for an entered card', () => {
    setup(doneCard({ id: 'c1', poolEntryId: 'e1' }), poolView({ myEntryCount: 1 }));
    expect(text()).toContain('In the pool ✓');
    expect(button()!.disabled).toBeTrue();
  });

  it('submits the card and emits the new pool', () => {
    setup(red, poolView());
    const after = poolView({ myEntryCount: 1, entries: [poolEntry({ id: 'e9', cardId: 'c1', mine: true })] });
    pools.submit.and.returnValue(of(after));
    let emitted: PoolView | undefined;
    component.submitted.subscribe(p => (emitted = p));
    button()!.click();
    expect(pools.submit).toHaveBeenCalledWith('c1');
    expect(emitted).toBe(after);
  });

  it("shows the server's message when the submit fails, and doesn't claim the card is in", () => {
    setup(red, poolView());
    pools.submit.and.returnValue(throwError(() => new HttpErrorResponse({
      status: 404, error: { error: 'No Knowledge Pool is open' }
    })));
    let emitted = false;
    component.submitted.subscribe(() => (emitted = true));
    button()!.click();
    fixture.detectChanges();
    expect(text()).toContain('No Knowledge Pool is open');
    expect(text()).not.toContain('In the pool');
    expect(emitted).toBeFalse();
  });
});
