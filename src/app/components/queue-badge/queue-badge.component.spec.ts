import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BehaviorSubject, map } from 'rxjs';
import { QueueStatus } from '../../models/api.model';
import { QueueService } from '../../services/queue.service';
import { QueueBadgeComponent } from './queue-badge.component';

describe('QueueBadgeComponent', () => {
  let fixture: ComponentFixture<QueueBadgeComponent>;
  let status$: BehaviorSubject<QueueStatus | null | undefined>;

  beforeEach(() => {
    status$ = new BehaviorSubject<QueueStatus | null | undefined>(undefined);
    const fake = {
      status$: status$.pipe(map(s => s as QueueStatus | null)),
      online$: status$.pipe(map(s => s !== null))
    };
    TestBed.configureTestingModule({
      declarations: [QueueBadgeComponent],
      providers: [{ provide: QueueService, useValue: fake }]
    });
    fixture = TestBed.createComponent(QueueBadgeComponent);
  });

  function text(): string {
    fixture.detectChanges();
    return (fixture.nativeElement as HTMLElement).textContent!.replace(/\s+/g, ' ').trim();
  }

  it('shows "Server idle" when not busy', () => {
    status$.next({ busy: false, cardsAhead: 0, generatingNow: 0, avgImageSeconds: 9, etaSeconds: 9 });
    expect(text()).toBe('Server idle');
    expect(fixture.nativeElement.querySelector('.queue-badge').classList).toContain('idle');
  });

  it('shows cards ahead and minutes (rounded up) when busy', () => {
    status$.next({ busy: true, cardsAhead: 4, generatingNow: 1, avgImageSeconds: 12, etaSeconds: 61 });
    expect(text()).toBe('Server busy · 4 cards ahead · ~2 min');
    expect(fixture.nativeElement.querySelector('.queue-badge').classList).toContain('busy');
  });

  it('shows "Server offline" when the status call fails', () => {
    status$.next(null);
    expect(text()).toBe('Server offline');
    expect(fixture.nativeElement.querySelector('.queue-badge').classList).toContain('offline');
  });

  it('formats the text with the brief formula', () => {
    const c = fixture.componentInstance;
    expect(c.text({ busy: true, cardsAhead: 1, generatingNow: 1, avgImageSeconds: 10, etaSeconds: 120 }))
      .toBe('Server busy · 1 cards ahead · ~2 min');
    expect(c.text(null)).toBe('Server offline');
  });
});
