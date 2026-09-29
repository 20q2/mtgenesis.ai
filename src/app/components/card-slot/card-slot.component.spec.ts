import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { CardView } from '../../models/api.model';
import { MediaPipe } from '../../pipes/media.pipe';
import { MediaService } from '../../services/media.service';
import { cardView, doneCard } from '../../testing/fixtures';
import { CardSlotComponent } from './card-slot.component';

describe('CardSlotComponent', () => {
  let fixture: ComponentFixture<CardSlotComponent>;
  let component: CardSlotComponent;

  beforeEach(() => {
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ? `resolved:${u}` : null));
    TestBed.configureTestingModule({
      declarations: [CardSlotComponent, MediaPipe],
      providers: [{ provide: MediaService, useValue: media }]
    });
    fixture = TestBed.createComponent(CardSlotComponent);
    component = fixture.componentInstance;
  });

  function render(view: CardView | null, canReroll = true): HTMLElement {
    component.view = view;
    component.canReroll = canReroll;
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }

  function text(el: HTMLElement): string {
    return el.textContent!.replace(/\s+/g, ' ');
  }

  it('shows the queue position and ETA while queued', () => {
    const el = render(cardView({ status: 'queued', queuePosition: 4, etaSeconds: 35 }));
    expect(text(el)).toContain('Queued #4 · ~35s');
  });

  it('shows the Rules text / Artwork checklist while generating', () => {
    const el = render(cardView({ status: 'generating', textReady: true, artReady: false, queuePosition: 0 }));
    const items = Array.from(el.querySelectorAll('.checklist li'));
    expect(items.map(li => li.textContent!.trim())).toEqual([
      jasmine.stringContaining('Rules text'), jasmine.stringContaining('Artwork')
    ] as any);
    expect(items[0].classList).toContain('ready');
    expect(items[1].classList).not.toContain('ready');
  });

  it('shows Rendering… while rendering', () => {
    const el = render(cardView({ status: 'rendering', textReady: true, artReady: true, queuePosition: null }));
    expect(text(el)).toContain('Rendering…');
  });

  it('shows the error when failed', () => {
    const el = render(cardView({ status: 'failed', error: 'Ollama unreachable', queuePosition: null }));
    expect(text(el)).toContain('Failed: Ollama unreachable');
  });

  it('shows the finished card image and enlarges it on tap', () => {
    const el = render(doneCard({ id: 'x' }));
    const img = el.querySelector('img.slot-image') as HTMLImageElement;
    expect(img.getAttribute('src')).toBe('resolved:/api/v1/media/cards/x.png');
    expect(el.querySelector('.lightbox')).toBeNull();

    img.click();
    fixture.detectChanges();
    expect(el.querySelector('.lightbox img')).not.toBeNull();

    (el.querySelector('.lightbox') as HTMLElement).click();
    fixture.detectChanges();
    expect(el.querySelector('.lightbox')).toBeNull();
  });

  it('emits reroll when allowed, and disables the button otherwise', () => {
    const emitted: CardView[] = [];
    component.reroll.subscribe((v: CardView) => emitted.push(v));
    const view = doneCard();
    let el = render(view, true);
    (el.querySelector('button.reroll') as HTMLButtonElement).click();
    expect(emitted).toEqual([view]);

    el = render(view, false);
    expect((el.querySelector('button.reroll') as HTMLButtonElement).disabled).toBeTrue();
  });

  it('hides the reroll button when showReroll is false', () => {
    component.showReroll = false;
    const el = render(doneCard());
    expect(el.querySelector('button.reroll')).toBeNull();
  });
});
