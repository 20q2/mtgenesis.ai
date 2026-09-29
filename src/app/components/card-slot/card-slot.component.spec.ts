import { Component } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { OverlayModule } from '@angular/cdk/overlay';
import { of } from 'rxjs';
import { CardView } from '../../models/api.model';
import { MediaPipe } from '../../pipes/media.pipe';
import { MediaService } from '../../services/media.service';
import { cardView, doneCard } from '../../testing/fixtures';
import { CardSlotComponent } from './card-slot.component';

/** Mirrors real usage: every card slot sits inside a .panel (whose backdrop-filter traps position: fixed). */
@Component({
  template: '<div class="panel" style="backdrop-filter: blur(6px)"><app-card-slot [view]="view"></app-card-slot></div>'
})
class PanelHostComponent {
  view: CardView | null = null;
}

describe('CardSlotComponent', () => {
  let fixture: ComponentFixture<CardSlotComponent>;
  let component: CardSlotComponent;

  beforeEach(() => {
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ? `resolved:${u}` : null));
    TestBed.configureTestingModule({
      imports: [OverlayModule],
      declarations: [CardSlotComponent, MediaPipe, PanelHostComponent],
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

  it('shows the finished card image', () => {
    const el = render(doneCard({ id: 'x' }));
    const img = el.querySelector('img.slot-image') as HTMLImageElement;
    expect(img.getAttribute('src')).toBe('resolved:/api/v1/media/cards/x.png');
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

  describe('tap to enlarge', () => {
    let host: ComponentFixture<PanelHostComponent>;

    beforeEach(() => {
      host = TestBed.createComponent(PanelHostComponent);
      host.componentInstance.view = doneCard({ id: 'big' });
      host.detectChanges();
    });

    afterEach(() => host.destroy());

    const lightbox = () => document.querySelector('.lightbox') as HTMLElement | null;

    it('opens the enlarged card in an overlay outside any .panel, covering the viewport', () => {
      const panel = host.nativeElement.querySelector('.panel') as HTMLElement;
      expect(lightbox()).toBeNull();

      (panel.querySelector('img.slot-image') as HTMLImageElement).click();
      host.detectChanges();

      const box = lightbox();
      expect(box).not.toBeNull();
      expect(panel.contains(box)).toBeFalse();
      expect(box!.closest('.panel')).toBeNull();
      expect(box!.closest('.cdk-overlay-container')).not.toBeNull();
      expect(box!.querySelector('img')!.getAttribute('src')).toBe('resolved:/api/v1/media/cards/big.png');

      const rect = box!.getBoundingClientRect();
      expect(rect.width).toBeCloseTo(document.documentElement.clientWidth, -1);
      expect(rect.height).toBeCloseTo(document.documentElement.clientHeight, -1);
    });

    it('closes on tap and on Escape, and removes the overlay when the slot is destroyed', () => {
      const img = host.nativeElement.querySelector('img.slot-image') as HTMLImageElement;
      img.click();
      host.detectChanges();
      lightbox()!.click();
      host.detectChanges();
      expect(lightbox()).toBeNull();

      img.click();
      host.detectChanges();
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
      host.detectChanges();
      expect(lightbox()).toBeNull();

      img.click();
      host.detectChanges();
      expect(lightbox()).not.toBeNull();
      host.destroy();
      expect(lightbox()).toBeNull();
    });
  });
});
