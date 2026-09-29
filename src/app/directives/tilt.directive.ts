import { Directive, ElementRef, HostBinding, HostListener, Input, NgZone } from '@angular/core';

/**
 * Tilts a card toward the pointer (the corner under the cursor lifts toward you) and
 * feeds a glare position for foil effects. Sets --tilt-x / --tilt-y (deg) and
 * --glare-x / --glare-y (%) on the host; the .tilt class in styles.scss applies them.
 * Mouse and pen only (touch would fight scrolling), and off under reduced motion.
 */
@Directive({
  selector: '[appTilt]'
})
export class TiltDirective {
  /** Maximum lean in degrees at the card's edges (a plain attribute: appTilt="8"). */
  @Input() appTilt: number | string = '';

  @HostBinding('class.tilt') readonly tiltClass = true;
  @HostBinding('class.tilting') tilting = false;

  private frame = 0;
  private readonly reducedMotion =
    typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

  constructor(private el: ElementRef<HTMLElement>, private zone: NgZone) {}

  @HostListener('pointermove', ['$event'])
  onMove(event: PointerEvent): void {
    if (this.reducedMotion || event.pointerType === 'touch') {
      return;
    }
    this.tilting = true;
    const { clientX, clientY } = event;
    cancelAnimationFrame(this.frame);
    this.zone.runOutsideAngular(() => {
      this.frame = requestAnimationFrame(() => this.apply(clientX, clientY));
    });
  }

  @HostListener('pointerleave')
  onLeave(): void {
    cancelAnimationFrame(this.frame);
    this.tilting = false;
    const style = this.el.nativeElement.style;
    style.setProperty('--tilt-x', '0deg');
    style.setProperty('--tilt-y', '0deg');
    style.setProperty('--glare-x', '50%');
    style.setProperty('--glare-y', '50%');
  }

  private apply(clientX: number, clientY: number): void {
    const host = this.el.nativeElement;
    const rect = host.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      return;
    }
    // 0..1 across the card, clamped so fast exits don't overshoot.
    const px = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    const py = Math.min(1, Math.max(0, (clientY - rect.top) / rect.height));
    const max = Number(this.appTilt) || 12;
    // The corner under the pointer lifts toward the viewer.
    host.style.setProperty('--tilt-y', `${((0.5 - px) * 2 * max).toFixed(2)}deg`);
    host.style.setProperty('--tilt-x', `${((py - 0.5) * 2 * max).toFixed(2)}deg`);
    host.style.setProperty('--glare-x', `${(px * 100).toFixed(1)}%`);
    host.style.setProperty('--glare-y', `${(py * 100).toFixed(1)}%`);
  }
}
