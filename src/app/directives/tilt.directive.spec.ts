import { Component } from '@angular/core';
import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { TiltDirective } from './tilt.directive';

@Component({
  template: `<div appTilt style="width: 200px; height: 280px"></div>`
})
class HostComponent {}

describe('TiltDirective', () => {
  let fixture: ComponentFixture<HostComponent>;
  let el: HTMLElement;

  beforeEach(() => {
    TestBed.configureTestingModule({ declarations: [HostComponent, TiltDirective] });
    fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
    el = fixture.nativeElement.querySelector('[appTilt]');
  });

  function moveTo(fx: number, fy: number, pointerType = 'mouse'): void {
    const r = el.getBoundingClientRect();
    el.dispatchEvent(new PointerEvent('pointermove', {
      clientX: r.left + r.width * fx,
      clientY: r.top + r.height * fy,
      pointerType
    }));
  }

  it('lifts the corner under the pointer toward the viewer', fakeAsync(() => {
    moveTo(1, 0); // top-right corner
    tick(20);
    // Negative rotateY brings the right edge forward; negative rotateX brings the top forward.
    expect(parseFloat(el.style.getPropertyValue('--tilt-y'))).toBeLessThan(0);
    expect(parseFloat(el.style.getPropertyValue('--tilt-x'))).toBeLessThan(0);
    expect(el.style.getPropertyValue('--glare-x')).toBe('100.0%');
  }));

  it('ignores touch so scrolling is not hijacked', fakeAsync(() => {
    moveTo(1, 0, 'touch');
    tick(20);
    expect(el.style.getPropertyValue('--tilt-y')).toBe('');
  }));

  it('settles flat when the pointer leaves', fakeAsync(() => {
    moveTo(0, 1);
    tick(20);
    el.dispatchEvent(new PointerEvent('pointerleave'));
    expect(el.style.getPropertyValue('--tilt-x')).toBe('0deg');
    expect(el.style.getPropertyValue('--tilt-y')).toBe('0deg');
    expect(el.classList).toContain('tilt');
  }));
});
