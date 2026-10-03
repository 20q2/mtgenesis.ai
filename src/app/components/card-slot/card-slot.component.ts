import {
  Component, EventEmitter, HostListener, Input, OnDestroy, Output, TemplateRef, ViewChild, ViewContainerRef
} from '@angular/core';
import { Overlay, OverlayRef } from '@angular/cdk/overlay';
import { TemplatePortal } from '@angular/cdk/portal';
import { CardView } from '../../models/api.model';
import { isPainting, queueLine } from '../../services/card-status';

/**
 * One card of a commander set (or a gallery tile): status line while pending
 * (spec §7), the finished card (tap to enlarge), or the failure reason, plus Reroll.
 *
 * The enlarged view is rendered through a CDK Overlay (attached to the overlay
 * container under <body>), because every slot sits inside a `.panel` whose
 * backdrop-filter would otherwise trap a `position: fixed` lightbox inside the panel.
 */
@Component({
  selector: 'app-card-slot',
  templateUrl: './card-slot.component.html',
  styleUrls: ['./card-slot.component.scss']
})
export class CardSlotComponent implements OnDestroy {
  @Input() view: CardView | null = null;
  @Input() canReroll = false;
  @Input() showReroll = true;
  @Input() rerollLabel = 'Reroll';
  /** Shown when there is no card yet, e.g. "Version 2". */
  @Input() label = '';
  @Output() reroll = new EventEmitter<CardView>();

  @ViewChild('lightbox') private lightboxTpl!: TemplateRef<unknown>;
  private overlayRef: OverlayRef | null = null;

  constructor(private overlay: Overlay, private viewContainerRef: ViewContainerRef) {}

  get enlarged(): boolean {
    return !!this.overlayRef;
  }

  queueText(view: CardView): string | null {
    return queueLine(view);
  }

  painting(view: CardView): boolean {
    return isPainting(view);
  }

  onReroll(): void {
    if (this.view && this.canReroll) {
      this.reroll.emit(this.view);
    }
  }

  open(): void {
    if (this.overlayRef || !this.view?.cardImageUrl) {
      return;
    }
    this.overlayRef = this.overlay.create({
      positionStrategy: this.overlay.position().global().top('0').left('0'),
      scrollStrategy: this.overlay.scrollStrategies.block(),
      width: '100vw',
      height: '100vh',
      panelClass: 'card-lightbox-pane'
    });
    this.overlayRef.attach(new TemplatePortal(this.lightboxTpl, this.viewContainerRef));
  }

  close(): void {
    this.overlayRef?.dispose();
    this.overlayRef = null;
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    this.close();
  }

  ngOnDestroy(): void {
    this.close();
  }
}
