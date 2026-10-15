import {
  AfterViewInit, Component, ElementRef, EventEmitter, Input, OnDestroy, Output, ViewChild
} from '@angular/core';
import { Subscription, finalize, take } from 'rxjs';
import { CardView, User } from '../../models/api.model';
import { apiErrorMessage } from '../../services/api.util';
import {
  Frame, MAX_ZOOM, displayRect, framedJpeg, initialFrame, panFrame, zoomFrame
} from '../../services/avatar-crop';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { UserService } from '../../services/user.service';

type Step = 'choose' | 'cards' | 'frame';

interface Source { url: string; width: number; height: number; image: HTMLImageElement; revoke: boolean; }

/**
 * Add, change or remove your profile picture (docs/superpowers/specs/2026-10-14-profile-pictures-design.md):
 * pick a photo or one of your cards' art, frame it in a circle, save. A native modal <dialog>,
 * like the vote page's player sheet. `prompted` is the after-login / first-visit ask.
 */
@Component({
  selector: 'app-profile-picture',
  templateUrl: './profile-picture.component.html',
  styleUrls: ['./profile-picture.component.scss']
})
export class ProfilePictureComponent implements AfterViewInit, OnDestroy {
  @Input() user!: User;
  /** Opened by the app (after login, or the one-time nudge) rather than by the player. */
  @Input() prompted = false;
  @Output() closed = new EventEmitter<void>();

  @ViewChild('dialog', { static: true }) private dialog!: ElementRef<HTMLDialogElement>;
  @ViewChild('fileInput') private fileInput?: ElementRef<HTMLInputElement>;

  readonly maxZoom = MAX_ZOOM;
  /** The framing window's size in CSS px (see .framer). */
  readonly viewSize = 260;

  step: Step = 'choose';
  busy = false;
  error: string | null = null;
  source: Source | null = null;
  frame: Frame = { zoom: 1, cx: 0, cy: 0 };
  /** My finished cards with art; null while loading. */
  cards: CardView[] | null = null;

  private readonly pointers = new Map<number, { x: number; y: number }>();
  private pinchDistance = 0;
  private subs = new Subscription();

  constructor(private users: UserService, private generations: GenerationService,
              private media: MediaService) {}

  ngAfterViewInit(): void {
    const dialog = this.dialog.nativeElement;
    if (!dialog.open) {
      if (typeof dialog.showModal === 'function') {
        dialog.showModal();
      } else {
        dialog.setAttribute('open', '');
      }
    }
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.releaseSource();
    const dialog = this.dialog.nativeElement;
    if (dialog.open && typeof dialog.close === 'function') {
      dialog.close();
    }
  }

  get title(): string {
    if (this.step === 'cards') {
      return 'Pick a card';
    }
    if (this.step === 'frame') {
      return 'Frame your picture';
    }
    return this.prompted && !this.user.avatarUrl ? 'Add a profile picture?' : 'Profile picture';
  }

  /** Where to draw the picture inside the framing window. */
  get placement(): { x: number; y: number; width: number; height: number } {
    const s = this.source;
    return s ? displayRect(s.width, s.height, this.frame, this.viewSize) : { x: 0, y: 0, width: 0, height: 0 };
  }

  close(): void {
    this.closed.emit();
  }

  /** Esc steps back a screen before it closes the dialog. */
  onCancel(event: Event): void {
    event.preventDefault();
    if (this.busy) {
      return;
    }
    if (this.step !== 'choose') {
      this.back();
    } else {
      this.close();
    }
  }

  onDialogClick(event: MouseEvent): void {
    if (event.target === this.dialog.nativeElement && !this.busy) {
      this.close();
    }
  }

  back(): void {
    this.error = null;
    this.releaseSource();
    this.step = 'choose';
  }

  chooseFile(): void {
    this.fileInput?.nativeElement.click();
  }

  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = ''; // picking the same file again still fires change
    if (!file) {
      return;
    }
    if (file.type && !file.type.startsWith('image/')) {
      this.error = "That file isn't a picture.";
      return;
    }
    this.load(URL.createObjectURL(file), true);
  }

  showCards(): void {
    this.error = null;
    this.step = 'cards';
    if (this.cards === null) {
      this.subs.add(this.generations.myCards().subscribe({
        next: cards => (this.cards = cards.filter(c => c.status === 'done' && !!c.artImageUrl)),
        error: err => {
          this.cards = [];
          this.error = apiErrorMessage(err, 'Could not load your cards.');
        }
      }));
    }
  }

  pickCard(card: CardView): void {
    this.error = null;
    this.busy = true;
    // Through MediaService: a same-origin blob URL, so the canvas can export it.
    this.media.src(card.artImageUrl).pipe(take(1), finalize(() => (this.busy = false))).subscribe(url => {
      if (url) {
        this.load(url, false);
      } else {
        this.error = 'Could not load that art.';
      }
    });
  }

  // ----- framing -----
  onZoomInput(event: Event): void {
    this.setZoom(Number((event.target as HTMLInputElement).value));
  }

  onWheel(event: WheelEvent): void {
    event.preventDefault();
    this.setZoom(this.frame.zoom * Math.exp(-event.deltaY * 0.0015));
  }

  onPointerDown(event: PointerEvent): void {
    (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
    this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    this.pinchDistance = this.distance();
  }

  onPointerMove(event: PointerEvent): void {
    const last = this.pointers.get(event.pointerId);
    if (!last || !this.source) {
      return;
    }
    const now = { x: event.clientX, y: event.clientY };
    this.pointers.set(event.pointerId, now);
    if (this.pointers.size >= 2) {
      const d = this.distance();
      if (this.pinchDistance > 0 && d > 0) {
        this.setZoom(this.frame.zoom * (d / this.pinchDistance));
      }
      this.pinchDistance = d;
      return;
    }
    const s = this.source;
    this.frame = panFrame(s.width, s.height, this.frame, now.x - last.x, now.y - last.y, this.viewSize);
  }

  onPointerUp(event: PointerEvent): void {
    this.pointers.delete(event.pointerId);
    this.pinchDistance = this.distance();
  }

  /** Arrow keys move the picture, + and - zoom. */
  onFramerKey(event: KeyboardEvent): void {
    const s = this.source;
    if (!s) {
      return;
    }
    const step = event.shiftKey ? 40 : 10;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step]
    };
    if (moves[event.key]) {
      const [dx, dy] = moves[event.key];
      this.frame = panFrame(s.width, s.height, this.frame, dx, dy, this.viewSize);
    } else if (event.key === '+' || event.key === '=') {
      this.setZoom(this.frame.zoom * 1.15);
    } else if (event.key === '-' || event.key === '_') {
      this.setZoom(this.frame.zoom / 1.15);
    } else {
      return;
    }
    event.preventDefault();
  }

  save(): void {
    const s = this.source;
    if (!s || this.busy) {
      return;
    }
    this.error = null;
    this.busy = true;
    framedJpeg(s.image, s.width, s.height, this.frame).then(
      blob => this.subs.add(this.users.setAvatar(blob).pipe(finalize(() => (this.busy = false))).subscribe({
        next: () => this.close(),
        error: err => (this.error = apiErrorMessage(err, 'Your picture did not save. Please try again.'))
      })),
      () => {
        this.busy = false;
        this.error = "Couldn't read that picture. Try another one.";
      });
  }

  remove(): void {
    if (this.busy) {
      return;
    }
    this.error = null;
    this.busy = true;
    this.subs.add(this.users.removeAvatar().pipe(finalize(() => (this.busy = false))).subscribe({
      error: err => (this.error = apiErrorMessage(err, 'Could not remove your picture.'))
    }));
  }

  trackCard(_index: number, card: CardView): string {
    return card.id;
  }

  private setZoom(zoom: number): void {
    const s = this.source;
    if (s) {
      this.frame = zoomFrame(s.width, s.height, this.frame, zoom);
    }
  }

  private distance(): number {
    const [a, b] = Array.from(this.pointers.values());
    return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
  }

  private load(url: string, revoke: boolean): void {
    const image = new Image();
    this.busy = true;
    image.onload = () => {
      this.busy = false;
      this.releaseSource();
      this.source = { url, width: image.naturalWidth, height: image.naturalHeight, image, revoke };
      this.frame = initialFrame(image.naturalWidth, image.naturalHeight);
      this.step = 'frame';
    };
    image.onerror = () => {
      this.busy = false;
      if (revoke) {
        URL.revokeObjectURL(url);
      }
      this.error = "Couldn't open that picture. Try a JPEG or PNG.";
    };
    image.src = url;
  }

  private releaseSource(): void {
    if (this.source?.revoke) {
      URL.revokeObjectURL(this.source.url);
    }
    this.source = null;
    this.pointers.clear();
  }
}
