import { Component, Input } from '@angular/core';

/**
 * A player's face: their profile picture as a circle, or the gold initial when they have
 * none (or it fails to load). Decorative: the player's name is always written next to it.
 */
@Component({
  selector: 'app-avatar',
  templateUrl: './avatar.component.html',
  styleUrls: ['./avatar.component.scss']
})
export class AvatarComponent {
  private pictureUrl: string | null = null;
  failed = false;

  @Input() set url(value: string | null | undefined) {
    if ((value ?? null) !== this.pictureUrl) {
      this.pictureUrl = value ?? null;
      this.failed = false;
    }
  }
  get url(): string | null {
    return this.pictureUrl;
  }

  @Input() name = '';
  /** Diameter in px. */
  @Input() size = 28;

  get initial(): string {
    return (this.name.trim().charAt(0) || '?').toUpperCase();
  }

  get showPicture(): boolean {
    return !!this.pictureUrl && !this.failed;
  }
}
