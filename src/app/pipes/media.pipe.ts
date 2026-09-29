import { Pipe, PipeTransform } from '@angular/core';
import { Observable } from 'rxjs';
import { MediaService } from '../services/media.service';

/** `[src]="view.cardImageUrl | media | async"` — see MediaService. */
@Pipe({ name: 'media' })
export class MediaPipe implements PipeTransform {
  constructor(private media: MediaService) {}

  transform(url: string | null | undefined): Observable<string | null> {
    return this.media.src(url);
  }
}
