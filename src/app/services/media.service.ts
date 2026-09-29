import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, map, of, shareReplay, take } from 'rxjs';
import { isApiRequest, mediaUrl } from './api.util';

/**
 * Card and art PNGs are served by the API. Loading them through HttpClient (as blobs)
 * sends the ngrok-skip-browser-warning header, which a plain <img src> cannot, so the
 * images still load when the host runs behind a free ngrok tunnel. Media is immutable
 * per card id, so object URLs are cached for the session.
 */
@Injectable({ providedIn: 'root' })
export class MediaService {
  private readonly cache = new Map<string, Observable<string>>();

  constructor(private http: HttpClient) {}

  /** A displayable src for a (possibly relative) media URL. */
  src(url: string | null | undefined): Observable<string | null> {
    const absolute = mediaUrl(url);
    if (!absolute) {
      return of(null);
    }
    if (!isApiRequest(absolute)) {
      return of(absolute);
    }
    let cached = this.cache.get(absolute);
    if (!cached) {
      cached = this.http.get(absolute, { responseType: 'blob' }).pipe(
        map(blob => URL.createObjectURL(blob)),
        catchError(() => {
          this.cache.delete(absolute);
          return of(absolute);
        }),
        shareReplay(1)
      );
      this.cache.set(absolute, cached);
    }
    return cached;
  }

  /** Saves the image to the device as `filename`. */
  download(url: string | null | undefined, filename: string): void {
    this.src(url).pipe(take(1)).subscribe(src => {
      if (!src) {
        return;
      }
      const link = document.createElement('a');
      link.href = src;
      link.download = filename;
      link.target = '_blank';
      link.rel = 'noopener';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }
}
