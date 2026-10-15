import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map, shareReplay } from 'rxjs';
import { CardView, TokenFramesResponse, TokenRequest } from '../models/api.model';
import { api } from './api.util';

const WUBRG = ['W', 'U', 'B', 'R', 'G'];

/**
 * The frame image a token uses: W/U/B/R/G for one color, M for two or more, L for a
 * colorless land, else A. Mirrors token_renderer.frame_code, so keep them in sync.
 */
export function tokenFrameCode(colors: string[], type: string): string {
  const found = WUBRG.filter(c => colors.includes(c));
  if (found.length > 1) {
    return 'M';
  }
  if (found.length === 1) {
    return found[0];
  }
  return type.toLowerCase().includes('land') ? 'L' : 'A';
}

/** Token Maker: frames, live previews and queuing a token (spec: proxy-server/tokens.py). */
@Injectable({ providedIn: 'root' })
export class TokenService {
  private frames$?: Observable<TokenFramesResponse>;

  constructor(private http: HttpClient) {}

  /** The frame list and token types (fetched once). */
  frames(): Observable<TokenFramesResponse> {
    this.frames$ ??= this.http.get<TokenFramesResponse>(api('/tokens/frames')).pipe(
      shareReplay({ bufferSize: 1, refCount: false })
    );
    return this.frames$;
  }

  /** A frame's sample thumbnail in a color code (public media, like card images). */
  thumbnailUrl(frameId: string, code: string): string {
    return `/api/v1/tokens/frames/${encodeURIComponent(frameId)}/${code}.png`;
  }

  /** The token as typed so far on placeholder art, as a data: URL. */
  preview(token: TokenRequest): Observable<string> {
    return this.http.post<{ image: string }>(api('/tokens/preview'), { token }).pipe(map(r => r.image));
  }

  /** Queues the token; poll it with GenerationService.watch. */
  create(token: TokenRequest): Observable<CardView> {
    return this.http.post<CardView>(api('/tokens'), { token });
  }
}
