import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { EMPTY, Observable, catchError, exhaustMap, takeWhile, throwError, timer } from 'rxjs';
import { CardParams, CardView, GenerationRequest, GenerationResponse } from '../models/api.model';
import { Card, Rarity } from '../models/card.model';
import { api, mediaUrl } from './api.util';
import { isFinished } from './card-status';

export const WATCH_INTERVAL_MS = 2000;

/** Submit, reroll and poll generation jobs (spec §4 /generations, /cards/<id>). */
@Injectable({ providedIn: 'root' })
export class GenerationService {
  constructor(private http: HttpClient) {}

  submit(req: GenerationRequest): Observable<GenerationResponse> {
    return this.http.post<GenerationResponse>(api('/generations'), req);
  }

  reroll(cardId: string): Observable<CardView> {
    return this.http.post<CardView>(api(`/cards/${encodeURIComponent(cardId)}/reroll`), {});
  }

  getCard(cardId: string): Observable<CardView> {
    return this.http.get<CardView>(api(`/cards/${encodeURIComponent(cardId)}`));
  }

  /**
   * Polls GET /cards/<id> every 2s and emits each CardView; completes after the
   * card is done or failed (that last view is emitted). Network blips and 5xx are
   * skipped so a flaky connection doesn't abandon the job; other errors propagate.
   */
  watch(cardId: string): Observable<CardView> {
    return timer(WATCH_INTERVAL_MS, WATCH_INTERVAL_MS).pipe(
      exhaustMap(() => this.getCard(cardId).pipe(
        catchError((err: unknown) =>
          err instanceof HttpErrorResponse && (err.status === 0 || err.status >= 500)
            ? EMPTY
            : throwError(() => err))
      )),
      takeWhile(view => !isFinished(view.status), true)
    );
  }

  /** Maps a CardView onto the display model, prefixing media URLs with environment.apiUrl. */
  toCard(view: CardView, base: Card): Card {
    const out: Card = { ...base };
    const c = view.card;
    if (c) {
      const fields: (keyof CardParams | 'flavorText')[] = [
        'name', 'manaCost', 'supertype', 'type', 'subtype', 'colors', 'cmc',
        'description', 'flavorText', 'power', 'toughness'
      ];
      for (const key of fields) {
        const value = (c as any)[key];
        if (value !== undefined && value !== null) {
          (out as any)[key] = value;
        }
      }
      if (c.rarity) {
        out.rarity = c.rarity as Rarity;
      }
    }
    const cardImage = mediaUrl(view.cardImageUrl);
    if (cardImage) {
      out.cardImageUrl = cardImage;
    }
    const art = mediaUrl(view.artImageUrl);
    if (art) {
      out.imageUrl = art;
    }
    return out;
  }

  /** The generator's card params (same shape as the legacy cardData) from a form card. */
  cardParams(card: Card): CardParams {
    const params: CardParams = {
      name: card.name ?? '',
      manaCost: card.manaCost ?? '',
      colors: card.colors ?? [],
      type: card.type ?? '',
      rarity: card.rarity ?? Rarity.COMMON,
      cmc: Number(card.cmc) || 0
    };
    if (card.supertype) { params.supertype = card.supertype; }
    if (card.subtype) { params.subtype = card.subtype; }
    if (card.description) { params.description = card.description; }
    if (card.power) { params.power = card.power; }
    if (card.toughness) { params.toughness = card.toughness; }
    return params;
  }

  /** The art prompt the form generated, or a fallback from name and type. */
  promptFor(card: Card, fallback?: Card): string {
    return card.artPrompt || fallback?.artPrompt || `Fantasy art of ${card.name || 'a card'}, ${card.type || 'fantasy'}`;
  }
}
