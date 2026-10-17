import { Component, OnInit } from '@angular/core';
import { Observable, catchError, firstValueFrom, forkJoin, map, of, switchMap } from 'rxjs';
import { EventView, PoolEntryView, PoolView } from '../../models/api.model';
import { apiErrorMessage, safeFileName } from '../../services/api.util';
import { EventService } from '../../services/event.service';
import { MediaService } from '../../services/media.service';
import { PoolService } from '../../services/pool.service';
import { SheetCard, SheetSpec, loadImage, sheetBlob } from '../../services/results-export';
import { PlayerCommanders, latestClosed, playerCommanders, pointsLabel, poolWinners } from '../../services/results';

/** Which export sheet: the pool, or one player's commanders. */
type SheetKey = 'pool' | `player:${string}`;

/**
 * /results: the last closed event's legal commanders for each player, and the cards that made the
 * last closed Knowledge Pool. Each group exports as a PNG sheet (or copies to the clipboard)
 * sized to post on Discord.
 */
@Component({
  selector: 'app-results-page',
  templateUrl: './results-page.component.html',
  styleUrls: ['./results-page.component.scss']
})
export class ResultsPageComponent implements OnInit {
  loading = true;
  error: string | null = null;
  event: EventView | null = null;
  pool: PoolView | null = null;
  players: PlayerCommanders[] = [];
  poolCards: PoolEntryView[] = [];

  /** The sheet being drawn right now. */
  busy: SheetKey | null = null;
  /** The sheet just copied, for a moment of "Copied". */
  copied: SheetKey | null = null;
  exportError: string | null = null;
  readonly canCopy = typeof ClipboardItem !== 'undefined' && !!navigator.clipboard?.write;

  private readonly sheets = new Map<SheetKey, Promise<Blob>>();

  constructor(private events: EventService, private pools: PoolService, private media: MediaService) {}

  ngOnInit(): void {
    forkJoin({ event: this.loadEvent(), pool: this.loadPool() }).subscribe({
      next: ({ event, pool }) => {
        this.event = event;
        this.pool = pool;
        this.players = playerCommanders(event?.sets ?? []);
        this.poolCards = poolWinners(pool?.entries ?? []);
        this.loading = false;
      },
      error: err => {
        this.error = apiErrorMessage(err, 'Could not load the results.');
        this.loading = false;
      }
    });
  }

  get commanderCount(): number {
    return this.players.reduce((n, p) => n + p.commanders.length, 0);
  }

  playerKey(player: PlayerCommanders): SheetKey {
    return `player:${player.userId}`;
  }

  pointsLabel(points: number): string {
    return pointsLabel(points);
  }

  // ----- export -----
  download(key: SheetKey): void {
    this.run(key, this.sheet(key).then(blob => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = this.fileName(key);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    }));
  }

  /** Straight to the clipboard, to paste into Discord. The blob promise goes in as-is so Safari
   *  still counts the click as the gesture. */
  copy(key: SheetKey): void {
    const write = navigator.clipboard.write([new ClipboardItem({ 'image/png': this.sheet(key) })]);
    this.run(key, write.then(() => {
      this.copied = key;
      setTimeout(() => {
        if (this.copied === key) {
          this.copied = null;
        }
      }, 2500);
    }));
  }

  private run(key: SheetKey, work: Promise<void>): void {
    this.busy = key;
    this.exportError = null;
    work.catch(() => {
      this.sheets.delete(key);
      this.exportError = 'Could not make that image. Please try again.';
    }).finally(() => {
      if (this.busy === key) {
        this.busy = null;
      }
    });
  }

  /** Each sheet is drawn once per visit; a copy after a download reuses it. */
  private sheet(key: SheetKey): Promise<Blob> {
    let blob = this.sheets.get(key);
    if (!blob) {
      blob = this.spec(key).then(spec => sheetBlob(spec));
      this.sheets.set(key, blob);
    }
    return blob;
  }

  private async spec(key: SheetKey): Promise<SheetSpec> {
    const logo = loadImage('assets/site_logo.png');
    const footer = this.closedDate();
    if (key === 'pool') {
      const pool = this.pool!;
      const cards = await Promise.all(this.poolCards.map(async (e): Promise<SheetCard> => ({
        image: await this.image(e.card.cardImageUrl),
        caption: `by ${e.username}`,
        detail: [`#${e.rank}`, pointsLabel(e.points), e.tiedAtCutoff ? 'tied at the line' : '',
                 e.card.errata ? 'errata' : ''].filter(Boolean).join(' · '),
        person: { image: await this.image(e.avatarUrl), name: e.username }
      })));
      return {
        eyebrow: pool.name, title: 'Knowledge Pool',
        subtitle: `${cards.length} ${cards.length === 1 ? 'card' : 'cards'} made the pool, ` +
                  `legal in every deck · ${pool.entries.length} entered`,
        cards, logo: await logo, footer
      };
    }
    const player = this.players.find(p => this.playerKey(p) === key)!;
    const cards: SheetCard[] = [];
    for (const lc of player.commanders) {
      for (const card of lc.cards) {
        const tie = lc.outcome !== 'winner' ? ' · host picks' : '';
        cards.push({
          image: await this.image(card.cardImageUrl),
          caption: lc.set.cmc != null ? `${lc.set.cmc} CMC${tie}` : `${lc.set.commanderName}${tie}`,
          detail: lc.label + (card.errata ? ' · errata' : '')
        });
      }
    }
    return {
      eyebrow: this.event!.name, title: player.username,
      subtitle: `Legal commanders · ${player.commanders.map(c => c.set.commanderName).join(' · ')}`,
      person: { image: await this.image(player.avatarUrl), name: player.username },
      cards, logo: await logo, footer
    };
  }

  private async image(url: string | null): Promise<HTMLImageElement | null> {
    return loadImage(await firstValueFrom(this.media.src(url)));
  }

  private fileName(key: SheetKey): string {
    if (key === 'pool') {
      return `${safeFileName(this.pool?.name, 'event')}_knowledge_pool.png`;
    }
    const player = this.players.find(p => this.playerKey(p) === key);
    return `${safeFileName(this.event?.name, 'event')}_${safeFileName(player?.username, 'player')}_commanders.png`;
  }

  private closedDate(): string {
    const at = this.event?.closedAt ?? this.pool?.closedAt;
    return at ? new Date(at).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' }) : '';
  }

  // ----- loading -----
  /** The newest closed event with its sets, or null when none has closed. */
  private loadEvent(): Observable<EventView | null> {
    return this.events.list().pipe(
      map(latestClosed),
      switchMap(ev => (ev ? this.events.get(ev.id) : of(null)))
    );
  }

  /** The newest closed pool; the page still shows the commanders if the pool can't load. */
  private loadPool(): Observable<PoolView | null> {
    return this.pools.list().pipe(
      map(latestClosed),
      switchMap(p => (p ? this.pools.get(p.id) : of(null))),
      catchError(() => of(null))
    );
  }
}
