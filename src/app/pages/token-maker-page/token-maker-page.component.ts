import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Subscription, catchError, debounceTime, finalize, map, of, startWith, switchMap } from 'rxjs';
import { CardView, TokenFrame, TokenRequest, TokenType } from '../../models/api.model';
import { apiErrorMessage, mediaUrl, safeFileName } from '../../services/api.util';
import { isPainting, queueLine } from '../../services/card-status';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { QueueService } from '../../services/queue.service';
import { TokenService, tokenFrameCode } from '../../services/token.service';

interface TokenPreset {
  label: string;
  type: TokenType;
  subtype: string;
  colors: string[];
  power?: string;
  toughness?: string;
  description?: string;
}

/** Quick starts: the tokens real cards make most. */
export const TOKEN_PRESETS: TokenPreset[] = [
  { label: 'Goblin', type: 'Creature', subtype: 'Goblin', colors: ['R'], power: '1', toughness: '1' },
  { label: 'Zombie', type: 'Creature', subtype: 'Zombie', colors: ['B'], power: '2', toughness: '2' },
  { label: 'Spirit', type: 'Creature', subtype: 'Spirit', colors: ['W'], power: '1', toughness: '1', description: 'Flying' },
  { label: 'Soldier', type: 'Creature', subtype: 'Soldier', colors: ['W'], power: '1', toughness: '1' },
  { label: 'Saproling', type: 'Creature', subtype: 'Saproling', colors: ['G'], power: '1', toughness: '1' },
  { label: 'Beast', type: 'Creature', subtype: 'Beast', colors: ['G'], power: '3', toughness: '3' },
  { label: 'Drake', type: 'Creature', subtype: 'Drake', colors: ['U'], power: '2', toughness: '2', description: 'Flying' },
  { label: 'Dragon', type: 'Creature', subtype: 'Dragon', colors: ['R'], power: '4', toughness: '4', description: 'Flying' },
  { label: 'Treasure', type: 'Artifact', subtype: 'Treasure', colors: [],
    description: '{T}, Sacrifice this artifact: Add one mana of any color.' },
  { label: 'Clue', type: 'Artifact', subtype: 'Clue', colors: [],
    description: '{2}, Sacrifice this artifact: Draw a card.' },
  { label: 'Food', type: 'Artifact', subtype: 'Food', colors: [],
    description: '{2}, {T}, Sacrifice this artifact: You gain 3 life.' }
];

export const TOKEN_COLORS = [
  { code: 'W', label: 'White' }, { code: 'U', label: 'Blue' }, { code: 'B', label: 'Black' },
  { code: 'R', label: 'Red' }, { code: 'G', label: 'Green' }
];

const PT_PATTERN = /^(\d{1,2}|\*|X|x)$/;
const PREVIEW_DEBOUNCE_MS = 350;

/**
 * Token Maker (/tokens): the player types a token and picks one of the frames, the AI paints
 * the art and writes a funny flavor text (and a name, when it's blank). Tokens go through the
 * same queue as cards and land in My Gallery. While the player types, the server draws the
 * token on placeholder art so the frame and text are exactly what will print.
 */
@Component({
  selector: 'app-token-maker-page',
  templateUrl: './token-maker-page.component.html',
  styleUrls: ['../create-page/create-page.component.scss', './token-maker-page.component.scss']
})
export class TokenMakerPageComponent implements OnInit, OnDestroy {
  readonly presets = TOKEN_PRESETS;
  readonly colorOptions = TOKEN_COLORS;
  frames: TokenFrame[] = [];
  types: TokenType[] = ['Creature', 'Artifact Creature', 'Enchantment Creature', 'Artifact', 'Enchantment', 'Land'];

  form: FormGroup;
  /** The live preview (data: URL), or null before the first one arrives. */
  previewImage: string | null = null;
  previewLoading = false;

  isGenerating = false;
  /** The running or finished token. */
  job: CardView | null = null;
  sharing = false;
  error: string | null = null;
  successMessage: string | null = null;
  online = false;

  private subs = new Subscription();
  private jobSub?: Subscription;
  private successTimer: any = null;

  constructor(
    fb: FormBuilder,
    private tokens: TokenService,
    private generation: GenerationService,
    private media: MediaService,
    private queue: QueueService
  ) {
    this.form = fb.group({
      frame: ['classic'],
      name: ['', Validators.maxLength(30)],
      type: ['Creature' as TokenType],
      subtype: ['Goblin', Validators.maxLength(40)],
      colors: [['R'] as string[]],
      power: ['1'],
      toughness: ['1'],
      description: ['', Validators.maxLength(300)],
      flavorText: ['', Validators.maxLength(160)],
      artPrompt: ['', Validators.maxLength(400)]
    });
  }

  ngOnInit(): void {
    this.subs.add(this.tokens.frames().subscribe({
      next: res => {
        this.frames = res.frames;
        this.types = res.types;
      },
      error: err => (this.error = apiErrorMessage(err, 'Could not load the token frames.'))
    }));
    this.subs.add(this.queue.online$.subscribe(online => (this.online = online)));
    this.subs.add(this.form.valueChanges.pipe(
      startWith(this.form.value),
      debounceTime(PREVIEW_DEBOUNCE_MS),
      switchMap(() => {
        this.previewLoading = true;
        return this.tokens.preview(this.request()).pipe(
          catchError(() => of(null)),
          finalize(() => (this.previewLoading = false))
        );
      })
    ).subscribe(image => {
      if (image) {
        this.previewImage = image;
      }
    }));
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.jobSub?.unsubscribe();
    clearTimeout(this.successTimer);
  }

  // ----- form helpers -----

  get isCreature(): boolean {
    return (this.form.value.type as string).toLowerCase().includes('creature');
  }

  get colors(): string[] {
    return this.form.value.colors as string[];
  }

  get frameCode(): string {
    return tokenFrameCode(this.colors, this.form.value.type);
  }

  get selectedFrame(): TokenFrame | undefined {
    return this.frames.find(f => f.id === this.form.value.frame);
  }

  thumbnail(frame: TokenFrame): string {
    return this.tokens.thumbnailUrl(frame.id, this.frameCode);
  }

  selectFrame(id: string): void {
    this.form.patchValue({ frame: id });
  }

  toggleColor(code: string): void {
    const colors = this.colors.includes(code) ? this.colors.filter(c => c !== code) : [...this.colors, code];
    this.form.patchValue({ colors });
  }

  selectType(type: TokenType): void {
    this.form.patchValue({ type });
  }

  applyPreset(preset: TokenPreset): void {
    this.form.patchValue({
      type: preset.type,
      subtype: preset.subtype,
      colors: preset.colors,
      power: preset.power ?? '',
      toughness: preset.toughness ?? '',
      description: preset.description ?? '',
      name: ''
    });
  }

  ptInvalid(key: 'power' | 'toughness'): boolean {
    return this.isCreature && !PT_PATTERN.test(String(this.form.value[key] ?? '').trim());
  }

  /** Why Make token is disabled, or null when it can go. */
  blocker(): string | null {
    const v = this.form.value;
    if (!this.online) {
      return 'The server is offline';
    }
    if (this.form.invalid) {
      return 'Some fields are too long';
    }
    if (this.ptInvalid('power') || this.ptInvalid('toughness')) {
      return 'Creature tokens need a power and toughness';
    }
    if (!v.name.trim() && !v.subtype.trim() && !v.artPrompt.trim()) {
      return 'Give it a name, a type or an art description';
    }
    return null;
  }

  request(): TokenRequest {
    const v = this.form.value;
    const req: TokenRequest = {
      frame: v.frame,
      name: v.name.trim(),
      type: v.type,
      subtype: v.subtype.trim(),
      colors: v.colors,
      description: v.description.trim(),
      flavorText: v.flavorText.trim(),
      artPrompt: v.artPrompt.trim()
    };
    if (this.isCreature) {
      req.power = String(v.power ?? '').trim();
      req.toughness = String(v.toughness ?? '').trim();
    }
    return req;
  }

  // ----- generation -----

  makeToken(): void {
    if (this.isGenerating || this.blocker()) {
      return;
    }
    this.error = null;
    this.successMessage = null;
    this.isGenerating = true;
    this.job = null;
    this.jobSub = this.tokens.create(this.request()).pipe(
      switchMap(job => {
        this.job = job;
        return this.generation.watch(job.id);
      }),
      finalize(() => (this.isGenerating = false))
    ).subscribe({
      next: view => {
        this.job = view;
        if (view.status === 'done') {
          this.showSuccess('Token made! It is saved in My Gallery.');
        } else if (view.status === 'failed') {
          this.error = `The token failed: ${view.error || 'unknown error'}`;
        }
      },
      error: err => (this.error = `The token failed: ${apiErrorMessage(err)}`)
    });
  }

  /** Back to the live preview to make another one. */
  startOver(): void {
    this.job = null;
  }

  get finished(): CardView | null {
    return this.job?.status === 'done' && this.job.cardImageUrl ? this.job : null;
  }

  /** Flavor text the frame doesn't print (Full Art), shown under the card instead. */
  get offCardFlavor(): string | null {
    const done = this.finished;
    if (!done || this.frames.find(f => f.id === done.card?.frame)?.hasText !== false) {
      return null;
    }
    return done.card?.flavorText || null;
  }

  statusLine(job: CardView): string {
    const queued = queueLine(job);
    if (queued) {
      return queued;
    }
    if (isPainting(job)) {
      return 'Painting your token…';
    }
    switch (job.status) {
      case 'queued': return 'Queued…';
      case 'generating': return job.artReady ? 'Thinking of something funny…' : 'Generating…';
      case 'rendering': return 'Framing it…';
      default: return '';
    }
  }

  downloadCard(): void {
    const done = this.finished;
    if (done) {
      this.media.download(done.cardImageUrl, `${safeFileName(done.card?.name, 'token')}_token.png`);
    }
  }

  downloadArt(): void {
    const done = this.finished;
    if (done?.artImageUrl) {
      this.media.download(done.artImageUrl, `${safeFileName(done.card?.name, 'token')}_artwork.png`);
    }
  }

  toggleShare(): void {
    const done = this.finished;
    if (!done || this.sharing) {
      return;
    }
    this.sharing = true;
    this.generation.share(done.id, !done.shared).pipe(finalize(() => (this.sharing = false))).subscribe({
      next: view => {
        this.job = view;
        if (view.shared) {
          this.showSuccess('Shared! Everyone can see it in the Community Gallery.');
        }
      },
      error: err => (this.error = apiErrorMessage(err, 'Could not share the token.'))
    });
  }

  cardImage(): string | null {
    return mediaUrl(this.finished?.cardImageUrl);
  }

  trackFrame(_i: number, frame: TokenFrame): string {
    return frame.id;
  }

  private showSuccess(message: string): void {
    this.successMessage = message;
    clearTimeout(this.successTimer);
    this.successTimer = setTimeout(() => (this.successMessage = null), 20000);
  }
}
