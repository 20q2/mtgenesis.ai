/**
 * API request/response types.
 * Legacy types reconstructed 2026-09-28 from usages (the original was never committed).
 */

export interface CardValidationRequest {
  card: Record<string, unknown>;
}

export interface CardValidationResponse {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

/** Legacy single-card request for POST /api/v1/create_card */
export interface CardGenerationRequest {
  prompt: string;
  width: number;
  height: number;
  cardData: {
    name: string;
    manaCost: string;
    supertype?: string;
    colors: string[];
    type: string;
    subtype?: string;
    rarity: string;
    cmc: number;
    description?: string;
    power?: string;
    toughness?: string;
  };
}

/** Legacy response for POST /api/v1/create_card */
export interface CardGenerationResponse {
  cardData: string | null;
  imageData: string | null;
  card_image: string | null;
  warning?: string;
  generation_time?: number;
}

export interface ApiErrorResponse {
  error: string;
  details?: string;
}

// ===== AI Night API (spec §4) =====

export type CardStatus = 'queued' | 'generating' | 'rendering' | 'done' | 'failed';

export interface User { id: string; username: string; }

/** Card properties sent to the generator (same shape as the legacy cardData) */
export interface CardParams {
  name: string;
  manaCost: string;
  supertype?: string;
  colors: string[];
  type: string;
  subtype?: string;
  rarity: string;
  cmc: number;
  description?: string;
  power?: string;
  toughness?: string;
}

export interface GeneratedCardData extends CardParams { flavorText?: string; }

export interface CardView {
  id: string;
  userId: string;
  setId: string | null;
  slot: number | null;
  replaced: boolean;
  status: CardStatus;
  error: string | null;
  textReady: boolean;
  artReady: boolean;
  /** 1-based place in the image queue, 0 while painting, null once art is ready or finished */
  queuePosition: number | null;
  etaSeconds: number | null;
  /** Final card data when done; the requested card params while pending */
  card: GeneratedCardData | null;
  /** Relative URL, e.g. /api/v1/media/cards/<id>.png; prefix with environment.apiUrl */
  cardImageUrl: string | null;
  artImageUrl: string | null;
  createdAt: string;
  /** True while the card is shared to the gallery's Community tab */
  shared: boolean;
  /** The card's entry in the open Knowledge Pool, if it has one */
  poolEntryId: string | null;
}

/** A card on the gallery's Community tab (GET /cards/shared), with its maker. */
export interface SharedCardView extends CardView { username: string; }

export interface SetCardView extends CardView { votes: number; leader: boolean; tied: boolean; }

export interface SetView {
  id: string;
  userId: string;
  username: string;
  eventId: string | null;
  commanderName: string;
  prompt: string;
  status: 'draft' | 'locked' | 'abandoned';
  lockedAt: string | null;
  cards: SetCardView[];
  myVoteCardId: string | null;
}

export interface EventSummary {
  id: string;
  name: string;
  status: 'open' | 'closed';
  createdAt: string;
  closedAt: string | null;
}

export interface EventView extends EventSummary { sets: SetView[]; }

export interface QueueStatus {
  busy: boolean;
  cardsAhead: number;
  generatingNow: number;
  avgImageSeconds: number;
  etaSeconds: number;
}

export interface GenerationRequest {
  prompt: string;
  cardData: CardParams;
  count: 1 | 3;
  commanderName?: string;
}

export interface GenerationResponse { setId: string | null; cards: CardView[]; }

// ===== Knowledge Pool (docs/superpowers/specs/2026-09-29-knowledge-pool-design.md) =====

/** One of each per player per pool: gold 3 points, silver 2, bronze 1. */
export type Medal = 'gold' | 'silver' | 'bronze';

/** Advisory power estimate: rules-text value vs. what the card's cost and rarity usually buy. */
export interface PowerCheck {
  estimate: number;
  budget: number;
  verdict: 'fair' | 'pushed' | 'over';
}

export interface PoolEntryView {
  id: string;
  cardId: string;
  card: CardView;
  /** Who submitted the card (everything is visible, open or closed). */
  username: string;
  /** The viewer submitted this card. */
  mine: boolean;
  power: PowerCheck | null;
  /** Medal counts from all voters, and the points they add up to. */
  gold: number;
  silver: number;
  bronze: number;
  points: number;
  /** 1-based; points, then golds, then silvers. Equal keys share a rank. */
  rank: number;
  /** Makes the pool: in the top `cutoff`, or level with the last card that is. */
  in: boolean;
  /** In, but level with other cards at the pool line. */
  tiedAtCutoff: boolean;
  myMedal: Medal | null;
  createdAt: string;
}

export interface PoolSummary {
  id: string;
  name: string;
  status: 'open' | 'closed';
  maxEntriesPerUser: number;
  createdAt: string;
  closedAt: string | null;
}

export interface PoolView extends PoolSummary {
  /** Players with at least one entry. */
  submitters: number;
  /** floor(submitters / 2): how many cards make the pool (ties at the line add more). */
  cutoff: number;
  myEntryCount: number;
  /** The entry id each of my medals is on. */
  myMedals: Record<Medal, string | null>;
  /** Sorted by rank, then oldest first. */
  entries: PoolEntryView[];
}
