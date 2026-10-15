/**
 * API request/response types.
 * Legacy types reconstructed 2026-09-28 from usages (the original was never committed).
 */
import { CommanderKind } from './card.model';

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

/** avatarUrl: the profile picture (GET /me); missing on users stored before pictures existed. */
export interface User { id: string; username: string; avatarUrl?: string | null; }

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
  /** Commander sets only: Legendary Creature or Legendary Artifact — Vehicle. */
  commanderKind?: CommanderKind;
}

export interface GeneratedCardData extends CardParams {
  flavorText?: string;
  /** "token" for Token Maker cards (their frame id in `frame`); absent on regular cards. */
  kind?: 'token';
  frame?: string;
}

// ===== Token Maker (proxy-server/tokens.py) =====

export type TokenType = 'Creature' | 'Artifact Creature' | 'Enchantment Creature' | 'Artifact' | 'Enchantment' | 'Land';

/** A frame the player can pick; hasText: false frames print no abilities or flavor. */
export interface TokenFrame { id: string; label: string; blurb: string; hasText: boolean; }

export interface TokenFramesResponse { frames: TokenFrame[]; types: TokenType[]; }

/** The Token Maker form, as POST /tokens and /tokens/preview take it. */
export interface TokenRequest {
  frame: string;
  name: string;
  type: TokenType;
  subtype: string;
  colors: string[];
  power?: string;
  toughness?: string;
  description: string;
  flavorText: string;
  artPrompt: string;
  /** The AI fills a blank name / flavor text (default true); false keeps it blank. */
  aiName?: boolean;
  aiFlavor?: boolean;
}

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
  /** True while the card is shared to the Community Gallery */
  shared: boolean;
  /** The card's entry in the open Knowledge Pool, if it has one */
  poolEntryId: string | null;
  /** The host changed its rules text or P/T after it was made (cardImageUrl then carries ?v=). */
  errata: boolean;
}

/** A card on the Community Gallery (GET /cards/shared), with its maker. */
export interface SharedCardView extends CardView { username: string; avatarUrl: string | null; }

/** votes are weighted: the set owner's own vote counts 2, and ownerVote marks that card. */
export interface SetCardView extends CardView {
  votes: number;
  leader: boolean;
  tied: boolean;
  ownerVote: boolean;
  /** Players asking the owner to reroll this version (docs/superpowers/specs/2026-10-14-reroll-requests-design.md). */
  rerollRequests: number;
  myRerollRequest: boolean;
  /** Only ever true for the owner: the version can be rerolled now (enough requests, or it failed). */
  canReroll: boolean;
}

export interface SetView {
  id: string;
  userId: string;
  username: string;
  /** The owner's profile picture, or null. */
  avatarUrl: string | null;
  eventId: string | null;
  commanderName: string;
  prompt: string;
  status: 'draft' | 'locked' | 'abandoned';
  lockedAt: string | null;
  /** A set is one commander at 3, 4 or 5 mana; null for sets made before the commander rules. */
  cmc: number | null;
  rarity: string | null;
  cards: SetCardView[];
  myVoteCardId: string | null;
  /** Reroll requests a version needs before its owner may reroll it during the vote. */
  rerollRequestsNeeded: number;
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
  /** Commander sets (count 3): the commander's mana value, 3, 4 or 5. */
  cmc?: number;
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
  avatarUrl: string | null;
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
  /** Who flagged the card as broken (usernames, oldest first). Advisory: never affects rank or `in`. */
  wtfBy: string[];
  myWtf: boolean;
  /** Who liked the card without a medal (usernames, oldest first). Advisory, like `wtfBy`. */
  likedBy: string[];
  myLike: boolean;
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

// ===== Host errata (docs/superpowers/specs/2026-10-14-errata-design.md) =====

/** A card the host may errata, with where it's from (GET /admin/errata). */
export interface ErrataItem {
  card: CardView;
  /** The card as printed, once it has errata; null otherwise. */
  original: GeneratedCardData | null;
  /** Commander name, or the card's name. */
  label: string;
  username: string;
  /** "Version 2", "Winner", "Tied", "In · 9 points"... */
  detail: string;
}

export interface ErrataGroup {
  id: 'vote' | 'legal' | 'pool';
  title: string;
  items: ErrataItem[];
}

/** What the host can change: name, cost, type and rarity stay as printed. */
export interface ErrataFields {
  description: string;
  power?: string;
  toughness?: string;
}

/** An erratum drawn but not saved. */
export interface ErrataPreview {
  /** data: URL of the rendered card */
  image: string;
  power: PowerCheck | null;
}
