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
