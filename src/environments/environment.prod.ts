import { apiConfig } from './config';

/**
 * Production environment configuration
 */
export const environment = {
  production: true,
  // Single API endpoint for unified card generation in production
  apiUrl: apiConfig.cardGenerationUrl,
  cardGenerationEndpoint: '/api/v1/create_card',
  cardValidationEndpoint: '/cards/validate',
  defaultCardWidth: 408,  // Magic card art box width (divisible by 8)
  defaultCardHeight: 336, // Magic card art box height (divisible by 8, ~1.21:1 aspect ratio)
  retryAttempts: 3,
  retryDelay: 1000
};
