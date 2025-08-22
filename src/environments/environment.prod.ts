import { apiConfig } from './config';

/**
 * Production environment configuration
 */
export const environment = {
  production: true,
  // Single API endpoint for unified card generation in production
  apiUrl: apiConfig.cardGenerationUrl,
  cardGenerationEndpoint: '',
  cardValidationEndpoint: '/cards/validate',
  defaultCardWidth: 384,
  defaultCardHeight: 288,
  retryAttempts: 3,
  retryDelay: 1000
};
