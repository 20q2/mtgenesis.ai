import { apiConfig } from './config';

/**
 * Development environment configuration
 */
export const environment = {
  production: false,
  // Single API endpoint for unified card generation
  apiUrl: 'http://localhost:5000',
  cardGenerationEndpoint: '/api/v1/create_card',
  cardValidationEndpoint: '/cards/validate',
  defaultCardWidth: 384,
  defaultCardHeight: 288,
  retryAttempts: 3,
  retryDelay: 1000
};

