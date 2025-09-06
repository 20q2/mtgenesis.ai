import { apiConfig } from './config';

/**
 * Development environment configuration
 */
export const environment = {
  production: false,
  // Single API endpoint for unified card generation
  apiUrl: 'http://127.0.0.1:5000',
  cardGenerationEndpoint: '/api/v1/create_card',
  cardValidationEndpoint: '/cards/validate',
  defaultCardWidth: 408,  // Magic card art box width (divisible by 8)
  defaultCardHeight: 336, // Magic card art box height (divisible by 8, ~1.21:1 aspect ratio)
  retryAttempts: 3,
  retryDelay: 1000,
  // HTTP timeout for card generation - dynamic based on cold start vs warm run
  httpTimeoutMs: 300000, // Default (unused, will be dynamic)
  // Cold start timeout (first job) - 3 minutes
  coldStartTimeoutMs: 180000,
  // Warm run timeout (subsequent jobs) - 30 seconds  
  warmRunTimeoutMs: 30000
};

