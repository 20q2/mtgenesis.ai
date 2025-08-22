export interface ApiConfig {
  cardGenerationUrl: string;
}

export const apiConfig: ApiConfig = {
  // Single API endpoint for card generation
  cardGenerationUrl: 'http://127.0.0.1:5000/api/v1/create_card'
};

