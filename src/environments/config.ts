export interface ApiConfig {
  cardGenerationUrl: string;
}

export const apiConfig: ApiConfig = {
  // Lambda proxy URL for card generation (proxies to ngrok)
  cardGenerationUrl: 'https://f559f10091f9.ngrok-free.app'
};

