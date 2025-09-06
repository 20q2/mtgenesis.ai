export interface ApiConfig {
  cardGenerationUrl: string;
}

export const apiConfig: ApiConfig = {
  // Lambda proxy URL for card generation (proxies to ngrok)
  cardGenerationUrl: 'https://0eccb5a667ac.ngrok-free.app'
};

