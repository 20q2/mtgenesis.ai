export interface ApiConfig {
  cardGenerationUrl: string;
}

export const apiConfig: ApiConfig = {
  // Build-time fallback only. In production, main.ts reads the live ngrok URL from the
  // api-config.json that deploy/start-site.ps1 publishes next to the site.
  cardGenerationUrl: 'https://0eccb5a667ac.ngrok-free.app'
};
