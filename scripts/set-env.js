const fs = require('fs');
const path = require('path');

// Get environment variables
const apiUrl = process.env['API_URL'] || 'https://api.mtgenesis.ai';
const environment = process.env['NODE_ENV'] || 'production';

// Create dynamic environment config
const envConfig = `/**
 * Production environment configuration - Generated at build time
 */
export const environment = {
  production: ${environment === 'production'},
  apiUrl: '${apiUrl}',
  cardGenerationEndpoint: '/api/v1/create_card',
  cardValidationEndpoint: '/cards/validate',
  defaultCardWidth: 408,
  defaultCardHeight: 336,
  retryAttempts: 3,
  retryDelay: ${environment === 'production' ? 2000 : 1000}
};
`;

// Write the environment file
const targetPath = path.resolve(__dirname, '../src/environments/environment.prod.ts');
fs.writeFileSync(targetPath, envConfig);

console.log(`✅ Environment configuration written to ${targetPath}`);
console.log(`🌐 API URL: ${apiUrl}`);
console.log(`🔧 Environment: ${environment}`);