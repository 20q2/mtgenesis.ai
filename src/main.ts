import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';

import { AppModule } from './app/app.module';
import { environment } from './environments/environment';

/**
 * In production the backend sits behind an ngrok tunnel whose URL can change on every
 * launch. deploy/start-site.ps1 publishes the live URL as api-config.json next to the
 * site, so read it before bootstrapping; the URL baked in at build time is the fallback.
 */
async function loadRuntimeApiUrl(): Promise<void> {
  if (!environment.production) {
    return;
  }
  try {
    // Cache-bust: GitHub Pages serves files with a 10-minute max-age.
    const res = await fetch(`api-config.json?t=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) {
      return;
    }
    const config = await res.json();
    if (typeof config?.apiUrl === 'string' && config.apiUrl) {
      environment.apiUrl = config.apiUrl.replace(/\/+$/, '');
    }
  } catch {
    // Keep the build-time URL.
  }
}

loadRuntimeApiUrl()
  .then(() => platformBrowserDynamic().bootstrapModule(AppModule))
  .catch(err => console.error(err));
