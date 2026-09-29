import { HttpErrorResponse } from '@angular/common/http';
import { environment } from '../../environments/environment';

/** Absolute URL for an AI Night API path, e.g. api('/me/cards'). */
export function api(path: string): string {
  return `${environment.apiUrl}/api/v1${path}`;
}

/** True when the request goes to our backend (and so gets the auth + ngrok headers). */
export function isApiRequest(url: string): boolean {
  return !!environment.apiUrl && url.startsWith(environment.apiUrl);
}

/**
 * CardView media URLs are relative ("/api/v1/media/cards/<id>.png").
 * Prefix them with environment.apiUrl; absolute, data: and blob: URLs pass through.
 */
export function mediaUrl(url: string | null | undefined): string | null {
  if (!url) {
    return null;
  }
  if (/^(https?:|data:|blob:)/i.test(url)) {
    return url;
  }
  return `${environment.apiUrl}${url.startsWith('/') ? '' : '/'}${url}`;
}

/** The server's `{error}` text when present, otherwise a readable fallback. */
export function apiErrorMessage(err: unknown, fallback = 'Something went wrong. Please try again.'): string {
  if (err instanceof HttpErrorResponse) {
    const body = err.error;
    if (body && typeof body === 'object' && typeof body.error === 'string' && body.error) {
      return body.error;
    }
    if (typeof body === 'string' && body && !body.trim().startsWith('<')) {
      return body;
    }
    if (err.status === 0) {
      return 'Cannot reach the server. Is it running?';
    }
    return fallback;
  }
  if (err instanceof Error && err.message) {
    return err.message;
  }
  return fallback;
}

/** Filename-safe version of a card name. */
export function safeFileName(name: string | null | undefined, fallback = 'magic-card'): string {
  const cleaned = (name || '').replace(/[^a-z0-9]+/gi, '_').replace(/^_+|_+$/g, '').toLowerCase();
  return cleaned || fallback;
}
