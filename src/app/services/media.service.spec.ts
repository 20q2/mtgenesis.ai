import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { environment } from '../../environments/environment';
import { MediaService } from './media.service';

describe('MediaService', () => {
  let service: MediaService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ imports: [HttpClientTestingModule] });
    service = TestBed.inject(MediaService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('fetches API media as a blob (so the ngrok header is sent) and caches the object URL', () => {
    const seen: (string | null)[] = [];
    service.src('/api/v1/media/cards/x.png').subscribe(u => seen.push(u));
    service.src(`${environment.apiUrl}/api/v1/media/cards/x.png`).subscribe(u => seen.push(u));

    const req = http.expectOne(`${environment.apiUrl}/api/v1/media/cards/x.png`);
    expect(req.request.responseType).toBe('blob');
    req.flush(new Blob(['png'], { type: 'image/png' }));

    expect(seen.length).toBe(2);
    expect(seen[0]).toMatch(/^blob:/);
    expect(seen[1]).toBe(seen[0]);
  });

  it('passes data: URLs and null through untouched', () => {
    const seen: (string | null)[] = [];
    service.src('data:image/png;base64,AAAA').subscribe(u => seen.push(u));
    service.src(null).subscribe(u => seen.push(u));
    expect(seen).toEqual(['data:image/png;base64,AAAA', null]);
  });

  it('falls back to the plain URL if the blob fetch fails', () => {
    let seen: string | null = null;
    service.src('/api/v1/media/art/y.png').subscribe(u => (seen = u));
    http.expectOne(`${environment.apiUrl}/api/v1/media/art/y.png`)
      .flush(null, { status: 404, statusText: 'Not Found' });
    expect(seen).toBe(`${environment.apiUrl}/api/v1/media/art/y.png` as any);
  });
});
