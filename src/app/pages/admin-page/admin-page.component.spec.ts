import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ReactiveFormsModule } from '@angular/forms';
import { RouterTestingModule } from '@angular/router/testing';
import { of } from 'rxjs';
import { environment } from '../../../environments/environment';
import { EventSummary } from '../../models/api.model';
import { WinnersBannerComponent } from '../../components/winners-banner/winners-banner.component';
import { CmcGroupsPipe } from '../../pipes/cmc-groups.pipe';
import { MediaPipe } from '../../pipes/media.pipe';
import { MediaService } from '../../services/media.service';
import { eventView, setCard, setView } from '../../testing/fixtures';
import { ADMIN_PIN_STORAGE_KEY, AdminPageComponent } from './admin-page.component';

describe('AdminPageComponent', () => {
  let fixture: ComponentFixture<AdminPageComponent>;
  let component: AdminPageComponent;
  let http: HttpTestingController;
  const base = `${environment.apiUrl}/api/v1`;
  const history: EventSummary[] = [
    { id: 'e-2', name: 'AI Night #2', status: 'open', createdAt: '2026-09-28T19:00:00+00:00', closedAt: null },
    { id: 'e-1', name: 'AI Night #1', status: 'closed', createdAt: '2026-09-21T19:00:00+00:00', closedAt: '2026-09-21T23:00:00+00:00' }
  ];

  function setup(current: ReturnType<typeof eventView> | null, events: EventSummary[] = history) {
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule, ReactiveFormsModule, RouterTestingModule],
      declarations: [AdminPageComponent, WinnersBannerComponent, MediaPipe, CmcGroupsPipe],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [{ provide: MediaService, useValue: media }]
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(AdminPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
    http.expectOne(`${base}/events/current`).flush(current);
    http.expectOne(`${base}/events`).flush(events);
    fixture.detectChanges();
  }

  beforeEach(() => sessionStorage.removeItem(ADMIN_PIN_STORAGE_KEY));
  afterEach(() => {
    http.verify();
    sessionStorage.removeItem(ADMIN_PIN_STORAGE_KEY);
  });

  const text = () => (fixture.nativeElement as HTMLElement).textContent!;

  it('keeps the PIN in sessionStorage', () => {
    setup(null);
    component.pin.setValue('4242');
    expect(sessionStorage.getItem(ADMIN_PIN_STORAGE_KEY)).toBe('4242');
  });

  it('restores the PIN from sessionStorage', () => {
    sessionStorage.setItem(ADMIN_PIN_STORAGE_KEY, '9999');
    setup(null);
    expect(component.pin.value).toBe('9999');
  });

  it('sends X-Admin-Pin when creating an event', () => {
    setup(null, []);
    component.pin.setValue('4242');
    component.eventName.setValue('  AI Night #3  ');
    component.createEvent();

    const req = http.expectOne(`${base}/admin/events`);
    expect(req.request.method).toBe('POST');
    expect(req.request.headers.get('X-Admin-Pin')).toBe('4242');
    expect(req.request.body).toEqual({ name: 'AI Night #3' });
    req.flush(eventView({ id: 'e-3', name: 'AI Night #3', sets: [] }));
    http.expectOne(`${base}/events`).flush(history);
    fixture.detectChanges();

    expect(component.current?.id).toBe('e-3');
    expect(text()).toContain('AI Night #3');
  });

  it('submitting the create form sends the request without a native page submit', () => {
    setup(null, []);
    component.pin.setValue('4242');
    component.eventName.setValue('AI Night #3');
    fixture.detectChanges();
    const event = new Event('submit', { cancelable: true });
    (fixture.nativeElement.querySelector('form.create-row') as HTMLFormElement).dispatchEvent(event);
    expect(event.defaultPrevented).toBeTrue();
    http.expectOne(`${base}/admin/events`).flush(eventView({ id: 'e-3', sets: [] }));
    http.expectOne(`${base}/events`).flush([]);
  });

  it('shows "Wrong PIN" on a 403', () => {
    setup(null, []);
    component.pin.setValue('0000');
    component.eventName.setValue('AI Night #3');
    component.createEvent();
    http.expectOne(`${base}/admin/events`).flush({ error: 'Forbidden' }, { status: 403, statusText: 'Forbidden' });
    fixture.detectChanges();
    expect(text()).toContain('Wrong PIN');
  });

  it('shows the lockout message on a 429 (too many wrong PINs)', () => {
    setup(null, []);
    component.pin.setValue('0000');
    component.eventName.setValue('AI Night #3');
    component.createEvent();
    http.expectOne(`${base}/admin/events`).flush(
      { error: 'Too many wrong PINs - wait a few minutes' }, { status: 429, statusText: 'Too Many Requests' });
    fixture.detectChanges();
    expect(text()).toContain('Too many wrong PINs - wait a few minutes');
  });

  it('shows the server text on a 409 (an event is already open)', () => {
    setup(null, []);
    component.pin.setValue('4242');
    component.eventName.setValue('AI Night #3');
    component.createEvent();
    http.expectOne(`${base}/admin/events`)
      .flush({ error: 'An event is already open' }, { status: 409, statusText: 'Conflict' });
    // The page resyncs the current event and history after a conflict.
    http.expectOne(`${base}/events/current`).flush(eventView({ id: 'e-9', name: 'Night by someone else' }));
    http.expectOne(`${base}/events`).flush(history);
    fixture.detectChanges();
    expect(text()).toContain('An event is already open');
    expect(component.current?.id).toBe('e-9');
  });

  it('closes the current event after confirming, sending X-Admin-Pin, and shows the winners', () => {
    setup(eventView({ id: 'e-2', name: 'AI Night #2' }));
    component.pin.setValue('4242');
    const confirmSpy = spyOn(window, 'confirm').and.returnValue(false);

    component.closeCurrent();
    expect(confirmSpy).toHaveBeenCalled();
    http.expectNone(`${base}/admin/events/e-2/close`);

    confirmSpy.and.returnValue(true);
    component.closeCurrent();
    const req = http.expectOne(`${base}/admin/events/e-2/close`);
    expect(req.request.method).toBe('POST');
    expect(req.request.headers.get('X-Admin-Pin')).toBe('4242');
    req.flush(eventView({
      id: 'e-2', name: 'AI Night #2', status: 'closed',
      sets: [setView({ commanderName: 'Grimbold', cards: [setCard({ slot: 1, leader: true, votes: 2 }), setCard({ slot: 2 }), setCard({ slot: 3 })] })]
    }));
    http.expectOne(`${base}/events`).flush(history);
    fixture.detectChanges();

    expect(component.current).toBeNull();
    expect((fixture.nativeElement as HTMLElement).querySelector('.winners-banner')!.textContent).toContain('Grimbold');
  });

  it('shows "Wrong PIN" when closing with a bad PIN', () => {
    setup(eventView({ id: 'e-2' }));
    component.pin.setValue('1');
    spyOn(window, 'confirm').and.returnValue(true);
    component.closeCurrent();
    http.expectOne(`${base}/admin/events/e-2/close`).flush({ error: 'Forbidden' }, { status: 403, statusText: 'Forbidden' });
    fixture.detectChanges();
    expect(text()).toContain('Wrong PIN');
  });

  it('asks for the PIN before calling the server', () => {
    setup(null, []);
    component.eventName.setValue('AI Night #3');
    component.createEvent();
    http.expectNone(`${base}/admin/events`);
    expect(component.error).toBe('Enter the host PIN.');
  });

  it('shows the error instead of "No events yet." when the history fails to load', () => {
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule, ReactiveFormsModule, RouterTestingModule],
      declarations: [AdminPageComponent, WinnersBannerComponent, MediaPipe, CmcGroupsPipe],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [{ provide: MediaService, useValue: media }]
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(AdminPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
    http.expectOne(`${base}/events/current`).flush(null);
    http.expectOne(`${base}/events`)
      .flush({ error: 'database is locked' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    const history = (fixture.nativeElement as HTMLElement).querySelector('.history-panel') as HTMLElement;
    expect(history.textContent).toContain('database is locked');
    expect(history.textContent).not.toContain('No events yet.');
  });

  it('shows a generic message when the history request fails without a body', () => {
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule, ReactiveFormsModule, RouterTestingModule],
      declarations: [AdminPageComponent, WinnersBannerComponent, MediaPipe, CmcGroupsPipe],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [{ provide: MediaService, useValue: media }]
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(AdminPageComponent);
    fixture.detectChanges();
    http.expectOne(`${base}/events/current`).flush(null);
    http.expectOne(`${base}/events`).error(new ProgressEvent('error'), { status: 0 });
    fixture.detectChanges();

    const history = (fixture.nativeElement as HTMLElement).querySelector('.history-panel') as HTMLElement;
    expect(history.textContent).toContain('Cannot reach the server');
    expect(history.textContent).not.toContain('No events yet.');
  });

  it('lists event history with links', () => {
    setup(null);
    const links = Array.from((fixture.nativeElement as HTMLElement).querySelectorAll('.history a'))
      .map(a => a.getAttribute('href'));
    expect(links).toEqual(['/events/e-2', '/events/e-1']);
  });
});
