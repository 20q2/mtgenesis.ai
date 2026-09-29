import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ReactiveFormsModule } from '@angular/forms';
import { RouterTestingModule } from '@angular/router/testing';
import { environment } from '../../../environments/environment';
import { poolView } from '../../testing/fixtures';
import { PoolAdminComponent } from './pool-admin.component';

describe('PoolAdminComponent', () => {
  let fixture: ComponentFixture<PoolAdminComponent>;
  let component: PoolAdminComponent;
  let http: HttpTestingController;
  const base = `${environment.apiUrl}/api/v1`;

  function setup(current: ReturnType<typeof poolView> | null, pin = '4321') {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule, ReactiveFormsModule, RouterTestingModule],
      declarations: [PoolAdminComponent]
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(PoolAdminComponent);
    component = fixture.componentInstance;
    component.pin = pin;
    fixture.detectChanges();
    http.expectOne(`${base}/pools/current`).flush(current);
    fixture.detectChanges();
  }

  afterEach(() => http.verify());

  const el = () => fixture.nativeElement as HTMLElement;

  it('prefills the default 16 slots', () => {
    setup(null);
    expect(el().querySelectorAll('.slot-edit').length).toBe(16);
  });

  it('opens a pool with the edited slots and the PIN', () => {
    setup(null);
    component.form.controls.name.setValue('Knowledge Pool 2026');
    component.form.controls.maxEntries.setValue(3);
    component.removeSlot(15);
    component.addSlot();
    (el().querySelector('.create-pool-btn') as HTMLButtonElement).click();
    const req = http.expectOne(`${base}/admin/pools`);
    expect(req.request.headers.get('X-Admin-Pin')).toBe('4321');
    expect(req.request.body.name).toBe('Knowledge Pool 2026');
    expect(req.request.body.maxEntriesPerUser).toBe(3);
    expect(req.request.body.slots.length).toBe(16);
    expect(req.request.body.slots[15]).toEqual({ label: '', colorRule: 'any', typeRule: 'any' });
    req.flush(poolView());
    fixture.detectChanges();
    expect(el().textContent).toContain('Close pool');
  });

  it('asks for the PIN before calling the server', () => {
    setup(null, '');
    component.form.controls.name.setValue('KP');
    component.createPool();
    expect(component.error).toBe('Enter the host PIN.');
  });

  it('closes the open pool after confirming', () => {
    setup(poolView());
    spyOn(window, 'confirm').and.returnValue(true);
    (el().querySelector('.close-pool') as HTMLButtonElement).click();
    const req = http.expectOne(`${base}/admin/pools/p-1/close`);
    expect(req.request.headers.get('X-Admin-Pin')).toBe('4321');
    req.flush(poolView({ status: 'closed', closedAt: '2026-10-10T23:00:00+00:00' }));
    fixture.detectChanges();
    expect(el().textContent).toContain('is closed');
    expect(el().textContent).toContain('Open Knowledge Pool');
  });

  it('shows "Wrong PIN" on a 403', () => {
    setup(poolView());
    spyOn(window, 'confirm').and.returnValue(true);
    component.closePool();
    http.expectOne(`${base}/admin/pools/p-1/close`).flush({ error: 'Wrong admin PIN' }, { status: 403, statusText: 'Forbidden' });
    expect(component.error).toBe('Wrong PIN');
  });
});
