import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { of } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AvatarComponent } from '../../components/avatar/avatar.component';
import { CardSlotComponent } from '../../components/card-slot/card-slot.component';
import { MediaPipe } from '../../pipes/media.pipe';
import { MediaService } from '../../services/media.service';
import { doneCard } from '../../testing/fixtures';
import { CommunityPageComponent } from './community-page.component';

describe('CommunityPageComponent', () => {
  let fixture: ComponentFixture<CommunityPageComponent>;
  let http: HttpTestingController;
  const url = `${environment.apiUrl}/api/v1/cards/shared`;

  beforeEach(() => {
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src', 'download']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule, RouterTestingModule],
      declarations: [CommunityPageComponent, CardSlotComponent, AvatarComponent, MediaPipe],
      providers: [{ provide: MediaService, useValue: media }]
    });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(CommunityPageComponent);
    fixture.detectChanges();
  });

  afterEach(() => fixture.destroy());

  const el = () => fixture.nativeElement as HTMLElement;

  it('lists shared cards with their maker, downloads only', () => {
    http.expectOne(url).flush([{ ...doneCard({ id: 'x1', shared: true }), username: 'Beth' }]);
    fixture.detectChanges();

    const tile = el().querySelector('[data-card-id="x1"]') as HTMLElement;
    expect(tile.querySelector('.maker')!.textContent).toContain('by');
    expect(tile.querySelector('.byline app-avatar')).not.toBeNull();
    expect(tile.querySelector('.byline')!.textContent!.trim()).toMatch(/Beth$/);
    expect(tile.querySelector('button.share')).toBeNull();
    expect(tile.querySelector('button.pool-submit')).toBeNull();
    expect(tile.querySelector('.download')).not.toBeNull();
  });

  it('shows an empty state when nobody has shared', () => {
    http.expectOne(url).flush([]);
    fixture.detectChanges();
    expect(el().textContent).toContain('Nobody has shared a card yet');
  });
});
