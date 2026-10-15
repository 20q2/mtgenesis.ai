import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { MediaPipe } from '../../pipes/media.pipe';
import { MediaService } from '../../services/media.service';
import { AvatarComponent } from './avatar.component';

describe('AvatarComponent', () => {
  let fixture: ComponentFixture<AvatarComponent>;

  beforeEach(() => {
    const media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) => of(u ?? null));
    TestBed.configureTestingModule({
      declarations: [AvatarComponent, MediaPipe],
      providers: [{ provide: MediaService, useValue: media }]
    });
    fixture = TestBed.createComponent(AvatarComponent);
  });

  const el = () => fixture.nativeElement as HTMLElement;

  it('shows the gold initial without a picture', () => {
    fixture.componentInstance.name = 'sam';
    fixture.detectChanges();
    expect(el().querySelector('img')).toBeNull();
    expect(el().textContent!.trim()).toBe('S');
  });

  it('shows the picture at the requested size', () => {
    fixture.componentInstance.url = '/api/v1/media/avatars/a.jpg';
    fixture.componentInstance.name = 'sam';
    fixture.componentInstance.size = 40;
    fixture.detectChanges();
    expect(el().querySelector('img')!.getAttribute('src')).toBe('/api/v1/media/avatars/a.jpg');
    expect((el().querySelector('.avatar') as HTMLElement).style.getPropertyValue('--size')).toBe('40px');
  });

  it('falls back to the initial when the picture fails, until a new one is given', () => {
    fixture.componentInstance.url = '/api/v1/media/avatars/broken.jpg';
    fixture.componentInstance.name = 'sam';
    fixture.detectChanges();
    el().querySelector('img')!.dispatchEvent(new Event('error'));
    fixture.detectChanges();
    expect(el().querySelector('img')).toBeNull();
    expect(el().textContent!.trim()).toBe('S');

    fixture.componentInstance.url = '/api/v1/media/avatars/new.jpg';
    fixture.detectChanges();
    expect(el().querySelector('img')).not.toBeNull();
  });
});
