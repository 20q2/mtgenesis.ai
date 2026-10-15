import { ComponentFixture, TestBed } from '@angular/core/testing';
import { RouterTestingModule } from '@angular/router/testing';
import { of } from 'rxjs';
import { User } from '../../models/api.model';
import { MediaPipe } from '../../pipes/media.pipe';
import { MAX_UPLOAD_BYTES } from '../../services/avatar-crop';
import { GenerationService } from '../../services/generation.service';
import { MediaService } from '../../services/media.service';
import { UserService } from '../../services/user.service';
import { cardView, doneCard } from '../../testing/fixtures';
import { AvatarComponent } from '../avatar/avatar.component';
import { ProfilePictureComponent } from './profile-picture.component';

/** A real little PNG, so the dialog really decodes and frames it. */
async function pngFile(width = 90, height = 60): Promise<File> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#c0392b';
  ctx.fillRect(0, 0, width, height);
  const blob = await new Promise<Blob>(resolve => canvas.toBlob(b => resolve(b!), 'image/png'));
  return new File([blob], 'me.png', { type: 'image/png' });
}

async function until(check: () => boolean, ms = 3000): Promise<void> {
  const start = Date.now();
  while (!check()) {
    if (Date.now() - start > ms) {
      throw new Error('timed out');
    }
    await new Promise(r => setTimeout(r, 10));
  }
}

describe('ProfilePictureComponent', () => {
  let fixture: ComponentFixture<ProfilePictureComponent>;
  let users: jasmine.SpyObj<UserService>;
  let generations: jasmine.SpyObj<GenerationService>;
  let media: jasmine.SpyObj<MediaService>;
  const sam: User = { id: 'u-1', username: 'sam', avatarUrl: null };
  let artUrl = '';

  beforeEach(async () => {
    users = jasmine.createSpyObj<UserService>('UserService', ['setAvatar', 'removeAvatar']);
    users.setAvatar.and.returnValue(of({ ...sam, avatarUrl: '/api/v1/media/avatars/n.jpg' }));
    users.removeAvatar.and.returnValue(of(sam));
    generations = jasmine.createSpyObj<GenerationService>('GenerationService', ['myCards']);
    artUrl = URL.createObjectURL(await pngFile(120, 100));
    media = jasmine.createSpyObj<MediaService>('MediaService', ['src']);
    media.src.and.callFake((u: string | null | undefined) =>
      of(u && u.includes('/media/art/') ? artUrl : u ?? null));
    TestBed.configureTestingModule({
      imports: [RouterTestingModule],
      declarations: [ProfilePictureComponent, AvatarComponent, MediaPipe],
      providers: [
        { provide: UserService, useValue: users },
        { provide: GenerationService, useValue: generations },
        { provide: MediaService, useValue: media }
      ]
    });
    fixture = TestBed.createComponent(ProfilePictureComponent);
  });

  afterEach(() => {
    fixture.destroy();
    URL.revokeObjectURL(artUrl);
  });

  function create(user: User = sam, prompted = false): ProfilePictureComponent {
    fixture.componentInstance.user = user;
    fixture.componentInstance.prompted = prompted;
    fixture.detectChanges();
    return fixture.componentInstance;
  }

  const el = () => fixture.nativeElement as HTMLElement;
  const text = (sel: string) => (el().querySelector(sel)?.textContent ?? '').replace(/\s+/g, ' ').trim();
  const button = (label: string) => Array.from(el().querySelectorAll('button'))
    .find(b => b.textContent!.includes(label)) as HTMLButtonElement;

  it('asks after login, opens as a modal dialog, and Not now closes it', () => {
    const c = create(sam, true);
    expect((el().querySelector('dialog') as HTMLDialogElement).open).toBeTrue();
    expect(text('.picture-title')).toBe('Add a profile picture?');
    const closed = jasmine.createSpy('closed');
    c.closed.subscribe(closed);
    button('Not now').click();
    expect(closed).toHaveBeenCalled();
  });

  it('opened from the avatar it is just "Profile picture" with Done', () => {
    create(sam, false);
    expect(text('.picture-title')).toBe('Profile picture');
    expect(button('Done')).toBeTruthy();
    expect(button('Remove picture')).toBeUndefined();
  });

  it('frames a chosen photo and uploads a small square JPEG', async () => {
    const c = create();
    const closed = jasmine.createSpy('closed');
    c.closed.subscribe(closed);
    const input = el().querySelector('input[type=file]') as HTMLInputElement;
    const files = new DataTransfer();
    files.items.add(await pngFile());
    input.files = files.files;
    input.dispatchEvent(new Event('change'));
    await until(() => c.step === 'frame');
    fixture.detectChanges();

    expect(text('.picture-title')).toBe('Frame your picture');
    expect(c.source!.width).toBe(90);
    expect(c.frame).toEqual({ zoom: 1, cx: 45, cy: 30 });
    // The short side fills the 260px window
    expect(c.placement.height).toBeCloseTo(260);

    button('Save picture').click();
    await until(() => users.setAvatar.calls.count() > 0);
    const blob = users.setAvatar.calls.mostRecent().args[0];
    expect(blob.type).toBe('image/jpeg');
    expect(blob.size).toBeLessThanOrEqual(MAX_UPLOAD_BYTES);
    await until(() => closed.calls.count() > 0);
  });

  it('says so when the file is not a picture', () => {
    const c = create();
    c.onFile({ target: { files: [new File(['x'], 'notes.txt', { type: 'text/plain' })], value: '' } } as unknown as Event);
    fixture.detectChanges();
    expect(text('.alert')).toContain("isn't a picture");
    expect(c.step).toBe('choose');
  });

  it("lists my finished cards' art and frames the one I pick", async () => {
    generations.myCards.and.returnValue(of([
      doneCard({ id: 'c1', card: { name: 'Ember Pup' } as never }),
      cardView({ id: 'c2' }) // still queued: no art
    ]));
    const c = create();
    button("Use a card's art").click();
    fixture.detectChanges();
    const picks = el().querySelectorAll('.art-pick');
    expect(picks.length).toBe(1);
    expect(picks[0].getAttribute('aria-label')).toContain('Ember Pup');

    (picks[0] as HTMLButtonElement).click();
    await until(() => c.step === 'frame');
    expect(media.src).toHaveBeenCalledWith('/api/v1/media/art/c1.png');
    expect(c.source!.width).toBe(120);
  });

  it('removes a picture I have', () => {
    create({ ...sam, avatarUrl: '/api/v1/media/avatars/a.jpg' });
    button('Remove picture').click();
    expect(users.removeAvatar).toHaveBeenCalled();
  });

  it('Esc steps back from framing before it closes', async () => {
    const c = create();
    c.step = 'cards';
    const closed = jasmine.createSpy('closed');
    c.closed.subscribe(closed);
    el().querySelector('dialog')!.dispatchEvent(new Event('cancel', { cancelable: true }));
    expect(c.step).toBe('choose');
    expect(closed).not.toHaveBeenCalled();
    el().querySelector('dialog')!.dispatchEvent(new Event('cancel', { cancelable: true }));
    expect(closed).toHaveBeenCalled();
  });
});
