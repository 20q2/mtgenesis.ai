# Profile pictures

Players can give themselves a profile picture: a photo (a selfie on phones) or the artwork from one of their own cards, framed in a circle. Faces replace the gold letter avatars wherever a player appears. It is optional and for fun.

## Decisions

- **Prompted, never required.** The picture dialog opens right after logging in when the player has no picture. Players who are already logged in get a one-time nudge on their next visit. "Not now" hides the nudge for that player on that device. The top-bar avatar always opens the dialog, to add, change or remove a picture.
- **Sources.** Upload a photo (phones offer the camera in the picker), or pick the art of one of your finished cards. Either way the player drags and zooms to frame it in a circle before saving.
- **Everywhere a player appears.** The top bar, the vote grid tiles and sheet, and a small face beside every "by alice" byline: gallery, Knowledge Pool, winners banner, past events. No picture means the gold letter avatar, as before.
- **Same trust model as everything else.** Identity is username-only, so whoever logs in as alice can change alice's picture. A host fixes a prank the same way: log in as that player and replace or remove it. No separate moderation tool.

## Storage and API

- `users.avatar_id` (nullable TEXT, added by `Storage._migrate`). The file is `data/avatars/<avatar_id>.jpg`.
- Every upload gets a new UUID, so a picture URL never changes content: `/api/v1/media/avatars/<uuid>.jpg` (public, like the other media, cached by the browser and `MediaService`). The previous file is deleted after the new one is saved.
- `GET /me` returns `{id, username, avatarUrl}`. The app calls it at startup to refresh the stored user (and to decide whether to nudge).
- `PUT /me/avatar` takes the raw image as the request body (`image/jpeg`, `image/png` or `image/webp`) and returns the updated user. The server opens it with Pillow, rejects anything that is not an image or is larger than 4096 px on a side, applies EXIF rotation, center-crops it square, resizes it to 256×256 and re-encodes it as JPEG (quality 85). Re-encoding drops all metadata, including GPS.
- `DELETE /me/avatar` removes the picture and returns the updated user.
- The global 64 KB request cap (`MAX_CONTENT_LENGTH`) stays. The browser sends a 320×320 JPEG, usually 20–40 KB, and lowers the quality until it fits under 60 KB.
- `avatarUrl` (string or null) is added next to `username` in the login response, set views, shared gallery cards and pool entries.

## Frontend

- `AvatarComponent` (`<app-avatar [url] [name] [size]>`): the picture as a circle (loaded through the media pipe), or the gold initial when there is none.
- `ProfilePictureComponent`: a native modal `<dialog>` hosted by `AppComponent`.
  - **Choose:** the current picture, "Choose a photo" (file input, `accept="image/*"`), "Use a card's art" (a grid of your finished cards' art from `/me/cards`), "Remove" when you have one, and "Not now".
  - **Frame:** the image in a circular window. Drag to move, use the slider, wheel or pinch to zoom. The image always covers the circle. Save crops to a 320×320 canvas, exports JPEG and uploads it.
  - The crop math (`avatar-crop.ts`) is pure and unit-tested.
- `UserService` stores `avatarUrl` with the user, refreshes it from `GET /me`, and uploads and removes the picture. It also tracks whether the dialog should open after a login, and the per-device "Not now" (`mtgenesis.avatarPrompt.<userId>` in localStorage, guarded like the other storage calls).

## Tests

- Backend: storage (set and clear the avatar, migration), API (upload a JPEG/PNG, reject a non-image and an oversized one, 401 without a user, the old file deleted on replace, the media route serves it, `avatarUrl` in the login, set, gallery and pool views).
- Frontend: crop math, the avatar component's fallback, the dialog's triggers (after login, the one-time nudge, "Not now" remembered) and upload flow.
