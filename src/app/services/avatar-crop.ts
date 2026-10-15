/**
 * Framing a profile picture (docs/superpowers/specs/2026-10-14-profile-pictures-design.md):
 * a square window over the picture. `zoom` 1 means the picture's short side just fills the
 * window; (cx, cy) is the picture point at the window's center. Every frame is clamped so
 * the picture always covers the window. All sizes are in picture pixels unless named "view".
 */
export interface Frame { zoom: number; cx: number; cy: number; }

export const MAX_ZOOM = 4;
/** The framed square sent to the server; it stores 256px. */
export const OUTPUT_SIZE = 320;
/** Stays under the server's 64 KB request cap. */
export const MAX_UPLOAD_BYTES = 60_000;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function initialFrame(width: number, height: number): Frame {
  return { zoom: 1, cx: width / 2, cy: height / 2 };
}

/** The side of the picture square the window shows. */
function cropSide(width: number, height: number, zoom: number): number {
  return Math.min(width, height) / zoom;
}

export function clampFrame(width: number, height: number, frame: Frame): Frame {
  const zoom = clamp(frame.zoom, 1, MAX_ZOOM);
  const half = cropSide(width, height, zoom) / 2;
  return { zoom, cx: clamp(frame.cx, half, width - half), cy: clamp(frame.cy, half, height - half) };
}

/** Zoom about the window's center. */
export function zoomFrame(width: number, height: number, frame: Frame, zoom: number): Frame {
  return clampFrame(width, height, { ...frame, zoom });
}

/** A drag of (dx, dy) window pixels: the picture follows the pointer. */
export function panFrame(width: number, height: number, frame: Frame,
                         dx: number, dy: number, viewSize: number): Frame {
  const perViewPx = cropSide(width, height, frame.zoom) / viewSize;
  return clampFrame(width, height, { ...frame, cx: frame.cx - dx * perViewPx, cy: frame.cy - dy * perViewPx });
}

/** The picture square to draw into the output canvas. */
export function cropRect(width: number, height: number, frame: Frame): { sx: number; sy: number; side: number } {
  const side = cropSide(width, height, frame.zoom);
  return { sx: frame.cx - side / 2, sy: frame.cy - side / 2, side };
}

/** Where to draw the whole picture inside a viewSize window (window pixels). */
export function displayRect(width: number, height: number, frame: Frame, viewSize: number):
    { x: number; y: number; width: number; height: number } {
  const { sx, sy, side } = cropRect(width, height, frame);
  const scale = viewSize / side;
  return { x: -sx * scale, y: -sy * scale, width: width * scale, height: height * scale };
}

/** The framed picture as a JPEG small enough to upload, stepping the quality down as needed. */
export async function framedJpeg(image: CanvasImageSource, width: number, height: number,
                                 frame: Frame): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = OUTPUT_SIZE;
  canvas.height = OUTPUT_SIZE;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas unavailable');
  }
  const { sx, sy, side } = cropRect(width, height, frame);
  ctx.fillStyle = '#fff'; // transparent PNGs flatten onto white, as on the server
  ctx.fillRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image, sx, sy, side, side, 0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  let blob: Blob | null = null;
  for (const quality of [0.9, 0.82, 0.74, 0.66, 0.58, 0.5]) {
    blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
    if (blob && blob.size <= MAX_UPLOAD_BYTES) {
      return blob;
    }
  }
  if (!blob) {
    throw new Error('Could not encode the picture');
  }
  return blob; // a 320px JPEG at quality 0.5 is far below the cap; the server re-checks anyway
}
