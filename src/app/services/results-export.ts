import { sheetColumns } from './results';

/**
 * Results sheets for Discord: a group of finished cards (the Knowledge Pool, or one player's legal
 * commanders) drawn on a dark canvas with a title and a caption under each card, saved as a PNG.
 * Card images come from MediaService as blob: URLs, so the canvas is never tainted.
 */

/** Discord's free upload limit is 10 MB; stay under it with some room. */
export const DISCORD_MAX_BYTES = 9.5 * 1024 * 1024;

export interface SheetCard {
  image: HTMLImageElement | null;
  /** First caption line (Beleren): "3 CMC", "by alice". */
  caption: string;
  /** Second caption line: "Version 2 · 6 votes", "#1 · 12 points". */
  detail: string;
  /** A face drawn before the caption (the pool's makers); the initial stands in without a picture. */
  person?: { image: HTMLImageElement | null; name: string };
}

export interface SheetSpec {
  /** Small gold caps above the title: the event name. */
  eyebrow: string;
  title: string;
  subtitle: string;
  /** The player's face beside the title on a commanders sheet. */
  person?: { image: HTMLImageElement | null; name: string };
  cards: SheetCard[];
  logo: HTMLImageElement | null;
  /** Right side of the footer, e.g. the date. */
  footer: string;
}

// Measured at scale 1 (cards at their native 496 px width).
const PAD = 56;
const CARD_W = 496;
const CARD_H = 693;
const GAP_X = 32;
const GAP_Y = 44;
const CAPTION_H = 84;
const HEADER_H = 178;
const FOOTER_H = 76;
const MIN_W = 1100;

const INK = '#15121f';
const TEXT = '#f3e9d2';
const MUTED = '#c4b89c';
const GOLD = '#d4af37';
const DISPLAY = '"Beleren", "Palatino Linotype", serif';
const CAPS = '"Beleren SC", "Beleren", "Palatino Linotype", serif';
const BODY = '"Alegreya Sans", "Segoe UI", sans-serif';

export interface SheetLayout {
  width: number;
  height: number;
  columns: number;
  rows: number;
  /** Top-left of each card. */
  slots: { x: number; y: number }[];
}

/** Where everything goes at scale 1; short last rows are centered. */
export function sheetLayout(count: number): SheetLayout {
  const columns = sheetColumns(count);
  const rows = Math.max(1, Math.ceil(count / columns));
  const gridW = columns * CARD_W + (columns - 1) * GAP_X;
  const width = Math.max(MIN_W, gridW + 2 * PAD);
  const height = PAD + HEADER_H + rows * (CARD_H + CAPTION_H) + (rows - 1) * GAP_Y + FOOTER_H + PAD / 2;
  const slots: { x: number; y: number }[] = [];
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / columns);
    const inRow = Math.min(columns, count - row * columns);
    const rowW = inRow * CARD_W + (inRow - 1) * GAP_X;
    const col = i - row * columns;
    slots.push({
      x: (width - rowW) / 2 + col * (CARD_W + GAP_X),
      y: PAD + HEADER_H + row * (CARD_H + CAPTION_H + GAP_Y)
    });
  }
  return { width, height, columns, rows, slots };
}

/** The sheet drawn at `scale` (1 = cards at native size). */
export function drawSheet(spec: SheetSpec, scale = 1): HTMLCanvasElement {
  const layout = sheetLayout(spec.cards.length);
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(layout.width * scale);
  canvas.height = Math.round(layout.height * scale);
  const ctx = canvas.getContext('2d')!;
  ctx.scale(scale, scale);
  ctx.imageSmoothingQuality = 'high';
  const { width, height } = layout;

  // Ink with a soft gold glow behind the title
  ctx.fillStyle = INK;
  ctx.fillRect(0, 0, width, height);
  const glow = ctx.createRadialGradient(width / 2, 0, 0, width / 2, 0, Math.max(width, 900) * 0.7);
  glow.addColorStop(0, 'rgba(212, 175, 55, 0.20)');
  glow.addColorStop(1, 'rgba(212, 175, 55, 0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);

  // Header: optional face, eyebrow, title, subtitle
  let textX = PAD;
  if (spec.person) {
    drawFace(ctx, spec.person.image, spec.person.name, PAD + 50, PAD + 62, 50);
    textX = PAD + 100 + 28;
  }
  const textW = width - textX - PAD;
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = GOLD;
  ctx.font = `26px ${CAPS}`;
  ctx.fillText(fit(ctx, spec.eyebrow, textW), textX, PAD + 24);
  ctx.fillStyle = TEXT;
  ctx.font = `60px ${DISPLAY}`;
  ctx.fillText(fit(ctx, spec.title, textW), textX, PAD + 90);
  ctx.fillStyle = MUTED;
  ctx.font = `26px ${BODY}`;
  ctx.fillText(fit(ctx, spec.subtitle, textW), textX, PAD + 130);
  const rule = ctx.createLinearGradient(PAD, 0, width - PAD, 0);
  rule.addColorStop(0, 'rgba(212, 175, 55, 0.55)');
  rule.addColorStop(1, 'rgba(212, 175, 55, 0)');
  ctx.fillStyle = rule;
  ctx.fillRect(PAD, PAD + HEADER_H - 26, width - 2 * PAD, 2);

  // Cards with captions
  spec.cards.forEach((card, i) => {
    const { x, y } = layout.slots[i];
    if (card.image) {
      ctx.save();
      ctx.shadowColor = 'rgba(0, 0, 0, 0.55)';
      ctx.shadowBlur = 24;
      ctx.shadowOffsetY = 10;
      ctx.drawImage(card.image, x, y, CARD_W, CARD_H);
      ctx.restore();
    } else {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
      roundRect(ctx, x, y, CARD_W, CARD_H, 22);
      ctx.fill();
    }
    let capX = x;
    const capY = y + CARD_H + 36;
    if (card.person) {
      drawFace(ctx, card.person.image, card.person.name, x + 15, capY - 9, 15);
      capX = x + 40;
    }
    ctx.fillStyle = TEXT;
    ctx.font = `26px ${DISPLAY}`;
    ctx.fillText(fit(ctx, card.caption, CARD_W - (capX - x)), capX, capY);
    ctx.fillStyle = MUTED;
    ctx.font = `22px ${BODY}`;
    ctx.fillText(fit(ctx, card.detail, CARD_W), x, capY + 32);
  });

  // Footer: logo and site name, the date on the right
  const footY = height - PAD / 2 - FOOTER_H / 2;
  let brandX = PAD;
  if (spec.logo) {
    ctx.drawImage(spec.logo, PAD, footY - 20, 40, 40);
    brandX += 52;
  }
  ctx.textBaseline = 'middle';
  ctx.fillStyle = MUTED;
  ctx.font = `26px ${DISPLAY}`;
  ctx.fillText('MTGenesis.AI', brandX, footY + 2);
  ctx.font = `22px ${BODY}`;
  ctx.textAlign = 'right';
  ctx.fillText(spec.footer, width - PAD, footY + 2);
  ctx.textAlign = 'left';
  return canvas;
}

/** The sheet as a PNG under `maxBytes`, shrinking it (never enlarging the cards) until it fits. */
export async function sheetBlob(spec: SheetSpec, maxBytes = DISCORD_MAX_BYTES): Promise<Blob> {
  await loadFonts();
  // A card is roughly 0.55 MB of PNG at native size: start near the size that fits.
  const guess = spec.cards.length * 580_000 + 250_000;
  let scale = Math.min(1, Math.sqrt((maxBytes * 0.85) / guess));
  let blob = await toPng(drawSheet(spec, scale));
  for (let attempt = 0; attempt < 4 && blob.size > maxBytes; attempt++) {
    scale *= Math.sqrt(maxBytes / blob.size) * 0.95;
    blob = await toPng(drawSheet(spec, scale));
  }
  return blob;
}

/** Loads an image for the canvas; null when it can't (the sheet then draws a blank slot). */
export function loadImage(src: string | null | undefined): Promise<HTMLImageElement | null> {
  if (!src) {
    return Promise.resolve(null);
  }
  return new Promise(resolve => {
    const img = new Image();
    if (!/^(blob:|data:)/.test(src)) {
      img.crossOrigin = 'anonymous';
    }
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

function toPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob(b => (b ? resolve(b) : reject(new Error('Could not draw the image'))), 'image/png'));
}

/** Canvas text only uses a web font once it has loaded. */
async function loadFonts(): Promise<void> {
  const fonts = (document as Document & { fonts?: FontFaceSet }).fonts;
  if (!fonts) {
    return;
  }
  try {
    await Promise.all([`60px ${DISPLAY}`, `26px ${CAPS}`, `22px ${BODY}`].map(f => fonts.load(f)));
  } catch {
    // Fallback fonts are fine.
  }
}

/** Text cut with an ellipsis to fit `maxWidth` in the current font. */
function fit(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) {
    return text;
  }
  let cut = text;
  while (cut.length > 1 && ctx.measureText(`${cut}…`).width > maxWidth) {
    cut = cut.slice(0, -1);
  }
  return `${cut.trimEnd()}…`;
}

/** A round face: the picture, or the gold initial like <app-avatar>. */
function drawFace(ctx: CanvasRenderingContext2D, image: HTMLImageElement | null, name: string,
                  cx: number, cy: number, r: number): void {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  if (image) {
    ctx.clip();
    const side = Math.min(image.naturalWidth, image.naturalHeight);
    ctx.drawImage(image, (image.naturalWidth - side) / 2, (image.naturalHeight - side) / 2, side, side,
                  cx - r, cy - r, 2 * r, 2 * r);
  } else {
    ctx.fillStyle = GOLD;
    ctx.fill();
    ctx.fillStyle = '#1b1608';
    ctx.font = `${Math.round(r * 1.1)}px ${DISPLAY}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText((name.trim()[0] || '?').toUpperCase(), cx, cy + r * 0.06);
  }
  ctx.restore();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(212, 175, 55, 0.7)';
  ctx.lineWidth = Math.max(2, r / 12);
  ctx.stroke();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
