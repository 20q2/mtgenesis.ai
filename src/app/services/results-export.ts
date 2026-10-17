import { sheetColumns } from './results';

/**
 * Results sheets for Discord: a group of finished cards (the Knowledge Pool, or one player's legal
 * commanders), or everything on one overview sheet, drawn on a dark canvas with titles and a
 * caption under each card, saved as a PNG. Card images come from MediaService as blob: URLs, so
 * the canvas is never tainted.
 */

/** Discord's free upload limit is 10 MB; stay under it with some room. */
export const DISCORD_MAX_BYTES = 9.5 * 1024 * 1024;

export interface SheetPerson {
  image: HTMLImageElement | null;
  name: string;
}

export interface SheetCard {
  image: HTMLImageElement | null;
  /** First caption line (Beleren): "3 CMC", "by alice". */
  caption: string;
  /** Second caption line: "Version 2 · 6 votes", "#1 · 12 points". */
  detail: string;
  /** A face drawn before the caption (the pool's makers); the initial stands in without a picture. */
  person?: SheetPerson;
}

export interface SheetSpec {
  /** Small gold caps above the title: the event name. */
  eyebrow: string;
  title: string;
  subtitle: string;
  /** The player's face beside the title on a commanders sheet. */
  person?: SheetPerson;
  cards: SheetCard[];
  logo: HTMLImageElement | null;
  /** Right side of the footer, e.g. the date. */
  footer: string;
}

/** Everything on one sheet: the pool, then a framed block per player. */
export interface OverviewSpec {
  eyebrow: string;
  title: string;
  subtitle: string;
  /** Null when no pool closed; an empty pool is left out too. */
  pool: { title: string; subtitle: string; cards: SheetCard[] } | null;
  players: { title: string; subtitle: string; groups: { person: SheetPerson; cards: SheetCard[] }[] } | null;
  logo: HTMLImageElement | null;
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
// Overview only
const SECTION_H = 92;
const SECTION_GAP = 64;
const BLOCK_PAD = 28;
const BLOCK_LABEL_H = 80;
const BLOCK_GAP = 40;
/** Cards per row inside a player's block (their three commanders). */
const BLOCK_CARDS = 3;
const BLOCK_W = BLOCK_CARDS * CARD_W + (BLOCK_CARDS - 1) * GAP_X + 2 * BLOCK_PAD;

const INK = '#15121f';
const TEXT = '#f3e9d2';
const MUTED = '#c4b89c';
const GOLD = '#d4af37';
const DISPLAY = '"Beleren", "Palatino Linotype", serif';
const CAPS = '"Beleren SC", "Beleren", "Palatino Linotype", serif';
const BODY = '"Alegreya Sans", "Segoe UI", sans-serif';

type Point = { x: number; y: number };

export interface SheetLayout {
  width: number;
  height: number;
  columns: number;
  rows: number;
  /** Top-left of each card. */
  slots: Point[];
}

export interface OverviewLayout {
  width: number;
  height: number;
  pool: { headY: number; slots: Point[] } | null;
  players: { headY: number; blocks: { x: number; y: number; w: number; h: number; slots: Point[] }[] } | null;
}

/** Where everything goes at scale 1; short last rows are centered. */
export function sheetLayout(count: number): SheetLayout {
  const columns = sheetColumns(count);
  const rows = Math.max(1, Math.ceil(count / columns));
  const gridW = columns * CARD_W + (columns - 1) * GAP_X;
  const width = Math.max(MIN_W, gridW + 2 * PAD);
  const height = PAD + HEADER_H + cardRowsHeight(rows) + FOOTER_H + PAD / 2;
  const top = PAD + HEADER_H;
  const slots = centeredRows(count, columns, CARD_W, GAP_X, 0, width)
    .map(c => ({ x: c.x, y: top + c.row * (CARD_H + CAPTION_H + GAP_Y) }));
  return { width, height, columns, rows, slots };
}

/** The pool spans the full width; players sit in blocks up to three a row (3×3 for nine players). */
export function overviewLayout(poolCount: number, playerCardCounts: number[]): OverviewLayout {
  const players = playerCardCounts.length;
  // One row up to three players, then 2×2, then rows of three
  const blockColumns = players <= 3 ? Math.max(1, players) : Math.min(3, Math.ceil(Math.sqrt(players)));
  const playersW = players ? blockColumns * BLOCK_W + (blockColumns - 1) * BLOCK_GAP : 0;
  const poolNaturalCols = sheetColumns(poolCount);
  const poolNaturalW = poolCount ? poolNaturalCols * CARD_W + (poolNaturalCols - 1) * GAP_X : 0;
  const contentW = Math.max(playersW, poolNaturalW, MIN_W - 2 * PAD);
  const width = contentW + 2 * PAD;
  let y = PAD + HEADER_H;

  let pool: OverviewLayout['pool'] = null;
  if (poolCount) {
    // Beside the player blocks the pool fills the width; on its own it keeps a sheet's grid.
    const columns = players ? Math.floor((contentW + GAP_X) / (CARD_W + GAP_X)) : poolNaturalCols;
    const headY = y;
    const top = y + SECTION_H;
    pool = {
      headY,
      slots: centeredRows(poolCount, columns, CARD_W, GAP_X, PAD, contentW)
        .map(c => ({ x: c.x, y: top + c.row * (CARD_H + CAPTION_H + GAP_Y) }))
    };
    y = top + cardRowsHeight(Math.ceil(poolCount / columns));
  }

  let playersLayout: OverviewLayout['players'] = null;
  if (players) {
    if (pool) {
      y += SECTION_GAP;
    }
    const headY = y;
    let rowTop = y + SECTION_H;
    const heights = playerCardCounts.map(n =>
      2 * BLOCK_PAD + BLOCK_LABEL_H + cardRowsHeight(Math.max(1, Math.ceil(n / BLOCK_CARDS))));
    const cells = centeredRows(players, blockColumns, BLOCK_W, BLOCK_GAP, PAD, contentW);
    const blocks: NonNullable<OverviewLayout['players']>['blocks'] = [];
    for (let row = 0; row * blockColumns < players; row++) {
      const inRow = cells.map((c, i) => ({ ...c, i })).filter(c => c.row === row);
      const rowH = Math.max(...inRow.map(c => heights[c.i]));
      for (const c of inRow) {
        const cardTop = rowTop + BLOCK_PAD + BLOCK_LABEL_H;
        const slots = centeredRows(playerCardCounts[c.i], BLOCK_CARDS, CARD_W, GAP_X,
                                   c.x + BLOCK_PAD, BLOCK_W - 2 * BLOCK_PAD)
          .map(s => ({ x: s.x, y: cardTop + s.row * (CARD_H + CAPTION_H + GAP_Y) }));
        blocks.push({ x: c.x, y: rowTop, w: BLOCK_W, h: rowH, slots });
      }
      rowTop += rowH + BLOCK_GAP;
    }
    playersLayout = { headY, blocks };
    y = rowTop - BLOCK_GAP;
  }
  return { width, height: y + 12 + FOOTER_H + PAD / 2, pool, players: playersLayout };
}

/** A group sheet drawn at `scale` (1 = cards at native size). */
export function drawSheet(spec: SheetSpec, scale = 1): HTMLCanvasElement {
  const layout = sheetLayout(spec.cards.length);
  const { canvas, ctx } = startCanvas(layout.width, layout.height, scale);
  drawHeader(ctx, spec, layout.width);
  spec.cards.forEach((card, i) => drawCard(ctx, card, layout.slots[i]));
  drawFooter(ctx, spec.logo, spec.footer, layout.width, layout.height);
  return canvas;
}

/** The overview drawn at `scale`. */
export function drawOverview(spec: OverviewSpec, scale = 1): HTMLCanvasElement {
  const pool = spec.pool?.cards.length ? spec.pool : null;
  const groups = spec.players?.groups ?? [];
  const layout = overviewLayout(pool?.cards.length ?? 0, groups.map(g => g.cards.length));
  const { canvas, ctx } = startCanvas(layout.width, layout.height, scale);
  drawHeader(ctx, spec, layout.width);
  if (pool && layout.pool) {
    const slots = layout.pool.slots;
    drawSectionTitle(ctx, pool.title, pool.subtitle, layout.pool.headY, layout.width);
    pool.cards.forEach((card, i) => drawCard(ctx, card, slots[i]));
  }
  if (spec.players && layout.players) {
    const blocks = layout.players.blocks;
    drawSectionTitle(ctx, spec.players.title, spec.players.subtitle, layout.players.headY, layout.width);
    groups.forEach((group, i) => {
      const block = blocks[i];
      ctx.fillStyle = 'rgba(255, 255, 255, 0.045)';
      roundRect(ctx, block.x, block.y, block.w, block.h, 26);
      ctx.fill();
      ctx.strokeStyle = 'rgba(212, 175, 55, 0.28)';
      ctx.lineWidth = 2;
      ctx.stroke();
      const r = 28;
      const cy = block.y + BLOCK_PAD + r;
      drawFace(ctx, group.person.image, group.person.name, block.x + BLOCK_PAD + r, cy, r);
      const nameX = block.x + BLOCK_PAD + 2 * r + 20;
      ctx.textBaseline = 'middle';
      ctx.fillStyle = TEXT;
      ctx.font = `44px ${DISPLAY}`;
      ctx.fillText(fit(ctx, group.person.name, block.x + block.w - BLOCK_PAD - nameX), nameX, cy + 2);
      group.cards.forEach((card, j) => drawCard(ctx, card, block.slots[j]));
    });
  }
  drawFooter(ctx, spec.logo, spec.footer, layout.width, layout.height);
  return canvas;
}

/** A group sheet as a PNG under `maxBytes`, shrinking it (never enlarging the cards) until it fits. */
export function sheetBlob(spec: SheetSpec, maxBytes = DISCORD_MAX_BYTES): Promise<Blob> {
  return fittedPng(scale => drawSheet(spec, scale), spec.cards.length, maxBytes);
}

/** The overview as a PNG under `maxBytes`; with ~30 cards they come out about two-thirds size. */
export function overviewBlob(spec: OverviewSpec, maxBytes = DISCORD_MAX_BYTES): Promise<Blob> {
  const cards = (spec.pool?.cards.length ?? 0) +
    (spec.players?.groups ?? []).reduce((n, g) => n + g.cards.length, 0);
  return fittedPng(scale => drawOverview(spec, scale), cards, maxBytes);
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

// ===== Layout and drawing pieces =====

/** `count` items of width `w` in rows of `columns`, each row centered in [left, left + span]. */
function centeredRows(count: number, columns: number, w: number, gap: number,
                      left: number, span: number): { x: number; row: number }[] {
  const out: { x: number; row: number }[] = [];
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / columns);
    const inRow = Math.min(columns, count - row * columns);
    const rowW = inRow * w + (inRow - 1) * gap;
    out.push({ x: left + (span - rowW) / 2 + (i - row * columns) * (w + gap), row });
  }
  return out;
}

function cardRowsHeight(rows: number): number {
  return rows * (CARD_H + CAPTION_H) + (rows - 1) * GAP_Y;
}

/** A canvas painted ink with a soft gold glow behind the title. */
function startCanvas(width: number, height: number, scale: number) {
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext('2d')!;
  ctx.scale(scale, scale);
  ctx.imageSmoothingQuality = 'high';
  ctx.fillStyle = INK;
  ctx.fillRect(0, 0, width, height);
  const glow = ctx.createRadialGradient(width / 2, 0, 0, width / 2, 0, Math.max(width, 900) * 0.7);
  glow.addColorStop(0, 'rgba(212, 175, 55, 0.20)');
  glow.addColorStop(1, 'rgba(212, 175, 55, 0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);
  return { canvas, ctx };
}

/** Optional face, eyebrow, title, subtitle, and a gold rule under them. */
function drawHeader(ctx: CanvasRenderingContext2D,
                    spec: { eyebrow: string; title: string; subtitle: string; person?: SheetPerson },
                    width: number): void {
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
}

/** An overview section's gold heading and its line of detail. */
function drawSectionTitle(ctx: CanvasRenderingContext2D, title: string, subtitle: string,
                          y: number, width: number): void {
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = GOLD;
  ctx.font = `46px ${CAPS}`;
  ctx.fillText(fit(ctx, title, width - 2 * PAD), PAD, y + 40);
  ctx.fillStyle = MUTED;
  ctx.font = `24px ${BODY}`;
  ctx.fillText(fit(ctx, subtitle, width - 2 * PAD), PAD, y + 74);
}

/** A card (or a blank slot) with its two caption lines under it. */
function drawCard(ctx: CanvasRenderingContext2D, card: SheetCard, { x, y }: Point): void {
  ctx.textBaseline = 'alphabetic';
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
}

/** Logo and site name on the left, the date on the right. */
function drawFooter(ctx: CanvasRenderingContext2D, logo: HTMLImageElement | null, footer: string,
                    width: number, height: number): void {
  const footY = height - PAD / 2 - FOOTER_H / 2;
  let brandX = PAD;
  if (logo) {
    ctx.drawImage(logo, PAD, footY - 20, 40, 40);
    brandX += 52;
  }
  ctx.textBaseline = 'middle';
  ctx.fillStyle = MUTED;
  ctx.font = `26px ${DISPLAY}`;
  ctx.fillText('MTGenesis.AI', brandX, footY + 2);
  ctx.font = `22px ${BODY}`;
  ctx.textAlign = 'right';
  ctx.fillText(footer, width - PAD, footY + 2);
  ctx.textAlign = 'left';
}

/** Draws at the scale that should fit, then shrinks until the PNG is under `maxBytes`. */
async function fittedPng(draw: (scale: number) => HTMLCanvasElement, cards: number,
                         maxBytes: number): Promise<Blob> {
  await loadFonts();
  // A card is roughly 0.55 MB of PNG at native size: start near the size that fits.
  const guess = cards * 580_000 + 250_000;
  let scale = Math.min(1, Math.sqrt((maxBytes * 0.85) / guess));
  let blob = await toPng(draw(scale));
  for (let attempt = 0; attempt < 5 && blob.size > maxBytes; attempt++) {
    scale *= Math.sqrt(maxBytes / blob.size) * 0.95;
    blob = await toPng(draw(scale));
  }
  return blob;
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
    await Promise.all([`60px ${DISPLAY}`, `46px ${CAPS}`, `22px ${BODY}`].map(f => fonts.load(f)));
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
