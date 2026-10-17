import { SheetSpec, drawOverview, drawSheet, overviewLayout, sheetBlob, sheetLayout } from './results-export';

/** A card-sized image of noise, so the PNG doesn't compress to nothing. */
function noiseCard(): Promise<HTMLImageElement> {
  const canvas = document.createElement('canvas');
  canvas.width = 496;
  canvas.height = 693;
  const ctx = canvas.getContext('2d')!;
  const data = ctx.createImageData(496, 693);
  for (let i = 0; i < data.data.length; i++) {
    data.data[i] = i % 4 === 3 ? 255 : Math.floor(Math.random() * 256);
  }
  ctx.putImageData(data, 0, 0);
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = canvas.toDataURL('image/png');
  });
}

function spec(cards: SheetSpec['cards']): SheetSpec {
  return { eyebrow: 'AI Night', title: 'Knowledge Pool', subtitle: 'Cards', cards, logo: null, footer: 'Today' };
}

describe('results export', () => {
  it('lays three cards out in one centered row', () => {
    const layout = sheetLayout(3);
    expect(layout.columns).toBe(3);
    expect(layout.rows).toBe(1);
    expect(layout.slots[0].y).toBe(layout.slots[2].y);
    const left = layout.slots[0].x;
    const right = layout.width - (layout.slots[2].x + 496);
    expect(left).toBeCloseTo(right, 5);
  });

  it('centers a short last row', () => {
    const layout = sheetLayout(7); // 4 columns: 4 + 3
    expect(layout.rows).toBe(2);
    const lastRow = layout.slots.slice(4);
    const left = lastRow[0].x;
    const right = layout.width - (lastRow[2].x + 496);
    expect(left).toBeCloseTo(right, 5);
    expect(lastRow[0].y).toBeGreaterThan(layout.slots[0].y);
  });

  it('draws at the requested scale, with or without images and faces', () => {
    const canvas = drawSheet({
      ...spec([{ image: null, caption: 'by alice', detail: '#1 · 9 points', person: { image: null, name: 'alice' } }]),
      person: { image: null, name: 'Alice' }
    }, 0.5);
    const layout = sheetLayout(1);
    expect(canvas.width).toBe(Math.round(layout.width * 0.5));
    expect(canvas.height).toBe(Math.round(layout.height * 0.5));
  });

  it('overview: the pool in one row above nine players in a 3×3 grid of blocks', () => {
    const layout = overviewLayout(7, Array(9).fill(3));
    const pool = layout.pool!;
    const blocks = layout.players!.blocks;
    expect(new Set(pool.slots.map(s => s.y)).size).toBe(1);
    expect(blocks.length).toBe(9);
    expect(new Set(blocks.map(b => b.y)).size).toBe(3);
    expect(new Set(blocks.map(b => b.x)).size).toBe(3);
    expect(layout.players!.headY).toBeGreaterThan(pool.slots[0].y);
    // Every card sits inside its block, and the sheet is as wide as three blocks
    for (const b of blocks) {
      for (const s of b.slots) {
        expect(s.x).toBeGreaterThanOrEqual(b.x);
        expect(s.x + 496).toBeLessThanOrEqual(b.x + b.w);
        expect(s.y + 693).toBeLessThan(b.y + b.h);
      }
    }
    expect(layout.height).toBeGreaterThan(blocks[8].y + blocks[8].h);
  });

  it('overview: a tie makes a taller block and its row grows with it', () => {
    const layout = overviewLayout(0, [3, 4, 3]);
    expect(layout.pool).toBeNull();
    const [a, b, c] = layout.players!.blocks;
    expect(b.slots.length).toBe(4);
    expect(b.slots[3].y).toBeGreaterThan(b.slots[0].y);
    expect(a.h).toBe(b.h);
    expect(c.h).toBe(b.h);
  });

  it('overview draws with only a pool or only players', () => {
    const card = { image: null, caption: 'x', detail: 'y' };
    const base = { eyebrow: 'Results', title: 'AI Night', subtitle: 's', logo: null, footer: 'f' };
    expect(drawOverview({ ...base, pool: { title: 'Pool', subtitle: '', cards: [card] }, players: null }, 0.25).width).toBeGreaterThan(0);
    expect(drawOverview({ ...base, pool: null, players: { title: 'Commanders', subtitle: '', groups: [
      { person: { image: null, name: 'amy' }, cards: [card, card, card] }] } }, 0.25).width).toBeGreaterThan(0);
  });

  it('shrinks the PNG until it fits under the limit', async () => {
    const img = await noiseCard();
    const cards = Array.from({ length: 4 }, () => ({ image: img, caption: '3 CMC', detail: 'Version 1' }));
    const limit = 1.5 * 1024 * 1024;
    const blob = await sheetBlob(spec(cards), limit);
    expect(blob.type).toBe('image/png');
    expect(blob.size).toBeLessThanOrEqual(limit);
  });
});
