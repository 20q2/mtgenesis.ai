import {
  MAX_ZOOM, clampFrame, cropRect, displayRect, initialFrame, panFrame, zoomFrame
} from './avatar-crop';

describe('avatar framing', () => {
  const W = 1200;
  const H = 800;

  it('starts centered, with the short side just filling the window', () => {
    const f = initialFrame(W, H);
    expect(f).toEqual({ zoom: 1, cx: 600, cy: 400 });
    expect(cropRect(W, H, f)).toEqual({ sx: 200, sy: 0, side: 800 });
  });

  it('zooming in narrows the crop around the same center', () => {
    const f = zoomFrame(W, H, initialFrame(W, H), 2);
    expect(cropRect(W, H, f)).toEqual({ sx: 400, sy: 200, side: 400 });
  });

  it('keeps zoom between 1 and MAX_ZOOM', () => {
    expect(zoomFrame(W, H, initialFrame(W, H), 0.2).zoom).toBe(1);
    expect(zoomFrame(W, H, initialFrame(W, H), 99).zoom).toBe(MAX_ZOOM);
  });

  it('never lets the crop leave the picture', () => {
    const f = clampFrame(W, H, { zoom: 1, cx: -500, cy: 9999 });
    expect(cropRect(W, H, f)).toEqual({ sx: 0, sy: 0, side: 800 });
    const g = clampFrame(W, H, { zoom: 2, cx: 5000, cy: -10 });
    expect(cropRect(W, H, g)).toEqual({ sx: 800, sy: 0, side: 400 });
  });

  it('dragging moves the picture with the pointer (the crop the other way), in window pixels', () => {
    // 300px window over an 800px crop: 1 window px = 8/3 picture px
    const f = panFrame(W, H, initialFrame(W, H), 30, 0, 300);
    expect(f.cx).toBeCloseTo(600 - 80);
    expect(f.cy).toBe(400);
  });

  it('zooming out after a pan pulls the crop back inside the picture', () => {
    const panned = clampFrame(W, H, { zoom: 3, cx: 1100, cy: 700 });
    const out = zoomFrame(W, H, panned, 1);
    expect(cropRect(W, H, out)).toEqual({ sx: 400, sy: 0, side: 800 });
  });

  it('places the picture so the crop fills the window', () => {
    const f = zoomFrame(W, H, initialFrame(W, H), 2); // crop 400 square at (400, 200)
    expect(displayRect(W, H, f, 300)).toEqual({ x: -300, y: -150, width: 900, height: 600 });
  });

  it('handles portrait pictures', () => {
    const f = initialFrame(600, 900);
    expect(cropRect(600, 900, f)).toEqual({ sx: 0, sy: 150, side: 600 });
  });
});
