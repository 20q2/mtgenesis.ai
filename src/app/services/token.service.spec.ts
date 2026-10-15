import { tokenFrameCode } from './token.service';

describe('tokenFrameCode (mirrors token_renderer.frame_code)', () => {
  it('uses the color for mono-colored tokens', () => {
    expect(tokenFrameCode(['G'], 'Creature')).toBe('G');
  });

  it('uses gold for two or more colors', () => {
    expect(tokenFrameCode(['W', 'U'], 'Creature')).toBe('M');
  });

  it('uses the land frame for colorless lands and the artifact frame otherwise', () => {
    expect(tokenFrameCode([], 'Land')).toBe('L');
    expect(tokenFrameCode([], 'Artifact')).toBe('A');
    expect(tokenFrameCode([], 'Creature')).toBe('A');
  });
});
