import { describe, expect, it } from 'vitest';
import {
  contrastRatio,
  deriveContrastColor,
  deriveTeacherPaletteFromClothing,
  hslToRgb,
  rgbToHsl,
} from './canvasPainter.js';

describe('teacher contrast color', () => {
  it('round-trips primary red through HSL conversion', () => {
    const hsl = rgbToHsl({ r: 255, g: 0, b: 0 });
    expect(hsl.h).toBeCloseTo(0);
    expect(hsl.s).toBeCloseTo(1);
    expect(hsl.l).toBeCloseTo(0.5);
    expect(hslToRgb(hsl.h, hsl.s, hsl.l)).toEqual({ r: 255, g: 0, b: 0 });
  });

  it('chooses a cyan/blue family contrast for red clothing', () => {
    const color = deriveContrastColor({ r: 210, g: 24, b: 36 });
    expect(color.g).toBeGreaterThan(color.r);
    expect(color.b).toBeGreaterThan(color.r);
  });

  it('keeps the teacher stroke readable against clothing and a dark stage', () => {
    const clothing = { r: 24, g: 92, b: 210 };
    const stroke = deriveContrastColor(clothing);
    expect(contrastRatio(stroke, clothing)).toBeGreaterThan(2.2);
    expect(contrastRatio(stroke, { r: 3, g: 6, b: 12 })).toBeGreaterThan(4.5);
  });

  it('keeps readable contrast for red, blue, green, white, and black clothing', () => {
    const darkStage = { r: 3, g: 6, b: 12 };
    const samples = [
      { r: 210, g: 24, b: 36 },
      { r: 24, g: 92, b: 210 },
      { r: 64, g: 160, b: 76 },
      { r: 238, g: 238, b: 232 },
      { r: 18, g: 20, b: 24 },
    ];

    for (const clothing of samples) {
      const stroke = deriveContrastColor(clothing);
      expect(contrastRatio(stroke, clothing)).toBeGreaterThan(1.9);
      expect(contrastRatio(stroke, darkStage)).toBeGreaterThan(4.5);
    }
  });

  it('returns a complete canvas palette', () => {
    const palette = deriveTeacherPaletteFromClothing({ r: 64, g: 160, b: 76 });
    expect(palette.fill).toMatch(/^rgba\(/);
    expect(palette.stroke).toMatch(/^rgba\(/);
    expect(palette.glow).toMatch(/^rgba\(/);
    expect(palette.accent).toMatch(/^rgba\(/);
  });
});
