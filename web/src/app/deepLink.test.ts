import { describe, expect, it } from 'vitest';
import { buildDeepLink, parseDeepLink } from './deepLink.js';
import { BUNDLED_POSE_PRESETS, findPresetById } from './data/presets.js';

describe('parseDeepLink', () => {
  it('returns null preset when no search params', () => {
    const intent = parseDeepLink('', findPresetById);
    expect(intent.preset).toBeNull();
    expect(intent.autoplay).toBe(false);
    expect(intent.unresolvedPresetName).toBeNull();
  });

  it('resolves preset by id', () => {
    const intent = parseDeepLink('?preset=angel&autoplay=1', findPresetById);
    expect(intent.preset?.id).toBe('angel');
    expect(intent.autoplay).toBe(true);
  });

  it('resolves preset by display name', () => {
    const intent = parseDeepLink('?preset=' + encodeURIComponent('鸿门旋律'), findPresetById);
    expect(intent.preset?.id).toBe('hongmen');
  });

  it('reports unresolved name when no match', () => {
    const intent = parseDeepLink('?preset=mystery', findPresetById);
    expect(intent.preset).toBeNull();
    expect(intent.unresolvedPresetName).toBe('mystery');
  });

  it('ignores autoplay when not "1"', () => {
    const intent = parseDeepLink('?preset=angel&autoplay=yes', findPresetById);
    expect(intent.autoplay).toBe(false);
  });

  it('preserves supported view parameters for router integration', () => {
    const intent = parseDeepLink('?preset=angel&view=motion', findPresetById);
    expect(intent.view).toBe('motion');
  });
});

describe('buildDeepLink', () => {
  it('omits params when no preset and no autoplay', () => {
    expect(buildDeepLink('https://x.test', null, false)).toBe('https://x.test/pose_viewer.html');
  });

  it('adds preset id and autoplay', () => {
    const url = buildDeepLink('https://x.test', BUNDLED_POSE_PRESETS[0], true);
    expect(url).toContain('preset=hongmen');
    expect(url).toContain('autoplay=1');
  });
});
