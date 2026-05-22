import { describe, expect, it } from 'vitest';
import {
  BUNDLED_POSE_PRESETS,
  findPresetById,
  findPresetByLabel,
  motionPathForPreset,
  type PosePreset,
} from './presets.js';

describe('preset metadata', () => {
  it('exports 9 bundled presets with unique ids', () => {
    expect(BUNDLED_POSE_PRESETS.length).toBe(9);
    const ids = new Set(BUNDLED_POSE_PRESETS.map((p) => p.id));
    expect(ids.size).toBe(9);
  });

  it('has exactly one featured preset', () => {
    expect(BUNDLED_POSE_PRESETS.filter((p) => p.featured).length).toBe(1);
  });

  it('paths point under wudao/ and end with _pose.json', () => {
    BUNDLED_POSE_PRESETS.forEach((p) => {
      expect(p.path.startsWith('wudao/')).toBe(true);
      expect(p.path.endsWith('_pose.json')).toBe(true);
    });
  });
});

describe('findPresetById / findPresetByLabel', () => {
  it('matches by stable id', () => {
    expect(findPresetById('hongmen')?.id).toBe('hongmen');
  });
  it('matches by display name', () => {
    expect(findPresetById('鸿门旋律')?.id).toBe('hongmen');
  });
  it('matches by full label', () => {
    expect(findPresetByLabel('鸿门旋律_pose.json')?.id).toBe('hongmen');
  });
  it('returns undefined for unknown', () => {
    expect(findPresetById('nope')).toBeUndefined();
    expect(findPresetByLabel('nope.json')).toBeUndefined();
  });
});

describe('motionPathForPreset', () => {
  it('replaces _pose.json with _motion.json', () => {
    const p = BUNDLED_POSE_PRESETS[0]!;
    expect(motionPathForPreset(p)).toBe(p.path.replace('_pose.json', '_motion.json'));
  });

  it('honors explicit override', () => {
    const p: PosePreset = {
      ...BUNDLED_POSE_PRESETS[0]!,
      motionPath: 'custom/elsewhere.json',
    };
    expect(motionPathForPreset(p)).toBe('custom/elsewhere.json');
  });

  it('falls back to extension swap when path does not end with _pose.json', () => {
    const p: PosePreset = {
      id: 'x',
      name: 'x',
      emoji: '🌟',
      tagline: '',
      path: 'wudao/x.json',
      label: 'x.json',
    };
    expect(motionPathForPreset(p)).toBe('wudao/x_motion.json');
  });

  it('appends suffix when no extension at all', () => {
    const p: PosePreset = {
      id: 'x',
      name: 'x',
      emoji: '🌟',
      tagline: '',
      path: 'wudao/x',
      label: 'x',
    };
    expect(motionPathForPreset(p)).toBe('wudao/x_motion.json');
  });
});
