import type { TeacherAvatarPalette, TeacherBasePalette } from './types.js';

export const DEFAULT_RESIN_MATERIAL = {
  resinFill: 'rgba(243, 234, 220, 0.74)',
  resinLight: 'rgba(255, 248, 236, 0.88)',
  resinShadow: 'rgba(216, 195, 168, 0.82)',
  shadowColor: 'rgba(64, 42, 32, 0.28)',
  highlightColor: 'rgba(255, 255, 255, 0.42)',
  jointStroke: 'rgba(90, 58, 42, 0.34)',
} as const;

export function buildTeacherAvatarPalette(base: TeacherBasePalette): TeacherAvatarPalette {
  return {
    ...DEFAULT_RESIN_MATERIAL,
    teacherStroke: base.stroke,
    teacherGlow: base.glow ?? base.stroke,
    teacherAccent: base.accent ?? base.stroke,
  };
}
