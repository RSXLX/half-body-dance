import { describe, expect, it } from 'vitest';
import type { NormalizedLandmark, PoseFrame } from '../../core/types.js';
import { buildTeacherAvatarPalette } from './avatarPalette.js';
import {
  findNearestRenderableFrame,
  getTeacherAvatarMetrics,
  getVisibleTeacherBodyParts,
} from './poseParts.js';
import { GLTF_BJD_PART_NAME_CANDIDATES } from './threeTeacherAvatar.js';

function point(x: number, y: number, visibility = 0.95): NormalizedLandmark {
  return { x, y, z: 0, visibility };
}

function makePose(overrides: Record<number, NormalizedLandmark> = {}): NormalizedLandmark[] {
  const pose = Array.from({ length: 33 }, () => point(0.5, 0.5, 0.95));
  Object.assign(pose, {
    0: point(0.5, 0.2),
    7: point(0.46, 0.21),
    8: point(0.54, 0.21),
    11: point(0.42, 0.34),
    12: point(0.58, 0.34),
    13: point(0.36, 0.48),
    14: point(0.64, 0.48),
    15: point(0.32, 0.62),
    16: point(0.68, 0.62),
    23: point(0.44, 0.62),
    24: point(0.56, 0.62),
    25: point(0.42, 0.78),
    26: point(0.58, 0.78),
    27: point(0.4, 0.94),
    28: point(0.6, 0.94),
  });
  for (const [index, value] of Object.entries(overrides)) {
    pose[Number(index)] = value;
  }
  return pose;
}

describe('teacher avatar pose parts', () => {
  it('falls back to the nearest frame with renderable pose landmarks', () => {
    const frames: PoseFrame[] = [
      { time: 0, pose_landmarks: [] },
      { time: 0.033, pose_landmarks: [] },
      { time: 0.066, pose_landmarks: makePose() },
    ];

    expect(findNearestRenderableFrame(frames, 0)).toBe(frames[2]);
    expect(findNearestRenderableFrame(frames, 2)).toBe(frames[2]);
  });

  it('skips low-confidence lower-body parts without dropping visible upper-body parts', () => {
    const pose = makePose({
      25: point(0.42, 0.78, 0.05),
      27: point(0.4, 0.94, 0.05),
    });
    const visible = getVisibleTeacherBodyParts(pose);

    expect(visible.some((part) => part.id === 'leftUpperArm')).toBe(true);
    expect(visible.some((part) => part.id === 'leftCalf')).toBe(false);
  });

  it('uses tapered radii for limbs and stable joint sizes from the torso scale', () => {
    const metrics = getTeacherAvatarMetrics(makePose(), { x: 0, y: 0, width: 720, height: 1280 });

    expect(metrics.limbs.leftThigh.startRadius).toBeGreaterThan(metrics.limbs.leftThigh.endRadius);
    expect(metrics.limbs.rightForearm.startRadius).toBeGreaterThan(metrics.limbs.rightForearm.endRadius);
    expect(metrics.joints.elbow).toBeGreaterThan(3);
    expect(metrics.joints.hip).toBeGreaterThan(metrics.joints.elbow);
  });
});

describe('teacher avatar palette', () => {
  it('preserves the dynamic teacher stroke while adding BJD resin material colors', () => {
    const palette = buildTeacherAvatarPalette({
      fill: 'rgba(12, 200, 220, 0.24)',
      stroke: 'rgba(12, 200, 220, 0.98)',
      glow: 'rgba(12, 200, 220, 0.58)',
      accent: 'rgba(180, 255, 255, 1)',
    });

    expect(palette.teacherStroke).toBe('rgba(12, 200, 220, 0.98)');
    expect(palette.resinFill).toMatch(/^rgba\(/);
    expect(palette.shadowColor).toMatch(/^rgba\(/);
    expect(palette.highlightColor).toMatch(/^rgba\(/);
    expect(palette.jointStroke).toMatch(/^rgba\(/);
  });
});

describe('three teacher avatar GLB mapping', () => {
  it('maps the checked-in BJD basemesh names to driven body regions', () => {
    expect(GLTF_BJD_PART_NAME_CANDIDATES.upperArm).toContain('upperArmMesh004');
    expect(GLTF_BJD_PART_NAME_CANDIDATES.foreArm).toContain('foreArmMesh002');
    expect(GLTF_BJD_PART_NAME_CANDIDATES.thigh).toContain('tightMesh002');
    expect(GLTF_BJD_PART_NAME_CANDIDATES.calf).toContain('calfMesh002');
    expect(GLTF_BJD_PART_NAME_CANDIDATES.chest).toContain('chestMesh002');
    expect(GLTF_BJD_PART_NAME_CANDIDATES.head).toContain('headMesh002');
  });
});
