import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import { BUNDLED_POSE_PRESETS, motionPathForPreset } from '../app/data/presets.js';
import {
  analyzeMotionFromPose,
  getSegmentStartTime,
  isMotionAnalysis,
  loadMotionFromPath,
} from './motionData.js';
import type { MotionAnalysis } from './motionTypes.js';

const validMotion = {
  schema_version: 2,
  source_pose: 'sample_pose.json',
  fps: 30,
  duration: 3.2,
  extracted_at: '2026-05-10T00:00:00Z',
  extract_config: {},
  trajectories: {},
  beats: [
    {
      index: 0,
      startTime: 0,
      endTime: 1,
      duration: 1,
      segmentId: 'segment-1',
      primaryJoint: 'leftWrist',
      primaryDirection: 'right',
      emoji: '👉',
      label: '左手向右',
      jointStats: {
        leftWrist: {
          distance: 0.42,
          pathLength: 0.58,
          peakSpeed: 1.2,
          cardinal: 'right',
          visibility: 0.98,
          angleDeg: 5.2,
        },
      },
    },
  ],
  limbs: {
    leftWrist: [
      {
        beatIndex: 0,
        startTime: 0,
        endTime: 1,
        cardinal: 'right',
        emoji: '👉',
        distance: 0.42,
        peakSpeed: 1.2,
        armPose: { elbowAngleDeg: 152.4, wristHeightBand: 'mid' },
      },
    ],
  },
  segments: [
    {
      id: 'segment-1',
      index: 0,
      startTime: 0.5,
      endTime: 1.25,
      duration: 0.75,
      emoji: '👋',
      title: '挥手',
      description: '手臂向外打开',
      primaryJoints: ['leftWrist'],
      primaryDirection: 'right',
      difficulty: 1,
      keyFrames: [0.5, 1.25],
      tips: ['手腕放松'],
    },
  ],
  hints: [],
  summary: {
    primaryJoints: ['leftWrist'],
    dominantDirections: [{ cardinal: 'right', share: 1 }],
  },
} satisfies MotionAnalysis;

const jsonResponse = (status: number, payload: unknown): Response =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { 'content-type': 'application/json' },
  });

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

const readRepoJson = (repoRelativePath: string): unknown =>
  JSON.parse(readFileSync(path.join(repoRoot, repoRelativePath), 'utf8'));

describe('bundled motion analysis files', () => {
  it('match the runtime motion schema for every bundled preset', () => {
    const motionPaths = BUNDLED_POSE_PRESETS.map(motionPathForPreset);

    expect(motionPaths).toHaveLength(9);
    for (const motionPath of motionPaths) {
      expect(isMotionAnalysis(readRepoJson(motionPath)), motionPath).toBe(true);
    }
  });
});

describe('isMotionAnalysis', () => {
  it('accepts valid motion data with beats and segments arrays', () => {
    expect(isMotionAnalysis(validMotion)).toBe(true);
  });

  it('rejects motion data without beats or segments arrays', () => {
    expect(isMotionAnalysis({ ...validMotion, beats: undefined })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, segments: undefined })).toBe(false);
  });

  it('rejects non-object values and wrong beats or segments types', () => {
    expect(isMotionAnalysis(null)).toBe(false);
    expect(isMotionAnalysis('motion')).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: {} })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, segments: 'bad' })).toBe(false);
  });

  it('rejects malformed motion payloads with missing required fields or invalid segment timing', () => {
    const withoutSummary = { ...validMotion } as Record<string, unknown>;
    delete withoutSummary.summary;
    const withoutHints = { ...validMotion } as Record<string, unknown>;
    delete withoutHints.hints;

    expect(isMotionAnalysis(withoutSummary)).toBe(false);
    expect(isMotionAnalysis(withoutHints)).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, schema_version: 1 })).toBe(false);
    expect(
      isMotionAnalysis({
        ...validMotion,
        segments: [{ ...validMotion.segments[0], startTime: Number.NaN }],
      }),
    ).toBe(false);
    expect(
      isMotionAnalysis({
        ...validMotion,
        segments: [{ ...validMotion.segments[0], startTime: 2, endTime: 1 }],
      }),
    ).toBe(false);
    expect(
      isMotionAnalysis({
        ...validMotion,
        segments: [{ ...validMotion.segments[0], duration: -0.1 }],
      }),
    ).toBe(false);
    expect(
      isMotionAnalysis({
        ...validMotion,
        segments: [{ ...validMotion.segments[0], difficulty: 4 }],
      }),
    ).toBe(false);
    expect(
      isMotionAnalysis({
        ...validMotion,
        segments: [{ ...validMotion.segments[0], primaryJoints: 'leftWrist' }],
      }),
    ).toBe(false);
  });

  it('rejects non-positive fps and negative duration values', () => {
    expect(isMotionAnalysis({ ...validMotion, fps: 0 })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, fps: -1 })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, duration: -0.1 })).toBe(false);
  });

  it('rejects payloads whose top-level fields do not match the motion contract', () => {
    expect(isMotionAnalysis({ ...validMotion, source_pose: undefined })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, source_pose: 123 })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, extracted_at: undefined })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, extracted_at: 123 })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, extract_config: undefined })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, extract_config: null })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, extract_config: [] })).toBe(false);
  });

  it('rejects beats that do not match the writer schema', () => {
    const validBeat = validMotion.beats[0];

    expect(isMotionAnalysis({ ...validMotion, beats: [null] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: [{ ...validBeat, index: Number.NaN }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: [{ ...validBeat, startTime: -0.1 }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: [{ ...validBeat, endTime: 0, startTime: 1 }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: [{ ...validBeat, duration: -0.1 }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: [{ ...validBeat, primaryJoint: 15 }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: [{ ...validBeat, primaryJoint: 'leftShoulder' }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: [{ ...validBeat, primaryDirection: 'north' }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: [{ ...validBeat, jointStats: null }] })).toBe(false);
  });

  it('rejects limb slices that do not match the writer schema', () => {
    const validSlice = validMotion.limbs.leftWrist[0];

    expect(isMotionAnalysis({ ...validMotion, limbs: null })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, limbs: { leftShoulder: [validSlice] } })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, limbs: { leftWrist: [null] } })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, limbs: { leftWrist: [{ ...validSlice, beatIndex: Number.NaN }] } })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, limbs: { leftWrist: [{ ...validSlice, endTime: 0, startTime: 1 }] } })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, limbs: { leftWrist: [{ ...validSlice, cardinal: 'north' }] } })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, limbs: { leftWrist: [{ ...validSlice, emoji: 123 }] } })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, limbs: { leftWrist: [{ ...validSlice, distance: -0.1 }] } })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, limbs: { leftWrist: [{ ...validSlice, peakSpeed: Infinity }] } })).toBe(false);
  });

  it('rejects segments missing writer-required fields or invalid joint and direction values', () => {
    expect(isMotionAnalysis({ ...validMotion, segments: [{ ...validMotion.segments[0], index: undefined }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, segments: [{ ...validMotion.segments[0], emoji: undefined }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, segments: [{ ...validMotion.segments[0], primaryDirection: undefined }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, segments: [{ ...validMotion.segments[0], primaryDirection: 'north' }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, segments: [{ ...validMotion.segments[0], primaryJoints: ['leftShoulder'] }] })).toBe(false);
  });

  it('accepts the real writer shape without legacy beatIndex timestamp or preview fields', () => {
    expect(validMotion.beats[0]).not.toHaveProperty('beatIndex');
    expect(validMotion.beats[0]).not.toHaveProperty('timestamp');
    expect(validMotion.beats[0]).not.toHaveProperty('preview');
    expect(validMotion.limbs.leftWrist[0]).not.toHaveProperty('joint');
    expect(validMotion.limbs.leftWrist[0]).not.toHaveProperty('direction');
    expect(validMotion.limbs.leftWrist[0]).not.toHaveProperty('visibility');
    expect(isMotionAnalysis(validMotion)).toBe(true);
  });

  it("accepts writer-valid segments whose primaryDirection is mixed", () => {
    expect(
      isMotionAnalysis({
        ...validMotion,
        segments: [{ ...validMotion.segments[0], primaryDirection: 'mixed' }],
      }),
    ).toBe(true);
  });

  it('rejects hints and summary fields that do not match the motion contract', () => {
    const hint = {
      triggerTime: 0.25,
      segmentId: 'segment-1',
      preview: '准备挥手',
      cue: 'beatPrep',
      leadMs: 250,
    };

    expect(isMotionAnalysis({ ...validMotion, hints: [hint] })).toBe(true);
    expect(isMotionAnalysis({ ...validMotion, hints: [null] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, hints: [{ ...hint, triggerTime: Number.NaN }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, hints: [{ ...hint, leadMs: Infinity }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, hints: [{ ...hint, cue: 123 }] })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, summary: { ...validMotion.summary, primaryJoints: 'leftWrist' } })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, summary: { ...validMotion.summary, dominantDirections: {} } })).toBe(false);
  });
});

describe('getSegmentStartTime', () => {
  it('returns a finite non-negative startTime', () => {
    expect(getSegmentStartTime(validMotion.segments[0])).toBe(0.5);
    expect(getSegmentStartTime({ startTime: 0 })).toBe(0);
  });

  it('returns null for invalid startTime values', () => {
    expect(getSegmentStartTime({ startTime: -0.1 })).toBeNull();
    expect(getSegmentStartTime({ startTime: Number.NaN })).toBeNull();
    expect(getSegmentStartTime({ startTime: Infinity })).toBeNull();
    expect(getSegmentStartTime({})).toBeNull();
    expect(getSegmentStartTime(null)).toBeNull();
  });
});

describe('loadMotionFromPath', () => {
  it('returns motion data when fetch succeeds with a valid schema', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(200, validMotion));

    await expect(loadMotionFromPath('/wudao/demo_motion.json', fetchImpl)).resolves.toStrictEqual(validMotion);
    expect(fetchImpl).toHaveBeenCalledWith('/wudao/demo_motion.json');
  });

  it('returns null when fetch returns 404', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(404, { error: 'not found' }));

    await expect(loadMotionFromPath('/missing_motion.json', fetchImpl)).resolves.toBeNull();
  });

  it('returns null when fetched JSON is not motion data', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(200, { ...validMotion, beats: {} }));

    await expect(loadMotionFromPath('/bad_motion.json', fetchImpl)).resolves.toBeNull();
  });
});

describe('analyzeMotionFromPose', () => {
  it('posts pose JSON to /api/analyze-motion and returns valid motion on ok payloads', async () => {
    const poseJson = { fps: 30, frames: [] };
    const fetchImpl = vi.fn(async () => jsonResponse(200, { ok: true, motion: validMotion }));

    await expect(analyzeMotionFromPose(poseJson, fetchImpl)).resolves.toStrictEqual(validMotion);
    expect(fetchImpl).toHaveBeenCalledWith('/api/analyze-motion', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ poseJson }),
    });
  });

  it('returns null when the analysis payload is not ok', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(200, { ok: false, error: 'failed' }));

    await expect(analyzeMotionFromPose({ frames: [] }, fetchImpl)).resolves.toBeNull();
  });

  it('returns null when the analysis motion schema is invalid', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(200, { ok: true, motion: { ...validMotion, segments: null } }));

    await expect(analyzeMotionFromPose({ frames: [] }, fetchImpl)).resolves.toBeNull();
  });
});
