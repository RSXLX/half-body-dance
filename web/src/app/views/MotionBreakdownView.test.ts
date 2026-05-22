// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import {
  bindMotionBreakdownView,
  renderMotionBreakdownView,
  type MotionBreakdownViewState,
} from './MotionBreakdownView.js';
import type { MotionAnalysis, MotionSegment } from '../../core/motionTypes.js';

const segment: MotionSegment = {
  id: 'seg-wave',
  index: 0,
  startTime: 0,
  endTime: 1.2,
  duration: 1.2,
  emoji: '👋',
  title: '右手挥动',
  description: '右手从胸前向上挥动。',
  primaryJoints: ['rightWrist'],
  primaryDirection: 'up',
  difficulty: 2,
  keyFrames: [0, 1.2],
  tips: ['手腕抬到肩膀上方'],
  beatIndices: [0],
};

const motion: MotionAnalysis = {
  schema_version: 2,
  source_pose: 'angel_pose.json',
  fps: 30,
  duration: 1.2,
  extracted_at: '2026-05-10T00:00:00Z',
  extract_config: {
    stride: 1,
    smoothingWindow: 5,
    stillSpeedThreshold: 0.02,
    visibilityThreshold: 0.5,
    minSegmentDuration: 0.4,
    maxSegments: 8,
  },
  trajectories: {},
  beats: [
    {
      index: 0,
      startTime: 0,
      endTime: 0.6,
      duration: 0.6,
      segmentId: 'seg-wave',
      primaryJoint: 'rightWrist',
      primaryDirection: 'up',
      emoji: '🙋',
      label: '右手向上',
      jointStats: {
        rightWrist: {
          distance: 0.28,
          pathLength: 0.32,
          peakSpeed: 0.9,
          cardinal: 'up',
          visibility: 0.98,
          angleDeg: 88,
        },
      },
    },
  ],
  segments: [segment],
  hints: [
    {
      triggerTime: 0,
      segmentId: 'seg-wave',
      preview: '准备抬右手',
      cue: 'beatPrep',
      leadMs: 300,
    },
  ],
  summary: {
    primaryJoints: ['rightWrist'],
    dominantDirections: [{ cardinal: 'up', share: 1 }],
    bpm: 100,
    beatTimes: [0],
  },
};

function createState(overrides: Partial<MotionBreakdownViewState> = {}): MotionBreakdownViewState {
  return {
    presetName: 'Angel',
    status: 'ready',
    errorMessage: null,
    motion,
    activeSegmentId: null,
    ...overrides,
  };
}

describe('renderMotionBreakdownView', () => {
  it('renders ready state with title, beat label, and segment title', () => {
    const html = renderMotionBreakdownView(createState());

    expect(html).toContain('Angel · 动作分解');
    expect(html).toContain('右手向上');
    expect(html).toContain('右手挥动');
  });

  it('renders unavailable state when status is empty and motion is null', () => {
    const html = renderMotionBreakdownView(
      createState({
        status: 'empty',
        motion: null,
      }),
    );

    expect(html).toContain('动作分解暂不可用');
  });
});

describe('bindMotionBreakdownView', () => {
  it('calls callbacks when back, practice, and segment card are clicked', () => {
    const state = createState();
    const root = document.createElement('div');
    root.innerHTML = renderMotionBreakdownView(state);
    const events: string[] = [];
    const selected: MotionSegment[] = [];

    const unbind = bindMotionBreakdownView(root, state, {
      onBack: () => events.push('back'),
      onPractice: () => events.push('practice'),
      onSelectSegment: (clickedSegment) => selected.push(clickedSegment),
    });

    root.querySelector<HTMLButtonElement>('#motionBack')?.click();
    root.querySelector<HTMLButtonElement>('#motionStartPractice')?.click();
    root.querySelector<HTMLButtonElement>('[data-segment-id="seg-wave"]')?.click();
    unbind();

    expect(events).toEqual(['back', 'practice']);
    expect(selected).toEqual([segment]);
  });
});
