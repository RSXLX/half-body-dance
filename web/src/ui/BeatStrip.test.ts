import { describe, expect, it } from 'vitest';
import { formatMotionTime, renderBeatStrip } from './BeatStrip.js';
import type { BeatAction } from '../core/motionTypes.js';

const beats: BeatAction[] = [
  {
    index: 0,
    startTime: 0,
    endTime: 1.2,
    duration: 1.2,
    segmentId: 'segment-1',
    primaryJoint: 'rightWrist',
    primaryDirection: 'up',
    emoji: '👆',
    label: '右手向上',
    jointStats: {
      rightWrist: {
        distance: 0.36,
        pathLength: 0.48,
        peakSpeed: 1.15,
        cardinal: 'up',
        visibility: 0.96,
        angleDeg: 87,
      },
    },
  },
  {
    index: 1,
    startTime: 1.2,
    endTime: 2.4,
    duration: 1.2,
    primaryJoint: 'leftWrist',
    primaryDirection: 'left',
    emoji: '👈',
    label: '左手向左',
    jointStats: {
      leftWrist: {
        distance: 0.31,
        pathLength: 0.44,
        peakSpeed: 1.02,
        cardinal: 'left',
        visibility: 0.94,
      },
    },
  },
];

describe('formatMotionTime', () => {
  it('formats seconds as mm:ss.cs', () => {
    expect(formatMotionTime(0)).toBe('00:00.00');
    expect(formatMotionTime(65.34)).toBe('01:05.34');
  });
});

describe('renderBeatStrip', () => {
  it('renders an empty state when no beat actions are available', () => {
    expect(renderBeatStrip([])).toContain('暂无拍点分解');
  });

  it('renders one motion beat item per BeatAction with labels from the writer schema', () => {
    const html = renderBeatStrip(beats);

    expect(html).toContain('role="list"');
    expect(html.match(/class="motion-beat motion-dir-/g)).toHaveLength(2);
    expect(html).toContain('右手向上');
    expect(html).toContain('左手向左');
    expect(html).toContain('00:00.00–00:01.20');
    expect(html).toContain('00:01.20–00:02.40');
  });
});
