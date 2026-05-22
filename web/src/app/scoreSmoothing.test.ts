import { describe, expect, it } from 'vitest';
import { createScoreSmoother, formatSmoothedScore } from './scoreSmoothing.js';

describe('createScoreSmoother', () => {
  it('starts at null', () => {
    const s = createScoreSmoother();
    expect(s.value).toBeNull();
  });

  it('alpha=1 acts as raw passthrough', () => {
    const s = createScoreSmoother({ alpha: 1 });
    expect(s.push(80)).toBe(80);
    expect(s.push(40)).toBe(40);
  });

  it('alpha=0.3 follows a step within ~6 ticks', () => {
    const s = createScoreSmoother({ alpha: 0.3 });
    s.push(0);
    let last = 0;
    for (let i = 0; i < 6; i++) last = s.push(100)!;
    expect(last).toBeGreaterThan(85);
  });

  it('alpha=0 freezes at the initial sample', () => {
    const s = createScoreSmoother({ alpha: 0 });
    s.push(50);
    s.push(80);
    s.push(20);
    expect(s.value).toBe(50);
  });

  it('null inputs do not update but preserve last value', () => {
    const s = createScoreSmoother({ alpha: 0.5 });
    s.push(80);
    s.push(null);
    expect(s.value).toBe(80);
  });

  it('reset clears state', () => {
    const s = createScoreSmoother({ alpha: 0.5 });
    s.push(80);
    s.reset();
    expect(s.value).toBeNull();
  });

  it('clamps invalid alpha into [0, 1]', () => {
    const s = createScoreSmoother({ alpha: 99 });
    s.push(50);
    s.push(100);
    // alpha clamped to 1 → passthrough
    expect(s.value).toBe(100);
  });
});

describe('formatSmoothedScore', () => {
  it('rounds to integer percent', () => {
    expect(formatSmoothedScore(82.7)).toBe('83%');
    expect(formatSmoothedScore(82.4)).toBe('82%');
  });
  it('returns -- on null/NaN', () => {
    expect(formatSmoothedScore(null)).toBe('--');
    expect(formatSmoothedScore(Number.NaN)).toBe('--');
  });
});
