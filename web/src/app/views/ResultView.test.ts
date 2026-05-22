import { describe, expect, it, vi } from 'vitest';
import {
  buildSubtitle,
  formatPercent,
  formatRate,
  heroTitle,
  renderResultView,
  reviewBody,
  reviewTitle,
  type ResultViewState,
} from './ResultView.js';

function baseState(overrides: Partial<ResultViewState> = {}): ResultViewState {
  return {
    status: 'pending',
    finalAverageText: '--',
    finalConclusionText: '待结算',
    leftArmText: '--',
    rightArmText: '--',
    beatHitRateText: '--',
    beatCompositeText: '--',
    beatStatsText: '0 / 0',
    matchLabel: '待机中',
    ...overrides,
  };
}

describe('formatPercent', () => {
  it('returns -- for null', () => {
    expect(formatPercent(null)).toBe('--');
  });
  it('defaults to 0 fraction digits', () => {
    expect(formatPercent(85.4)).toBe('85%');
  });
  it('honors fraction digits', () => {
    expect(formatPercent(85.4, 1)).toBe('85.4%');
  });
});

describe('formatRate', () => {
  it('returns -- when total is 0', () => {
    expect(formatRate(0, 0)).toBe('--');
  });
  it('computes integer percentage', () => {
    expect(formatRate(3, 4)).toBe('75%');
  });
});

describe('buildSubtitle', () => {
  it('combines the three labels', () => {
    expect(
      buildSubtitle({
        finalAverageText: '82%',
        beatHitRateText: '70%',
        beatCompositeText: '78%',
      }),
    ).toBe('过程均分 82% · 拍点命中 70% · 拍点击分 78%');
  });
});

describe('hero and review copy', () => {
  it('uses the waiting copy before results', () => {
    const s = baseState();
    expect(heroTitle(s)).toBe('等待练习完成');
    expect(reviewTitle(s)).toBe('还没有结算结果');
  });
  it('switches copy when completed without score', () => {
    const s = baseState({ status: 'completed-no-score', beatHitRateText: '60%' });
    expect(heroTitle(s)).toBe('本轮播放已完成');
    expect(reviewTitle(s)).toBe('未产生有效评分');
    expect(reviewBody(s)).toContain('60%');
  });
  it('includes arm and match label when completed', () => {
    const s = baseState({
      status: 'completed',
      finalAverageText: '82%',
      finalConclusionText: '动作到位',
      leftArmText: '80%',
      rightArmText: '84%',
      matchLabel: '动作流畅',
      beatStatsText: '12 / 16',
    });
    expect(reviewTitle(s)).toBe('动作到位');
    expect(reviewBody(s)).toContain('80%');
    expect(reviewBody(s)).toContain('动作流畅');
  });
});

describe('renderResultView', () => {
  it('includes the three anchor ids and the two main CTAs', () => {
    const html = renderResultView(baseState({ status: 'completed', finalAverageText: '80%' }));
    expect(html).toContain('id="finalAverageScore"');
    expect(html).toContain('id="leftArmScore"');
    expect(html).toContain('id="resultReplay"');
    expect(html).toContain('id="resultToSetup"');
  });
  it('disables replay when pending', () => {
    const html = renderResultView(baseState({ status: 'pending' }));
    expect(html).toMatch(/id="resultReplay"[^>]*disabled/);
  });
});

// Quick sanity check on escape behavior for adversarial input
describe('xss safety', () => {
  it('escapes conclusion text before injecting into innerHTML', () => {
    const html = renderResultView(
      baseState({ status: 'completed', finalConclusionText: '<img src=x onerror=1>' }),
    );
    expect(html).not.toContain('<img src=x');
    expect(html).toContain('&lt;img src=x onerror=1&gt;');
  });
});

// Ensures vi is imported to avoid unused warning and allow future mocks.
void vi;
