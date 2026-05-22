// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { bindSegmentCards, renderSegmentCards } from './SegmentCard.js';
import type { MotionSegment } from '../core/motionTypes.js';

const segment: MotionSegment = {
  id: 'seg-1',
  index: 0,
  startTime: 0,
  endTime: 1.5,
  duration: 1.5,
  emoji: '👋',
  title: '挥手开场',
  description: '右手从身体侧面向上挥动。',
  primaryJoints: ['rightWrist'],
  primaryDirection: 'up',
  difficulty: 2,
  keyFrames: [0, 1.5],
  tips: ['右手抬到肩膀上方', '保持身体稳定'],
};

function createSegment(overrides: Partial<MotionSegment>): MotionSegment {
  return {
    ...segment,
    ...overrides,
  };
}

describe('renderSegmentCards', () => {
  it('renders an empty state when no motion segments are available', () => {
    expect(renderSegmentCards([])).toContain('暂无片段分解');
  });

  it('renders segment title, time range, first tip, and difficulty', () => {
    const html = renderSegmentCards([segment]);

    expect(html).toContain('挥手开场');
    expect(html).toContain('00:00.00 – 00:01.50');
    expect(html).toContain('右手抬到肩膀上方');
    expect(html).toContain('难度 2');
  });

  it('escapes text fields before rendering HTML', () => {
    const unsafeSegment = createSegment({
      title: '<script>alert("title")</script> & title',
      description: '<script>alert("description")</script> & description',
      tips: ['<script>alert("tip")</script> & tip'],
    });

    const html = renderSegmentCards([unsafeSegment]);

    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;script&gt;alert("title")&lt;/script&gt; &amp; title');
    expect(html).toContain('&lt;script&gt;alert("tip")&lt;/script&gt; &amp; tip');
  });

  it('escapes description when it is used as the fallback tip', () => {
    const fallbackSegment = createSegment({
      description: '<script>alert("description")</script> & description',
      tips: [],
    });

    const html = renderSegmentCards([fallbackSegment]);

    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;script&gt;alert("description")&lt;/script&gt; &amp; description');
  });

  it('escapes segment id inside the data attribute and preserves the clicked segment id', () => {
    const unsafeIdSegment = createSegment({ id: 'seg-"<&>\'-1' });
    const root = document.createElement('div');
    root.innerHTML = renderSegmentCards([unsafeIdSegment]);
    const selected: MotionSegment[] = [];

    const card = root.querySelector<HTMLButtonElement>('.motion-segment-card');
    const unbind = bindSegmentCards(root, [unsafeIdSegment], (clickedSegment) => {
      selected.push(clickedSegment);
    });

    expect(card?.getAttribute('data-segment-id')).toBe(unsafeIdSegment.id);
    card?.click();
    unbind();

    expect(selected).toEqual([unsafeIdSegment]);
  });

  it('marks the active segment card', () => {
    const html = renderSegmentCards([segment], 'seg-1');
    const root = document.createElement('div');
    root.innerHTML = html;

    expect(root.querySelector('.motion-segment-card')?.classList.contains('is-active')).toBe(true);
  });

  it('accepts null active segment id without marking a card active', () => {
    const html = renderSegmentCards([segment], null);
    const root = document.createElement('div');
    root.innerHTML = html;

    expect(root.querySelector('.motion-segment-card')?.classList.contains('is-active')).toBe(false);
  });

  it('uses description as the tip when tips are empty', () => {
    const fallbackSegment = createSegment({
      description: '用描述作为提示。',
      tips: [],
    });

    expect(renderSegmentCards([fallbackSegment])).toContain('用描述作为提示。');
  });
});

describe('bindSegmentCards', () => {
  it('calls back with the clicked segment', () => {
    const root = document.createElement('div');
    root.innerHTML = renderSegmentCards([segment]);
    const selected: MotionSegment[] = [];

    const unbind = bindSegmentCards(root, [segment], (clickedSegment) => {
      selected.push(clickedSegment);
    });

    root.querySelector<HTMLButtonElement>('[data-segment-id="seg-1"]')?.click();
    unbind();

    expect(selected).toEqual([segment]);
  });

  it('does not call back after unbind is returned and invoked', () => {
    const root = document.createElement('div');
    root.innerHTML = renderSegmentCards([segment]);
    const selected: MotionSegment[] = [];

    const unbind = bindSegmentCards(root, [segment], (clickedSegment) => {
      selected.push(clickedSegment);
    });
    unbind();

    root.querySelector<HTMLButtonElement>('[data-segment-id="seg-1"]')?.click();

    expect(selected).toEqual([]);
  });

  it('ignores unknown segment ids', () => {
    const root = document.createElement('div');
    root.innerHTML = '<button class="motion-segment-card" type="button" data-segment-id="unknown">未知</button>';
    const selected: MotionSegment[] = [];

    const unbind = bindSegmentCards(root, [segment], (clickedSegment) => {
      selected.push(clickedSegment);
    });

    root.querySelector<HTMLButtonElement>('.motion-segment-card')?.click();
    unbind();

    expect(selected).toEqual([]);
  });

  it('ignores non-card elements with data-segment-id', () => {
    const root = document.createElement('div');
    root.innerHTML = '<span data-segment-id="seg-1">不是卡片</span>';
    const selected: MotionSegment[] = [];

    const unbind = bindSegmentCards(root, [segment], (clickedSegment) => {
      selected.push(clickedSegment);
    });

    root.querySelector<HTMLElement>('[data-segment-id="seg-1"]')?.click();
    unbind();

    expect(selected).toEqual([]);
  });
});
