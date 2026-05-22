import type { MotionSegment } from '../core/motionTypes.js';
import { formatMotionTime } from './BeatStrip.js';

export type { MotionSegment } from '../core/motionTypes.js';

export function renderSegmentCards(
  segments: readonly MotionSegment[],
  activeSegmentId?: string | null,
): string {
  if (!segments.length) {
    return '<div class="motion-empty">暂无片段分解。</div>';
  }

  const items = segments
    .map((segment) => {
      const activeClass = segment.id === activeSegmentId ? ' is-active' : '';
      const tip = segment.tips[0] ?? segment.description;
      return `
        <button class="motion-segment-card${activeClass}" type="button" role="listitem" data-segment-id="${escapeAttr(segment.id)}">
          <span class="motion-segment-emoji" aria-hidden="true">${escapeText(segment.emoji)}</span>
          <span class="motion-segment-body">
            <span class="motion-segment-title">${escapeText(segment.title)}</span>
            <span class="motion-segment-time">${escapeText(formatSegmentTime(segment))}</span>
            <span class="motion-segment-difficulty">难度 ${escapeText(String(segment.difficulty))}</span>
            <span class="motion-segment-tip">${escapeText(tip)}</span>
          </span>
        </button>`;
    })
    .join('');

  return `<div class="motion-segment-list" role="list">${items}</div>`;
}

export function bindSegmentCards(
  root: Element,
  segments: readonly MotionSegment[],
  onSelectSegment: (segment: MotionSegment) => void,
): () => void {
  const handleClick = (event: Event) => {
    const target = event.target;
    if (!(target instanceof Element)) {
      return;
    }

    const card = target.closest<HTMLElement>('.motion-segment-card[data-segment-id]');
    if (!card || !root.contains(card)) {
      return;
    }

    const segmentId = card.dataset.segmentId;
    const segment = segments.find((candidate) => candidate.id === segmentId);
    if (segment) {
      onSelectSegment(segment);
    }
  };

  root.addEventListener('click', handleClick);
  return () => root.removeEventListener('click', handleClick);
}

function formatSegmentTime(segment: MotionSegment): string {
  return `${formatMotionTime(segment.startTime)} – ${formatMotionTime(segment.endTime)}`;
}

function escapeText(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function escapeAttr(input: string): string {
  return escapeText(input).replace(/"/g, '&quot;');
}
