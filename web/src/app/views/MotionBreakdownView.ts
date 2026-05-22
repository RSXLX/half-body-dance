import type { MotionAnalysis, MotionSegment } from '../../core/motionTypes.js';
import { renderBeatStrip } from '../../ui/BeatStrip.js';
import { bindSegmentCards, renderSegmentCards } from '../../ui/SegmentCard.js';

export type MotionBreakdownStatus = 'idle' | 'loading' | 'ready' | 'empty' | 'error';

export interface MotionBreakdownViewState {
  presetName: string;
  status: MotionBreakdownStatus;
  errorMessage: string | null;
  motion: MotionAnalysis | null;
  activeSegmentId: string | null;
}

export interface MotionBreakdownViewCallbacks {
  onBack: () => void;
  onPractice: () => void;
  onSelectSegment: (segment: MotionSegment) => void;
}

export function renderMotionBreakdownView(state: MotionBreakdownViewState): string {
  return `
    <section class="motion-breakdown-view" data-view="motion-breakdown">
      ${renderHeader(state)}
      ${renderBody(state)}
    </section>
  `;
}

export function bindMotionBreakdownView(
  root: HTMLElement,
  state: MotionBreakdownViewState,
  callbacks: MotionBreakdownViewCallbacks,
): () => void {
  const handleClick = (event: Event) => {
    const target = event.target;
    if (!(target instanceof Element)) {
      return;
    }

    if (target.closest('#motionBack')) {
      callbacks.onBack();
    } else if (target.closest('#motionStartPractice')) {
      callbacks.onPractice();
    }
  };

  root.addEventListener('click', handleClick);
  const unbindSegments = bindSegmentCards(root, state.motion?.segments ?? [], callbacks.onSelectSegment);

  return () => {
    root.removeEventListener('click', handleClick);
    unbindSegments();
  };
}

function renderHeader(state: MotionBreakdownViewState): string {
  const presetName = state.presetName.trim() || '当前动作';
  return `
    <header class="motion-breakdown-header">
      <button id="motionBack" class="ghost" type="button" aria-label="返回">返回</button>
      <div class="motion-breakdown-heading">
        <strong>${escapeHtml(presetName)} · 动作分解</strong>
      </div>
      <button id="motionStartPractice" class="primary" type="button">开始练习</button>
    </header>
  `;
}

function renderBody(state: MotionBreakdownViewState): string {
  if (state.status === 'loading') {
    return '<div class="motion-breakdown-loading">正在生成动作分解…</div>';
  }

  if (state.status === 'error' || state.status === 'empty' || !state.motion) {
    const message = state.errorMessage ?? '动作分解暂不可用，可继续普通练习。';
    return `<div class="motion-breakdown-empty">${escapeHtml(message)}</div>`;
  }

  return `
    <div class="motion-breakdown-body">
      <section class="motion-breakdown-beats" aria-label="拍点分解">
        ${renderBeatStrip(state.motion.beats)}
      </section>
      <section class="motion-breakdown-segments" aria-label="动作片段">
        ${renderSegmentCards(state.motion.segments, state.activeSegmentId)}
      </section>
    </div>
  `;
}

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
