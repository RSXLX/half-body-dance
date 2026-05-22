/**
 * BeatTimeline — 可复用节拍时间轴组件。
 *
 * 渲染节拍序列为水平刻度，按状态着色（normal / warning / error），
 * 当前节拍高亮，点击触发 onSeek 回调。
 *
 * 设计：纯 DOM + data-* 委托，不依赖任何框架；保持 SSR/无 React 依赖一致。
 */

import type { Beat } from '../core/beats.js';
import type { BeatAnalysis, BeatStatus } from '../core/beatAnalysis.js';

export interface BeatTimelineState {
  beats: readonly Beat[];
  analyses: readonly BeatAnalysis[];
  /** 当前播放/选中的节拍序号；-1 表示无。 */
  activeBeatIndex: number;
  /** 当前播放时间，绘制刻度线用（秒）。 */
  currentTime: number;
  /** 总时长，归一化布局用（秒）；<=0 时不渲染主轴线。 */
  duration: number;
}

export interface BeatTimelineCallbacks {
  onSeekToBeat: (beatIndex: number) => void;
}

const STATUS_CLASS: Record<BeatStatus, string> = {
  normal: 'is-normal',
  warning: 'is-warning',
  error: 'is-error',
};

const STATUS_LABEL: Record<BeatStatus, string> = {
  normal: '正常',
  warning: '轻微问题',
  error: '严重问题',
};

function statusOf(beatIndex: number, analyses: readonly BeatAnalysis[]): BeatStatus {
  return analyses[beatIndex]?.status ?? 'normal';
}

export function renderBeatTimeline(state: BeatTimelineState): string {
  const { beats, analyses, activeBeatIndex, currentTime, duration } = state;
  if (!beats.length) {
    return `<div class="beat-timeline beat-timeline--empty">未检测到节拍</div>`;
  }
  const total = duration > 0 ? duration : (beats[beats.length - 1]!.endTime || 1);
  const cursorLeft = clampPercent((currentTime / total) * 100);
  const markers = beats
    .map((b) => {
      const status = statusOf(b.beatIndex, analyses);
      const left = clampPercent((b.timestamp / total) * 100);
      const isActive = b.beatIndex === activeBeatIndex;
      const ana = analyses[b.beatIndex];
      const tooltip = ana
        ? `第 ${b.beatIndex + 1} 拍 · ${STATUS_LABEL[ana.status]} · ${ana.actionName}`
        : `第 ${b.beatIndex + 1} 拍`;
      return `
        <button
          type="button"
          class="beat-marker ${STATUS_CLASS[status]} ${isActive ? 'is-active' : ''}"
          style="left:${left}%"
          data-beat-index="${b.beatIndex}"
          title="${escapeAttr(tooltip)}"
          aria-label="${escapeAttr(tooltip)}"
        >
          <span class="beat-marker-dot"></span>
          <span class="beat-marker-label">${b.beatIndex + 1}</span>
        </button>`;
    })
    .join('');

  return `
    <div class="beat-timeline" data-role="beat-timeline">
      <div class="beat-timeline-track">
        <div class="beat-timeline-cursor" style="left:${cursorLeft}%"></div>
        ${markers}
      </div>
      <div class="beat-timeline-legend">
        <span><i class="dot is-normal"></i>正常</span>
        <span><i class="dot is-warning"></i>轻微问题</span>
        <span><i class="dot is-error"></i>严重问题</span>
      </div>
    </div>
  `;
}

export function bindBeatTimeline(
  root: HTMLElement,
  callbacks: BeatTimelineCallbacks,
): () => void {
  const handler = (event: Event) => {
    const target = (event.target as HTMLElement | null)?.closest<HTMLElement>('.beat-marker');
    if (!target) return;
    const idx = Number(target.dataset.beatIndex);
    if (Number.isFinite(idx)) callbacks.onSeekToBeat(idx);
  };
  root.addEventListener('click', handler);
  return () => root.removeEventListener('click', handler);
}

/** 节拍列表（侧栏式），适合在窄屏 / 详情面板下使用。 */
export function renderBeatList(state: BeatTimelineState): string {
  const { beats, analyses, activeBeatIndex } = state;
  if (!beats.length) return '<div class="beat-list beat-list--empty">无节拍</div>';
  const items = beats
    .map((b) => {
      const ana = analyses[b.beatIndex];
      const status = ana?.status ?? 'normal';
      const isActive = b.beatIndex === activeBeatIndex;
      return `
        <li
          class="beat-list-item ${STATUS_CLASS[status]} ${isActive ? 'is-active' : ''}"
          data-beat-index="${b.beatIndex}"
        >
          <span class="beat-list-index">#${b.beatIndex + 1}</span>
          <span class="beat-list-name">${escapeText(ana?.actionName ?? `第 ${b.beatIndex + 1} 拍`)}</span>
          <span class="beat-list-meta">${b.timestamp.toFixed(2)}s · ${STATUS_LABEL[status]}</span>
        </li>`;
    })
    .join('');
  return `<ul class="beat-list" data-role="beat-list">${items}</ul>`;
}

export function bindBeatList(root: HTMLElement, callbacks: BeatTimelineCallbacks): () => void {
  const handler = (event: Event) => {
    const target = (event.target as HTMLElement | null)?.closest<HTMLElement>('.beat-list-item');
    if (!target) return;
    const idx = Number(target.dataset.beatIndex);
    if (Number.isFinite(idx)) callbacks.onSeekToBeat(idx);
  };
  root.addEventListener('click', handler);
  return () => root.removeEventListener('click', handler);
}

function clampPercent(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(100, v));
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
