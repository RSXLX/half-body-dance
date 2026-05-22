/**
 * BeatAnalysisView — 节拍级动作分析视图。
 *
 * 职责：呈现节拍时间轴 / 节拍列表 / 当前节拍详情 / 修复入口。
 * 视图本身是无状态壳：所有可变状态由调用方在 BeatAnalysisViewState 中给出，
 * 实际播放/跳转/重新生成由 main.ts 的 controller 负责。
 */

import type { Beat } from '../../core/beats.js';
import type { BeatAnalysis, BeatAnalysisSummary } from '../../core/beatAnalysis.js';
import { directionLabelCN } from '../../core/beatAnalysis.js';
import {
  renderBeatTimeline,
  renderBeatList,
  bindBeatTimeline,
  bindBeatList,
} from '../../ui/BeatTimeline.js';

export type BeatAnalysisLoadStatus = 'idle' | 'loading' | 'ready' | 'empty' | 'error';

export interface BeatAnalysisViewState {
  presetName: string;
  status: BeatAnalysisLoadStatus;
  errorMessage?: string;
  beats: readonly Beat[];
  analyses: readonly BeatAnalysis[];
  summary: BeatAnalysisSummary | null;
  /** 当前选中/播放的节拍序号；-1 表示无。 */
  activeBeatIndex: number;
  /** 当前播放时刻（秒）。 */
  currentTime: number;
  /** 视频总时长（秒）。 */
  duration: number;
  /** 是否正在播放。 */
  isPlaying: boolean;
  /** 是否正在重新生成当前节拍动画。 */
  regenerating: boolean;
  /** 资源（视频）路径，可空（无视频时只用姿态预览画布）。 */
  videoSrc?: string | null;
}

export interface BeatAnalysisViewCallbacks {
  onBack: () => void;
  onTogglePlayback: () => void;
  onStepFrame: (direction: 1 | -1) => void;
  onSeekToBeat: (beatIndex: number) => void;
  onPrevBeat: () => void;
  onNextBeat: () => void;
  onRegenerateBeat: (beatIndex: number) => void;
  onApplyFix: (beatIndex: number) => void;
}

const STATUS_LABEL_CN: Record<'normal' | 'warning' | 'error', string> = {
  normal: '正常',
  warning: '轻微问题',
  error: '严重问题',
};

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function metricText(v: number | null | undefined, suffix = ''): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${v}${suffix}`;
}

function renderHeader(state: BeatAnalysisViewState): string {
  const sum = state.summary;
  return `
    <header class="beat-view-header">
      <button id="beatViewBack" class="ghost" type="button" aria-label="返回">←</button>
      <div class="beat-view-heading">
        <strong>${escapeHtml(state.presetName)} · 节拍分析</strong>
        <span class="beat-view-sub">
          ${sum ? `共 ${sum.total} 拍 · ` : ''}
          ${sum ? `<i class="dot is-normal"></i>${sum.normal}` : ''}
          ${sum ? ` <i class="dot is-warning"></i>${sum.warning}` : ''}
          ${sum ? ` <i class="dot is-error"></i>${sum.error}` : ''}
        </span>
      </div>
    </header>
  `;
}

function renderStage(state: BeatAnalysisViewState): string {
  if (state.videoSrc) {
    return `
      <div class="beat-stage">
        <video id="beatVideo" preload="metadata" playsinline ${state.isPlaying ? 'autoplay' : ''} src="${escapeHtml(state.videoSrc)}"></video>
        <canvas id="beatPoseCanvas" class="beat-pose-canvas"></canvas>
        <div class="beat-stage-time" id="beatStageTime">${state.currentTime.toFixed(2)}s / ${state.duration.toFixed(2)}s</div>
      </div>
    `;
  }
  return `
    <div class="beat-stage beat-stage--canvas-only">
      <canvas id="beatPoseCanvas" class="beat-pose-canvas"></canvas>
      <div class="beat-stage-time" id="beatStageTime">${state.currentTime.toFixed(2)}s / ${state.duration.toFixed(2)}s</div>
    </div>
  `;
}

function renderControls(state: BeatAnalysisViewState): string {
  const playable = state.status === 'ready';
  return `
    <div class="beat-controls">
      <button id="beatStepBack" class="ghost" type="button" ${playable ? '' : 'disabled'} aria-label="上一帧">⏮帧</button>
      <button id="beatPrevBeat" class="ghost" type="button" ${playable ? '' : 'disabled'}>上一拍</button>
      <button id="beatTogglePlay" class="primary" type="button" ${playable ? '' : 'disabled'}>
        ${state.isPlaying ? '暂停' : '播放'}
      </button>
      <button id="beatNextBeat" class="ghost" type="button" ${playable ? '' : 'disabled'}>下一拍</button>
      <button id="beatStepForward" class="ghost" type="button" ${playable ? '' : 'disabled'} aria-label="下一帧">帧⏭</button>
    </div>
  `;
}

function renderDetailPanel(state: BeatAnalysisViewState): string {
  const ana = state.activeBeatIndex >= 0 ? state.analyses[state.activeBeatIndex] : undefined;
  if (!ana) {
    return `
      <aside class="beat-detail beat-detail--empty">
        <p>${state.status === 'loading' ? '正在加载节拍分析…' : '点击时间轴上的节拍以查看详细解析。'}</p>
      </aside>
    `;
  }
  const issues = ana.issues.length
    ? `<ul class="beat-detail-issues">${ana.issues
        .map(
          (it) => `
            <li class="is-${it.severity}">
              <span class="badge">${escapeHtml(STATUS_LABEL_CN[it.severity])}</span>
              <span>${escapeHtml(it.message)}</span>
            </li>`,
        )
        .join('')}</ul>`
    : '<p class="beat-detail-empty">未检出异常。</p>';

  const suggestions = ana.suggestions.length
    ? `<ul class="beat-detail-suggestions">${ana.suggestions
        .map((s) => `<li>${escapeHtml(s)}</li>`)
        .join('')}</ul>`
    : '<p class="beat-detail-empty">无修复建议。</p>';

  return `
    <aside class="beat-detail">
      <header class="beat-detail-header is-${ana.status}">
        <strong>第 ${ana.beatIndex + 1} 拍 · ${escapeHtml(ana.actionName)}</strong>
        <span class="badge">${escapeHtml(STATUS_LABEL_CN[ana.status])}</span>
      </header>

      <dl class="beat-detail-grid">
        <div><dt>节拍区间</dt><dd>${ana.startTime.toFixed(2)}s – ${ana.endTime.toFixed(2)}s</dd></div>
        <div><dt>中心时刻</dt><dd>${ana.timestamp.toFixed(2)}s</dd></div>
        <div><dt>动作偏差</dt><dd>${metricText(ana.poseDeviation, ' 分')}</dd></div>
        <div><dt>节奏匹配</dt><dd>${metricText(ana.rhythmMatch)}</dd></div>
        <div><dt>动画流畅度</dt><dd>${metricText(ana.smoothness)}</dd></div>
        <div><dt>主导方向</dt><dd>${escapeHtml(directionLabelCN(ana.primaryDirection))}</dd></div>
      </dl>

      <section class="beat-detail-section">
        <h4>标准动作</h4>
        <p>${escapeHtml(ana.standardDescription)}</p>
      </section>

      <section class="beat-detail-section">
        <h4>用户表现</h4>
        <p>${escapeHtml(ana.userPerformance)}</p>
      </section>

      <section class="beat-detail-section">
        <h4>问题</h4>
        ${issues}
      </section>

      <section class="beat-detail-section">
        <h4>修复建议</h4>
        ${suggestions}
      </section>

      <footer class="beat-detail-actions">
        <button id="beatRegenerate" class="primary" type="button" ${state.regenerating ? 'disabled' : ''}>
          ${state.regenerating ? '正在重新生成…' : '重新生成该节拍动画'}
        </button>
        <button id="beatApplyFix" class="ghost" type="button">应用修复建议</button>
      </footer>
    </aside>
  `;
}

export function renderBeatAnalysisView(state: BeatAnalysisViewState): string {
  if (state.status === 'loading') {
    return `
      <section class="beat-view" data-view="analysis">
        ${renderHeader(state)}
        <div class="beat-view-loading">正在分析节拍，请稍候…</div>
      </section>
    `;
  }
  if (state.status === 'error') {
    return `
      <section class="beat-view" data-view="analysis">
        ${renderHeader(state)}
        <div class="beat-view-error">
          <p>节拍分析失败：${escapeHtml(state.errorMessage ?? '未知错误')}</p>
          <button id="beatViewBack" class="ghost" type="button">返回</button>
        </div>
      </section>
    `;
  }
  if (state.status === 'empty' || !state.beats.length) {
    return `
      <section class="beat-view" data-view="analysis">
        ${renderHeader(state)}
        <div class="beat-view-empty">
          <p>未能从该动作中提取节拍。请尝试另一个标准动作或重跑动作分析。</p>
        </div>
      </section>
    `;
  }
  return `
    <section class="beat-view" data-view="analysis">
      ${renderHeader(state)}
      <div class="beat-view-body">
        <div class="beat-view-main">
          ${renderStage(state)}
          ${renderControls(state)}
          <div id="beatTimelineHost">
            ${renderBeatTimeline({
              beats: state.beats,
              analyses: state.analyses,
              activeBeatIndex: state.activeBeatIndex,
              currentTime: state.currentTime,
              duration: state.duration,
            })}
          </div>
          <div id="beatListHost" class="beat-list-host">
            ${renderBeatList({
              beats: state.beats,
              analyses: state.analyses,
              activeBeatIndex: state.activeBeatIndex,
              currentTime: state.currentTime,
              duration: state.duration,
            })}
          </div>
        </div>
        ${renderDetailPanel(state)}
      </div>
    </section>
  `;
}

export function bindBeatAnalysisView(
  root: HTMLElement,
  state: BeatAnalysisViewState,
  callbacks: BeatAnalysisViewCallbacks,
): () => void {
  const onClick = (event: Event) => {
    const target = event.target as HTMLElement | null;
    if (!target) return;
    if (target.closest('#beatViewBack')) callbacks.onBack();
    else if (target.closest('#beatTogglePlay')) callbacks.onTogglePlayback();
    else if (target.closest('#beatStepBack')) callbacks.onStepFrame(-1);
    else if (target.closest('#beatStepForward')) callbacks.onStepFrame(1);
    else if (target.closest('#beatPrevBeat')) callbacks.onPrevBeat();
    else if (target.closest('#beatNextBeat')) callbacks.onNextBeat();
    else if (target.closest('#beatRegenerate') && state.activeBeatIndex >= 0) {
      callbacks.onRegenerateBeat(state.activeBeatIndex);
    } else if (target.closest('#beatApplyFix') && state.activeBeatIndex >= 0) {
      callbacks.onApplyFix(state.activeBeatIndex);
    }
  };
  root.addEventListener('click', onClick);

  const timelineHost = root.querySelector<HTMLElement>('#beatTimelineHost');
  const unbindTimeline = timelineHost
    ? bindBeatTimeline(timelineHost, { onSeekToBeat: callbacks.onSeekToBeat })
    : () => {};

  const listHost = root.querySelector<HTMLElement>('#beatListHost');
  const unbindList = listHost
    ? bindBeatList(listHost, { onSeekToBeat: callbacks.onSeekToBeat })
    : () => {};

  return () => {
    root.removeEventListener('click', onClick);
    unbindTimeline();
    unbindList();
  };
}

/** 用于增量更新时间轴 / 详情而无需重渲染整页。 */
export function patchBeatAnalysisView(root: HTMLElement, state: BeatAnalysisViewState): void {
  const timelineHost = root.querySelector<HTMLElement>('#beatTimelineHost');
  if (timelineHost) {
    timelineHost.innerHTML = renderBeatTimeline({
      beats: state.beats,
      analyses: state.analyses,
      activeBeatIndex: state.activeBeatIndex,
      currentTime: state.currentTime,
      duration: state.duration,
    });
  }
  const listHost = root.querySelector<HTMLElement>('#beatListHost');
  if (listHost) {
    listHost.innerHTML = renderBeatList({
      beats: state.beats,
      analyses: state.analyses,
      activeBeatIndex: state.activeBeatIndex,
      currentTime: state.currentTime,
      duration: state.duration,
    });
  }
  const time = root.querySelector<HTMLElement>('#beatStageTime');
  if (time) time.textContent = `${state.currentTime.toFixed(2)}s / ${state.duration.toFixed(2)}s`;
  const playBtn = root.querySelector<HTMLElement>('#beatTogglePlay');
  if (playBtn) playBtn.textContent = state.isPlaying ? '暂停' : '播放';
  // 详情面板切换：直接重渲染右栏
  const old = root.querySelector<HTMLElement>('.beat-detail');
  if (old) {
    const wrap = document.createElement('div');
    wrap.innerHTML = renderDetailPanel(state).trim();
    const next = wrap.firstElementChild;
    if (next) old.replaceWith(next);
  }
}
