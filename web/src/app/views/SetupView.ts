/**
 * SetupView — Phase 2 first slice.
 *
 * The view is split into pure render functions that return HTML strings (so
 * they can be unit tested without a DOM) plus a thin event-binding helper that
 * wires DOM listeners to a callback bag. The legacy implementation in
 * pose_viewer.html keeps working in parallel.
 */

import type { PosePreset } from '../data/presets.js';

export interface SetupViewState {
  presets: readonly PosePreset[];
  /** id of the currently selected preset, null when none chosen yet. */
  selectedPresetId: string | null;
  /** id of the preset currently being fetched, null when idle. */
  loadingPresetId: string | null;
  /** Whether the user already granted camera access. */
  cameraRunning: boolean;
  /** True while a preset JSON has loaded successfully. */
  poseDataReady: boolean;
  /** True while motion breakdown data has loaded successfully. */
  motionReady: boolean;
  /** True while motion breakdown data is being generated. */
  motionLoading: boolean;
  /** Motion breakdown generation error, null when none. */
  motionError: string | null;
}

export interface SetupViewCallbacks {
  onPresetClick: (preset: PosePreset) => void;
  onEnterPractice: () => void;
  onEnterMotionBreakdown: () => void;
}

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function renderPresetGrid(state: SetupViewState): string {
  const cards = state.presets.map((preset) => {
    const classes = [
      'preset-card',
      state.selectedPresetId === preset.id ? 'is-selected' : '',
      state.loadingPresetId === preset.id ? 'is-loading' : '',
    ].filter(Boolean).join(' ');
    return `
      <button type="button"
              class="${classes}"
              data-preset-id="${escapeHtml(preset.id)}"
              role="listitem"
              aria-pressed="${state.selectedPresetId === preset.id}">
        <span class="preset-emoji" aria-hidden="true">${escapeHtml(preset.emoji)}</span>
        <strong>${escapeHtml(preset.name)}</strong>
        <small>${escapeHtml(preset.tagline)}</small>
        ${preset.featured ? '<span class="preset-badge">推荐</span>' : ''}
      </button>
    `;
  }).join('');
  return `<div id="presetGrid" class="preset-grid" role="list" aria-label="示例动作选择">${cards}</div>`;
}

export function ctaLabel(state: SetupViewState): string {
  if (!state.poseDataReady) return '先选择一个标准动作';
  return state.cameraRunning ? '开始练习' : '开始练习（自动开启摄像头）';
}

export function renderCTA(state: SetupViewState): string {
  const disabled = state.poseDataReady ? '' : 'disabled';
  const ready = state.poseDataReady;
  const titleText = state.cameraRunning ? '动作与摄像头已就绪' : '动作已就绪';
  const copyText = state.cameraRunning
    ? '点击开始练习会直接进入全屏舞台并自动播放。'
    : '点击开始练习会尝试打开摄像头；允许权限后自动进入练习。';
  const motionButton = ready && state.motionReady
    ? '<button id="enterMotionBreakdown" class="ghost" type="button">先看动作分解</button>'
    : '';
  const motionStatus = state.motionLoading
    ? '<span class="setup-cta-copy">正在生成动作分解…</span>'
    : '';
  const motionError = state.motionError
    ? `<span class="setup-cta-copy">${escapeHtml(state.motionError)}</span>`
    : '';
  return `
    <div class="setup-cta">
      <button id="enterPractice" class="primary" type="button" ${disabled}>${escapeHtml(ctaLabel(state))}</button>
      ${motionButton}
      ${ready ? `<span class="setup-cta-title">${escapeHtml(titleText)}</span>` : ''}
      ${ready ? `<span class="setup-cta-copy">${escapeHtml(copyText)}</span>` : ''}
      ${motionStatus}
      ${motionError}
    </div>
  `;
}

export function renderSetupView(state: SetupViewState): string {
  return `
    <section class="setup-view" data-view="setup">
      <article class="app-card">
        <h2 class="panel-title"><span class="step-badge">1</span>选择标准动作</h2>
        ${renderPresetGrid(state)}
      </article>
      ${renderCTA(state)}
    </section>
  `;
}

/**
 * Wire DOM events on a previously rendered SetupView. Returns a disposer so
 * callers can clean up before re-rendering.
 */
export function bindSetupEvents(root: HTMLElement, state: SetupViewState, callbacks: SetupViewCallbacks): () => void {
  const presetMap = new Map(state.presets.map((p) => [p.id, p] as const));
  const onClick = (event: Event) => {
    const target = event.target as HTMLElement | null;
    const card = target?.closest<HTMLButtonElement>('.preset-card');
    if (card && card.dataset.presetId) {
      const preset = presetMap.get(card.dataset.presetId);
      if (preset) callbacks.onPresetClick(preset);
      return;
    }
    if (target?.closest('#enterMotionBreakdown')) {
      callbacks.onEnterMotionBreakdown();
      return;
    }
    if (target?.closest('#enterPractice')) {
      callbacks.onEnterPractice();
    }
  };
  root.addEventListener('click', onClick);
  return () => root.removeEventListener('click', onClick);
}
