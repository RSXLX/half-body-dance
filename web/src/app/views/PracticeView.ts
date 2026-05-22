/**
 * PracticeView — Phase 2 third slice.
 *
 * The view intentionally ships as a "shell": renders the stage, HUD, sheet,
 * and floating controls. The actual render-loop integration (camera feed +
 * MediaPipe scoring) lives in main.ts so we can swap implementations behind
 * these stable DOM anchors.
 */

export interface PracticeViewState {
  poseDataReady: boolean;
  cameraRunning: boolean;
  isPlaying: boolean;
  presetName: string;
  /** Score label (e.g. "82%"). */
  scoreText: string;
  /** Left/right arm summary. */
  armSummaryText: string;
  /** Stage status headline. */
  stageLabel: string;
  stageSubLabel: string;
  /** Sheet expanded flag — callers own persistence. */
  sheetExpanded: boolean;
  recordingState: 'idle' | 'recording' | 'finalizing';
  recordingAvailable: boolean;
  debugPalette: {
    clothingCss: string;
    teacherCss: string;
    performanceText?: string;
  } | null;
}

export interface PracticeViewCallbacks {
  onTogglePlayback: () => void;
  onBackToSetup: () => void;
  onSheetToggle: () => void;
  onReplay: () => void;
  onCameraAction: () => void;
  onOpenResult: () => void;
  onOpenBeatAnalysis?: () => void;
  onToggleRecording?: () => void;
  onOpenRecording?: () => void;
}

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function togglePlaybackLabel(state: PracticeViewState): string {
  if (!state.poseDataReady) return '未加载标准动作';
  if (state.isPlaying) return '暂停';
  return '开始播放';
}

export function renderStage(state: PracticeViewState): string {
  const debugPalette = state.debugPalette
    ? `
        <div class="teacher-debug-palette" aria-label="教师颜色调试">
          <span class="palette-chip"><i style="background:${escapeHtml(state.debugPalette.clothingCss)}"></i>衣着</span>
          <span class="palette-chip"><i style="background:${escapeHtml(state.debugPalette.teacherCss)}"></i>教师</span>
          ${state.debugPalette.performanceText ? `<span class="palette-perf">${escapeHtml(state.debugPalette.performanceText)}</span>` : ''}
        </div>
      `
    : '';
  return `
    <div class="practice-stage-shell">
      <div class="stage">
        <div id="stageMedia" class="stage-media">
          <video id="cameraVideo" playsinline autoplay muted></video>
          <canvas id="teacherWebglCanvas" class="teacher-webgl-canvas" width="720" height="1280" aria-hidden="true"></canvas>
          <canvas id="overlayCanvas" width="720" height="1280"></canvas>
        </div>
        <audio id="bgmAudio" preload="metadata"></audio>

        <div id="stageReticle" class="stage-reticle" hidden aria-hidden="true">
          <div class="stage-reticle-frame"><i></i></div>
          <div class="stage-reticle-copy" id="stageReticleCopy">请站到画面中央</div>
        </div>

        <div id="stageStatusBanner" class="stage-status" data-kind="loading" hidden role="status" aria-live="polite">
          <span class="stage-status-dot"></span>
          <div class="stage-status-body">
            <strong class="stage-status-title" id="stageStatusTitle">正在准备</strong>
            <span class="stage-status-hint" id="stageStatusHint">加载识别模型与摄像头</span>
          </div>
          <div class="stage-status-progress" aria-hidden="true"></div>
        </div>

        <div class="practice-topbar">
          <button id="backToSetup" class="nav-icon ghost" type="button" aria-label="返回准备页">←</button>
          <div class="practice-heading">
            <strong id="practiceHeaderTitle">${escapeHtml(state.presetName || '练习中')}</strong>
          </div>
        </div>

        <div class="stage-score-hud">
          <div class="score-dial" id="scoreDial">
            <svg viewBox="0 0 72 72" aria-hidden="true">
              <circle cx="36" cy="36" r="30" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="7"/>
              <circle id="matchRing" cx="36" cy="36" r="30" fill="none" stroke="#ffbe3b" stroke-width="7" stroke-dasharray="188.5" stroke-dashoffset="188.5"/>
            </svg>
            <div class="score-dial-value">
              <strong id="scoreDialValue">${escapeHtml(state.scoreText)}</strong>
              <span>手臂分</span>
            </div>
          </div>
        </div>
        ${debugPalette}
      </div>
    </div>
  `;
}

export function renderDock(state: PracticeViewState): string {
  const expanded = state.sheetExpanded ? 'true' : 'false';
  const recordLabel =
    state.recordingState === 'recording'
      ? '停止录屏'
      : state.recordingState === 'finalizing'
        ? '生成中'
        : '开始录屏';
  return `
    <div class="practice-dock">
      <div class="sheet-shell" data-expanded="${expanded}">
        <button id="practiceSheetToggle" class="sheet-handle" type="button" aria-expanded="${expanded}">
          <span class="sheet-handle-bar"></span>
          <span class="sheet-handle-copy">${state.sheetExpanded ? '收起详情' : '查看详情'}</span>
        </button>

        <div class="sheet-peek">
          <div class="practice-score-peek">
            <span class="dock-kicker">当前总分</span>
            <strong id="practiceScoreLabel">${escapeHtml(state.scoreText)}</strong>
            <span id="practiceArmSummary">${escapeHtml(state.armSummaryText)}</span>
          </div>
          <div class="sheet-mini-actions">
            <button id="practiceCameraActionMini" class="ghost compact" type="button">${state.cameraRunning ? '关摄像头' : '开摄像头'}</button>
            <button id="openResultMini" class="ghost compact" type="button">结果</button>
          </div>
        </div>

        <div class="sheet-expand-content" ${state.sheetExpanded ? '' : 'hidden'}>
          <div class="dock-panel">
            <span class="dock-kicker">状态</span>
            <strong id="stageLabel">${escapeHtml(state.stageLabel)}</strong>
            <span id="stageSubLabel">${escapeHtml(state.stageSubLabel)}</span>
          </div>
          <div class="dock-panel practice-actions">
            <button id="practiceReplay" class="ghost" type="button" ${state.poseDataReady ? '' : 'disabled'}>重新播放</button>
            <button id="practiceCameraAction" class="ghost" type="button">${state.cameraRunning ? '关闭摄像头' : '打开摄像头'}</button>
            <button id="openResult" class="ghost" type="button">查看结果</button>
            <button id="openBeatAnalysis" class="ghost" type="button" ${state.poseDataReady ? '' : 'disabled'}>节拍分析</button>
            <button id="togglePracticeRecording" class="ghost" type="button" ${state.poseDataReady && state.recordingState !== 'finalizing' ? '' : 'disabled'}>${recordLabel}</button>
            <button id="openPracticeRecording" class="ghost" type="button" ${state.recordingAvailable ? '' : 'disabled'}>打开录屏</button>
          </div>
        </div>
      </div>

      <div class="practice-floating-controls">
        <button id="togglePlayback" class="playback-glass" type="button" aria-pressed="${state.isPlaying}" ${state.poseDataReady ? '' : 'disabled'}>${escapeHtml(togglePlaybackLabel(state))}</button>
      </div>
    </div>
  `;
}

export function renderPracticeView(state: PracticeViewState): string {
  return `
    <section class="practice-view is-active" data-view="practice">
      ${renderStage(state)}
      ${renderDock(state)}
    </section>
  `;
}

export function bindPracticeEvents(
  root: HTMLElement,
  callbacks: PracticeViewCallbacks,
): () => void {
  const onClick = (event: Event) => {
    const target = event.target as HTMLElement | null;
    if (!target) return;
    if (target.closest('#togglePlayback')) callbacks.onTogglePlayback();
    else if (target.closest('#backToSetup')) callbacks.onBackToSetup();
    else if (target.closest('#practiceSheetToggle')) callbacks.onSheetToggle();
    else if (target.closest('#practiceReplay')) callbacks.onReplay();
    else if (target.closest('#practiceCameraAction') || target.closest('#practiceCameraActionMini')) callbacks.onCameraAction();
    else if (target.closest('#openResult') || target.closest('#openResultMini')) callbacks.onOpenResult();
    else if (target.closest('#openBeatAnalysis')) callbacks.onOpenBeatAnalysis?.();
    else if (target.closest('#togglePracticeRecording')) callbacks.onToggleRecording?.();
    else if (target.closest('#openPracticeRecording')) callbacks.onOpenRecording?.();
  };
  root.addEventListener('click', onClick);
  return () => root.removeEventListener('click', onClick);
}
