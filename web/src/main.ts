/**
 * Phase 2 entry — Setup / Practice / Result router with live scoring.
 *
 * The practice screen now drives a real render loop via practiceLoop +
 * poseDetector + EMA smoothing. Canvas drawing of the teacher skeleton stays
 * in pose_viewer.html for now; the new shell focuses on the score read-out.
 * See docs/optimization-mediapipe-scoring.md.
 */
import './styles/setup.css';
import './styles/result.css';
import './styles/practice.css';
import './styles/beat-analysis.css';
import './styles/motion-breakdown.css';

import { BUNDLED_POSE_PRESETS, findPresetById, motionPathForPreset, type PosePreset } from './app/data/presets.js';
import { buildDeepLink, parseDeepLink } from './app/deepLink.js';
import { createCameraController, describeCameraError } from './app/cameraController.js';
import {
  bindSetupEvents,
  renderSetupView,
  type SetupViewState,
} from './app/views/SetupView.js';
import {
  bindPracticeEvents,
  renderPracticeView,
  type PracticeViewState,
} from './app/views/PracticeView.js';
import {
  bindResultEvents,
  renderResultView,
  type ResultViewState,
} from './app/views/ResultView.js';
import {
  renderBeatAnalysisView,
  bindBeatAnalysisView,
  patchBeatAnalysisView,
  type BeatAnalysisViewState,
} from './app/views/BeatAnalysisView.js';
import {
  bindMotionBreakdownView,
  renderMotionBreakdownView,
  type MotionBreakdownViewState,
} from './app/views/MotionBreakdownView.js';
import { detectBeats, type Beat } from './core/beats.js';
import {
  analyzeBeats,
  summarizeBeatAnalyses,
  type BeatAnalysis,
  type BeatAnalysisSummary,
} from './core/beatAnalysis.js';
import type { MotionAnalysis, MotionSegment } from './core/motionTypes.js';
import { getSegmentStartTime, loadMotionFromPath } from './core/motionData.js';
import {
  createNoopPoseDetector,
  createTasksPoseDetector,
  type PoseDetector,
} from './app/poseDetector.js';
import { startPracticeLoop, type PracticeLoopHandle } from './app/practiceLoop.js';
import { formatSmoothedScore } from './app/scoreSmoothing.js';
import {
  DEFAULT_TEACHER_PALETTE,
  blendRgbColor,
  clearStage,
  computeStageStatus,
  deriveTeacherPaletteFromClothing,
  syncCanvasSize,
  type BodyPalette,
  type RgbColor,
  type StageStatusOutput,
} from './app/canvasPainter.js';
import { drawTeacherAvatar } from './app/teacherAvatar/canvasTeacherAvatar.js';
import { findNearestRenderableFrame } from './app/teacherAvatar/poseParts.js';
import type { ThreeTeacherAvatarRenderer } from './app/teacherAvatar/threeTeacherAvatar.js';
import {
  getStageContentRect,
  resolveStageAspectRatio,
} from './app/stageRenderer.js';
import { getVisibility } from './core/geometry.js';
import { getDefaultStageReference, getPoseReference, transformPointsWithReferences } from './core/reference.js';
import type { PoseData, PoseFrame, NormalizedLandmark } from './core/types.js';
import { ARM_SCORE_SYSTEM_LABEL } from './core/poseCompare.js';

type View = 'setup' | 'practice' | 'result' | 'analysis' | 'motion';

interface BeatAnalysisRuntime {
  status: BeatAnalysisViewState['status'];
  errorMessage?: string;
  beats: Beat[];
  analyses: BeatAnalysis[];
  summary: BeatAnalysisSummary | null;
  motion: MotionAnalysis | null;
  activeBeatIndex: number;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  regenerating: boolean;
}

interface AppState {
  view: View;
  selectedPresetId: string | null;
  loadingPresetId: string | null;
  poseData: PoseData | null;
  poseDataReady: boolean;
  cameraRunning: boolean;
  cameraError: string | null;
  poseReady: boolean;
  motion: MotionAnalysis | null;
  motionStatus: 'idle' | 'loading' | 'ready' | 'empty' | 'error';
  motionError: string | null;
  activeMotionSegmentId: string | null;
  practiceStartTime: number;
  practiceIsPlaying: boolean;
  practiceSheetExpanded: boolean;
  recordingState: 'idle' | 'recording' | 'finalizing';
  recordingAvailable: boolean;
  /** Index of the current target frame, advanced by the playback clock. */
  currentFrameIndex: number;
  playbackStartedAt: number | null;
  /** Last MediaPipe visibility (>= 0.4 ⇒ in-frame). */
  personVisible: boolean;
  beat: BeatAnalysisRuntime;
}

const intent = parseDeepLink(window.location.search, findPresetById);

const state: AppState = {
  view: intent.view ?? 'setup',
  selectedPresetId: intent.preset?.id ?? null,
  loadingPresetId: null,
  poseData: null,
  poseDataReady: false,
  cameraRunning: false,
  cameraError: null,
  poseReady: false,
  motion: null,
  motionStatus: 'idle',
  motionError: null,
  activeMotionSegmentId: null,
  practiceStartTime: 0,
  practiceIsPlaying: false,
  practiceSheetExpanded: false,
  recordingState: 'idle',
  recordingAvailable: false,
  currentFrameIndex: 0,
  playbackStartedAt: null,
  personVisible: false,
  beat: {
    status: 'idle',
    beats: [],
    analyses: [],
    summary: null,
    motion: null,
    activeBeatIndex: -1,
    currentTime: 0,
    duration: 0,
    isPlaying: false,
    regenerating: false,
  },
};

const camera = createCameraController();

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) throw new Error('#app container missing');

let unbind: (() => void) | null = null;
let practiceLoop: PracticeLoopHandle | null = null;
let detector: PoseDetector | null = null;
let detectorPromise: Promise<PoseDetector> | null = null;
let teacherPalette: BodyPalette = DEFAULT_TEACHER_PALETTE;
let clothingColor: RgbColor | null = null;
let clothingSampleCanvas: HTMLCanvasElement | null = null;
let lastClothingSampleAt = 0;
const CLOTHING_SAMPLE_INTERVAL_MS = 650;
const MIRROR_X = true;
const debugEnabled = new URLSearchParams(window.location.search).get('debug') === '1';
let practiceEntryToken = 0;

export function createCameraStartGate() {
  let latestToken = 0;
  return {
    begin() {
      latestToken += 1;
      return latestToken;
    },
    isLatest(token: number) {
      return token === latestToken;
    },
  };
}

const cameraStartGate = createCameraStartGate();
let teacherFrameAvgMs: number | null = null;
let lastTeacherFrameAt = 0;
let teacherHighDetail = true;
let teacher3dRenderer: ThreeTeacherAvatarRenderer | null = null;
let teacher3dRendererPromise: Promise<ThreeTeacherAvatarRenderer | null> | null = null;
let teacher3dRendererPromiseCanvas: HTMLCanvasElement | null = null;
let teacher3dRendererRequestId = 0;
let teacher3dUnavailable = false;

function getTeacherDisplayReference(): ReturnType<typeof getDefaultStageReference> {
  const reference = getDefaultStageReference();
  return {
    ...reference,
    center: { x: 0.5, y: 0.44 },
    scale: 0.25,
  };
}

interface PracticeRecordingRuntime {
  recorder: MediaRecorder | null;
  chunks: Blob[];
  stream: MediaStream | null;
  canvas: HTMLCanvasElement | null;
  ctx: CanvasRenderingContext2D | null;
  rafId: number;
  blobUrl: string;
  mimeType: string;
}

const practiceRecording: PracticeRecordingRuntime = {
  recorder: null,
  chunks: [],
  stream: null,
  canvas: null,
  ctx: null,
  rafId: 0,
  blobUrl: '',
  mimeType: '',
};

function ensureDetector(): Promise<PoseDetector> {
  if (detector) return Promise.resolve(detector);
  if (detectorPromise) return detectorPromise;
  detectorPromise = createTasksPoseDetector()
    .then((d) => {
      detector = d;
      state.poseReady = true;
      applyStageStatusDom();
      return d;
    })
    .catch((err) => {
      console.warn('[main] pose detector failed to init, using noop', err);
      detector = createNoopPoseDetector();
      state.poseReady = true;
      applyStageStatusDom();
      return detector;
    });
  return detectorPromise;
}

function setupViewState(): SetupViewState {
  return {
    presets: BUNDLED_POSE_PRESETS,
    selectedPresetId: state.selectedPresetId,
    loadingPresetId: state.loadingPresetId,
    cameraRunning: state.cameraRunning,
    poseDataReady: state.poseDataReady,
    motionReady: state.motionStatus === 'ready',
    motionLoading: state.motionStatus === 'loading',
    motionError: state.motionError,
  };
}

function selectedPresetName(): string {
  if (!state.selectedPresetId) return '';
  return findPresetById(state.selectedPresetId)?.name ?? '';
}

function practiceViewState(scoreText = '--', armSummary = '左臂 -- / 右臂 --', stageLabel?: string): PracticeViewState {
  return {
    poseDataReady: state.poseDataReady,
    cameraRunning: state.cameraRunning,
    isPlaying: state.practiceIsPlaying,
    presetName: selectedPresetName() || '练习中',
    scoreText,
    armSummaryText: armSummary,
    stageLabel: stageLabel ?? (state.cameraRunning ? '准备开始' : '摄像头未开启'),
    stageSubLabel: state.poseDataReady
      ? '点击底部"开始播放"按钮启动跟练。'
      : '先选择标准动作再进入练习。',
    sheetExpanded: state.practiceSheetExpanded,
    recordingState: state.recordingState,
    recordingAvailable: state.recordingAvailable,
    debugPalette: debugEnabled
      ? {
          clothingCss: clothingColor ? rgbDebugCss(clothingColor) : 'rgba(255, 255, 255, 0.18)',
          teacherCss: teacherPalette.stroke,
          performanceText: teacherFrameAvgMs
            ? `${teacherHighDetail ? '高细节' : '降级'} ${Math.round(1000 / Math.max(1, teacherFrameAvgMs))}fps`
            : undefined,
        }
      : null,
  };
}

function resultViewState(): ResultViewState {
  return {
    status: 'pending',
    finalAverageText: '--',
    finalConclusionText: '待结算',
    leftArmText: '--',
    rightArmText: '--',
    beatHitRateText: '--',
    beatCompositeText: '--',
    beatStatsText: '0 / 0',
    matchLabel: ARM_SCORE_SYSTEM_LABEL,
  };
}

function motionBreakdownViewState(): MotionBreakdownViewState {
  return {
    presetName: selectedPresetName() || '当前动作',
    status: state.motionStatus,
    errorMessage: state.motionError,
    motion: state.motion,
    activeSegmentId: state.activeMotionSegmentId,
  };
}

function mount() {
  if (unbind) unbind();
  if (state.view === 'setup') {
    root!.innerHTML = renderSetupView(setupViewState());
    unbind = bindSetupEvents(root!, setupViewState(), {
      onPresetClick: handlePresetClick,
      onEnterPractice: handleEnterPractice,
      onEnterMotionBreakdown: () => navigate('motion'),
    });
    stopPracticeLoop();
  } else if (state.view === 'motion') {
    root!.innerHTML = renderMotionBreakdownView(motionBreakdownViewState());
    unbind = bindMotionBreakdownView(root!, motionBreakdownViewState(), {
      onBack: () => navigate('setup'),
      onPractice: handleEnterPractice,
      onSelectSegment: handleMotionSegmentSelect,
    });
    stopPracticeLoop();
  } else if (state.view === 'practice') {
    stopPracticeLoop();
    root!.innerHTML = renderPracticeView(practiceViewState());
    unbind = bindPracticeEvents(root!, {
      onTogglePlayback: handleTogglePlayback,
      onBackToSetup: () => navigate('setup'),
      onSheetToggle: () => {
        state.practiceSheetExpanded = !state.practiceSheetExpanded;
        mount();
      },
      onReplay: handleReplay,
      onCameraAction: handleCameraAction,
      onOpenResult: () => navigate('result'),
      onOpenBeatAnalysis: () => navigate('analysis'),
      onToggleRecording: togglePracticeRecording,
      onOpenRecording: openLatestPracticeRecording,
    });
    void wirePracticeRuntime();
  } else if (state.view === 'result') {
    root!.innerHTML = renderResultView(resultViewState());
    unbind = bindResultEvents(root!, {
      onReplay: handleReplay,
      onChangeAction: () => navigate('setup'),
      onBackToPractice: () => navigate('practice'),
      onOpenBeatAnalysis: () => navigate('analysis'),
    });
    stopPracticeLoop();
  } else {
    root!.innerHTML = renderBeatAnalysisView(beatAnalysisViewState());
    unbind = bindBeatAnalysisView(root!, beatAnalysisViewState(), {
      onBack: () => navigate('result'),
      onTogglePlayback: handleBeatTogglePlayback,
      onStepFrame: handleBeatStepFrame,
      onSeekToBeat: handleSeekToBeat,
      onPrevBeat: () => handleStepBeat(-1),
      onNextBeat: () => handleStepBeat(1),
      onRegenerateBeat: handleRegenerateBeat,
      onApplyFix: handleApplyFix,
    });
    stopPracticeLoop();
    void ensureBeatAnalysisLoaded();
  }
  syncUrl();
}

function navigate(view: View) {
  state.view = view;
  if (view === 'setup') {
    state.practiceStartTime = 0;
    state.activeMotionSegmentId = null;
    state.currentFrameIndex = 0;
  }
  if (view !== 'practice') {
    state.practiceIsPlaying = false;
    state.playbackStartedAt = null;
  }
  mount();
}

function syncUrl() {
  const url = new URL(window.location.href);
  url.searchParams.delete('view');
  if (state.view !== 'setup') url.searchParams.set('view', state.view);
  if (state.selectedPresetId) url.searchParams.set('preset', state.selectedPresetId);
  history.replaceState({}, '', url.toString());
}

async function handlePresetClick(preset: PosePreset, options: { autoplay?: boolean } = {}) {
  state.selectedPresetId = preset.id;
  state.loadingPresetId = preset.id;
  state.poseDataReady = false;
  state.poseData = null;
  state.motion = null;
  state.motionStatus = 'loading';
  state.motionError = null;
  state.activeMotionSegmentId = null;
  state.practiceStartTime = 0;
  state.currentFrameIndex = 0;
  mount();

  const posePromise = fetch(preset.path).then(async (res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as PoseData;
  });
  const motionPromise = loadMotionFromPath(motionPathForPreset(preset));

  try {
    const [poseResult, motionResult] = await Promise.allSettled([posePromise, motionPromise]);
    if (state.selectedPresetId !== preset.id) return;

    if (poseResult.status === 'fulfilled') {
      state.poseData = poseResult.value;
      state.poseDataReady = Array.isArray(poseResult.value.frames) && poseResult.value.frames.length > 0;
    } else {
      console.warn('[main] preset load failed', poseResult.reason);
      state.poseData = null;
      state.poseDataReady = false;
    }

    if (motionResult.status === 'fulfilled') {
      state.motion = motionResult.value;
      state.motionStatus = motionResult.value ? 'ready' : 'empty';
      state.motionError = null;
    } else {
      console.warn('[main] motion load failed', motionResult.reason);
      state.motion = null;
      state.motionStatus = 'error';
      state.motionError = '动作分解暂不可用，可继续普通练习。';
    }
  } finally {
    if (state.selectedPresetId !== preset.id) return;
    state.loadingPresetId = null;
    if (options.autoplay && state.poseDataReady) {
      await handleEnterPractice();
    } else {
      mount();
    }
  }
}

async function handleEnterPractice() {
  if (!state.poseDataReady || !state.selectedPresetId) return;
  const entryToken = ++practiceEntryToken;
  const entryPresetId = state.selectedPresetId;
  const entryPoseData = state.poseData;
  const shouldStartCamera = !state.cameraRunning;
  const isStalePracticeEntry = () =>
    state.view !== 'practice' ||
    entryToken !== practiceEntryToken ||
    state.selectedPresetId !== entryPresetId ||
    state.poseData !== entryPoseData;

  navigate('practice');
  if (shouldStartCamera) {
    const cameraStartToken = cameraStartGate.begin();
    const isLatestCameraStart = () => cameraStartGate.isLatest(cameraStartToken);
    try {
      await camera.start();
      if (!isLatestCameraStart()) return;
      if (isStalePracticeEntry()) {
        camera.stop();
        return;
      }
      state.cameraRunning = true;
      state.cameraError = null;
    } catch (err) {
      if (!isLatestCameraStart() || isStalePracticeEntry()) return;
      state.cameraError = describeCameraError(err);
    }
    if (state.view === 'practice') mount();
  }
}

function handleTogglePlayback() {
  if (!state.poseDataReady) return;
  state.practiceIsPlaying = !state.practiceIsPlaying;
  if (state.practiceIsPlaying) {
    state.playbackStartedAt = performance.now();
    state.currentFrameIndex = frameIndexAtTime(state.practiceStartTime);
  }
  mount();
}

function handleReplay() {
  if (!state.poseDataReady) return;
  state.practiceIsPlaying = true;
  state.playbackStartedAt = performance.now();
  state.currentFrameIndex = frameIndexAtTime(state.practiceStartTime);
  navigate('practice');
}

function handleMotionSegmentSelect(segment: MotionSegment) {
  const startTime = getSegmentStartTime(segment);
  if (startTime === null) return;
  state.activeMotionSegmentId = segment.id;
  state.practiceStartTime = startTime;
  state.currentFrameIndex = frameIndexAtTime(startTime);
  void handleEnterPractice();
}

function frameIndexAtTime(seconds: number): number {
  const frames = state.poseData?.frames ?? [];
  if (!frames.length) return 0;
  const targetTime = Math.max(0, seconds);
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if ((frames[mid]?.time ?? 0) < targetTime) low = mid + 1;
    else high = mid;
  }
  return low;
}

async function handleCameraAction() {
  if (state.cameraRunning) {
    camera.stop();
    state.cameraRunning = false;
    state.personVisible = false;
  } else {
    try {
      await camera.start();
      state.cameraRunning = true;
      state.cameraError = null;
    } catch (err) {
      state.cameraRunning = false;
      state.cameraError = describeCameraError(err);
    }
  }
  mount();
}

function rgbDebugCss(color: RgbColor): string {
  return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
}

function getCurrentTargetFrame(): PoseFrame | null {
  const data = state.poseData;
  if (!data || !data.frames.length) return null;
  if (!state.practiceIsPlaying || state.playbackStartedAt === null) {
    return data.frames[Math.min(state.currentFrameIndex, data.frames.length - 1)] ?? null;
  }
  const elapsed = state.practiceStartTime + (performance.now() - state.playbackStartedAt) / 1000;
  // Assume the JSON is sorted by time; do an O(log n) lookup but simple linear
  // scan from the cursor is fine for typical 1k-frame clips.
  const frames = data.frames;
  const last = frames[frames.length - 1]!;
  if (elapsed >= (last.time ?? 0)) {
    state.practiceIsPlaying = false;
    return last;
  }
  let idx = state.currentFrameIndex;
  while (idx > 0 && (frames[idx]!.time ?? 0) > elapsed) {
    idx--;
  }
  while (idx < frames.length - 1 && (frames[idx + 1]!.time ?? 0) <= elapsed) {
    idx++;
  }
  state.currentFrameIndex = idx;
  return frames[idx] ?? null;
}

function midpoint(
  a: { x: number; y: number } | null | undefined,
  b: { x: number; y: number } | null | undefined,
): { x: number; y: number } | null {
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function getClothingSampleBounds(points: readonly (NormalizedLandmark | null | undefined)[] | null): {
  x: number;
  y: number;
  width: number;
  height: number;
} | null {
  if (!points) return null;
  const visiblePoint = (index: number, minVisibility = 0.18) => {
    const point = points[index];
    return point && getVisibility(point) >= minVisibility ? point : null;
  };
  const leftShoulder = visiblePoint(11);
  const rightShoulder = visiblePoint(12);
  if (!leftShoulder || !rightShoulder) return null;

  const leftHip = visiblePoint(23, 0.12);
  const rightHip = visiblePoint(24, 0.12);
  const shoulderMid = midpoint(leftShoulder, rightShoulder)!;
  const shoulderWidth = Math.max(0.12, Math.abs(leftShoulder.x - rightShoulder.x));
  const hipY = leftHip && rightHip ? (leftHip.y + rightHip.y) / 2 : shoulderMid.y + shoulderWidth * 1.45;
  const minX = Math.min(leftShoulder.x, rightShoulder.x, leftHip?.x ?? leftShoulder.x, rightHip?.x ?? rightShoulder.x);
  const maxX = Math.max(leftShoulder.x, rightShoulder.x, leftHip?.x ?? leftShoulder.x, rightHip?.x ?? rightShoulder.x);
  const minY = Math.min(leftShoulder.y, rightShoulder.y);
  const maxY = Math.max(hipY, minY + shoulderWidth * 1.2);
  const expandX = Math.max(0.035, (maxX - minX) * 0.18);
  const expandY = Math.max(0.035, (maxY - minY) * 0.12);
  return {
    x: clamp01(minX - expandX),
    y: clamp01(minY - expandY),
    width: Math.min(0.7, Math.max(0.08, (maxX - minX) + expandX * 2)),
    height: Math.min(0.7, Math.max(0.08, (maxY - minY) + expandY * 2)),
  };
}

function getDominantColorFromImageData(imageData: ImageData): RgbColor | null {
  const bins = new Map<string, { r: number; g: number; b: number; weight: number }>();
  const { data, width, height } = imageData;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const r = data[offset]!;
      const g = data[offset + 1]!;
      const b = data[offset + 2]!;
      const alpha = data[offset + 3]!;
      if (alpha < 180) continue;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const lightness = (max + min) / 510;
      const saturation = max === min ? 0 : (max - min) / 255;
      if (lightness < 0.08 || lightness > 0.94) continue;
      const distanceFromCenter = Math.hypot((x + 0.5) / width - 0.5, (y + 0.5) / height - 0.5);
      const centerWeight = 1 - Math.min(0.62, distanceFromCenter);
      const weight = centerWeight * (0.28 + saturation * 1.2 + (1 - Math.abs(lightness - 0.5)) * 0.35);
      const key = `${r >> 4}-${g >> 4}-${b >> 4}`;
      const item = bins.get(key) ?? { r: 0, g: 0, b: 0, weight: 0 };
      item.r += r * weight;
      item.g += g * weight;
      item.b += b * weight;
      item.weight += weight;
      bins.set(key, item);
    }
  }
  let best: { r: number; g: number; b: number; weight: number } | null = null;
  for (const item of bins.values()) {
    if (!best || item.weight > best.weight) best = item;
  }
  if (!best || best.weight <= 0) return null;
  return {
    r: best.r / best.weight,
    g: best.g / best.weight,
    b: best.b / best.weight,
  };
}

function sampleClothingColorFromCamera(
  video: HTMLVideoElement,
  userPose: readonly (NormalizedLandmark | null | undefined)[] | null,
): RgbColor | null {
  if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) return null;
  const bounds = getClothingSampleBounds(userPose);
  if (!bounds) return null;

  const sampleSize = 48;
  clothingSampleCanvas ??= document.createElement('canvas');
  clothingSampleCanvas.width = sampleSize;
  clothingSampleCanvas.height = sampleSize;
  const ctx = clothingSampleCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  const sx = bounds.x * video.videoWidth;
  const sy = bounds.y * video.videoHeight;
  const sw = Math.max(1, bounds.width * video.videoWidth);
  const sh = Math.max(1, bounds.height * video.videoHeight);
  try {
    ctx.clearRect(0, 0, sampleSize, sampleSize);
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, sampleSize, sampleSize);
    return getDominantColorFromImageData(ctx.getImageData(0, 0, sampleSize, sampleSize));
  } catch (err) {
    console.warn('[main] clothing color sample failed', err);
    return null;
  }
}

function refreshTeacherPalette(
  video: HTMLVideoElement,
  userPose: readonly (NormalizedLandmark | null | undefined)[] | null,
): void {
  const now = performance.now();
  if (now - lastClothingSampleAt < CLOTHING_SAMPLE_INTERVAL_MS) return;
  lastClothingSampleAt = now;
  const sampled = sampleClothingColorFromCamera(video, userPose);
  if (!sampled) return;
  clothingColor = blendRgbColor(clothingColor, sampled);
  teacherPalette = deriveTeacherPaletteFromClothing(clothingColor);
}

function ensureTeacher3dRenderer(canvas: HTMLCanvasElement | null): Promise<ThreeTeacherAvatarRenderer | null> {
  if (!canvas || teacher3dUnavailable) return Promise.resolve(null);
  if (teacher3dRenderer?.canvas === canvas && canvas.isConnected) return Promise.resolve(teacher3dRenderer);
  if (teacher3dRendererPromise && teacher3dRendererPromiseCanvas === canvas) return teacher3dRendererPromise;
  if (teacher3dRenderer && teacher3dRenderer.canvas !== canvas) {
    teacher3dRenderer.dispose();
    teacher3dRenderer = null;
  }
  const requestId = ++teacher3dRendererRequestId;
  teacher3dRendererPromiseCanvas = canvas;
  teacher3dRendererPromise = import('./app/teacherAvatar/threeTeacherAvatar.js')
    .then(({ ThreeTeacherAvatarRenderer: Renderer }) => {
      const renderer = new Renderer({ canvas });
      if (!canvas.isConnected || requestId !== teacher3dRendererRequestId) {
        renderer.dispose();
        return null;
      }
      teacher3dRenderer = renderer;
      if (teacher3dRendererPromiseCanvas === canvas) {
        teacher3dRendererPromise = null;
        teacher3dRendererPromiseCanvas = null;
      }
      canvas.hidden = false;
      canvas.dataset.renderer = 'three';
      return renderer;
    })
    .catch((err) => {
      console.warn('[main] three teacher renderer unavailable, falling back to canvas', err);
      if (requestId === teacher3dRendererRequestId) teacher3dUnavailable = true;
      canvas.hidden = true;
      delete canvas.dataset.renderer;
      if (teacher3dRenderer?.canvas === canvas) teacher3dRenderer = null;
      if (teacher3dRendererPromiseCanvas === canvas) {
        teacher3dRendererPromise = null;
        teacher3dRendererPromiseCanvas = null;
      }
      return null;
    });
  return teacher3dRendererPromise;
}

function disposeTeacher3dRenderer(): void {
  teacher3dRendererRequestId += 1;
  teacher3dRenderer?.dispose();
  teacher3dRenderer = null;
  teacher3dRendererPromise = null;
  teacher3dRendererPromiseCanvas = null;
}

function renderTeacherStage(args: {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  video: HTMLVideoElement;
  userPose: readonly (NormalizedLandmark | null | undefined)[] | null;
  targetFrame: PoseFrame | null;
  threeRenderer?: ThreeTeacherAvatarRenderer | null;
}): void {
  const { canvas, ctx, video, userPose, targetFrame, threeRenderer } = args;
  const now = performance.now();
  if (lastTeacherFrameAt) {
    const frameMs = now - lastTeacherFrameAt;
    teacherFrameAvgMs = teacherFrameAvgMs == null ? frameMs : teacherFrameAvgMs * 0.9 + frameMs * 0.1;
    if (teacherFrameAvgMs > 45) teacherHighDetail = false;
    else if (teacherFrameAvgMs < 30) teacherHighDetail = true;
  }
  lastTeacherFrameAt = now;

  syncCanvasSize(canvas);
  clearStage(ctx, canvas);
  refreshTeacherPalette(video, userPose);

  const aspect = resolveStageAspectRatio({
    videoReady: false,
    videoWidth: 0,
    videoHeight: 0,
    poseVideoWidth: state.poseData?.video_width ?? null,
    poseVideoHeight: state.poseData?.video_height ?? null,
    fallback: video.readyState >= 2 && video.videoWidth && video.videoHeight
      ? video.videoWidth / video.videoHeight
      : 9 / 16,
  });
  const rect = getStageContentRect(
    { width: canvas.width, height: canvas.height },
    aspect,
  );

  const displayTargetFrame = targetFrame?.pose_landmarks?.length
    ? targetFrame
    : findNearestRenderableFrame(state.poseData?.frames, state.currentFrameIndex);
  if (!displayTargetFrame?.pose_landmarks?.length) {
    threeRenderer?.clear();
    return;
  }

  const targetRef = getPoseReference(displayTargetFrame.pose_landmarks);
  const teacherRef = getTeacherDisplayReference();
  const projected = transformPointsWithReferences(
    displayTargetFrame.pose_landmarks,
    targetRef,
    teacherRef,
  ) as NormalizedLandmark[];
  const projectedHands = (displayTargetFrame.hands ?? []).map((hand) => ({
    ...hand,
    landmarks: transformPointsWithReferences(
      hand.landmarks ?? [],
      targetRef,
      teacherRef,
    ) as NormalizedLandmark[],
  }));

  const rendered3d = threeRenderer?.render({
    poseLandmarks: projected,
    hands: projectedHands,
    stageRect: rect,
    palette: teacherPalette,
    mirrorX: MIRROR_X,
    highDetail: teacherHighDetail,
    showHead: true,
  }) ?? false;
  if (rendered3d) return;

  drawTeacherAvatar({
    ctx,
    poseLandmarks: projected,
    hands: projectedHands,
    stageRect: rect,
    palette: teacherPalette,
    mirrorX: MIRROR_X,
    highDetail: teacherHighDetail,
    showHead: false,
  });
}

function refreshRecordingUi(): void {
  const toggle = document.querySelector<HTMLButtonElement>('#togglePracticeRecording');
  const open = document.querySelector<HTMLButtonElement>('#openPracticeRecording');
  if (toggle) {
    toggle.textContent =
      state.recordingState === 'recording'
        ? '停止录屏'
        : state.recordingState === 'finalizing'
          ? '生成中'
          : '开始录屏';
    toggle.disabled = !state.poseDataReady || state.recordingState === 'finalizing';
  }
  if (open) {
    open.disabled = !state.recordingAvailable || state.recordingState !== 'idle';
  }
}

function pickRecordingMimeType(): string {
  if (!('MediaRecorder' in window)) return '';
  return [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
    'video/mp4',
  ].find((type) => MediaRecorder.isTypeSupported(type)) ?? '';
}

function ensureRecordingCanvas(): void {
  if (practiceRecording.canvas && practiceRecording.ctx) return;
  const canvas = document.createElement('canvas');
  canvas.width = 720;
  canvas.height = 1280;
  practiceRecording.canvas = canvas;
  practiceRecording.ctx = canvas.getContext('2d');
}

function composeRecordingFrame(): void {
  if (state.recordingState !== 'recording' || !practiceRecording.canvas || !practiceRecording.ctx) return;
  const overlay = document.querySelector<HTMLCanvasElement>('#overlayCanvas');
  const teacherWebgl = document.querySelector<HTMLCanvasElement>('#teacherWebglCanvas');
  const canvas = practiceRecording.canvas;
  const ctx = practiceRecording.ctx;
  if (teacherWebgl?.width && teacherWebgl?.height) {
    canvas.width = teacherWebgl.width;
    canvas.height = teacherWebgl.height;
  } else if (overlay?.width && overlay?.height) {
    canvas.width = overlay.width;
    canvas.height = overlay.height;
  }
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  gradient.addColorStop(0, '#060b13');
  gradient.addColorStop(1, '#020409');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (teacherWebgl && !teacherWebgl.hidden) ctx.drawImage(teacherWebgl, 0, 0, canvas.width, canvas.height);
  if (overlay) ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.fillStyle = 'rgba(5, 10, 18, 0.72)';
  ctx.fillRect(24, 24, 220, 82);
  ctx.fillStyle = '#f5f8ff';
  ctx.font = '700 28px system-ui, sans-serif';
  ctx.fillText(document.querySelector('#scoreDialValue')?.textContent || '--', 42, 64);
  ctx.font = '500 16px system-ui, sans-serif';
  ctx.fillText(document.querySelector('#practiceArmSummary')?.textContent || '左臂 -- / 右臂 --', 42, 92);
  ctx.restore();
  practiceRecording.rafId = requestAnimationFrame(composeRecordingFrame);
}

function clearPracticeRecordingBlob(): void {
  if (!practiceRecording.blobUrl) return;
  URL.revokeObjectURL(practiceRecording.blobUrl);
  practiceRecording.blobUrl = '';
  state.recordingAvailable = false;
}

function startPracticeRecording(): void {
  if (state.recordingState !== 'idle' || !state.poseDataReady) return;
  const mimeType = pickRecordingMimeType();
  if (!mimeType || !HTMLCanvasElement.prototype.captureStream) {
    state.cameraError = '当前浏览器不支持练习录屏，请换用 Chrome 或 Edge';
    applyStageStatusDom();
    return;
  }
  clearPracticeRecordingBlob();
  ensureRecordingCanvas();
  if (!practiceRecording.canvas) return;
  practiceRecording.mimeType = mimeType;
  practiceRecording.chunks = [];
  state.recordingState = 'recording';
  state.recordingAvailable = false;
  refreshRecordingUi();
  composeRecordingFrame();
  const stream = practiceRecording.canvas.captureStream(30);
  practiceRecording.stream = stream;
  const recorder = new MediaRecorder(stream, { mimeType });
  practiceRecording.recorder = recorder;
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) practiceRecording.chunks.push(event.data);
  };
  recorder.onstop = () => {
    const blob = new Blob(practiceRecording.chunks, { type: practiceRecording.mimeType });
    practiceRecording.chunks = [];
    practiceRecording.stream?.getTracks().forEach((track) => track.stop());
    practiceRecording.stream = null;
    practiceRecording.recorder = null;
    practiceRecording.blobUrl = URL.createObjectURL(blob);
    state.recordingState = 'idle';
    state.recordingAvailable = true;
    refreshRecordingUi();
  };
  recorder.start(1000);
}

function stopPracticeRecording(): void {
  if (state.recordingState !== 'recording') return;
  state.recordingState = 'finalizing';
  refreshRecordingUi();
  if (practiceRecording.rafId) cancelAnimationFrame(practiceRecording.rafId);
  practiceRecording.rafId = 0;
  if (practiceRecording.recorder && practiceRecording.recorder.state !== 'inactive') {
    practiceRecording.recorder.onstop = null;
    practiceRecording.recorder.ondataavailable = null;
    practiceRecording.recorder.stop();
  }
}

function cleanupPracticeRecording(): void {
  if (practiceRecording.rafId) cancelAnimationFrame(practiceRecording.rafId);
  practiceRecording.rafId = 0;
  if (practiceRecording.recorder && practiceRecording.recorder.state !== 'inactive') {
    practiceRecording.recorder.stop();
  }
  practiceRecording.stream?.getTracks().forEach((track) => track.stop());
  practiceRecording.stream = null;
  practiceRecording.recorder = null;
  practiceRecording.chunks = [];
  state.recordingState = 'idle';
  clearPracticeRecordingBlob();
}

function togglePracticeRecording(): void {
  if (state.recordingState === 'recording') stopPracticeRecording();
  else startPracticeRecording();
}

function openLatestPracticeRecording(): void {
  if (!practiceRecording.blobUrl) return;
  window.open(practiceRecording.blobUrl, '_blank', 'noopener');
}

let stageStatusOkSince = 0;
let stageStatusHideTimer: number | null = null;

function applyStageStatusDom(isPlaying = state.practiceIsPlaying) {
  if (state.view !== 'practice') return;
  const banner = document.querySelector<HTMLElement>('#stageStatusBanner');
  const titleEl = document.querySelector<HTMLElement>('#stageStatusTitle');
  const hintEl = document.querySelector<HTMLElement>('#stageStatusHint');
  const reticle = document.querySelector<HTMLElement>('#stageReticle');
  const reticleCopy = document.querySelector<HTMLElement>('#stageReticleCopy');

  const next: StageStatusOutput = computeStageStatus({
    poseReady: state.poseReady,
    handsReady: true,
    mediapipeReady: state.poseReady,
    cameraRunning: state.cameraRunning,
    cameraError: state.cameraError,
    personVisible: state.personVisible,
    poseDataLoaded: state.poseDataReady,
    matchDisabledReason: null,
    handsDisabledReason: null,
    isPlaying,
  });

  if (banner) {
    let visible = true;
    if (next.kind === 'ok') {
      const now = performance.now();
      if (banner.dataset.kind !== 'ok') stageStatusOkSince = now;
      if (now - stageStatusOkSince > 2400) visible = false;
    } else {
      stageStatusOkSince = 0;
    }

    if (visible) {
      if (stageStatusHideTimer !== null) {
        clearTimeout(stageStatusHideTimer);
        stageStatusHideTimer = null;
      }
      banner.hidden = false;
      banner.dataset.kind = next.kind;
      banner.classList.add('is-visible');
      if (titleEl) titleEl.textContent = next.title;
      if (hintEl) hintEl.textContent = next.hint;
    } else {
      banner.classList.remove('is-visible');
      if (stageStatusHideTimer === null) {
        stageStatusHideTimer = window.setTimeout(() => {
          if (!banner.classList.contains('is-visible')) banner.hidden = true;
          stageStatusHideTimer = null;
        }, 280);
      }
    }
  }

  if (reticle) {
    const showReticle =
      state.cameraRunning && !state.personVisible && !state.cameraError && state.poseReady;
    if (showReticle) {
      reticle.hidden = false;
      reticle.classList.add('is-visible');
      if (reticleCopy) {
        reticleCopy.textContent = state.poseDataReady
          ? '请站到画面中央，保持上半身完全入镜'
          : '请站到画面中央';
      }
    } else {
      reticle.classList.remove('is-visible');
      window.setTimeout(() => {
        if (!reticle.classList.contains('is-visible')) reticle.hidden = true;
      }, 320);
    }
  }
}

async function wirePracticeRuntime() {
  applyStageStatusDom();
  const video = document.querySelector<HTMLVideoElement>('#cameraVideo');
  if (!video) return;
  if (!video.srcObject && camera.stream) {
    video.srcObject = camera.stream;
    video.play().catch(() => {});
  }

  const scoreEl = document.querySelector<HTMLElement>('#scoreDialValue');
  const peekEl = document.querySelector<HTMLElement>('#practiceScoreLabel');
  const armEl = document.querySelector<HTMLElement>('#practiceArmSummary');
  const stageEl = document.querySelector<HTMLElement>('#stageLabel');
  const canvas = document.querySelector<HTMLCanvasElement>('#overlayCanvas');
  const ctx = canvas?.getContext('2d') ?? null;
  const teacherWebglCanvas = document.querySelector<HTMLCanvasElement>('#teacherWebglCanvas');
  let threeRenderer: ThreeTeacherAvatarRenderer | null = null;
  const redrawTeacherFrame = () => {
    if (!canvas || !ctx) return;
    renderTeacherStage({
      canvas,
      ctx,
      video,
      userPose: null,
      targetFrame: getCurrentTargetFrame(),
      threeRenderer,
    });
  };
  void ensureTeacher3dRenderer(teacherWebglCanvas).then((renderer) => {
    threeRenderer = renderer;
    redrawTeacherFrame();
    if (renderer) void renderer.modelReady.then(redrawTeacherFrame);
  });
  const matchRing = document.querySelector<SVGCircleElement>('#matchRing');
  const scoreDial = document.querySelector<HTMLElement>('#scoreDial');
  let lastDisplayedScore: number | null = null;
  let lastBumpAt = 0;
  let activeDetector: PoseDetector = detector ?? createNoopPoseDetector();

  void ensureDetector().then((readyDetector) => {
    activeDetector = readyDetector;
  });

  practiceLoop = startPracticeLoop({
    detector: {
      detectForVideo: (frameVideo, timestamp) => activeDetector.detectForVideo(frameVideo, timestamp),
      close: () => {},
    },
    video,
    getTargetFrame: getCurrentTargetFrame,
    isCameraRunning: () => state.cameraRunning,
    detectionStride: 2,
    evaluateEveryMs: 100,
    smootherAlpha: 0.3,
    onScore: ({ match, smoothed }) => {
      const text = formatSmoothedScore(smoothed);
      if (scoreEl) scoreEl.textContent = text;
      if (peekEl) peekEl.textContent = text;
      if (armEl) {
        if (match.kind === 'scored') {
          const left = match.armScores.left.score ?? '--';
          const right = match.armScores.right.score ?? '--';
          armEl.textContent = `左臂 ${left === '--' ? '--' : `${left}%`} / 右臂 ${right === '--' ? '--' : `${right}%`}`;
        } else {
          armEl.textContent = '左臂 -- / 右臂 --';
        }
      }
      if (stageEl) stageEl.textContent = match.label;
      // 进度环 + 状态色 + 跳变动画
      if (matchRing) {
        const ringLen = 2 * Math.PI * 30;
        const has = typeof smoothed === 'number';
        const progress = has ? Math.max(0, Math.min(1, smoothed! / 100)) : 0;
        matchRing.setAttribute('stroke-dasharray', ringLen.toFixed(2));
        matchRing.setAttribute('stroke-dashoffset', (ringLen * (1 - progress)).toFixed(2));
        let color = 'rgba(255,190,59,0.55)';
        if (has) {
          if (smoothed! >= 85) color = '#54f3a8';
          else if (smoothed! >= 60) color = '#ffd36e';
          else color = '#ff7b8a';
        }
        matchRing.setAttribute('stroke', color);
      }
      if (scoreDial) {
        scoreDial.classList.toggle('is-perfect', typeof smoothed === 'number' && smoothed >= 85);
        scoreDial.classList.toggle('is-miss', typeof smoothed === 'number' && smoothed < 55);
        if (typeof smoothed === 'number') {
          if (lastDisplayedScore == null || Math.abs(smoothed - lastDisplayedScore) >= 5) {
            const now = performance.now();
            if (now - lastBumpAt > 400) {
              scoreDial.classList.remove('score-dial-bump');
              void scoreDial.offsetWidth;
              scoreDial.classList.add('score-dial-bump');
              lastBumpAt = now;
            }
          }
          lastDisplayedScore = smoothed;
        }
      }
      // 用 personVisible / matchDisabled 驱动 stage status
      const visible = match.kind !== 'no-person';
      if (state.personVisible !== visible) {
        state.personVisible = visible;
      }
      applyStageStatusDom(state.practiceIsPlaying);
    },
    onFrame: ({ userPose, targetFrame }) => {
      if (!canvas || !ctx) return;
      renderTeacherStage({ canvas, ctx, video, userPose, targetFrame, threeRenderer });
    },
  });
}

function stopPracticeLoop() {
  if (practiceLoop) {
    practiceLoop.stop();
    practiceLoop = null;
  }
  disposeTeacher3dRenderer();
}

// ────────── 节拍分析 ──────────

function beatAnalysisViewState(): BeatAnalysisViewState {
  const beat = state.beat;
  return {
    presetName: selectedPresetName() || '动作',
    status: beat.status,
    errorMessage: beat.errorMessage,
    beats: beat.beats,
    analyses: beat.analyses,
    summary: beat.summary,
    activeBeatIndex: beat.activeBeatIndex,
    currentTime: beat.currentTime,
    duration: beat.duration,
    isPlaying: beat.isPlaying,
    regenerating: beat.regenerating,
    videoSrc: deriveVideoSrc(),
  };
}

function deriveVideoSrc(): string | null {
  const preset = state.selectedPresetId ? findPresetById(state.selectedPresetId) : null;
  if (!preset) return null;
  // 约定：与 *_pose.json 同目录的 *.mp4。
  if (preset.path.endsWith('_pose.json')) {
    return preset.path.slice(0, -'_pose.json'.length) + '.mp4';
  }
  return null;
}

let beatAnalysisLoadingFor: string | null = null;

async function ensureBeatAnalysisLoaded() {
  const preset = state.selectedPresetId ? findPresetById(state.selectedPresetId) : null;
  if (!preset) {
    state.beat.status = 'error';
    state.beat.errorMessage = '尚未选择标准动作';
    mount();
    return;
  }
  if (state.beat.status === 'ready' && state.beat.beats.length) return;
  if (beatAnalysisLoadingFor === preset.id) return;
  beatAnalysisLoadingFor = preset.id;

  state.beat.status = 'loading';
  patchAnalysisDom();

  try {
    if (!state.poseData) {
      const res = await fetch(preset.path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.poseData = (await res.json()) as PoseData;
      state.poseDataReady = Array.isArray(state.poseData.frames) && state.poseData.frames.length > 0;
    }
    let motion: MotionAnalysis | null = null;
    try {
      const motionRes = await fetch(motionPathForPreset(preset));
      if (motionRes.ok) motion = (await motionRes.json()) as MotionAnalysis;
    } catch {
      motion = null;
    }
    const beats = detectBeats(state.poseData, motion);
    const analyses = analyzeBeats(beats, state.poseData!, motion);
    const summary = summarizeBeatAnalyses(analyses);
    const lastFrame = state.poseData!.frames[state.poseData!.frames.length - 1];

    state.beat.beats = beats;
    state.beat.analyses = analyses;
    state.beat.summary = summary;
    state.beat.motion = motion;
    state.beat.duration = lastFrame?.time ?? 0;
    state.beat.activeBeatIndex = beats.length ? 0 : -1;
    state.beat.currentTime = beats[0]?.timestamp ?? 0;
    state.beat.status = beats.length ? 'ready' : 'empty';
    state.beat.errorMessage = undefined;
  } catch (err) {
    console.error('[main] beat analysis failed', err);
    state.beat.status = 'error';
    state.beat.errorMessage = err instanceof Error ? err.message : String(err);
  } finally {
    beatAnalysisLoadingFor = null;
    if (state.view === 'analysis') {
      // 全量重渲染一次以替换 loading 占位。
      mount();
      attachBeatVideoHandlers();
    }
  }
}

function patchAnalysisDom() {
  if (state.view !== 'analysis' || !root) return;
  patchBeatAnalysisView(root, beatAnalysisViewState());
}

function attachBeatVideoHandlers() {
  if (state.view !== 'analysis' || !root) return;
  const video = root.querySelector<HTMLVideoElement>('#beatVideo');
  if (!video) return;
  video.addEventListener('timeupdate', () => {
    state.beat.currentTime = video.currentTime;
    state.beat.activeBeatIndex = findBeatIndexAtTime(state.beat.currentTime);
    patchAnalysisDom();
  });
  video.addEventListener('play', () => {
    state.beat.isPlaying = true;
    patchAnalysisDom();
  });
  video.addEventListener('pause', () => {
    state.beat.isPlaying = false;
    patchAnalysisDom();
  });
  video.addEventListener('loadedmetadata', () => {
    if (Number.isFinite(video.duration)) state.beat.duration = video.duration;
    patchAnalysisDom();
  });
}

function findBeatIndexAtTime(t: number): number {
  const bs = state.beat.beats;
  if (!bs.length) return -1;
  for (const b of bs) {
    if (t >= b.startTime && t < b.endTime) return b.beatIndex;
  }
  return bs[bs.length - 1]!.beatIndex;
}

function getBeatVideo(): HTMLVideoElement | null {
  return root?.querySelector<HTMLVideoElement>('#beatVideo') ?? null;
}

function handleBeatTogglePlayback() {
  const video = getBeatVideo();
  if (video) {
    if (video.paused) video.play().catch(() => {});
    else video.pause();
    return;
  }
  // 无视频时仅切换状态用作 UI 提示
  state.beat.isPlaying = !state.beat.isPlaying;
  patchAnalysisDom();
}

function handleBeatStepFrame(direction: 1 | -1) {
  const fps = state.poseData?.fps ?? 30;
  const dt = (1 / Math.max(1, fps)) * direction;
  const video = getBeatVideo();
  if (video) {
    video.pause();
    video.currentTime = Math.max(0, Math.min(video.duration || state.beat.duration, video.currentTime + dt));
    return;
  }
  state.beat.currentTime = Math.max(
    0,
    Math.min(state.beat.duration, state.beat.currentTime + dt),
  );
  state.beat.activeBeatIndex = findBeatIndexAtTime(state.beat.currentTime);
  patchAnalysisDom();
}

function handleSeekToBeat(beatIndex: number) {
  const beat = state.beat.beats[beatIndex];
  if (!beat) return;
  state.beat.activeBeatIndex = beatIndex;
  state.beat.currentTime = beat.timestamp;
  const video = getBeatVideo();
  if (video) {
    video.pause();
    try {
      video.currentTime = beat.timestamp;
    } catch {
      /* noop */
    }
  }
  patchAnalysisDom();
}

function handleStepBeat(direction: 1 | -1) {
  const next = state.beat.activeBeatIndex + direction;
  if (next < 0 || next >= state.beat.beats.length) return;
  handleSeekToBeat(next);
}

async function handleRegenerateBeat(beatIndex: number) {
  if (!state.beat.beats[beatIndex] || !state.poseData) return;
  state.beat.regenerating = true;
  patchAnalysisDom();
  try {
    // 重新分析单拍：重跑该拍的分析逻辑（标准动画修复实际依赖后端，
    // 这里给出客户端可立即生效的"重算+平滑"占位，以便 UI 流程闭环）。
    const fresh = analyzeBeats(
      [state.beat.beats[beatIndex]!],
      state.poseData,
      state.beat.motion,
    );
    if (fresh[0]) {
      state.beat.analyses = state.beat.analyses.map((it, i) =>
        i === beatIndex ? fresh[0]! : it,
      );
      state.beat.summary = summarizeBeatAnalyses(state.beat.analyses);
    }
  } finally {
    state.beat.regenerating = false;
    patchAnalysisDom();
  }
}

function handleApplyFix(beatIndex: number) {
  const ana = state.beat.analyses[beatIndex];
  if (!ana) return;
  // 占位：把第一条建议附加到 userPerformance，便于跨节拍持续可见。
  const next = {
    ...ana,
    userPerformance: ana.userPerformance + ` 已记录修复建议：${ana.suggestions[0] ?? '—'}`,
  };
  state.beat.analyses = state.beat.analyses.map((it, i) => (i === beatIndex ? next : it));
  patchAnalysisDom();
}

if (intent.preset) {
  const preset = intent.preset;
  state.selectedPresetId = preset.id;
  void handlePresetClick(preset, { autoplay: intent.autoplay });
} else {
  mount();
}

const fallback = document.createElement('a');
fallback.href = buildDeepLink(window.location.origin, null, false);
fallback.textContent = '回到旧版入口';
fallback.style.cssText =
  'position:fixed;bottom:6px;left:8px;font-size:.72rem;color:rgba(245,248,255,.4);text-decoration:underline;';
document.body.appendChild(fallback);

window.addEventListener('beforeunload', () => {
  cleanupPracticeRecording();
  camera.stop();
  stopPracticeLoop();
  detector?.close();
});
