/**
 * 节拍级动作分析。
 *
 * 输入：节拍序列 + 标准 PoseData + 可选 MotionAnalysis（提供动作命名/描述）
 *      + 可选 用户姿态记录（按时间索引的 NormalizedLandmark[]）
 *
 * 输出：每个节拍一份 BeatAnalysis，含动作命名、标准描述、用户表现、偏差、
 *      节奏匹配度、动画流畅度、状态分级、修复建议。
 *
 * 设计：纯函数 + 可注入的 USER 数据，便于在练习态/复盘态/无用户数据态复用。
 */

import type { NormalizedLandmark, PoseData } from './types.js';
import type { MotionAnalysis, MotionSegment, Cardinal8 } from './motionTypes.js';
import type { Beat } from './beats.js';
import { compareArmPoses } from './poseCompare.js';

export type BeatStatus = 'normal' | 'warning' | 'error';

export type BeatIssueKind =
  | 'pose-deviation'
  | 'rhythm-drift'
  | 'animation-stutter'
  | 'transition-jump'
  | 'low-coverage';

export interface BeatIssue {
  kind: BeatIssueKind;
  severity: BeatStatus;
  message: string;
}

export interface BeatUserSample {
  /** 用户姿态采样所在时刻（秒）。 */
  time: number;
  pose: NormalizedLandmark[];
}

export interface BeatAnalysis {
  beatIndex: number;
  startTime: number;
  endTime: number;
  timestamp: number;
  /** 当前节拍动作中文命名。 */
  actionName: string;
  /** 标准动作描述（来自 MotionAnalysis 或自动归纳）。 */
  standardDescription: string;
  /** 用户实际动作描述（无用户数据时给出占位/原因）。 */
  userPerformance: string;
  /** 0..100 越小表示偏差越大；null 表示无法评估。 */
  poseDeviation: number | null;
  /** 0..100，节奏匹配度。 */
  rhythmMatch: number | null;
  /** 0..100，标准动画在该拍内的流畅度。 */
  smoothness: number;
  /** 节拍区间内的代表关键帧时刻列表（用于逐帧浏览）。 */
  keyFrameTimes: number[];
  /** 节拍状态分级。 */
  status: BeatStatus;
  /** 检出问题。 */
  issues: BeatIssue[];
  /** 修复建议（自然语言）。 */
  suggestions: string[];
  /** 关联的标准 segment id（若有）。 */
  segmentId?: string;
  /** 该拍主导方向（若可推断）。 */
  primaryDirection?: Cardinal8;
}

export interface AnalyzeBeatsOptions {
  /** 用户实际姿态采样；空表示仅做"标准动作"分析。 */
  userSamples?: readonly BeatUserSample[];
  /** 节奏匹配允许的时间窗口（秒）。 */
  rhythmToleranceSeconds?: number;
  /** 偏差低于该值视为严重问题。 */
  errorThreshold?: number;
  /** 偏差低于该值视为轻微问题。 */
  warningThreshold?: number;
}

const DEFAULTS: Required<Omit<AnalyzeBeatsOptions, 'userSamples'>> = {
  rhythmToleranceSeconds: 0.18,
  errorThreshold: 55,
  warningThreshold: 75,
};

const DIRECTION_CN: Record<Cardinal8, string> = {
  up: '上',
  down: '下',
  left: '左',
  right: '右',
  upLeft: '左上',
  upRight: '右上',
  downLeft: '左下',
  downRight: '右下',
  still: '静止',
  mixed: '多方向',
};

export function analyzeBeats(
  beats: readonly Beat[],
  poseData: PoseData,
  motion: MotionAnalysis | null | undefined,
  options: AnalyzeBeatsOptions = {},
): BeatAnalysis[] {
  const opt = { ...DEFAULTS, ...options };
  const samples = options.userSamples ?? [];

  return beats.map((beat) => buildOneBeatAnalysis(beat, poseData, motion ?? null, samples, opt));
}

function buildOneBeatAnalysis(
  beat: Beat,
  poseData: PoseData,
  motion: MotionAnalysis | null,
  samples: readonly BeatUserSample[],
  opt: Required<Omit<AnalyzeBeatsOptions, 'userSamples'>>,
): BeatAnalysis {
  const segment = findSegmentForBeat(beat, motion);
  const targetFrame = poseData.frames[beat.representativeFrameIndex] ?? null;

  const actionName = segment?.title ?? autoActionName(beat, poseData, segment);
  const standardDescription = segment?.llmDescription || segment?.description || autoStandardDescription(beat);

  const keyFrameTimes = collectKeyFrameTimes(beat, segment);
  const smoothness = estimateSmoothnessInRange(poseData, beat.startTime, beat.endTime);

  const userMatch = matchUserSample(samples, beat, opt.rhythmToleranceSeconds);
  const poseDeviation = userMatch && targetFrame
    ? scoreUserAgainstTarget(userMatch.pose, targetFrame.pose_landmarks)
    : null;
  const rhythmMatch = userMatch ? scoreRhythm(userMatch.time, beat.timestamp, opt.rhythmToleranceSeconds) : null;
  const userPerformance = userMatch
    ? describeUserPerformance(poseDeviation, rhythmMatch)
    : '未捕捉到该节拍的用户姿态采样。';

  const issues = collectIssues({
    poseDeviation,
    rhythmMatch,
    smoothness,
    targetExists: !!targetFrame,
    samples,
    opt,
  });
  const status = aggregateStatus(issues);
  const suggestions = buildSuggestions(issues, segment, beat);

  return {
    beatIndex: beat.beatIndex,
    startTime: beat.startTime,
    endTime: beat.endTime,
    timestamp: beat.timestamp,
    actionName,
    standardDescription,
    userPerformance,
    poseDeviation,
    rhythmMatch,
    smoothness,
    keyFrameTimes,
    status,
    issues,
    suggestions,
    segmentId: segment?.id,
    primaryDirection: segment?.primaryDirection,
  };
}

function findSegmentForBeat(beat: Beat, motion: MotionAnalysis | null): MotionSegment | undefined {
  if (!motion?.segments?.length) return undefined;
  return motion.segments.find((s) => beat.timestamp >= s.startTime && beat.timestamp < s.endTime)
    ?? motion.segments.find((s) => Math.abs(s.startTime - beat.timestamp) < 0.05);
}

function autoActionName(beat: Beat, poseData: PoseData, segment?: MotionSegment): string {
  if (segment?.title) return segment.title;
  // 退化命名：以左/右腕在节拍代表帧的位置高度作描述
  const f = poseData.frames[beat.representativeFrameIndex];
  if (!f?.pose_landmarks?.length) return `第 ${beat.beatIndex + 1} 拍动作`;
  const lw = f.pose_landmarks[15];
  const rw = f.pose_landmarks[16];
  const ls = f.pose_landmarks[11];
  const rs = f.pose_landmarks[12];
  if (!lw || !rw || !ls || !rs) return `第 ${beat.beatIndex + 1} 拍动作`;
  const leftHigh = (lw.y ?? 0) < (ls.y ?? 0);
  const rightHigh = (rw.y ?? 0) < (rs.y ?? 0);
  if (leftHigh && rightHigh) return `第 ${beat.beatIndex + 1} 拍 · 双臂上举`;
  if (leftHigh) return `第 ${beat.beatIndex + 1} 拍 · 左臂抬起`;
  if (rightHigh) return `第 ${beat.beatIndex + 1} 拍 · 右臂抬起`;
  return `第 ${beat.beatIndex + 1} 拍 · 双臂下落`;
}

function autoStandardDescription(beat: Beat): string {
  const dur = Math.max(0, beat.endTime - beat.startTime);
  return `节拍区间 ${beat.startTime.toFixed(2)}s–${beat.endTime.toFixed(2)}s（约 ${dur.toFixed(2)}s），中心时刻 ${beat.timestamp.toFixed(2)}s。`;
}

function collectKeyFrameTimes(beat: Beat, segment?: MotionSegment): number[] {
  if (!segment?.keyFrames?.length) return [beat.timestamp];
  return segment.keyFrames.filter((t) => t >= beat.startTime && t <= beat.endTime);
}

function estimateSmoothnessInRange(poseData: PoseData, start: number, end: number): number {
  const frames = poseData.frames;
  if (!frames.length || end <= start) return 80;
  // 取该区间的腕速序列，计算抖动 = 速度差分的标准差
  const seq: number[] = [];
  for (let i = 1; i < frames.length; i++) {
    const t = frames[i]!.time ?? 0;
    if (t < start || t > end) continue;
    const prev = frames[i - 1]!;
    const cur = frames[i]!;
    const dt = Math.max(1e-3, t - (prev.time ?? 0));
    let v = 0;
    let n = 0;
    for (const idx of [15, 16]) {
      const a = prev.pose_landmarks?.[idx];
      const b = cur.pose_landmarks?.[idx];
      if (!a || !b) continue;
      v += Math.hypot((b.x ?? 0) - (a.x ?? 0), (b.y ?? 0) - (a.y ?? 0)) / dt;
      n++;
    }
    if (n) seq.push(v / n);
  }
  if (seq.length < 3) return 80;
  const diffs = seq.slice(1).map((v, i) => v - seq[i]!);
  const mean = diffs.reduce((s, x) => s + x, 0) / diffs.length;
  const variance = diffs.reduce((s, x) => s + (x - mean) ** 2, 0) / diffs.length;
  const std = Math.sqrt(variance);
  // std 0..1 映射到 100..50
  return Math.max(0, Math.min(100, Math.round(100 - std * 60)));
}

function matchUserSample(
  samples: readonly BeatUserSample[],
  beat: Beat,
  tolerance: number,
): BeatUserSample | null {
  if (!samples.length) return null;
  let best: BeatUserSample | null = null;
  let bestDelta = Infinity;
  for (const s of samples) {
    if (s.time < beat.startTime - tolerance || s.time > beat.endTime + tolerance) continue;
    const d = Math.abs(s.time - beat.timestamp);
    if (d < bestDelta) {
      best = s;
      bestDelta = d;
    }
  }
  return best;
}

function scoreUserAgainstTarget(
  userPose: NormalizedLandmark[],
  targetPose: NormalizedLandmark[],
): number | null {
  const arm = compareArmPoses(userPose, targetPose);
  return arm ? arm.total : null;
}

function scoreRhythm(userTime: number, beatTime: number, tolerance: number): number {
  const d = Math.abs(userTime - beatTime);
  if (d <= tolerance * 0.25) return 100;
  if (d >= tolerance * 2) return 0;
  // 线性衰减
  const span = tolerance * 2 - tolerance * 0.25;
  return Math.max(0, Math.min(100, Math.round((1 - (d - tolerance * 0.25) / span) * 100)));
}

function describeUserPerformance(deviation: number | null, rhythm: number | null): string {
  const parts: string[] = [];
  if (deviation === null) parts.push('动作偏差未评估');
  else if (deviation >= 80) parts.push(`动作姿态贴合标准（${deviation} 分）`);
  else if (deviation >= 60) parts.push(`姿态略有偏差（${deviation} 分）`);
  else parts.push(`姿态偏差较大（${deviation} 分）`);
  if (rhythm === null) parts.push('节奏未评估');
  else if (rhythm >= 80) parts.push(`节奏精准（${rhythm}）`);
  else if (rhythm >= 50) parts.push(`节奏稍微滞后/超前（${rhythm}）`);
  else parts.push(`节奏明显错拍（${rhythm}）`);
  return parts.join('，') + '。';
}

interface IssueCtx {
  poseDeviation: number | null;
  rhythmMatch: number | null;
  smoothness: number;
  targetExists: boolean;
  samples: readonly BeatUserSample[];
  opt: Required<Omit<AnalyzeBeatsOptions, 'userSamples'>>;
}

function collectIssues(ctx: IssueCtx): BeatIssue[] {
  const issues: BeatIssue[] = [];

  if (!ctx.targetExists) {
    issues.push({ kind: 'low-coverage', severity: 'warning', message: '该节拍缺少标准代表帧。' });
  }

  if (ctx.poseDeviation !== null) {
    if (ctx.poseDeviation < ctx.opt.errorThreshold) {
      issues.push({
        kind: 'pose-deviation',
        severity: 'error',
        message: `动作偏差较大（${ctx.poseDeviation} 分），关键关节未对齐。`,
      });
    } else if (ctx.poseDeviation < ctx.opt.warningThreshold) {
      issues.push({
        kind: 'pose-deviation',
        severity: 'warning',
        message: `动作存在轻微偏差（${ctx.poseDeviation} 分）。`,
      });
    }
  } else if (ctx.samples.length > 0) {
    issues.push({ kind: 'low-coverage', severity: 'warning', message: '该节拍未匹配到用户姿态样本。' });
  }

  if (ctx.rhythmMatch !== null) {
    if (ctx.rhythmMatch < 40) {
      issues.push({ kind: 'rhythm-drift', severity: 'error', message: `节奏明显错拍（${ctx.rhythmMatch}）。` });
    } else if (ctx.rhythmMatch < 75) {
      issues.push({ kind: 'rhythm-drift', severity: 'warning', message: `节奏稍有偏差（${ctx.rhythmMatch}）。` });
    }
  }

  if (ctx.smoothness < 55) {
    issues.push({
      kind: 'animation-stutter',
      severity: 'error',
      message: `标准动画在该拍存在卡顿（流畅度 ${ctx.smoothness}）。`,
    });
  } else if (ctx.smoothness < 70) {
    issues.push({
      kind: 'animation-stutter',
      severity: 'warning',
      message: `动画过渡略不顺滑（流畅度 ${ctx.smoothness}）。`,
    });
  }

  return issues;
}

function aggregateStatus(issues: readonly BeatIssue[]): BeatStatus {
  if (issues.some((i) => i.severity === 'error')) return 'error';
  if (issues.some((i) => i.severity === 'warning')) return 'warning';
  return 'normal';
}

function buildSuggestions(
  issues: readonly BeatIssue[],
  segment: MotionSegment | undefined,
  beat: Beat,
): string[] {
  const out: string[] = [];
  for (const issue of issues) {
    switch (issue.kind) {
      case 'pose-deviation':
        out.push(
          segment?.tips?.[0]
            ? `跟随提示：${segment.tips[0]}`
            : `对照标准动作"${segment?.title ?? '该节拍'}"，调整手臂方向与幅度。`,
        );
        break;
      case 'rhythm-drift':
        out.push(`提前 ${(0.18).toFixed(2)}s 起手，让动作落点对齐节拍 ${beat.timestamp.toFixed(2)}s。`);
        break;
      case 'animation-stutter':
        out.push('对该节拍区间重新生成插值，或在卡顿点补充关键帧。');
        break;
      case 'transition-jump':
        out.push('在节拍交界处增加过渡帧或缓动函数。');
        break;
      case 'low-coverage':
        out.push('补录该节拍的用户姿态样本，或重跑标准动作提取。');
        break;
    }
  }
  if (!out.length) out.push('动作正常，保持。');
  return Array.from(new Set(out));
}

/** 便于 UI 展示的中文方向。 */
export function directionLabelCN(dir: Cardinal8 | undefined | null): string {
  if (!dir) return '—';
  return DIRECTION_CN[dir] ?? dir;
}

export interface BeatAnalysisSummary {
  total: number;
  normal: number;
  warning: number;
  error: number;
  averageDeviation: number | null;
  averageRhythm: number | null;
  averageSmoothness: number;
}

export function summarizeBeatAnalyses(items: readonly BeatAnalysis[]): BeatAnalysisSummary {
  const total = items.length;
  let normal = 0;
  let warning = 0;
  let error = 0;
  const dev: number[] = [];
  const rhy: number[] = [];
  const smo: number[] = [];
  for (const it of items) {
    if (it.status === 'normal') normal++;
    else if (it.status === 'warning') warning++;
    else error++;
    if (it.poseDeviation !== null) dev.push(it.poseDeviation);
    if (it.rhythmMatch !== null) rhy.push(it.rhythmMatch);
    smo.push(it.smoothness);
  }
  const avg = (xs: number[]): number | null =>
    xs.length ? Math.round(xs.reduce((s, x) => s + x, 0) / xs.length) : null;
  return {
    total,
    normal,
    warning,
    error,
    averageDeviation: avg(dev),
    averageRhythm: avg(rhy),
    averageSmoothness: avg(smo) ?? 0,
  };
}
