/**
 * Beat detection over PoseData.
 *
 * 节拍来源（按可用性优先级）：
 *   1) MotionAnalysis.summary.beatTimes 显式给出的节拍点（若以后补齐）
 *   2) MotionAnalysis.segments 的边界（每个 segment 起点视为一个节拍）
 *   3) 基于腕部 (15/16) 速度的峰值检测：在归一化 (x,y) 上做平滑，取
 *      局部极大值且高于动态阈值的点作为节拍
 *
 * 输出的 Beat 用于驱动节拍级别的动作分析 / 前端时间轴。
 */

import type { PoseData, PoseFrame } from './types.js';
import type { MotionAnalysis } from './motionTypes.js';

export interface Beat {
  /** 0-based 节拍序号。 */
  beatIndex: number;
  /** 节拍触发时刻，即 timestamp（秒）。 */
  timestamp: number;
  /** 节拍区间起点（秒），通常等于上一拍的中点 / 0。 */
  startTime: number;
  /** 节拍区间终点（秒），通常等于下一拍的中点 / duration。 */
  endTime: number;
  /** 区间内的代表帧索引；用于动作命名 / 比对。 */
  representativeFrameIndex: number;
  /** 节拍强度 0..1（无强度信息时填 0.5）。 */
  intensity: number;
  /** 节拍来源标签，便于调试与降级。 */
  source: 'beatTimes' | 'segment' | 'velocity-peak' | 'fallback-uniform';
}

export interface DetectBeatsOptions {
  /** 强制使用某种来源，主要给测试用。默认 auto 从优至劣选取。 */
  source?: 'auto' | 'beatTimes' | 'segment' | 'velocity-peak' | 'fallback-uniform';
  /** 节拍数兜底：当帧数不足或无信号时，至少生成几个均匀拍。 */
  fallbackBeats?: number;
  /** 速度峰值检测的最小相邻间隔（秒）。 */
  minBeatGapSeconds?: number;
  /** 速度峰值检测：相对于全局最大速度的阈值 (0..1)。 */
  velocityPeakThreshold?: number;
}

const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;

const DEFAULTS: Required<DetectBeatsOptions> = {
  source: 'auto',
  fallbackBeats: 8,
  minBeatGapSeconds: 0.32,
  velocityPeakThreshold: 0.32,
};

/** 主入口：综合 PoseData + MotionAnalysis 推导节拍。 */
export function detectBeats(
  poseData: PoseData | null | undefined,
  motion?: MotionAnalysis | null,
  options: DetectBeatsOptions = {},
): Beat[] {
  const opt = { ...DEFAULTS, ...options };
  if (!poseData || !Array.isArray(poseData.frames) || poseData.frames.length === 0) {
    return [];
  }

  const totalDuration = lastFrameTime(poseData);
  if (totalDuration <= 0) return [];

  const tryOrder: DetectBeatsOptions['source'][] =
    opt.source && opt.source !== 'auto'
      ? [opt.source]
      : ['beatTimes', 'segment', 'velocity-peak', 'fallback-uniform'];

  for (const src of tryOrder) {
    const timestamps = collectTimestamps(src!, poseData, motion ?? null, opt);
    if (timestamps.length >= 2) {
      return finalizeBeats(timestamps, src!, poseData, totalDuration);
    }
  }
  return [];
}

function lastFrameTime(poseData: PoseData): number {
  const frames = poseData.frames;
  return frames[frames.length - 1]?.time ?? 0;
}

function collectTimestamps(
  source: NonNullable<DetectBeatsOptions['source']>,
  poseData: PoseData,
  motion: MotionAnalysis | null,
  opt: Required<DetectBeatsOptions>,
): number[] {
  switch (source) {
    case 'beatTimes': {
      const beats = motion?.summary?.beatTimes;
      return Array.isArray(beats) ? sanitizeTimestamps(beats, lastFrameTime(poseData)) : [];
    }
    case 'segment': {
      const segs = motion?.segments;
      if (!Array.isArray(segs) || segs.length === 0) return [];
      const out = segs.map((s) => s.startTime).filter((t) => Number.isFinite(t));
      const last = lastFrameTime(poseData);
      if (segs[segs.length - 1]) out.push(segs[segs.length - 1]!.endTime);
      return sanitizeTimestamps(out, last);
    }
    case 'velocity-peak':
      return detectVelocityPeaks(poseData, opt);
    case 'fallback-uniform':
      return uniformTimestamps(lastFrameTime(poseData), opt.fallbackBeats);
    default:
      return [];
  }
}

function sanitizeTimestamps(values: readonly number[], maxTime: number): number[] {
  const cleaned = values
    .filter((v) => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= maxTime + 1e-3)
    .map((v) => Math.min(v, maxTime));
  cleaned.sort((a, b) => a - b);
  // 去掉过近的拍点
  const out: number[] = [];
  for (const t of cleaned) {
    if (!out.length || t - out[out.length - 1]! > 1e-3) out.push(t);
  }
  return out;
}

function uniformTimestamps(duration: number, count: number): number[] {
  if (duration <= 0 || count < 2) return [];
  const step = duration / count;
  const out: number[] = [];
  for (let i = 0; i < count; i++) out.push(step * (i + 0.5));
  return out;
}

function detectVelocityPeaks(poseData: PoseData, opt: Required<DetectBeatsOptions>): number[] {
  const frames = poseData.frames;
  const speeds = computeWristSpeeds(frames);
  if (speeds.length < 4) return [];

  const smoothed = smooth1D(speeds, 3);
  const maxSpeed = smoothed.reduce((m, v) => (v > m ? v : m), 0);
  if (maxSpeed <= 0) return [];
  const threshold = maxSpeed * opt.velocityPeakThreshold;

  const peaks: number[] = [];
  let lastPeakTime = -Infinity;
  for (let i = 1; i < smoothed.length - 1; i++) {
    const v = smoothed[i]!;
    if (v < threshold) continue;
    if (v <= smoothed[i - 1]! || v <= smoothed[i + 1]!) continue;
    const t = frames[i]!.time ?? 0;
    if (t - lastPeakTime < opt.minBeatGapSeconds) {
      // 同一峰值簇，保留更高的
      if (peaks.length && smoothed[i]! > smoothed[indexOfTime(frames, peaks[peaks.length - 1]!)]!) {
        peaks[peaks.length - 1] = t;
        lastPeakTime = t;
      }
      continue;
    }
    peaks.push(t);
    lastPeakTime = t;
  }
  return peaks;
}

function indexOfTime(frames: readonly PoseFrame[], t: number): number {
  // 线性查找对小数组而言成本可忽略；这里只在节拍点合并时使用。
  for (let i = 0; i < frames.length; i++) {
    if ((frames[i]!.time ?? 0) >= t) return i;
  }
  return frames.length - 1;
}

function computeWristSpeeds(frames: readonly PoseFrame[]): number[] {
  const out = new Array<number>(frames.length).fill(0);
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1]!;
    const cur = frames[i]!;
    const dt = Math.max(1e-3, (cur.time ?? 0) - (prev.time ?? 0));
    let total = 0;
    let count = 0;
    for (const idx of [LEFT_WRIST, RIGHT_WRIST]) {
      const a = prev.pose_landmarks?.[idx];
      const b = cur.pose_landmarks?.[idx];
      if (!a || !b) continue;
      if ((a.visibility ?? 1) < 0.3 || (b.visibility ?? 1) < 0.3) continue;
      const dx = (b.x ?? 0) - (a.x ?? 0);
      const dy = (b.y ?? 0) - (a.y ?? 0);
      total += Math.hypot(dx, dy) / dt;
      count++;
    }
    out[i] = count ? total / count : 0;
  }
  return out;
}

function smooth1D(values: readonly number[], radius: number): number[] {
  const n = values.length;
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    let count = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = i + k;
      if (j < 0 || j >= n) continue;
      sum += values[j]!;
      count++;
    }
    out[i] = count ? sum / count : 0;
  }
  return out;
}

function finalizeBeats(
  timestamps: number[],
  source: NonNullable<DetectBeatsOptions['source']>,
  poseData: PoseData,
  duration: number,
): Beat[] {
  const frames = poseData.frames;
  const sorted = [...timestamps].sort((a, b) => a - b);
  const beats: Beat[] = [];
  for (let i = 0; i < sorted.length; i++) {
    const t = sorted[i]!;
    const prev = i === 0 ? 0 : (sorted[i - 1]! + t) / 2;
    const next = i === sorted.length - 1 ? duration : (sorted[i + 1]! + t) / 2;
    const repIdx = nearestFrameIndex(frames, t);
    beats.push({
      beatIndex: i,
      timestamp: round3(t),
      startTime: round3(prev),
      endTime: round3(next),
      representativeFrameIndex: repIdx,
      intensity: estimateIntensity(frames, repIdx),
      source: source === 'auto' ? 'velocity-peak' : source,
    });
  }
  return beats;
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

function nearestFrameIndex(frames: readonly PoseFrame[], t: number): number {
  if (!frames.length) return -1;
  if (t <= (frames[0]!.time ?? 0)) return 0;
  const last = frames.length - 1;
  if (t >= (frames[last]!.time ?? 0)) return last;
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    const tm = frames[mid]!.time ?? 0;
    if (tm <= t) lo = mid;
    else hi = mid;
  }
  const tl = frames[lo]!.time ?? 0;
  const th = frames[hi]!.time ?? 0;
  return Math.abs(t - tl) <= Math.abs(t - th) ? lo : hi;
}

function estimateIntensity(frames: readonly PoseFrame[], idx: number): number {
  if (idx < 1 || idx >= frames.length) return 0.5;
  const prev = frames[idx - 1]!;
  const cur = frames[idx]!;
  const dt = Math.max(1e-3, (cur.time ?? 0) - (prev.time ?? 0));
  let total = 0;
  let count = 0;
  for (const i of [LEFT_WRIST, RIGHT_WRIST]) {
    const a = prev.pose_landmarks?.[i];
    const b = cur.pose_landmarks?.[i];
    if (!a || !b) continue;
    total += Math.hypot((b.x ?? 0) - (a.x ?? 0), (b.y ?? 0) - (a.y ?? 0)) / dt;
    count++;
  }
  if (!count) return 0.5;
  const v = total / count;
  // 经验缩放：0.0..2.0 单位/秒映射到 0..1
  return Math.max(0, Math.min(1, v / 2));
}

/** 给定一个时间点，返回它落在哪个节拍内（找不到时返回 -1）。 */
export function beatIndexAtTime(beats: readonly Beat[], time: number): number {
  if (!beats.length) return -1;
  if (time < beats[0]!.startTime) return 0;
  for (const b of beats) {
    if (time >= b.startTime && time < b.endTime) return b.beatIndex;
  }
  return beats[beats.length - 1]!.beatIndex;
}
