/**
 * Canvas painter — pure drawing primitives ported from pose_viewer.html.
 *
 * Stays decoupled from the practice loop / view: callers pass in a context, a
 * stage rect (already letterboxed) and landmark arrays. No DOM access, no
 * state, so it's straightforward to unit-test if needed later.
 */

import type { NormalizedLandmark, HandFrame } from '../core/types.js';
import { getVisibility } from '../core/geometry.js';
import { getPoseReference } from '../core/reference.js';
import { mapPointToStage, type StageRect } from './stageRenderer.js';

export const POSE_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15],
  [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20],
  [16, 22], [18, 20], [11, 23], [12, 24],
  [23, 24], [23, 25], [24, 26], [25, 27],
  [27, 29], [29, 31], [26, 28], [28, 30],
  [30, 32], [27, 31], [28, 32],
];

export const HAND_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export interface BodyPalette {
  fill: string;
  stroke: string;
  glow?: string;
  /** Mid-stop for gradient skeleton accents. Falls back to stroke. */
  accent?: string;
}

export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

export const BODY_PALETTES = {
  target: { fill: 'rgba(255,184,59,0.28)', stroke: 'rgba(255,214,122,0.98)', glow: 'rgba(255,184,59,0.45)', accent: 'rgba(255,245,200,1)' },
  targetStrong: { fill: 'rgba(255,184,59,0.38)', stroke: 'rgba(255,238,196,1)', glow: 'rgba(255,184,59,0.66)', accent: 'rgba(255,250,220,1)' },
  user: { fill: 'rgba(84,243,168,0.14)', stroke: 'rgba(84,243,168,0.9)', glow: 'rgba(84,243,168,0.45)', accent: 'rgba(110,231,255,1)' },
  userPerfect: { fill: 'rgba(84,243,168,0.22)', stroke: 'rgba(84,243,168,0.98)', glow: 'rgba(84,243,168,0.6)', accent: 'rgba(180,255,220,1)' },
  miss: { fill: 'rgba(255,123,138,0.16)', stroke: 'rgba(255,123,138,0.95)', glow: 'rgba(255,123,138,0.45)', accent: 'rgba(255,200,210,1)' },
} as const satisfies Record<string, BodyPalette>;

export const DEFAULT_TEACHER_PALETTE: BodyPalette = BODY_PALETTES.targetStrong;

export const HAND_PALETTES: Record<string, BodyPalette> = {
  Left: { stroke: 'rgba(125,193,255,0.92)', fill: 'rgba(125,193,255,0.92)' },
  Right: { stroke: 'rgba(255,152,169,0.92)', fill: 'rgba(255,152,169,0.92)' },
  default: { stroke: 'rgba(255,205,120,0.92)', fill: 'rgba(255,205,120,0.92)' },
};

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function rgbToHsl(color: RgbColor): { h: number; s: number; l: number } {
  const r = clampNumber(color.r, 0, 255) / 255;
  const g = clampNumber(color.g, 0, 255) / 255;
  const b = clampNumber(color.b, 0, 255) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h: number;
  if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return { h: h * 60, s, l };
}

export function hslToRgb(h: number, s: number, l: number): RgbColor {
  const hue = ((h % 360) + 360) % 360;
  const sat = clampNumber(s, 0, 1);
  const light = clampNumber(l, 0, 1);
  const c = (1 - Math.abs(2 * light - 1)) * sat;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = light - c / 2;
  let r1 = 0;
  let g1 = 0;
  let b1 = 0;
  if (hue < 60) [r1, g1, b1] = [c, x, 0];
  else if (hue < 120) [r1, g1, b1] = [x, c, 0];
  else if (hue < 180) [r1, g1, b1] = [0, c, x];
  else if (hue < 240) [r1, g1, b1] = [0, x, c];
  else if (hue < 300) [r1, g1, b1] = [x, 0, c];
  else [r1, g1, b1] = [c, 0, x];
  return {
    r: Math.round((r1 + m) * 255),
    g: Math.round((g1 + m) * 255),
    b: Math.round((b1 + m) * 255),
  };
}

function rgbToCss(color: RgbColor, alpha = 1): string {
  return `rgba(${Math.round(clampNumber(color.r, 0, 255))}, ${Math.round(clampNumber(color.g, 0, 255))}, ${Math.round(clampNumber(color.b, 0, 255))}, ${alpha})`;
}

function srgbToLinear(value: number): number {
  const v = clampNumber(value, 0, 255) / 255;
  return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

export function contrastRatio(a: RgbColor, b: RgbColor): number {
  const luminanceA = 0.2126 * srgbToLinear(a.r) + 0.7152 * srgbToLinear(a.g) + 0.0722 * srgbToLinear(a.b);
  const luminanceB = 0.2126 * srgbToLinear(b.r) + 0.7152 * srgbToLinear(b.g) + 0.0722 * srgbToLinear(b.b);
  const lighter = Math.max(luminanceA, luminanceB);
  const darker = Math.min(luminanceA, luminanceB);
  return (lighter + 0.05) / (darker + 0.05);
}

export function deriveContrastColor(sourceColor: RgbColor): RgbColor {
  const hsl = rgbToHsl(sourceColor);
  const darkStage = { r: 3, g: 6, b: 12 };
  const candidates: RgbColor[] = [];
  for (const delta of [180, 150, 210, 120, 240, 300]) {
    for (const lightness of [0.42, 0.5, 0.58, 0.66, 0.74, 0.82]) {
      candidates.push(hslToRgb(
        hsl.h + delta,
        Math.max(0.72, Math.min(0.96, hsl.s + 0.28)),
        lightness,
      ));
    }
  }
  return candidates
    .map((color) => ({
      color,
      score:
        contrastRatio(color, sourceColor) * 0.78 +
        Math.min(contrastRatio(color, darkStage), 8) * 0.22 -
        Math.max(0, 4.5 - contrastRatio(color, darkStage)) * 2,
    }))
    .sort((a, b) => b.score - a.score)[0]?.color ?? hslToRgb(hsl.h + 180, 0.86, 0.66);
}

export function deriveTeacherPaletteFromClothing(sourceColor: RgbColor): BodyPalette {
  const stroke = deriveContrastColor(sourceColor);
  const hsl = rgbToHsl(stroke);
  const accent = hslToRgb(hsl.h, Math.min(1, hsl.s + 0.08), 0.84);
  const fill = hslToRgb(hsl.h, Math.max(0.62, hsl.s), 0.58);
  return {
    fill: rgbToCss(fill, 0.24),
    stroke: rgbToCss(stroke, 0.98),
    glow: rgbToCss(stroke, 0.58),
    accent: rgbToCss(accent, 1),
  };
}

export function blendRgbColor(previous: RgbColor | null, next: RgbColor, alpha = 0.28): RgbColor {
  if (!previous) return next;
  return {
    r: previous.r + (next.r - previous.r) * alpha,
    g: previous.g + (next.g - previous.g) * alpha,
    b: previous.b + (next.b - previous.b) * alpha,
  };
}

/** Pick palette by smoothed score, mirrors pose_viewer.html banding. */
export function paletteForScore(score: number | null): BodyPalette {
  if (score == null) return BODY_PALETTES.user;
  if (score >= 85) return BODY_PALETTES.userPerfect;
  if (score >= 60) return BODY_PALETTES.user;
  return BODY_PALETTES.miss;
}

/** Resize canvas backing store to match its CSS size × DPR. Returns true if changed. */
export function syncCanvasSize(canvas: HTMLCanvasElement): boolean {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssWidth = canvas.clientWidth || canvas.width;
  const cssHeight = canvas.clientHeight || canvas.height;
  const w = Math.max(1, Math.round(cssWidth * dpr));
  const h = Math.max(1, Math.round(cssHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

export function clearStage(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

export interface DrawSkeletonOptions {
  lineWidth?: number;
  dashed?: boolean;
  jointRadius?: number;
  /** Optional horizontal mirror — useful when video is rendered with scaleX(-1). */
  mirrorX?: boolean;
}

/** Draw a basic 33-point pose skeleton (lines + dots). */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  points: readonly (NormalizedLandmark | null | undefined)[],
  rect: StageRect,
  palette: BodyPalette,
  options: DrawSkeletonOptions = {},
): void {
  if (!points || points.length < 11) return;
  const lineWidth = options.lineWidth ?? Math.max(2, rect.width * 0.006);
  const jointRadius = options.jointRadius ?? Math.max(2, rect.width * 0.005);

  ctx.save();
  if (options.mirrorX) {
    ctx.translate(rect.x + rect.width, 0);
    ctx.scale(-1, 1);
    ctx.translate(-rect.x, 0);
  }
  ctx.strokeStyle = palette.stroke;
  ctx.fillStyle = palette.fill;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(options.dashed ? [8, 5] : []);
  if (palette.glow) {
    ctx.shadowColor = palette.glow;
    ctx.shadowBlur = Math.max(4, rect.width * 0.012);
  }

  for (const [a, b] of POSE_CONNECTIONS) {
    const p1 = points[a];
    const p2 = points[b];
    if (!p1 || !p2 || getVisibility(p1) < 0.2 || getVisibility(p2) < 0.2) continue;
    const m1 = mapPointToStage(p1, rect);
    const m2 = mapPointToStage(p2, rect);
    if (!m1 || !m2) continue;
    ctx.beginPath();
    ctx.moveTo(m1.x, m1.y);
    ctx.lineTo(m2.x, m2.y);
    ctx.stroke();
  }

  ctx.fillStyle = palette.stroke; // dots use the stroke colour for contrast
  for (const point of points) {
    if (!point || getVisibility(point) < 0.2) continue;
    const m = mapPointToStage(point, rect);
    if (!m) continue;
    ctx.beginPath();
    ctx.arc(m.x, m.y, jointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/**
 * Capsule-style "human body" — torso polygon + head ellipse + limb capsules.
 * Used for the teacher avatar so it reads as a translucent silhouette rather
 * than line-art. Visibility threshold matches pose_viewer.html.
 */
export function drawHumanBody(
  ctx: CanvasRenderingContext2D,
  points: readonly (NormalizedLandmark | null | undefined)[],
  rect: StageRect,
  palette: BodyPalette,
  options: { mirrorX?: boolean; showHead?: boolean } = {},
): void {
  if (!points || points.length < 29) return;
  const reference = getPoseReference(points);
  if (!reference) {
    drawSkeleton(ctx, points, rect, palette, { mirrorX: options.mirrorX });
    return;
  }
  const W = rect.width;
  const H = rect.height;
  const scalePx = (reference.scale || 0.22) * Math.min(W, H);
  const upperArmR = Math.max(4, scalePx * 0.11);
  const forearmR = Math.max(3, scalePx * 0.085);
  const thighR = Math.max(5, scalePx * 0.14);
  const calfR = Math.max(4, scalePx * 0.1);
  const handR = forearmR * 0.95;
  const footR = calfR * 0.9;
  const jointR = Math.max(3.5, scalePx * 0.075);
  const headBaseR = Math.max(12, scalePx * 0.22);
  const showHead = options.showHead ?? true;

  const vis = (p: NormalizedLandmark | null | undefined) => !!p && getVisibility(p) >= 0.25;
  const map = (p: NormalizedLandmark | null | undefined) => mapPointToStage(p, rect);

  ctx.save();
  if (options.mirrorX) {
    ctx.translate(rect.x + rect.width, 0);
    ctx.scale(-1, 1);
    ctx.translate(-rect.x, 0);
  }
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.fillStyle = palette.fill;
  ctx.strokeStyle = palette.stroke;
  ctx.lineWidth = Math.max(1.2, scalePx * 0.012);
  if (palette.glow) {
    ctx.shadowColor = palette.glow;
    ctx.shadowBlur = Math.max(4, scalePx * 0.08);
  }

  // Curved torso: shoulder/waist/hip contour instead of a straight polygon.
  const torso = [11, 12, 24, 23].map((i) => points[i]);
  if (torso.every(vis)) {
    const ls = map(points[11])!;
    const rs = map(points[12])!;
    const rh = map(points[24])!;
    const lh = map(points[23])!;
    const shoulderMid = midpoint(ls, rs)!;
    const hipMid = midpoint(lh, rh)!;
    const leftWaist = moveToward(lerpPoint(ls, lh, 0.58), lerpPoint(shoulderMid, hipMid, 0.58), 0.18);
    const rightWaist = moveToward(lerpPoint(rs, rh, 0.58), lerpPoint(shoulderMid, hipMid, 0.58), 0.18);
    ctx.beginPath();
    ctx.moveTo(ls.x, ls.y);
    ctx.quadraticCurveTo(shoulderMid.x, shoulderMid.y - scalePx * 0.06, rs.x, rs.y);
    ctx.quadraticCurveTo(rightWaist.x, rightWaist.y, rh.x, rh.y);
    ctx.quadraticCurveTo(hipMid.x, hipMid.y + scalePx * 0.05, lh.x, lh.y);
    ctx.quadraticCurveTo(leftWaist.x, leftWaist.y, ls.x, ls.y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // Head ellipse
  if (showHead && vis(points[0])) {
    const c = map(points[0])!;
    let rx = headBaseR;
    let ry = headBaseR * 1.2;
    let rot = 0;
    if (vis(points[7]) && vis(points[8])) {
      const dxE = (points[8]!.x - points[7]!.x) * W;
      const dyE = (points[8]!.y - points[7]!.y) * H;
      const earDist = Math.hypot(dxE, dyE);
      rot = Math.atan2(dyE, dxE);
      rx = Math.max(headBaseR * 0.8, earDist * 0.6);
      ry = rx * 1.25;
    }
    ctx.beginPath();
    ctx.ellipse(c.x, c.y, rx, ry, rot, 0, Math.PI * 2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // Organic limb contours with subtle bends and tapered ends.
  const limbs: Array<[number, number, number, number, number]> = [
    [11, 13, upperArmR * 1.08, upperArmR * 0.9, -0.035],
    [13, 15, forearmR * 1.04, forearmR * 0.82, 0.032],
    [12, 14, upperArmR * 1.08, upperArmR * 0.9, 0.035],
    [14, 16, forearmR * 1.04, forearmR * 0.82, -0.032],
    [23, 25, thighR * 1.08, thighR * 0.86, 0.022],
    [25, 27, calfR * 1.04, calfR * 0.78, -0.025],
    [24, 26, thighR * 1.08, thighR * 0.86, -0.022],
    [26, 28, calfR * 1.04, calfR * 0.78, 0.025],
  ];
  for (const [a, b, r1, r2, curve] of limbs) {
    if (vis(points[a]) && vis(points[b])) {
      drawOrganicLimb(ctx, map(points[a])!, map(points[b])!, r1, r2, curve);
    }
  }

  for (const idx of [11, 12, 13, 14, 23, 24, 25, 26]) {
    if (!vis(points[idx])) continue;
    const m = map(points[idx])!;
    drawJointOrb(ctx, m, idx === 13 || idx === 14 || idx === 25 || idx === 26 ? jointR : jointR * 1.12);
  }

  // Hand / foot caps
  for (const [idx, r] of [[15, handR], [16, handR], [27, footR], [28, footR]] as const) {
    const p = points[idx];
    if (!vis(p)) continue;
    const m = map(p)!;
    ctx.beginPath();
    ctx.arc(m.x, m.y, r, 0, Math.PI * 2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function midpoint(
  a: { x: number; y: number } | null | undefined,
  b: { x: number; y: number } | null | undefined,
): { x: number; y: number } | null {
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function lerpPoint(
  a: { x: number; y: number },
  b: { x: number; y: number },
  t: number,
): { x: number; y: number } {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

function moveToward(
  point: { x: number; y: number },
  target: { x: number; y: number },
  amount: number,
): { x: number; y: number } {
  return {
    x: point.x + (target.x - point.x) * amount,
    y: point.y + (target.y - point.y) * amount,
  };
}

function drawOrganicLimb(
  ctx: CanvasRenderingContext2D,
  p1: { x: number; y: number },
  p2: { x: number; y: number },
  startRadius: number,
  endRadius: number,
  curvature: number,
): void {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const len = Math.hypot(dx, dy);
  if (!len) return;
  const ux = dx / len;
  const uy = dy / len;
  const nx = -uy;
  const ny = ux;
  const maxRadius = Math.max(startRadius, endRadius);
  const control = {
    x: (p1.x + p2.x) / 2 + nx * len * curvature,
    y: (p1.y + p2.y) / 2 + ny * len * curvature,
  };

  ctx.beginPath();
  ctx.moveTo(p1.x + nx * startRadius, p1.y + ny * startRadius);
  ctx.quadraticCurveTo(
    control.x + nx * maxRadius * 0.82,
    control.y + ny * maxRadius * 0.82,
    p2.x + nx * endRadius,
    p2.y + ny * endRadius,
  );
  ctx.quadraticCurveTo(
    p2.x + ux * endRadius * 0.95,
    p2.y + uy * endRadius * 0.95,
    p2.x - nx * endRadius,
    p2.y - ny * endRadius,
  );
  ctx.quadraticCurveTo(
    control.x - nx * maxRadius * 0.82,
    control.y - ny * maxRadius * 0.82,
    p1.x - nx * startRadius,
    p1.y - ny * startRadius,
  );
  ctx.quadraticCurveTo(
    p1.x - ux * startRadius * 0.95,
    p1.y - uy * startRadius * 0.95,
    p1.x + nx * startRadius,
    p1.y + ny * startRadius,
  );
  ctx.closePath();
  ctx.fill();
  if (ctx.lineWidth > 0) ctx.stroke();
}

function drawJointOrb(
  ctx: CanvasRenderingContext2D,
  point: { x: number; y: number },
  radius: number,
): void {
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

/** Lightweight 21-point hand skeleton overlay. */
export function drawHandSkeleton(
  ctx: CanvasRenderingContext2D,
  hand: HandFrame,
  rect: StageRect,
  options: { mirrorX?: boolean; palette?: BodyPalette } = {},
): void {
  const palette = options.palette ?? HAND_PALETTES[hand.handedness] ?? HAND_PALETTES.default!;
  const points = hand.landmarks;
  if (!points || points.length < 21) return;

  ctx.save();
  if (options.mirrorX) {
    ctx.translate(rect.x + rect.width, 0);
    ctx.scale(-1, 1);
    ctx.translate(-rect.x, 0);
  }
  ctx.strokeStyle = palette.stroke;
  ctx.fillStyle = palette.fill;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = Math.max(1.5, rect.width * 0.004);

  for (const [a, b] of HAND_CONNECTIONS) {
    const p1 = points[a];
    const p2 = points[b];
    if (!p1 || !p2) continue;
    const m1 = mapPointToStage(p1, rect);
    const m2 = mapPointToStage(p2, rect);
    if (!m1 || !m2) continue;
    ctx.beginPath();
    ctx.moveTo(m1.x, m1.y);
    ctx.lineTo(m2.x, m2.y);
    ctx.stroke();
  }
  const dotR = Math.max(1.5, rect.width * 0.0035);
  for (const p of points) {
    if (!p) continue;
    const m = mapPointToStage(p, rect);
    if (!m) continue;
    ctx.beginPath();
    ctx.arc(m.x, m.y, dotR, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/**
 * EMA smoothing for landmark arrays — used for the *display* skeleton only;
 * matching code should keep using the raw frame to avoid lag.
 */
export function smoothLandmarks(
  prev: readonly (NormalizedLandmark | null | undefined)[] | null | undefined,
  next: readonly (NormalizedLandmark | null | undefined)[] | null | undefined,
  alpha = 0.45,
): NormalizedLandmark[] | null {
  if (!Array.isArray(next)) return null;
  if (!prev || prev.length !== next.length) {
    return next.map((p) => (p ? { ...(p as NormalizedLandmark) } : (null as unknown as NormalizedLandmark)));
  }
  const out: NormalizedLandmark[] = new Array(next.length) as NormalizedLandmark[];
  for (let i = 0; i < next.length; i += 1) {
    const p = next[i];
    const q = prev[i];
    if (!p && !q) { (out as any)[i] = null; continue; }
    if (!p) { out[i] = q as NormalizedLandmark; continue; }
    if (!q) { out[i] = { ...(p as NormalizedLandmark) }; continue; }
    const vp = getVisibility(p);
    const a = Math.max(0.18, Math.min(0.7, alpha * (0.4 + vp * 0.7)));
    out[i] = {
      x: q.x + (p.x - q.x) * a,
      y: q.y + (p.y - q.y) * a,
      z: ((q.z ?? 0) + (((p.z ?? 0)) - (q.z ?? 0)) * a) as number,
      visibility: q.visibility != null
        ? q.visibility * 0.55 + (p.visibility ?? q.visibility) * 0.45
        : (p.visibility ?? 1),
    } as NormalizedLandmark;
  }
  return out;
}

/**
 * Gradient + glow accent skeleton — drawn on top of the capsule body to add a
 * "tech" look and convey per-joint confidence.
 */
export function drawSkeletonAccents(
  ctx: CanvasRenderingContext2D,
  points: readonly (NormalizedLandmark | null | undefined)[],
  rect: StageRect,
  palette: BodyPalette,
): void {
  if (!points || points.length < 11) return;
  const baseW = Math.min(rect.width, rect.height) || 720;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = palette.glow ?? palette.stroke;
  ctx.shadowBlur = Math.max(8, baseW * 0.018);

  for (const [a, b] of POSE_CONNECTIONS) {
    const p1 = points[a]; const p2 = points[b];
    if (!p1 || !p2) continue;
    const v1 = getVisibility(p1); const v2 = getVisibility(p2);
    if (v1 < 0.2 || v2 < 0.2) continue;
    const m1 = mapPointToStage(p1, rect);
    const m2 = mapPointToStage(p2, rect);
    if (!m1 || !m2) continue;
    const conf = Math.min(v1, v2);
    const grad = ctx.createLinearGradient(m1.x, m1.y, m2.x, m2.y);
    grad.addColorStop(0, palette.stroke);
    grad.addColorStop(0.5, palette.accent ?? palette.stroke);
    grad.addColorStop(1, palette.stroke);
    ctx.strokeStyle = grad;
    ctx.lineWidth = Math.max(2, baseW * 0.0036) + conf * Math.max(1.5, baseW * 0.0024);
    ctx.globalAlpha = 0.35 + conf * 0.55;
    ctx.beginPath();
    ctx.moveTo(m1.x, m1.y);
    ctx.lineTo(m2.x, m2.y);
    ctx.stroke();
  }

  ctx.shadowBlur = 0;
  for (const p of points) {
    if (!p) continue;
    const v = getVisibility(p);
    if (v < 0.25) continue;
    const m = mapPointToStage(p, rect);
    if (!m) continue;
    const r = Math.max(2.5, baseW * 0.0045) + v * Math.max(1.5, baseW * 0.0028);
    const halo = ctx.createRadialGradient(m.x, m.y, 0, m.x, m.y, r * 3.4);
    halo.addColorStop(0, palette.glow ?? palette.stroke);
    halo.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = halo;
    ctx.globalAlpha = 0.42 * v;
    ctx.beginPath();
    ctx.arc(m.x, m.y, r * 3.4, 0, Math.PI * 2);
    ctx.fill();

    ctx.globalAlpha = 0.85 + 0.15 * v;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(m.x, m.y, r * 0.6, 0, Math.PI * 2);
    ctx.fill();

    ctx.globalAlpha = 0.55 + 0.4 * v;
    ctx.strokeStyle = palette.accent ?? palette.stroke;
    ctx.lineWidth = Math.max(1, baseW * 0.0014);
    ctx.beginPath();
    ctx.arc(m.x, m.y, r * 0.95, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Stage status banner kinds — surface model loading, camera authorization,
 * person-in-frame and degraded scoring through a single string-derived state.
 */
export type StageStatusKind = 'loading' | 'idle' | 'ok' | 'warn' | 'error';
export interface StageStatusInputs {
  poseReady: boolean;
  handsReady: boolean;
  mediapipeReady: boolean;
  cameraRunning: boolean;
  cameraError: string | null;
  personVisible: boolean;
  poseDataLoaded: boolean;
  matchDisabledReason: string | null;
  handsDisabledReason: string | null;
  isPlaying: boolean;
}
export interface StageStatusOutput { kind: StageStatusKind; title: string; hint: string; }

export function computeStageStatus(s: StageStatusInputs): StageStatusOutput {
  if (!s.mediapipeReady && !s.poseReady) {
    return { kind: 'loading', title: '识别模型加载中', hint: '首次进入需要加载 MediaPipe，仅几秒钟…' };
  }
  if (!s.poseReady) return { kind: 'loading', title: '初始化姿态识别', hint: '正在挂载 PoseLandmarker' };
  if (s.cameraError) return { kind: 'error', title: '摄像头无法打开', hint: `${s.cameraError} · 请检查浏览器授权` };
  if (!s.cameraRunning) {
    if (s.poseDataLoaded) return { kind: 'warn', title: '摄像头未开启', hint: '可在右下角切换到练习并开启摄像头' };
    return { kind: 'idle', title: '准备就绪', hint: '加载一个标准动作即可开始' };
  }
  if (!s.personVisible) return { kind: 'warn', title: '搜索人体中', hint: '请退后半步、保持上半身完全入镜' };
  if (s.matchDisabledReason) return { kind: 'warn', title: '评分已降级', hint: s.matchDisabledReason };
  if (s.handsDisabledReason) return { kind: 'warn', title: '手部识别已降级', hint: `${s.handsDisabledReason} · 仍按躯干评分` };
  return { kind: 'ok', title: '人像已锁定', hint: s.isPlaying ? '继续保持节拍' : '随时点击开始播放' };
}
