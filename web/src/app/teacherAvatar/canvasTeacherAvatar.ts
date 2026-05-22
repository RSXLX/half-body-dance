import type { HandFrame, NormalizedLandmark } from '../../core/types.js';
import { getVisibility } from '../../core/geometry.js';
import { getPoseReference } from '../../core/reference.js';
import { mapPointToStage, type StageRect } from '../stageRenderer.js';
import { buildTeacherAvatarPalette } from './avatarPalette.js';
import {
  getTeacherAvatarMetrics,
  getVisibleTeacherBodyParts,
} from './poseParts.js';
import type { TeacherAvatarPalette, TeacherAvatarRenderInput, TeacherBasePalette } from './types.js';

const HAND_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

interface Vec2Point {
  x: number;
  y: number;
}

function midpoint(a: Vec2Point, b: Vec2Point): Vec2Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function lerpPoint(a: Vec2Point, b: Vec2Point, t: number): Vec2Point {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

function moveToward(point: Vec2Point, target: Vec2Point, amount: number): Vec2Point {
  return {
    x: point.x + (target.x - point.x) * amount,
    y: point.y + (target.y - point.y) * amount,
  };
}

function withMirror(
  ctx: CanvasRenderingContext2D,
  rect: StageRect,
  mirrorX: boolean,
  draw: () => void,
): void {
  ctx.save();
  if (mirrorX) {
    ctx.translate(rect.x + rect.width, 0);
    ctx.scale(-1, 1);
    ctx.translate(-rect.x, 0);
  }
  draw();
  ctx.restore();
}

function traceOrganicLimbPath(
  ctx: CanvasRenderingContext2D,
  p1: Vec2Point,
  p2: Vec2Point,
  startRadius: number,
  endRadius: number,
  curvature: number,
): { control: Vec2Point; nx: number; ny: number; len: number } | null {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const len = Math.hypot(dx, dy);
  if (!len) return null;
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
  return { control, nx, ny, len };
}

function strokeCurve(
  ctx: CanvasRenderingContext2D,
  p1: Vec2Point,
  p2: Vec2Point,
  control: Vec2Point,
  offsetX: number,
  offsetY: number,
): void {
  ctx.beginPath();
  ctx.moveTo(p1.x + offsetX, p1.y + offsetY);
  ctx.quadraticCurveTo(control.x + offsetX, control.y + offsetY, p2.x + offsetX, p2.y + offsetY);
  ctx.stroke();
}

function drawOrganicLimb(
  ctx: CanvasRenderingContext2D,
  p1: Vec2Point,
  p2: Vec2Point,
  metric: { startRadius: number; endRadius: number; curvature: number },
  palette: TeacherAvatarPalette,
  highDetail: boolean,
): void {
  const shape = traceOrganicLimbPath(ctx, p1, p2, metric.startRadius, metric.endRadius, metric.curvature);
  if (!shape) return;

  const fill = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
  fill.addColorStop(0, palette.resinLight);
  fill.addColorStop(0.52, palette.resinFill);
  fill.addColorStop(1, palette.resinShadow);
  ctx.fillStyle = fill;
  ctx.strokeStyle = palette.shadowColor;
  ctx.lineWidth = Math.max(1, Math.min(metric.startRadius, metric.endRadius) * 0.18);
  ctx.fill();
  ctx.stroke();

  if (highDetail) {
    const side = Math.max(1, Math.min(metric.startRadius, metric.endRadius) * 0.42);
    ctx.lineCap = 'round';
    ctx.strokeStyle = palette.shadowColor;
    ctx.lineWidth = Math.max(1, side * 0.72);
    strokeCurve(ctx, p1, p2, shape.control, -shape.nx * side, -shape.ny * side);

    ctx.strokeStyle = palette.highlightColor;
    ctx.lineWidth = Math.max(1, side * 0.48);
    strokeCurve(ctx, p1, p2, shape.control, shape.nx * side * 0.9, shape.ny * side * 0.9);
  }

  ctx.strokeStyle = palette.teacherStroke;
  ctx.lineWidth = Math.max(1.15, Math.min(metric.startRadius, metric.endRadius) * 0.2);
  ctx.globalAlpha = 0.72;
  strokeCurve(ctx, p1, p2, shape.control, 0, 0);
  ctx.globalAlpha = 1;
}

function drawJointSeam(
  ctx: CanvasRenderingContext2D,
  point: Vec2Point,
  radius: number,
  palette: TeacherAvatarPalette,
  axis = 0,
): void {
  ctx.save();
  ctx.translate(point.x, point.y);
  ctx.rotate(axis);
  ctx.strokeStyle = palette.jointStroke;
  ctx.lineWidth = Math.max(1, radius * 0.12);
  ctx.beginPath();
  ctx.ellipse(0, 0, radius * 0.72, radius * 0.34, 0, 0.12 * Math.PI, 0.88 * Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(0, 0, radius * 0.72, radius * 0.34, 0, 1.12 * Math.PI, 1.88 * Math.PI);
  ctx.stroke();
  ctx.restore();
}

function drawJointOrb(
  ctx: CanvasRenderingContext2D,
  point: Vec2Point,
  radius: number,
  palette: TeacherAvatarPalette,
  highDetail: boolean,
): void {
  const fill = ctx.createRadialGradient(
    point.x - radius * 0.32,
    point.y - radius * 0.38,
    radius * 0.1,
    point.x,
    point.y,
    radius * 1.14,
  );
  fill.addColorStop(0, palette.highlightColor);
  fill.addColorStop(0.38, palette.resinLight);
  fill.addColorStop(0.72, palette.resinFill);
  fill.addColorStop(1, palette.resinShadow);

  ctx.fillStyle = fill;
  ctx.strokeStyle = palette.jointStroke;
  ctx.lineWidth = Math.max(1, radius * 0.15);
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  if (highDetail) drawJointSeam(ctx, point, radius, palette);

  ctx.strokeStyle = palette.teacherStroke;
  ctx.lineWidth = Math.max(1, radius * 0.12);
  ctx.globalAlpha = 0.84;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius * 1.08, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawTorsoBlock(
  ctx: CanvasRenderingContext2D,
  points: Vec2Point[],
  palette: TeacherAvatarPalette,
  stroke = true,
): void {
  if (points.length < 4) return;
  const top = midpoint(points[0]!, points[1]!);
  const bottom = midpoint(points[2]!, points[3]!);
  const grad = ctx.createLinearGradient(top.x, top.y, bottom.x, bottom.y);
  grad.addColorStop(0, palette.resinLight);
  grad.addColorStop(0.55, palette.resinFill);
  grad.addColorStop(1, palette.resinShadow);
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(points[0]!.x, points[0]!.y);
  ctx.quadraticCurveTo(top.x, top.y - Math.abs(points[1]!.x - points[0]!.x) * 0.16, points[1]!.x, points[1]!.y);
  ctx.quadraticCurveTo(points[2]!.x, points[2]!.y, points[2]!.x, points[2]!.y);
  ctx.quadraticCurveTo(bottom.x, bottom.y + Math.abs(points[2]!.x - points[3]!.x) * 0.12, points[3]!.x, points[3]!.y);
  ctx.quadraticCurveTo(points[0]!.x, points[0]!.y, points[0]!.x, points[0]!.y);
  ctx.closePath();
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = palette.jointStroke;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
}

function drawSegmentedTorso(
  ctx: CanvasRenderingContext2D,
  points: readonly (NormalizedLandmark | null | undefined)[],
  rect: StageRect,
  palette: TeacherAvatarPalette,
  scalePx: number,
  highDetail: boolean,
): void {
  const map = (index: number) => mapPointToStage(points[index], rect);
  const ls = map(11);
  const rs = map(12);
  const lh = map(23);
  const rh = map(24);
  if (!ls || !rs || !lh || !rh) return;

  const shoulderMid = midpoint(ls, rs);
  const hipMid = midpoint(lh, rh);
  const leftWaist = moveToward(lerpPoint(ls, lh, 0.58), lerpPoint(shoulderMid, hipMid, 0.58), 0.22);
  const rightWaist = moveToward(lerpPoint(rs, rh, 0.58), lerpPoint(shoulderMid, hipMid, 0.58), 0.22);
  const leftChest = lerpPoint(ls, leftWaist, 0.52);
  const rightChest = lerpPoint(rs, rightWaist, 0.52);
  const leftPelvis = lerpPoint(leftWaist, lh, 0.58);
  const rightPelvis = lerpPoint(rightWaist, rh, 0.58);

  drawTorsoBlock(ctx, [ls, rs, rightChest, leftChest], palette, highDetail);
  drawTorsoBlock(ctx, [leftChest, rightChest, rightPelvis, leftPelvis], palette, highDetail);
  drawTorsoBlock(ctx, [leftPelvis, rightPelvis, rh, lh], palette, highDetail);

  if (highDetail) {
    ctx.strokeStyle = palette.shadowColor;
    ctx.lineWidth = Math.max(1, scalePx * 0.01);
    ctx.beginPath();
    ctx.moveTo(leftChest.x, leftChest.y);
    ctx.quadraticCurveTo(shoulderMid.x, shoulderMid.y + scalePx * 0.52, rightChest.x, rightChest.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(leftPelvis.x, leftPelvis.y);
    ctx.quadraticCurveTo(hipMid.x, hipMid.y - scalePx * 0.18, rightPelvis.x, rightPelvis.y);
    ctx.stroke();
  }

  ctx.strokeStyle = palette.teacherStroke;
  ctx.lineWidth = Math.max(1.2, scalePx * 0.012);
  ctx.globalAlpha = 0.78;
  ctx.beginPath();
  ctx.moveTo(ls.x, ls.y);
  ctx.quadraticCurveTo(shoulderMid.x, shoulderMid.y - scalePx * 0.06, rs.x, rs.y);
  ctx.quadraticCurveTo(rightWaist.x, rightWaist.y, rh.x, rh.y);
  ctx.quadraticCurveTo(hipMid.x, hipMid.y + scalePx * 0.05, lh.x, lh.y);
  ctx.quadraticCurveTo(leftWaist.x, leftWaist.y, ls.x, ls.y);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawHead(
  ctx: CanvasRenderingContext2D,
  points: readonly (NormalizedLandmark | null | undefined)[],
  rect: StageRect,
  palette: TeacherAvatarPalette,
  baseRadius: number,
): void {
  if (!points[0] || getVisibility(points[0]) < 0.25) return;
  const center = mapPointToStage(points[0], rect);
  if (!center) return;
  let rx = baseRadius;
  let ry = baseRadius * 1.2;
  let rot = 0;
  if (points[7] && points[8] && getVisibility(points[7]) >= 0.25 && getVisibility(points[8]) >= 0.25) {
    const dx = (points[8].x - points[7].x) * rect.width;
    const dy = (points[8].y - points[7].y) * rect.height;
    const earDist = Math.hypot(dx, dy);
    rot = Math.atan2(dy, dx);
    rx = Math.max(baseRadius * 0.8, earDist * 0.6);
    ry = rx * 1.25;
  }
  const fill = ctx.createRadialGradient(center.x - rx * 0.25, center.y - ry * 0.3, rx * 0.15, center.x, center.y, ry);
  fill.addColorStop(0, palette.highlightColor);
  fill.addColorStop(0.44, palette.resinLight);
  fill.addColorStop(1, palette.resinShadow);
  ctx.fillStyle = fill;
  ctx.strokeStyle = palette.jointStroke;
  ctx.lineWidth = Math.max(1, baseRadius * 0.12);
  ctx.beginPath();
  ctx.ellipse(center.x, center.y, rx, ry, rot, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawTeacherGuideAccents(
  ctx: CanvasRenderingContext2D,
  points: readonly (NormalizedLandmark | null | undefined)[],
  rect: StageRect,
  palette: TeacherAvatarPalette,
): void {
  const visibleParts = getVisibleTeacherBodyParts(points, 0.25);
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = palette.teacherGlow;
  ctx.shadowBlur = Math.max(4, Math.min(rect.width, rect.height) * 0.008);
  ctx.strokeStyle = palette.teacherStroke;
  ctx.globalAlpha = 0.38;
  for (const part of visibleParts) {
    const a = mapPointToStage(points[part.from], rect);
    const b = mapPointToStage(points[part.to], rect);
    if (!a || !b) continue;
    const width = Math.max(1.2, Math.min(rect.width, rect.height) * 0.003);
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawEndpoint(
  ctx: CanvasRenderingContext2D,
  point: Vec2Point,
  radius: number,
  palette: TeacherAvatarPalette,
): void {
  ctx.fillStyle = palette.resinFill;
  ctx.strokeStyle = palette.teacherStroke;
  ctx.lineWidth = Math.max(1, radius * 0.14);
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

export function drawTeacherAvatarBody(
  ctx: CanvasRenderingContext2D,
  points: readonly (NormalizedLandmark | null | undefined)[],
  rect: StageRect,
  basePalette: TeacherBasePalette,
  options: { mirrorX?: boolean; highDetail?: boolean; showHead?: boolean } = {},
): void {
  if (!points || points.length < 29) return;
  const palette = buildTeacherAvatarPalette(basePalette);
  const metrics = getTeacherAvatarMetrics(points, rect);
  const highDetail = options.highDetail ?? Math.min(rect.width, rect.height) >= 420;
  const showHead = options.showHead ?? false;

  withMirror(ctx, rect, Boolean(options.mirrorX), () => {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (basePalette.glow) {
      ctx.shadowColor = basePalette.glow;
      ctx.shadowBlur = Math.max(4, metrics.scalePx * 0.06);
    }

    if (getPoseReference(points)) {
      drawSegmentedTorso(ctx, points, rect, palette, metrics.scalePx, highDetail);
    }

    for (const part of getVisibleTeacherBodyParts(points)) {
      const start = mapPointToStage(points[part.from], rect);
      const end = mapPointToStage(points[part.to], rect);
      if (!start || !end) continue;
      drawOrganicLimb(ctx, start, end, metrics.limbs[part.id], palette, highDetail);
    }

    const jointRadii: Partial<Record<number, number>> = {
      11: metrics.joints.shoulder,
      12: metrics.joints.shoulder,
      13: metrics.joints.elbow,
      14: metrics.joints.elbow,
      15: metrics.joints.wrist,
      16: metrics.joints.wrist,
      23: metrics.joints.hip,
      24: metrics.joints.hip,
      25: metrics.joints.knee,
      26: metrics.joints.knee,
      27: metrics.joints.ankle,
      28: metrics.joints.ankle,
    };
    for (const [indexText, radius] of Object.entries(jointRadii)) {
      const index = Number(indexText);
      const point = points[index];
      if (!point || getVisibility(point) < 0.25 || !radius) continue;
      const mapped = mapPointToStage(point, rect);
      if (!mapped) continue;
      drawJointOrb(ctx, mapped, radius, palette, highDetail);
    }

    for (const [index, radius] of [[15, metrics.extremities.hand], [16, metrics.extremities.hand], [27, metrics.extremities.foot], [28, metrics.extremities.foot]] as const) {
      const point = points[index];
      if (!point || getVisibility(point) < 0.25) continue;
      const mapped = mapPointToStage(point, rect);
      if (mapped) drawEndpoint(ctx, mapped, radius, palette);
    }

    if (showHead) drawHead(ctx, points, rect, palette, metrics.head.baseRadius);
    drawTeacherGuideAccents(ctx, points, rect, palette);
  });
}

export function drawTeacherAvatarHands(
  ctx: CanvasRenderingContext2D,
  hands: readonly HandFrame[] | null | undefined,
  rect: StageRect,
  basePalette: TeacherBasePalette,
  options: { mirrorX?: boolean } = {},
): void {
  if (!hands?.length) return;
  const palette = buildTeacherAvatarPalette(basePalette);
  withMirror(ctx, rect, Boolean(options.mirrorX), () => {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = palette.teacherStroke;
    ctx.fillStyle = palette.teacherAccent;
    ctx.shadowColor = palette.teacherGlow;
    ctx.shadowBlur = Math.max(3, rect.width * 0.006);
    ctx.lineWidth = Math.max(1.5, rect.width * 0.004);

    for (const hand of hands) {
      const landmarks = hand.landmarks;
      if (!landmarks || landmarks.length < 21) continue;
      for (const [a, b] of HAND_CONNECTIONS) {
        const p1 = mapPointToStage(landmarks[a], rect);
        const p2 = mapPointToStage(landmarks[b], rect);
        if (!p1 || !p2) continue;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
      const dotR = Math.max(1.5, rect.width * 0.0035);
      for (const point of landmarks) {
        const mapped = mapPointToStage(point, rect);
        if (!mapped) continue;
        ctx.beginPath();
        ctx.arc(mapped.x, mapped.y, dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });
}

export function drawTeacherAvatar(input: TeacherAvatarRenderInput): void {
  drawTeacherAvatarBody(input.ctx, input.poseLandmarks, input.stageRect, input.palette, {
    mirrorX: input.mirrorX,
    highDetail: input.highDetail,
    showHead: input.showHead,
  });
  drawTeacherAvatarHands(input.ctx, input.hands, input.stageRect, input.palette, {
    mirrorX: input.mirrorX,
  });
}
