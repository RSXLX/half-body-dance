import type { HandFrame, NormalizedLandmark } from '../../core/types.js';
import type { StageRect } from '../stageRenderer.js';

export interface TeacherBasePalette {
  fill: string;
  stroke: string;
  glow?: string;
  accent?: string;
}

export interface TeacherAvatarPalette {
  resinFill: string;
  resinLight: string;
  resinShadow: string;
  shadowColor: string;
  highlightColor: string;
  jointStroke: string;
  teacherStroke: string;
  teacherGlow: string;
  teacherAccent: string;
}

export interface TeacherAvatarRenderInput {
  ctx: CanvasRenderingContext2D;
  poseLandmarks: readonly (NormalizedLandmark | null | undefined)[];
  hands?: readonly HandFrame[];
  stageRect: StageRect;
  palette: TeacherBasePalette;
  mirrorX?: boolean;
  highDetail?: boolean;
  showHead?: boolean;
}
