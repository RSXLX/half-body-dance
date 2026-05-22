/**
 * Types shared between Python extraction output and the browser scoring loop.
 * Source of truth: extract_pose.py emitted JSON.
 *
 * Phase 1 keeps these loose and permissive; Phase 2 will tighten them once we
 * move more of the legacy pose_viewer.html logic into TypeScript modules.
 */

export interface NormalizedLandmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

export interface HandFrame {
  handedness: 'Left' | 'Right' | string;
  landmarks: NormalizedLandmark[]; // expected length 21
  world_landmarks?: NormalizedLandmark[];
  finger_count?: number;
}

export interface PoseFrame {
  time: number;
  pose_landmarks: NormalizedLandmark[];
  hands?: HandFrame[];
}

export interface PoseData {
  fps: number;
  frames: PoseFrame[];
  aspect_ratio?: number;
  video_width?: number;
  video_height?: number;
  audio_source?: string;
  extract_config?: Record<string, unknown>;
  postprocess?: Record<string, unknown>;
  stats?: Record<string, unknown>;
  quality_report?: Record<string, unknown>;
}

/**
 * Reference frame derived from shoulder/hip midpoints, used to reproject the
 * teacher skeleton onto the current user's body (see pose_viewer.html:
 * getPoseReference + transformPointsWithReferences).
 */
export interface PoseReference {
  center: { x: number; y: number };
  scale: number;
  xAxis: { x: number; y: number };
  yAxis: { x: number; y: number };
}
