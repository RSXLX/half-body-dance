/**
 * Bundled standard-action library.
 *
 * Single source of truth for the new web/ workspace. Mirrors the array in
 * pose_viewer.html (BUNDLED_POSE_PRESETS) — keep both in sync until Phase 2
 * fully replaces the legacy entry, after which this file becomes authoritative
 * and the legacy HTML can be deleted (see docs/refactor-roadmap.md).
 */

export interface PosePreset {
  /** Stable URL-friendly id, used in `?preset=...` deep links. */
  id: string;
  /** Display name. */
  name: string;
  /** Lead emoji shown on the card. */
  emoji: string;
  /** One-line description shown under the title. */
  tagline: string;
  /** Path served by dev_server.py / Vite proxy. */
  path: string;
  /** Filename label, used by status messages and history. */
  label: string;
  /** Highlighted with a "推荐" badge when true. */
  featured?: boolean;
  /**
   * Path to the companion `*_motion.json` (analyze_motion.py output).
   * Defaults to deriving from `path`; override only when the file lives
   * elsewhere. UI gracefully degrades when fetching this 404s.
   */
  motionPath?: string;
}

export const BUNDLED_POSE_PRESETS: readonly PosePreset[] = [
  { id: 'hongmen',    name: '鸿门旋律',  emoji: '🥁', tagline: '节拍鲜明，可同步背景音乐',     path: 'wudao/鸿门旋律_pose.json',   label: '鸿门旋律_pose.json',   featured: true },
  { id: 'timoteam',   name: '提莫队长',  emoji: '🎖️', tagline: '动作切换清晰，看对拍衔接',      path: 'wudao/提莫队长_pose.json',   label: '提莫队长_pose.json' },
  { id: 'shoushi',    name: '手势舞',    emoji: '🫶', tagline: '手势集中，看手部和节奏提示',    path: 'wudao/shoushi_pose.json',    label: 'shoushi_pose.json' },
  { id: 'angel',      name: 'Angel',     emoji: '👼', tagline: '半身舞默认示例，节奏稳',         path: 'wudao/angel_pose.json',      label: 'angel_pose.json' },
  { id: 'ladada',     name: '啦哒哒',    emoji: '🎵', tagline: '上身律动，热身友好',           path: 'wudao/ladada_pose.json',     label: 'ladada_pose.json' },
  { id: 'xingqiyao',  name: '星奇摇 2.0', emoji: '✨', tagline: '幅度较大，适合体验评分',       path: 'wudao/星奇摇2.0_pose.json',  label: '星奇摇2.0_pose.json' },
  { id: 'migaoyao',   name: '米糕摇',    emoji: '🍡', tagline: '小臂摆动，看左右臂分项',       path: 'wudao/米糕摇_pose.json',     label: '米糕摇_pose.json' },
  { id: 'huajiliao',  name: '花寂寥',    emoji: '🌸', tagline: '动作轻缓，跟练门槛低',         path: 'wudao/花寂寥_pose.json',     label: '花寂寥_pose.json' },
  { id: 'mofa',       name: '魔法城堡',  emoji: '🏰', tagline: '段落分明，看节拍命中',         path: 'wudao/魔法城堡_pose.json',   label: '魔法城堡_pose.json' },
];

export const DEFAULT_PRESET = BUNDLED_POSE_PRESETS[0];

export function findPresetById(id: string | null | undefined): PosePreset | undefined {
  if (!id) return undefined;
  return BUNDLED_POSE_PRESETS.find((p) => p.id === id || p.name === id || p.label === id);
}

export function findPresetByLabel(label: string | null | undefined): PosePreset | undefined {
  if (!label) return undefined;
  return BUNDLED_POSE_PRESETS.find((p) => p.label === label);
}

/**
 * Derive the companion motion JSON path: `wudao/foo_pose.json` → `wudao/foo_motion.json`.
 * Honors a preset-level override when present.
 */
export function motionPathForPreset(preset: PosePreset): string {
  if (preset.motionPath) return preset.motionPath;
  if (preset.path.endsWith('_pose.json')) {
    return preset.path.slice(0, -'_pose.json'.length) + '_motion.json';
  }
  // Fallback: append before the extension.
  const dot = preset.path.lastIndexOf('.');
  if (dot < 0) return `${preset.path}_motion.json`;
  return `${preset.path.slice(0, dot)}_motion.json`;
}
