/**
 * Deep link parsing for `/pose_viewer.html?preset=<id>&autoplay=1`.
 *
 * Pure functions only — no window / navigator access. Callers pass in the
 * search string and the preset lookup so the module stays unit-testable.
 */

import type { PosePreset } from './data/presets.js';

export type DeepLinkView = 'setup' | 'practice' | 'result' | 'analysis' | 'motion';

export interface DeepLinkIntent {
  preset: PosePreset | null;
  autoplay: boolean;
  view: DeepLinkView | null;
  /** True when the URL carried a preset but we could not resolve it. */
  unresolvedPresetName: string | null;
}

export function parseDeepLink(
  searchString: string,
  lookup: (name: string) => PosePreset | undefined,
): DeepLinkIntent {
  const params = new URLSearchParams(searchString || '');
  const presetParam = (params.get('preset') || '').trim();
  const autoplay = params.get('autoplay') === '1';
  const view = parseView(params.get('view'));
  if (!presetParam) {
    return { preset: null, autoplay, view, unresolvedPresetName: null };
  }
  const preset = lookup(presetParam) ?? null;
  return {
    preset,
    autoplay,
    view,
    unresolvedPresetName: preset ? null : presetParam,
  };
}

function parseView(view: string | null): DeepLinkView | null {
  return view === 'setup' || view === 'practice' || view === 'result' || view === 'analysis' || view === 'motion'
    ? view
    : null;
}

export function buildDeepLink(origin: string, preset: PosePreset | null, autoplay: boolean): string {
  const url = new URL('/pose_viewer.html', origin);
  if (preset) url.searchParams.set('preset', preset.id);
  if (autoplay) url.searchParams.set('autoplay', '1');
  return url.toString();
}
