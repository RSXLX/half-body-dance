# MotionBreakdown MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a MotionBreakdown MVP in the new `web/` entry so preset and uploaded videos can show rule-based motion beats/segments and jump into practice from a segment.

**Architecture:** Add a small motion data layer, pure UI components, and a route-level `MotionBreakdownView`. Preset motion loads from `*_motion.json`; uploaded video motion is produced by calling `/api/analyze-motion` after `/api/extract-pose`; failures degrade to normal practice.

**Tech Stack:** Vite, TypeScript, DOM string renderers, Vitest/jsdom, existing Python `dev_server.py` APIs.

---

## File structure

- Modify: `web/src/core/motionTypes.ts` — align frontend types with current schema v2 fields from `analyze_motion.py`, especially optional `beats` and limb slices.
- Create: `web/src/core/motionData.ts` — motion guards, path loading, API call helpers, and safe segment start extraction.
- Test: `web/src/core/motionData.test.ts` — guard, preset load, upload analyze success/failure paths.
- Create: `web/src/ui/BeatStrip.ts` — pure beat strip renderer and time formatting helper.
- Test: `web/src/ui/BeatStrip.test.ts` — empty and populated beat rendering.
- Create: `web/src/ui/SegmentCard.ts` — pure segment card/list renderer and event binding.
- Test: `web/src/ui/SegmentCard.test.ts` — rendering and click callback.
- Create: `web/src/app/views/MotionBreakdownView.ts` — route view composed from `BeatStrip` and `SegmentCard`.
- Test: `web/src/app/views/MotionBreakdownView.test.ts` — ready/empty/error states and callback wiring.
- Modify: `web/src/app/views/SetupView.ts` — expose motion status and buttons for “动作分解” / “继续练习”.
- Test: `web/src/app/views/SetupView.test.ts` — motion entry appears only when motion is ready, normal CTA remains usable.
- Modify: `web/src/main.ts` — add `motion` route/state, preset motion loading, upload analyze call, and segment-to-practice jump.
- Modify/Create CSS: `web/src/styles/motion-breakdown.css` and import from `web/src/main.ts`.

---

## Task 1: Motion data guard and API helpers

**Files:**
- Modify: `web/src/core/motionTypes.ts`
- Create: `web/src/core/motionData.ts`
- Test: `web/src/core/motionData.test.ts`

- [ ] **Step 1: Write failing tests for motion guard and segment time extraction**

Create `web/src/core/motionData.test.ts` with:

```ts
import { describe, expect, it, vi } from 'vitest';
import {
  getSegmentStartTime,
  isMotionAnalysis,
  loadMotionFromPath,
  analyzeMotionFromPose,
} from './motionData.js';

const validMotion = {
  schema_version: 2,
  source_pose: 'wudao/angel_pose.json',
  fps: 30,
  duration: 12,
  extracted_at: '2026-05-11T00:00:00Z',
  extract_config: {},
  trajectories: {},
  limbs: {},
  beats: [
    {
      index: 0,
      startTime: 0,
      endTime: 0.5,
      duration: 0.5,
      segmentId: 'seg-1',
      primaryJoint: 'rightWrist',
      primaryDirection: 'up',
      emoji: '⬆️',
      label: '右手向上',
      jointStats: {
        rightWrist: {
          distance: 0.42,
          pathLength: 0.58,
          peakSpeed: 1.2,
          cardinal: 'up',
          visibility: 0.98,
          angleDeg: 88.4,
        },
      },
    },
  ],
  segments: [
    {
      id: 'seg-1',
      index: 0,
      startTime: 1.2,
      endTime: 2.4,
      duration: 1.2,
      emoji: '👋',
      title: '右手挥动',
      description: '右手向上挥动。',
      primaryJoints: ['rightWrist'],
      primaryDirection: 'up',
      difficulty: 1,
      keyFrames: [1.2],
      tips: ['手臂抬高'],
    },
  ],
  hints: [],
  summary: { primaryJoints: ['rightWrist'], dominantDirections: [], beatTimes: [0.25] },
};

describe('isMotionAnalysis', () => {
  it('accepts motion with beats and segments arrays', () => {
    expect(isMotionAnalysis(validMotion)).toBe(true);
  });

  it('rejects missing beats or segments arrays', () => {
    expect(isMotionAnalysis({ ...validMotion, beats: undefined })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, segments: undefined })).toBe(false);
    expect(isMotionAnalysis({ ...validMotion, beats: {} })).toBe(false);
  });
});

describe('getSegmentStartTime', () => {
  it('returns finite non-negative startTime', () => {
    expect(getSegmentStartTime(validMotion.segments[0])).toBe(1.2);
  });

  it('returns null for invalid segment startTime', () => {
    expect(getSegmentStartTime({ ...validMotion.segments[0], startTime: Number.NaN })).toBeNull();
    expect(getSegmentStartTime({ ...validMotion.segments[0], startTime: -1 })).toBeNull();
  });
});

describe('loadMotionFromPath', () => {
  it('returns motion when fetch succeeds with valid schema', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => validMotion });
    const motion = await loadMotionFromPath('wudao/angel_motion.json', fetchMock as typeof fetch);
    expect(motion).toEqual(validMotion);
  });

  it('returns null when fetch fails or schema is invalid', async () => {
    const notFound = vi.fn().mockResolvedValue({ ok: false, status: 404 });
    await expect(loadMotionFromPath('missing.json', notFound as typeof fetch)).resolves.toBeNull();

    const invalid = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ segments: [] }) });
    await expect(loadMotionFromPath('bad.json', invalid as typeof fetch)).resolves.toBeNull();
  });
});

describe('analyzeMotionFromPose', () => {
  it('posts poseJson and returns motion on success', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true, motion: validMotion }) });
    const motion = await analyzeMotionFromPose({ frames: [] }, fetchMock as typeof fetch);
    expect(fetchMock).toHaveBeenCalledWith('/api/analyze-motion', expect.objectContaining({ method: 'POST' }));
    expect(motion).toEqual(validMotion);
  });

  it('returns null when API reports failure', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: false, error: 'bad pose' }) });
    await expect(analyzeMotionFromPose({ frames: [] }, fetchMock as typeof fetch)).resolves.toBeNull();
  });
});
```

- [ ] **Step 2: Run the failing test**

Run: `npm --prefix web run test -- motionData`

Expected: FAIL because `web/src/core/motionData.ts` does not exist.

- [ ] **Step 3: Update motion types to accept schema v2 beats**

Modify `web/src/core/motionTypes.ts` by replacing the schema version and adding beat/limb interfaces. Keep existing interfaces, but add these definitions after `MotionHint` and update `MotionAnalysis`:

```ts
export const MOTION_SCHEMA_VERSION = 2;

export interface JointBeatStats {
  distance: number;
  pathLength?: number;
  peakSpeed: number;
  cardinal: Cardinal8;
  visibility: number;
  angleDeg?: number;
}

export interface BeatAction {
  index: number;
  startTime: number;
  endTime: number;
  duration: number;
  segmentId?: string;
  primaryJoint: JointId;
  primaryDirection: Cardinal8;
  emoji: string;
  label: string;
  jointStats: Partial<Record<JointId, JointBeatStats>>;
  visibilityWarning?: 'lowFoot' | 'lowHand' | string;
}

export interface LimbBeatSlice {
  beatIndex: number;
  startTime: number;
  endTime: number;
  cardinal: Cardinal8;
  emoji: string;
  distance: number;
  peakSpeed: number;
  armPose?: { elbowAngleDeg: number; wristHeightBand: 'overhead' | 'high' | 'mid' | 'low' | string };
  legPose?: { kneeBent: boolean; ankleSide: 'center' | 'outside' | 'inside' | string };
}

export interface MotionAnalysis {
  schema_version: number;
  source_pose: string;
  fps: number;
  duration: number;
  extracted_at: string;
  extract_config: Partial<MotionExtractConfig>;
  trajectories: Partial<Record<JointId, JointTrajectory>>;
  limbs?: Partial<Record<JointId, LimbBeatSlice[]>>;
  beats: BeatAction[];
  segments: MotionSegment[];
  hints: MotionHint[];
  summary: MotionSummary;
}
```

Ensure there is only one `MOTION_SCHEMA_VERSION` declaration and only one `MotionAnalysis` interface.

- [ ] **Step 4: Implement motionData helpers**

Create `web/src/core/motionData.ts`:

```ts
import type { MotionAnalysis, MotionSegment } from './motionTypes.js';
import type { PoseData } from './types.js';

export type MotionFetch = typeof fetch;

export function isMotionAnalysis(value: unknown): value is MotionAnalysis {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<MotionAnalysis>;
  return Array.isArray(candidate.beats) && Array.isArray(candidate.segments);
}

export function getSegmentStartTime(segment: Pick<MotionSegment, 'startTime'> | null | undefined): number | null {
  if (!segment || typeof segment.startTime !== 'number') return null;
  if (!Number.isFinite(segment.startTime) || segment.startTime < 0) return null;
  return segment.startTime;
}

export async function loadMotionFromPath(path: string, fetchImpl: MotionFetch = fetch): Promise<MotionAnalysis | null> {
  try {
    const response = await fetchImpl(path);
    if (!response.ok) return null;
    const json = await response.json();
    return isMotionAnalysis(json) ? json : null;
  } catch (err) {
    console.warn('[motionData] failed to load motion', err);
    return null;
  }
}

export async function analyzeMotionFromPose(poseJson: PoseData | Record<string, unknown>, fetchImpl: MotionFetch = fetch): Promise<MotionAnalysis | null> {
  try {
    const response = await fetchImpl('/api/analyze-motion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ poseJson }),
    });
    if (!response.ok) return null;
    const payload = await response.json();
    return payload?.ok && isMotionAnalysis(payload.motion) ? payload.motion : null;
  } catch (err) {
    console.warn('[motionData] failed to analyze motion', err);
    return null;
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npm --prefix web run test -- motionData`

Expected: PASS.

- [ ] **Step 6: Commit**

Do not commit unless the user explicitly asks for commits. If authorized, run:

```bash
git add web/src/core/motionTypes.ts web/src/core/motionData.ts web/src/core/motionData.test.ts
git commit -m "Add motion data helpers"
```

---

## Task 2: BeatStrip UI component

**Files:**
- Create: `web/src/ui/BeatStrip.ts`
- Test: `web/src/ui/BeatStrip.test.ts`

- [ ] **Step 1: Write failing BeatStrip tests**

Create `web/src/ui/BeatStrip.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { formatMotionTime, renderBeatStrip } from './BeatStrip.js';
import type { BeatAction } from '../core/motionTypes.js';

const beats: BeatAction[] = [
  {
    beatIndex: 0,
    startTime: 0,
    endTime: 0.5,
    timestamp: 0.25,
    primaryJoint: 'rightWrist',
    primaryDirection: 'up',
    emoji: '⬆️',
    label: '右手向上',
    preview: '下一拍：⬆️ 右手向上',
  },
  {
    beatIndex: 1,
    startTime: 0.5,
    endTime: 1,
    timestamp: 0.75,
    primaryJoint: 'leftWrist',
    primaryDirection: 'left',
    emoji: '⬅️',
    label: '左手向左',
    preview: '下一拍：⬅️ 左手向左',
  },
];

describe('formatMotionTime', () => {
  it('formats seconds as mm:ss.cs', () => {
    expect(formatMotionTime(0)).toBe('00:00.00');
    expect(formatMotionTime(65.34)).toBe('01:05.34');
  });
});

describe('renderBeatStrip', () => {
  it('renders an empty state without beats', () => {
    expect(renderBeatStrip([])).toContain('暂无拍点分解');
  });

  it('renders one item per beat with labels', () => {
    const html = renderBeatStrip(beats);
    expect((html.match(/class="motion-beat/g) ?? []).length).toBe(2);
    expect(html).toContain('右手向上');
    expect(html).toContain('左手向左');
  });
});
```

- [ ] **Step 2: Run failing test**

Run: `npm --prefix web run test -- BeatStrip`

Expected: FAIL because `BeatStrip.ts` does not exist.

- [ ] **Step 3: Implement BeatStrip**

Create `web/src/ui/BeatStrip.ts`:

```ts
import type { BeatAction } from '../core/motionTypes.js';

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function formatMotionTime(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${rest.toFixed(2).padStart(5, '0')}`;
}

export function renderBeatStrip(beats: readonly BeatAction[]): string {
  if (!beats.length) {
    return '<div class="motion-empty">暂无拍点分解。</div>';
  }
  return `
    <div class="motion-beat-strip" role="list" aria-label="动作拍点">
      ${beats
        .map(
          (beat) => `
            <div class="motion-beat motion-dir-${escapeHtml(beat.primaryDirection)}" role="listitem">
              <span class="motion-beat-time">${formatMotionTime(beat.timestamp)}</span>
              <strong class="motion-beat-emoji">${escapeHtml(beat.emoji || '•')}</strong>
              <span class="motion-beat-label">${escapeHtml(beat.label || beat.preview || '动作拍点')}</span>
            </div>
          `,
        )
        .join('')}
    </div>
  `;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm --prefix web run test -- BeatStrip`

Expected: PASS.

- [ ] **Step 5: Commit**

Do not commit unless the user explicitly asks for commits. If authorized, run:

```bash
git add web/src/ui/BeatStrip.ts web/src/ui/BeatStrip.test.ts
git commit -m "Add motion beat strip"
```

---

## Task 3: SegmentCard UI component

**Files:**
- Create: `web/src/ui/SegmentCard.ts`
- Test: `web/src/ui/SegmentCard.test.ts`

- [ ] **Step 1: Write failing SegmentCard tests**

Create `web/src/ui/SegmentCard.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { bindSegmentCards, renderSegmentCards } from './SegmentCard.js';
import type { MotionSegment } from '../core/motionTypes.js';

const segments: MotionSegment[] = [
  {
    id: 'seg-1',
    index: 0,
    startTime: 0,
    endTime: 1.5,
    duration: 1.5,
    emoji: '👋',
    title: '右手挥动',
    description: '右手向上挥动。',
    primaryJoints: ['rightWrist'],
    primaryDirection: 'up',
    difficulty: 1,
    keyFrames: [0.5],
    tips: ['手臂抬高'],
  },
];

describe('renderSegmentCards', () => {
  it('renders empty state', () => {
    expect(renderSegmentCards([])).toContain('暂无片段分解');
  });

  it('renders title, time, and tip', () => {
    const html = renderSegmentCards(segments);
    expect(html).toContain('右手挥动');
    expect(html).toContain('00:00.00 – 00:01.50');
    expect(html).toContain('手臂抬高');
  });
});

describe('bindSegmentCards', () => {
  it('calls back with clicked segment', () => {
    const root = document.createElement('div');
    root.innerHTML = renderSegmentCards(segments);
    let selected: MotionSegment | null = null;
    const unbind = bindSegmentCards(root, segments, (segment) => {
      selected = segment;
    });
    root.querySelector<HTMLButtonElement>('[data-segment-id="seg-1"]')!.click();
    unbind();
    expect(selected?.id).toBe('seg-1');
  });
});
```

- [ ] **Step 2: Run failing test**

Run: `npm --prefix web run test -- SegmentCard`

Expected: FAIL because `SegmentCard.ts` does not exist.

- [ ] **Step 3: Implement SegmentCard**

Create `web/src/ui/SegmentCard.ts`:

```ts
import type { MotionSegment } from '../core/motionTypes.js';
import { formatMotionTime } from './BeatStrip.js';

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function segmentTip(segment: MotionSegment): string {
  return segment.tips?.[0] || segment.description || '跟随节奏完成这一段动作。';
}

export function renderSegmentCards(segments: readonly MotionSegment[], activeSegmentId?: string | null): string {
  if (!segments.length) return '<div class="motion-empty">暂无片段分解。</div>';
  return `
    <div class="motion-segment-list" role="list" aria-label="动作片段">
      ${segments
        .map((segment) => {
          const active = activeSegmentId === segment.id ? ' is-active' : '';
          return `
            <button class="motion-segment-card${active}" type="button" data-segment-id="${escapeHtml(segment.id)}" role="listitem">
              <span class="motion-segment-emoji">${escapeHtml(segment.emoji || '🎬')}</span>
              <span class="motion-segment-main">
                <strong>${escapeHtml(segment.title || `第 ${segment.index + 1} 段`)}</strong>
                <small>${formatMotionTime(segment.startTime)} – ${formatMotionTime(segment.endTime)} · 难度 ${segment.difficulty}</small>
                <em>${escapeHtml(segmentTip(segment))}</em>
              </span>
            </button>
          `;
        })
        .join('')}
    </div>
  `;
}

export function bindSegmentCards(
  root: HTMLElement,
  segments: readonly MotionSegment[],
  onSelectSegment: (segment: MotionSegment) => void,
): () => void {
  const segmentMap = new Map(segments.map((segment) => [segment.id, segment] as const));
  const onClick = (event: Event) => {
    const target = event.target as HTMLElement | null;
    const button = target?.closest<HTMLButtonElement>('[data-segment-id]');
    if (!button?.dataset.segmentId) return;
    const segment = segmentMap.get(button.dataset.segmentId);
    if (segment) onSelectSegment(segment);
  };
  root.addEventListener('click', onClick);
  return () => root.removeEventListener('click', onClick);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm --prefix web run test -- SegmentCard`

Expected: PASS.

- [ ] **Step 5: Commit**

Do not commit unless the user explicitly asks for commits. If authorized, run:

```bash
git add web/src/ui/SegmentCard.ts web/src/ui/SegmentCard.test.ts
git commit -m "Add motion segment cards"
```

---

## Task 4: MotionBreakdownView

**Files:**
- Create: `web/src/app/views/MotionBreakdownView.ts`
- Test: `web/src/app/views/MotionBreakdownView.test.ts`

- [ ] **Step 1: Write failing MotionBreakdownView tests**

Create `web/src/app/views/MotionBreakdownView.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { bindMotionBreakdownView, renderMotionBreakdownView, type MotionBreakdownViewState } from './MotionBreakdownView.js';
import type { MotionAnalysis, MotionSegment } from '../../core/motionTypes.js';

const segment: MotionSegment = {
  id: 'seg-1',
  index: 0,
  startTime: 1,
  endTime: 2,
  duration: 1,
  emoji: '👋',
  title: '右手挥动',
  description: '右手向上挥动。',
  primaryJoints: ['rightWrist'],
  primaryDirection: 'up',
  difficulty: 1,
  keyFrames: [1.2],
  tips: ['手臂抬高'],
};

const motion: MotionAnalysis = {
  schema_version: 2,
  source_pose: 'wudao/angel_pose.json',
  fps: 30,
  duration: 8,
  extracted_at: '2026-05-11T00:00:00Z',
  extract_config: {},
  trajectories: {},
  limbs: {},
  beats: [
    {
      beatIndex: 0,
      startTime: 1,
      endTime: 2,
      timestamp: 1.5,
      primaryJoint: 'rightWrist',
      primaryDirection: 'up',
      emoji: '⬆️',
      label: '右手向上',
      preview: '下一拍：⬆️ 右手向上',
    },
  ],
  segments: [segment],
  hints: [],
  summary: { primaryJoints: ['rightWrist'], dominantDirections: [] },
};

function state(overrides: Partial<MotionBreakdownViewState> = {}): MotionBreakdownViewState {
  return {
    presetName: 'Angel',
    status: 'ready',
    errorMessage: null,
    motion,
    activeSegmentId: null,
    ...overrides,
  };
}

describe('renderMotionBreakdownView', () => {
  it('renders ready state with beat strip and segment card', () => {
    const html = renderMotionBreakdownView(state());
    expect(html).toContain('Angel · 动作分解');
    expect(html).toContain('右手向上');
    expect(html).toContain('右手挥动');
  });

  it('renders unavailable state', () => {
    const html = renderMotionBreakdownView(state({ status: 'empty', motion: null }));
    expect(html).toContain('动作分解暂不可用');
  });
});

describe('bindMotionBreakdownView', () => {
  it('wires back, practice, and segment callbacks', () => {
    const root = document.createElement('div');
    root.innerHTML = renderMotionBreakdownView(state());
    const calls: string[] = [];
    const unbind = bindMotionBreakdownView(root, state(), {
      onBack: () => calls.push('back'),
      onPractice: () => calls.push('practice'),
      onSelectSegment: (s) => calls.push(s.id),
    });
    root.querySelector<HTMLButtonElement>('#motionBack')!.click();
    root.querySelector<HTMLButtonElement>('#motionStartPractice')!.click();
    root.querySelector<HTMLButtonElement>('[data-segment-id="seg-1"]')!.click();
    unbind();
    expect(calls).toEqual(['back', 'practice', 'seg-1']);
  });
});
```

- [ ] **Step 2: Run failing test**

Run: `npm --prefix web run test -- MotionBreakdownView`

Expected: FAIL because `MotionBreakdownView.ts` does not exist.

- [ ] **Step 3: Implement MotionBreakdownView**

Create `web/src/app/views/MotionBreakdownView.ts`:

```ts
import type { MotionAnalysis, MotionSegment } from '../../core/motionTypes.js';
import { renderBeatStrip } from '../../ui/BeatStrip.js';
import { bindSegmentCards, renderSegmentCards } from '../../ui/SegmentCard.js';

export type MotionBreakdownStatus = 'idle' | 'loading' | 'ready' | 'empty' | 'error';

export interface MotionBreakdownViewState {
  presetName: string;
  status: MotionBreakdownStatus;
  errorMessage: string | null;
  motion: MotionAnalysis | null;
  activeSegmentId: string | null;
}

export interface MotionBreakdownCallbacks {
  onBack: () => void;
  onPractice: () => void;
  onSelectSegment: (segment: MotionSegment) => void;
}

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderHeader(state: MotionBreakdownViewState): string {
  return `
    <header class="motion-header">
      <button id="motionBack" class="ghost" type="button" aria-label="返回">←</button>
      <div class="motion-heading">
        <strong>${escapeHtml(state.presetName || '当前动作')} · 动作分解</strong>
        <span>先看每拍和片段，再进入跟练。</span>
      </div>
      <button id="motionStartPractice" class="primary" type="button">进入练习</button>
    </header>
  `;
}

export function renderMotionBreakdownView(state: MotionBreakdownViewState): string {
  if (state.status === 'loading') {
    return `
      <section class="motion-view" data-view="motion">
        ${renderHeader(state)}
        <div class="motion-panel motion-empty">正在生成动作分解…</div>
      </section>
    `;
  }

  if (state.status === 'error' || state.status === 'empty' || !state.motion) {
    const message = state.errorMessage || '动作分解暂不可用，可继续普通练习。';
    return `
      <section class="motion-view" data-view="motion">
        ${renderHeader(state)}
        <div class="motion-panel motion-empty">${escapeHtml(message)}</div>
      </section>
    `;
  }

  return `
    <section class="motion-view" data-view="motion">
      ${renderHeader(state)}
      <div class="motion-body">
        <article class="motion-panel motion-beats-panel">
          <h2>按拍提示</h2>
          ${renderBeatStrip(state.motion.beats)}
        </article>
        <article class="motion-panel motion-segments-panel">
          <h2>动作片段</h2>
          ${renderSegmentCards(state.motion.segments, state.activeSegmentId)}
        </article>
      </div>
    </section>
  `;
}

export function bindMotionBreakdownView(
  root: HTMLElement,
  state: MotionBreakdownViewState,
  callbacks: MotionBreakdownCallbacks,
): () => void {
  const onClick = (event: Event) => {
    const target = event.target as HTMLElement | null;
    if (target?.closest('#motionBack')) callbacks.onBack();
    else if (target?.closest('#motionStartPractice')) callbacks.onPractice();
  };
  root.addEventListener('click', onClick);
  const unbindSegments = state.motion
    ? bindSegmentCards(root, state.motion.segments, callbacks.onSelectSegment)
    : () => {};
  return () => {
    root.removeEventListener('click', onClick);
    unbindSegments();
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm --prefix web run test -- MotionBreakdownView`

Expected: PASS.

- [ ] **Step 5: Commit**

Do not commit unless the user explicitly asks for commits. If authorized, run:

```bash
git add web/src/app/views/MotionBreakdownView.ts web/src/app/views/MotionBreakdownView.test.ts
git commit -m "Add motion breakdown view"
```

---

## Task 5: SetupView motion entry

**Files:**
- Modify: `web/src/app/views/SetupView.ts`
- Test: `web/src/app/views/SetupView.test.ts`

- [ ] **Step 1: Extend SetupView tests first**

Append to `web/src/app/views/SetupView.test.ts`:

```ts
describe('motion breakdown entry', () => {
  it('shows motion entry when motion is ready', () => {
    const html = renderSetupView(baseState({ poseDataReady: true, motionReady: true }));
    expect(html).toContain('id="enterMotionBreakdown"');
    expect(html).toContain('先看动作分解');
  });

  it('does not show motion entry before motion is ready', () => {
    const html = renderSetupView(baseState({ poseDataReady: true, motionReady: false }));
    expect(html).not.toContain('enterMotionBreakdown');
  });
});
```

Then update `baseState` in the same file to include:

```ts
motionReady: false,
motionLoading: false,
motionError: null,
```

- [ ] **Step 2: Run failing SetupView test**

Run: `npm --prefix web run test -- SetupView`

Expected: FAIL because `SetupViewState` lacks motion fields and renderer lacks button.

- [ ] **Step 3: Update SetupView state and callbacks**

Modify `web/src/app/views/SetupView.ts`:

Add to `SetupViewState`:

```ts
  /** True when companion motion analysis is ready. */
  motionReady: boolean;
  /** True while companion motion analysis is loading. */
  motionLoading: boolean;
  /** Motion analysis failure message; normal practice remains usable. */
  motionError: string | null;
```

Add to `SetupViewCallbacks`:

```ts
  onEnterMotionBreakdown: () => void;
```

- [ ] **Step 4: Render motion entry in CTA**

Replace `renderCTA` in `web/src/app/views/SetupView.ts` with:

```ts
export function renderCTA(state: SetupViewState): string {
  const disabled = state.poseDataReady ? '' : 'disabled';
  const ready = state.poseDataReady;
  const titleText = state.cameraRunning ? '动作与摄像头已就绪' : '动作已就绪';
  const copyText = state.cameraRunning
    ? '点击开始练习会直接进入全屏舞台并自动播放。'
    : '点击开始练习会尝试打开摄像头；允许权限后自动进入练习。';
  const motionEntry = state.motionReady
    ? '<button id="enterMotionBreakdown" class="ghost" type="button">先看动作分解</button>'
    : state.motionLoading
      ? '<span class="setup-cta-copy">正在生成动作分解…</span>'
      : state.motionError
        ? `<span class="setup-cta-copy">${escapeHtml(state.motionError)}</span>`
        : '';
  return `
    <div class="setup-cta">
      <button id="enterPractice" class="primary" type="button" ${disabled}>${ctaLabel(state)}</button>
      ${motionEntry}
      ${ready ? `<span class="setup-cta-title">${escapeHtml(titleText)}</span>` : ''}
      ${ready ? `<span class="setup-cta-copy">${escapeHtml(copyText)}</span>` : ''}
    </div>
  `;
}
```

- [ ] **Step 5: Wire motion button**

In `bindSetupEvents`, add before `#enterPractice` handling:

```ts
    if (target?.closest('#enterMotionBreakdown')) {
      callbacks.onEnterMotionBreakdown();
      return;
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `npm --prefix web run test -- SetupView`

Expected: PASS.

- [ ] **Step 7: Commit**

Do not commit unless the user explicitly asks for commits. If authorized, run:

```bash
git add web/src/app/views/SetupView.ts web/src/app/views/SetupView.test.ts
git commit -m "Expose motion breakdown entry"
```

---

## Task 6: Main route and data integration

**Files:**
- Modify: `web/src/main.ts`
- Modify: `web/src/styles/motion-breakdown.css`
- Test: run existing integration-relevant tests, no direct full main test currently exists.

- [ ] **Step 1: Import motion view, helpers, and CSS**

Modify the top of `web/src/main.ts`:

```ts
import './styles/motion-breakdown.css';
```

Add imports:

```ts
import {
  bindMotionBreakdownView,
  renderMotionBreakdownView,
  type MotionBreakdownViewState,
} from './app/views/MotionBreakdownView.js';
import {
  analyzeMotionFromPose,
  getSegmentStartTime,
  loadMotionFromPath,
} from './core/motionData.js';
import type { MotionSegment } from './core/motionTypes.js';
```

If `MotionAnalysis` is already imported from `./core/motionTypes.js`, combine the type imports into one import.

- [ ] **Step 2: Extend view and AppState**

Change:

```ts
type View = 'setup' | 'practice' | 'result' | 'analysis';
```

to:

```ts
type View = 'setup' | 'practice' | 'result' | 'analysis' | 'motion';
```

Add to `AppState`:

```ts
  motion: MotionAnalysis | null;
  motionStatus: 'idle' | 'loading' | 'ready' | 'empty' | 'error';
  motionError: string | null;
  activeMotionSegmentId: string | null;
  practiceStartTime: number;
```

Add to initial `state`:

```ts
  motion: null,
  motionStatus: 'idle',
  motionError: null,
  activeMotionSegmentId: null,
  practiceStartTime: 0,
```

- [ ] **Step 3: Update setupViewState and mount bindings**

In `setupViewState()`, return:

```ts
    motionReady: state.motionStatus === 'ready',
    motionLoading: state.motionStatus === 'loading',
    motionError: state.motionStatus === 'error' ? state.motionError : null,
```

In setup branch of `mount()`, update `bindSetupEvents` callbacks:

```ts
      onEnterMotionBreakdown: () => navigate('motion'),
```

- [ ] **Step 4: Add motionViewState and mount branch**

Add function near `resultViewState()`:

```ts
function motionViewState(): MotionBreakdownViewState {
  return {
    presetName: selectedPresetName() || '上传动作',
    status: state.motionStatus,
    errorMessage: state.motionError,
    motion: state.motion,
    activeSegmentId: state.activeMotionSegmentId,
  };
}
```

Add a `motion` branch in `mount()` before the final `else` analysis branch:

```ts
  } else if (state.view === 'motion') {
    root!.innerHTML = renderMotionBreakdownView(motionViewState());
    unbind = bindMotionBreakdownView(root!, motionViewState(), {
      onBack: () => navigate('setup'),
      onPractice: () => void handleEnterPractice(),
      onSelectSegment: handleMotionSegmentSelect,
    });
    stopPracticeLoop();
```

- [ ] **Step 5: Load preset motion alongside pose**

In `handlePresetClick`, before `mount()` after clearing pose, add:

```ts
  state.motion = null;
  state.motionStatus = 'loading';
  state.motionError = null;
  state.activeMotionSegmentId = null;
  state.practiceStartTime = 0;
```

Replace the existing fetch chain with a pose+motion `Promise.allSettled` flow:

```ts
  const posePromise = fetch(preset.path)
    .then(async (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return (await res.json()) as PoseData;
    });
  const motionPromise = loadMotionFromPath(motionPathForPreset(preset));

  return Promise.allSettled([posePromise, motionPromise])
    .then(([poseResult, motionResult]) => {
      if (poseResult.status === 'fulfilled') {
        const json = poseResult.value;
        state.poseData = json;
        state.poseDataReady = Array.isArray(json.frames) && json.frames.length > 0;
      } else {
        console.warn('[main] preset load failed', poseResult.reason);
        state.poseData = null;
        state.poseDataReady = false;
      }

      if (motionResult.status === 'fulfilled' && motionResult.value) {
        state.motion = motionResult.value;
        state.motionStatus = 'ready';
        state.motionError = null;
      } else {
        state.motion = null;
        state.motionStatus = 'empty';
        state.motionError = null;
      }
    })
    .finally(() => {
      state.loadingPresetId = null;
      if (options.autoplay && state.poseDataReady) {
        void handleEnterPractice();
      } else {
        mount();
      }
    });
```

- [ ] **Step 6: Add segment-to-practice handling**

Add below `handleEnterPractice()`:

```ts
function handleMotionSegmentSelect(segment: MotionSegment): void {
  const startTime = getSegmentStartTime(segment);
  if (startTime === null) return;
  state.activeMotionSegmentId = segment.id;
  state.practiceStartTime = startTime;
  void handleEnterPractice();
}
```

Modify `handleEnterPractice()` after `navigate('practice')` to preserve the start time; no extra code is needed here if `practiceStartTime` is consumed by playback initialization.

Modify `handleTogglePlayback()` and `handleReplay()`:

```ts
    state.currentFrameIndex = frameIndexAtTime(state.practiceStartTime);
```

instead of `0`.

Add helper near `getCurrentTargetFrame()`:

```ts
function frameIndexAtTime(seconds: number): number {
  const frames = state.poseData?.frames ?? [];
  if (!frames.length || !Number.isFinite(seconds) || seconds <= 0) return 0;
  let best = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < frames.length; i++) {
    const distance = Math.abs((frames[i]!.time ?? 0) - seconds);
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}
```

Modify `getCurrentTargetFrame()` elapsed calculation:

```ts
  const elapsed = state.practiceStartTime + (performance.now() - state.playbackStartedAt) / 1000;
```

Reset `state.practiceStartTime = 0` in `handlePresetClick` and any “change action/back to setup” path where a new action starts.

- [ ] **Step 7: Add upload analyze hook**

Find the existing upload handler in `web/src/main.ts` if present. If it does not exist in the new entry, skip this step and add a note to the final summary that upload UI is not yet present in `web/`.

If there is an upload handler that receives extracted `PoseData`, add after pose success:

```ts
  state.motionStatus = 'loading';
  state.motionError = null;
  const motion = await analyzeMotionFromPose(state.poseData);
  if (motion) {
    state.motion = motion;
    state.motionStatus = 'ready';
  } else {
    state.motion = null;
    state.motionStatus = 'error';
    state.motionError = '动作已识别，但分解生成失败，可继续练习。';
  }
```

- [ ] **Step 8: Create MotionBreakdown CSS**

Create `web/src/styles/motion-breakdown.css`:

```css
.motion-view {
  min-height: 100vh;
  padding: 18px;
  color: #f6f8ff;
  background: radial-gradient(circle at top, rgba(92, 140, 255, 0.2), transparent 38%), #050814;
}

.motion-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 auto 16px;
  max-width: 1120px;
}

.motion-heading {
  display: grid;
  gap: 4px;
  flex: 1;
}

.motion-heading strong {
  font-size: clamp(18px, 3vw, 26px);
}

.motion-heading span,
.motion-beat-time,
.motion-segment-card small,
.motion-segment-card em {
  color: rgba(246, 248, 255, 0.68);
}

.motion-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 0.8fr);
  gap: 16px;
  max-width: 1120px;
  margin: 0 auto;
}

.motion-panel {
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 24px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.08);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22);
}

.motion-beat-strip {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
}

.motion-beat,
.motion-segment-card {
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 18px;
  background: rgba(8, 14, 28, 0.72);
}

.motion-beat {
  display: grid;
  gap: 6px;
  padding: 12px;
}

.motion-beat-emoji {
  font-size: 24px;
}

.motion-segment-list {
  display: grid;
  gap: 10px;
}

.motion-segment-card {
  display: flex;
  width: 100%;
  gap: 12px;
  padding: 12px;
  color: inherit;
  text-align: left;
  cursor: pointer;
}

.motion-segment-card.is-active,
.motion-segment-card:hover {
  border-color: rgba(117, 191, 255, 0.74);
  background: rgba(41, 95, 178, 0.28);
}

.motion-segment-emoji {
  font-size: 28px;
}

.motion-segment-main {
  display: grid;
  gap: 4px;
}

.motion-empty {
  padding: 28px;
  text-align: center;
}

@media (max-width: 760px) {
  .motion-header,
  .motion-body {
    grid-template-columns: 1fr;
  }

  .motion-header {
    align-items: stretch;
    flex-wrap: wrap;
  }
}
```

- [ ] **Step 9: Run focused tests**

Run:

```bash
npm --prefix web run test -- MotionBreakdownView SetupView motionData BeatStrip SegmentCard
```

Expected: PASS.

- [ ] **Step 10: Commit**

Do not commit unless the user explicitly asks for commits. If authorized, run:

```bash
git add web/src/main.ts web/src/styles/motion-breakdown.css web/src/app/views/SetupView.ts web/src/app/views/SetupView.test.ts
git commit -m "Wire motion breakdown flow"
```

---

## Task 7: Full verification and current-state notes

**Files:**
- Modify only if verification reveals a direct bug from previous tasks.

- [ ] **Step 1: Run web tests**

Run: `npm run web:test`

Expected: all tests PASS.

- [ ] **Step 2: Run web build**

Run: `npm run web:build`

Expected: TypeScript build and Vite build PASS.

- [ ] **Step 3: Run Python motion smoke test**

Run:

```bash
python3 scripts/analysis/analyze_motion.py wudao/angel_pose.json --output /tmp/angel_motion_test.json
```

Expected: command exits 0 and writes `/tmp/angel_motion_test.json`.

- [ ] **Step 4: Manual UI smoke test**

Run servers in separate terminals:

```bash
npm run dev
npm run web:dev
```

Open `http://127.0.0.1:4300`.

Expected preset path:
- Select “Angel” or another preset with a `*_motion.json` in `wudao/`.
- See “先看动作分解”.
- Open MotionBreakdownView.
- See beat strip and segment cards.
- Click a segment.
- PracticeView opens and starts near that segment’s time when playback begins.

Expected upload path:
- If upload UI exists in `web/`, upload a short video.
- Pose extraction succeeds.
- Motion analysis succeeds or fails gracefully.
- If upload UI does not exist in `web/`, record this as not implemented in the final status and do not claim upload flow is complete.

- [ ] **Step 5: Final status**

Report:
- Tests/build commands run and their result.
- Whether preset path works.
- Whether upload path is wired or blocked by missing upload UI in `web/`.
- Any remaining follow-up tasks.

- [ ] **Step 6: Commit**

Do not commit unless the user explicitly asks for commits. If authorized, run:

```bash
git status --short
git add web/src docs/superpowers/specs/2026-05-11-motionbreakdown-mvp-design.md docs/superpowers/plans/2026-05-11-motionbreakdown-mvp.md
git commit -m "Add motion breakdown MVP"
```

---

## Self-review

- Spec coverage: The plan covers MotionBreakdownView, preset motion loading, upload analyze hook when upload UI exists, segment-to-practice jump, graceful failures, and tests/build verification.
- Scope check: The plan intentionally excludes LLM, Web Audio beat detection, JointLane, KeyframeStrip, and legacy entry removal.
- Placeholder scan: No implementation step uses TBD/TODO-style placeholders. The upload integration includes an explicit branch for the current codebase condition where `web/` may not yet have upload UI.
- Type consistency: `MotionAnalysis`, `BeatAction`, `MotionSegment`, and route names are used consistently across tasks.
