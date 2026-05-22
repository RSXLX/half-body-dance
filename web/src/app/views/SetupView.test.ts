import { describe, expect, it, vi } from 'vitest';
import { BUNDLED_POSE_PRESETS } from '../data/presets.js';
import {
  bindSetupEvents,
  ctaLabel,
  renderCTA,
  renderPresetGrid,
  renderSetupView,
  type SetupViewState,
} from './SetupView.js';

function baseState(overrides: Partial<SetupViewState> = {}): SetupViewState {
  return {
    presets: BUNDLED_POSE_PRESETS,
    selectedPresetId: null,
    loadingPresetId: null,
    cameraRunning: false,
    poseDataReady: false,
    motionReady: false,
    motionLoading: false,
    motionError: null,
    ...overrides,
  };
}

describe('renderPresetGrid', () => {
  it('emits one button per preset', () => {
    const html = renderPresetGrid(baseState());
    const cardCount = (html.match(/class="preset-card[^"]*"/g) ?? []).length;
    expect(cardCount).toBe(BUNDLED_POSE_PRESETS.length);
  });

  it('shows the recommended badge only for featured presets', () => {
    const html = renderPresetGrid(baseState());
    const badgeCount = (html.match(/preset-badge/g) ?? []).length;
    const featured = BUNDLED_POSE_PRESETS.filter((p) => p.featured).length;
    expect(badgeCount).toBe(featured);
  });

  it('marks the selected preset with is-selected', () => {
    const html = renderPresetGrid(baseState({ selectedPresetId: 'angel' }));
    expect(html).toMatch(/data-preset-id="angel"[^>]*aria-pressed="true"/);
  });

  it('marks the loading preset with is-loading', () => {
    const html = renderPresetGrid(baseState({ loadingPresetId: 'shoushi' }));
    expect(html).toMatch(/class="[^"]*is-loading[^"]*"\s+data-preset-id="shoushi"/);
  });
});

describe('ctaLabel', () => {
  it('asks for a preset when none chosen', () => {
    expect(ctaLabel(baseState())).toBe('先选择一个标准动作');
  });

  it('hints camera will auto-open when ready but camera off', () => {
    expect(ctaLabel(baseState({ poseDataReady: true }))).toBe('开始练习（自动开启摄像头）');
  });

  it('drops the camera hint once camera is running', () => {
    expect(ctaLabel(baseState({ poseDataReady: true, cameraRunning: true }))).toBe('开始练习');
  });
});

describe('renderCTA', () => {
  it('disables the button until pose data is ready', () => {
    expect(renderCTA(baseState())).toContain('disabled');
    expect(renderCTA(baseState({ poseDataReady: true }))).not.toContain(' disabled');
  });

  it('keeps practice available while showing motion errors', () => {
    const html = renderCTA(baseState({ poseDataReady: true, motionError: '动作分解失败' }));
    expect(html).toContain('动作分解失败');
    expect(html).toContain('id="enterPractice"');
    expect(html).not.toContain('id="enterPractice" class="primary" type="button" disabled');
  });
});

describe('renderSetupView', () => {
  it('contains both the grid and the CTA', () => {
    const html = renderSetupView(baseState({ poseDataReady: true }));
    expect(html).toContain('id="presetGrid"');
    expect(html).toContain('id="enterPractice"');
  });

  it('shows the motion breakdown entry when pose and motion data are ready', () => {
    const html = renderSetupView(baseState({ poseDataReady: true, motionReady: true }));
    expect(html).toContain('id="enterMotionBreakdown"');
    expect(html).toContain('先看动作分解');
  });

  it('hides the motion breakdown entry until motion data is ready', () => {
    const html = renderSetupView(baseState({ poseDataReady: true, motionReady: false }));
    expect(html).not.toContain('id="enterMotionBreakdown"');
  });

  it('shows motion generation progress', () => {
    const html = renderSetupView(baseState({ motionLoading: true }));
    expect(html).toContain('正在生成动作分解…');
  });

  it('escapes dynamic preset fields and motion errors', () => {
    const html = renderSetupView(baseState({
      presets: [{
        id: 'unsafe-preset',
        name: '动作 <script>alert("name")</script> & more',
        emoji: '<script>alert("emoji")</script>&',
        tagline: '描述 <script>alert("tagline")</script> & more',
        path: 'wudao/unsafe_pose.json',
        label: 'unsafe_pose.json',
      }],
      poseDataReady: true,
      motionError: '错误 <script>alert("motion")</script> & more',
    }));

    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;script&gt;alert(&quot;name&quot;)&lt;/script&gt; &amp; more');
    expect(html).toContain('&lt;script&gt;alert(&quot;emoji&quot;)&lt;/script&gt;&amp;');
    expect(html).toContain('&lt;script&gt;alert(&quot;tagline&quot;)&lt;/script&gt; &amp; more');
    expect(html).toContain('&lt;script&gt;alert(&quot;motion&quot;)&lt;/script&gt; &amp; more');
  });
});

describe('bindSetupEvents', () => {
  it('calls onEnterMotionBreakdown when clicking the motion breakdown entry', () => {
    let clickHandler: EventListener | undefined;
    const root = {
      addEventListener: (_event: string, handler: EventListener) => {
        clickHandler = handler;
      },
      removeEventListener: vi.fn(),
    } as unknown as HTMLElement;
    const onEnterMotionBreakdown = vi.fn();

    const dispose = bindSetupEvents(root, baseState({ poseDataReady: true, motionReady: true }), {
      onPresetClick: vi.fn(),
      onEnterPractice: vi.fn(),
      onEnterMotionBreakdown,
    });

    if (!clickHandler) throw new Error('click handler was not registered');
    clickHandler({
      target: {
        closest: (selector: string) => selector === '#enterMotionBreakdown' ? {} : null,
      },
    } as unknown as Event);

    expect(onEnterMotionBreakdown).toHaveBeenCalledTimes(1);
    dispose();
  });
});
