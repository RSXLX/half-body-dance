// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import {
  bindPracticeEvents,
  renderPracticeView,
  togglePlaybackLabel,
  type PracticeViewState,
} from './PracticeView.js';

function state(overrides: Partial<PracticeViewState> = {}): PracticeViewState {
  return {
    poseDataReady: false,
    cameraRunning: false,
    isPlaying: false,
    presetName: '鸿门旋律',
    scoreText: '--',
    armSummaryText: '左臂 -- / 右臂 --',
    stageLabel: '准备开始',
    stageSubLabel: '先选择一个标准动作',
    sheetExpanded: false,
    recordingState: 'idle',
    recordingAvailable: false,
    debugPalette: null,
    ...overrides,
  };
}

describe('togglePlaybackLabel', () => {
  it('nudges the user when no pose data loaded', () => {
    expect(togglePlaybackLabel(state())).toBe('未加载标准动作');
  });
  it('shows "开始播放" when paused', () => {
    expect(togglePlaybackLabel(state({ poseDataReady: true }))).toBe('开始播放');
  });
  it('shows "暂停" when playing', () => {
    expect(togglePlaybackLabel(state({ poseDataReady: true, isPlaying: true }))).toBe('暂停');
  });
});

describe('renderPracticeView', () => {
  it('exposes the expected DOM anchors', () => {
    const html = renderPracticeView(state({ poseDataReady: true }));
    ['cameraVideo', 'teacherWebglCanvas', 'overlayCanvas', 'bgmAudio', 'scoreDialValue', 'togglePlayback', 'practiceSheetToggle', 'backToSetup'].forEach((id) => {
      expect(html).toContain(`id="${id}"`);
    });
  });
  it('disables toggle and replay until pose data loads', () => {
    const html = renderPracticeView(state());
    expect(html).toMatch(/id="togglePlayback"[^>]*disabled/);
    expect(html).toMatch(/id="practiceReplay"[^>]*disabled/);
  });
  it('hides expanded sheet content when collapsed', () => {
    expect(renderPracticeView(state())).toContain('hidden');
  });
  it('keeps the collapsed mobile sheet compact', () => {
    const html = renderPracticeView(state({ poseDataReady: true }));
    expect(html).toContain('sheet-mini-actions');
    expect(html).toContain('class="sheet-expand-content" hidden');
  });
  it('renders debug palette swatches only when debug data is supplied', () => {
    expect(renderPracticeView(state())).not.toContain('teacher-debug-palette');
    const html = renderPracticeView(state({
      debugPalette: {
        clothingCss: 'rgb(20, 30, 40)',
        teacherCss: 'rgba(200, 220, 240, 0.98)',
      },
    }));
    expect(html).toContain('teacher-debug-palette');
    expect(html).toContain('rgb(20, 30, 40)');
    expect(html).toContain('rgba(200, 220, 240, 0.98)');
  });
  it('exposes recording controls for the new entry', () => {
    const html = renderPracticeView(state({ poseDataReady: true, recordingAvailable: true }));
    expect(html).toContain('id="togglePracticeRecording"');
    expect(html).toContain('id="openPracticeRecording"');
  });
  it('escapes preset name into header', () => {
    const html = renderPracticeView(state({ presetName: '<script>' }));
    expect(html).toContain('&lt;script&gt;');
    expect(html).not.toContain('<script>');
  });
});

describe('bindPracticeEvents', () => {
  it('routes click targets to the right callbacks', () => {
    const root = document.createElement('div');
    root.innerHTML = renderPracticeView(state({
      poseDataReady: true,
      isPlaying: true,
      sheetExpanded: true,
      recordingAvailable: true,
    }));
    const callbacks = {
      onTogglePlayback: vi.fn(),
      onBackToSetup: vi.fn(),
      onSheetToggle: vi.fn(),
      onReplay: vi.fn(),
      onCameraAction: vi.fn(),
      onOpenResult: vi.fn(),
      onToggleRecording: vi.fn(),
      onOpenRecording: vi.fn(),
    };
    const unbind = bindPracticeEvents(root, callbacks);
    const click = (id: string) => {
      root.querySelector<HTMLElement>(`#${id}`)!.click();
    };
    click('togglePlayback');
    click('backToSetup');
    click('practiceSheetToggle');
    click('practiceReplay');
    click('practiceCameraAction');
    click('openResult');
    click('togglePracticeRecording');
    click('openPracticeRecording');
    unbind();

    expect(callbacks.onTogglePlayback).toHaveBeenCalled();
    expect(callbacks.onBackToSetup).toHaveBeenCalled();
    expect(callbacks.onSheetToggle).toHaveBeenCalled();
    expect(callbacks.onReplay).toHaveBeenCalled();
    expect(callbacks.onCameraAction).toHaveBeenCalled();
    expect(callbacks.onOpenResult).toHaveBeenCalled();
    expect(callbacks.onToggleRecording).toHaveBeenCalled();
    expect(callbacks.onOpenRecording).toHaveBeenCalled();
  });
});
