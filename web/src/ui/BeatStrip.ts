import type { BeatAction } from '../core/motionTypes.js';

export type { BeatAction } from '../core/motionTypes.js';

export function formatMotionTime(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const totalCentiseconds = Math.round(safeSeconds * 100);
  const minutes = Math.floor(totalCentiseconds / 6000);
  const secondsPart = Math.floor((totalCentiseconds % 6000) / 100);
  const centiseconds = totalCentiseconds % 100;

  return `${pad2(minutes)}:${pad2(secondsPart)}.${pad2(centiseconds)}`;
}

export function renderBeatStrip(beats: readonly BeatAction[]): string {
  if (!beats.length) {
    return '<div class="motion-empty">暂无拍点分解。</div>';
  }

  const items = beats
    .map((beat) => {
      const directionClass = safeClassToken(beat.primaryDirection);
      const timeLabel = formatBeatTime(beat);
      return `
        <div class="motion-beat motion-dir-${directionClass}" role="listitem" data-beat-index="${escapeAttr(String(beat.index))}">
          <span class="motion-beat-time">${escapeText(timeLabel)}</span>
          <span class="motion-beat-emoji" aria-hidden="true">${escapeText(beat.emoji)}</span>
          <span class="motion-beat-label">${escapeText(beat.label)}</span>
        </div>`;
    })
    .join('');

  return `<div class="motion-beat-strip" role="list">${items}</div>`;
}

function formatBeatTime(beat: BeatAction): string {
  if (Number.isFinite(beat.endTime) && beat.endTime > beat.startTime) {
    return `${formatMotionTime(beat.startTime)}–${formatMotionTime(beat.endTime)}`;
  }
  return formatMotionTime(beat.startTime);
}

function pad2(value: number): string {
  return String(value).padStart(2, '0');
}

function safeClassToken(input: string): string {
  return input.replace(/[^a-zA-Z0-9_-]/g, '');
}

function escapeText(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function escapeAttr(input: string): string {
  return escapeText(input).replace(/"/g, '&quot;');
}
