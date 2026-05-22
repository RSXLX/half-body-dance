// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';

describe('camera start ownership', () => {
  it('marks an earlier camera start stale after a newer start begins', async () => {
    document.body.innerHTML = '<div id="app"></div>';

    const { createCameraStartGate } = await import('./main.js');
    const gate = createCameraStartGate();

    const firstStart = gate.begin();
    const secondStart = gate.begin();

    expect(gate.isLatest(firstStart)).toBe(false);
    expect(gate.isLatest(secondStart)).toBe(true);
  });
});
