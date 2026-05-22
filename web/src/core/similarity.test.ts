import { describe, expect, it } from 'vitest';
import { cosineSimilarity } from './similarity.js';

describe('cosineSimilarity', () => {
  it('maps identical vectors to 1', () => {
    expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1);
  });

  it('maps opposite vectors to 0', () => {
    expect(cosineSimilarity([1, 0, 0], [-1, 0, 0])).toBeCloseTo(0);
  });

  it('maps orthogonal vectors to 0.5', () => {
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBeCloseTo(0.5);
  });

  it('returns 0 on zero-length input', () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0);
  });

  it('returns 0 on mismatched lengths', () => {
    expect(cosineSimilarity([1, 2], [1, 2, 3])).toBe(0);
  });
});
