/**
 * Similarity primitives used by the scoring pipeline.
 *
 * `cosineSimilarityN` works on flat number arrays and returns a value mapped
 * to [0, 1]. It is the generic primitive used in future generalized scoring.
 *
 * `cosineSimilarity2D` mirrors the legacy pose_viewer.html helper: takes two
 * 2D vectors ({x, y}) and returns the raw cosine in [-1, 1]. The caller is
 * expected to map to [0, 1] (usually via `(cos + 1) / 2`) as the legacy code
 * does.
 */

export interface Vec2 {
  x: number;
  y: number;
}

export function cosineSimilarityN(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  if (na === 0 || nb === 0) return 0;
  const raw = dot / (Math.sqrt(na) * Math.sqrt(nb));
  return (Math.max(-1, Math.min(1, raw)) + 1) / 2;
}

export function cosineSimilarity2D(v1: Vec2 | null | undefined, v2: Vec2 | null | undefined): number {
  if (!v1 || !v2) return 0;
  const dot = v1.x * v2.x + v1.y * v2.y;
  const len1 = Math.hypot(v1.x, v1.y);
  const len2 = Math.hypot(v2.x, v2.y);
  if (!len1 || !len2) return 0;
  return dot / (len1 * len2);
}

/** Back-compat alias for Phase 1 callers that used the array signature. */
export const cosineSimilarity = cosineSimilarityN;
