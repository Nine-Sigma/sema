/* The Signal Cartography constellation as data: 46 points around an amber core in a 2400×800 field,
   each joined to its two nearest neighbours. Deterministic per seed, so the hero art, the gallery
   specimen and the landing page's graph plane all draw the same figure. */

export type Pt = { x: number; y: number };

export type ConstellationGeometry = {
  pts: Pt[];
  core: Pt;
  edges: [number, number][];
  /** Point indices ordered by distance to the core (nearest first). */
  byDist: number[];
  /** rank[i] = position of point i in byDist. */
  rank: number[];
  /** The five nearest points: amber, spoked to the core. */
  amber: Set<number>;
  /** Ten points that carry the slow pulse loop. */
  pulse: Set<number>;
  radii: number[];
  phase: number[];
};

function mulberry(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function scatter(rnd: () => number, core: Pt): Pt[] {
  const pts: Pt[] = [];
  let guard = 0;
  while (pts.length < 46 && guard++ < 20000) {
    const p = { x: 780 + rnd() * 1560, y: 50 + rnd() * 700 };
    const far =
      pts.every((q) => Math.hypot(p.x - q.x, p.y - q.y) > 105) &&
      Math.hypot(p.x - core.x, p.y - core.y) > 150;
    if (far) pts.push(p);
  }
  return pts;
}

function nearestEdges(pts: Pt[]): [number, number][] {
  const edges = new Set<string>();
  pts.forEach((p, i) => {
    pts
      .map((q, j) => ({ j, d: Math.hypot(p.x - q.x, p.y - q.y) }))
      .filter((o) => o.j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, 2)
      .forEach((o) => edges.add(i < o.j ? `${i}-${o.j}` : `${o.j}-${i}`));
  });
  return [...edges].map((e) => e.split("-").map(Number) as [number, number]);
}

const cache = new Map<number, ConstellationGeometry>();

/* The draw order matters: radii and phases are drawn after the points, in the sequence the SVG
   renderer consumed them, so the rendered art is unchanged from the v5 hero. */
export function constellationGeometry(seed = 11): ConstellationGeometry {
  const hit = cache.get(seed);
  if (hit) return hit;
  const rnd = mulberry(seed);
  const core: Pt = { x: 1640, y: 400 };
  const pts = scatter(rnd, core);
  const edges = nearestEdges(pts);
  const byDist = pts
    .map((p, i) => ({ i, d: Math.hypot(p.x - core.x, p.y - core.y) }))
    .sort((a, b) => a.d - b.d)
    .map((o) => o.i);
  const rank: number[] = [];
  byDist.forEach((i, k) => (rank[i] = k));
  const amber = new Set(byDist.slice(0, 5));
  const pulse = new Set(byDist.filter((_, k) => k % 4 === 1).slice(0, 10));
  const radii: number[] = [];
  const phase: number[] = [];
  byDist.forEach((i) => {
    radii[i] = amber.has(i) ? 6 : 3.2 + rnd() * 2.4;
    phase[i] = pulse.has(i) ? rnd() * 6 : 0;
  });
  const geo: ConstellationGeometry = { pts, core, edges, byDist, rank, amber, pulse, radii, phase };
  cache.set(seed, geo);
  return geo;
}
