import { constellationGeometry, type ConstellationGeometry, type Pt } from "@sema/design";
import { castFor, fieldPoint, type Layout, type Role } from "./plane-geometry";

/* Builds the plane's SVG for one layout. Three depth sub-planes (grid 0.2×, field 0.6×, glow 0.9×),
   and inside the field: the background (36 nodes, their edges), the cast (the ten protagonists and
   the edges between them) and the G3 landing edges, drawn hidden. The class names match motion.css,
   so the v5 load moment (nodes by rank, edges draw, spokes, rings, core) plays on the plane unchanged. */

const NS = "http://www.w3.org/2000/svg";
const f = (n: number): string => n.toFixed(1);

function el<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
  parent: Element,
): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent.appendChild(e);
  return e;
}

export type PlaneNode = { el: SVGCircleElement; i: number; from: Pt; r0: number; role: Role | null };
export type PlaneEdge = { el: SVGPathElement; keep: boolean };
export type G3Edge = { el: SVGPathElement; kind: "graph" | "source" | "target" | "targetCurve"; q: number };

export type Built = {
  svg: SVGSVGElement;
  layout: Layout;
  G: ConstellationGeometry;
  core: Pt;
  depth: { grid: SVGGElement; field: SVGGElement; glow: SVGGElement };
  gridFill: SVGRectElement;
  bg: SVGGElement;
  cast: SVGGElement;
  g3: SVGGElement;
  g3Edges: G3Edge[];
  orbit: SVGGElement;
  sun: SVGGElement;
  rings: SVGCircleElement[];
  spokes: SVGPathElement[];
  mask: SVGRectElement;
  nodes: PlaneNode[];
  edges: PlaneEdge[];
};

function defs(svg: SVGSVGElement, L: Layout, core: Pt): void {
  const d = el("defs", {}, svg);
  const glow = el("radialGradient", { id: "sema-glow" }, d);
  el("stop", { offset: 0, "stop-color": "var(--accent)", "stop-opacity": 0.14 }, glow);
  el("stop", { offset: 1, "stop-color": "var(--accent)", "stop-opacity": 0 }, glow);
  const m = L.mask;
  const fade = el("linearGradient", { id: "sema-fade", x1: m.x1, y1: m.y1, x2: m.x2, y2: m.y2 }, d);
  el("stop", { offset: m.a, "stop-color": "var(--ground)", "stop-opacity": 1 }, fade);
  el("stop", { offset: m.b, "stop-color": "var(--ground)", "stop-opacity": 0 }, fade);
  const step = 120 * L.k;
  const grid = el(
    "pattern",
    { id: "sema-grid", width: f(step), height: f(step), patternUnits: "userSpaceOnUse", x: f(L.ox), y: f(L.oy + 80 * L.k) },
    d,
  );
  const g = el("g", { stroke: "var(--rule)", "stroke-opacity": 0.55 }, grid);
  el("path", { d: `M0 0V${f(step)}` }, g);
  el("path", { d: `M0 0H${f(step)}` }, g);
  void core;
}

function nodesAndEdges(b: Built, cast: Map<number, Role>): void {
  const { G, layout: L } = b;
  const P = (i: number): Pt => fieldPoint(L, G.pts[i] ?? G.core);
  const rk = (i: number): number => G.rank[i] ?? 0;
  const bgEdges = el("g", { stroke: "var(--blue)", "stroke-opacity": 0.32, "stroke-linecap": "round" }, b.bg);
  const castEdges = el("g", { stroke: "var(--blue)", "stroke-opacity": 0.32, "stroke-linecap": "round" }, b.cast);
  [...G.edges]
    .sort((u, v) => Math.min(rk(u[0]), rk(u[1])) - Math.min(rk(v[0]), rk(v[1])))
    .forEach(([a, c], q) => {
      const keep = cast.has(a) && cast.has(c);
      const path = el(
        "path",
        { class: "sema-edge", pathLength: 1, style: `--i:${q}`, d: `M${f(P(a).x)} ${f(P(a).y)}L${f(P(c).x)} ${f(P(c).y)}` },
        keep ? castEdges : bgEdges,
      );
      b.edges.push({ el: path, keep });
    });
  const bgNodes = el("g", {}, b.bg);
  const castNodes = el("g", {}, b.cast);
  let lit = 0;
  G.byDist.forEach((i) => {
    const p = P(i);
    const amber = G.amber.has(i);
    const r = amber ? Math.max(3.5, 6 * L.k) : Math.max(2, (G.radii[i] ?? 4) * L.k);
    const role = cast.get(i) ?? null;
    const isLit = !role && lit < 5;
    const cls = ["sema-node", "sema-tr", G.pulse.has(i) ? "sema-pulse" : "", isLit ? "sema-lit" : "", role ? "sema-cast-node" : ""]
      .filter(Boolean)
      .join(" ");
    let style = `--i:${rk(i)}`;
    if (G.pulse.has(i)) style += `;--pd:${f(G.phase[i] ?? 0)}s`;
    if (isLit) style += `;--l:${lit++}`;
    const attrs: Record<string, string | number> = amber
      ? { fill: "var(--accent)", "fill-opacity": 0.18, stroke: "var(--accent)", "stroke-width": 1.5 }
      : { fill: "var(--ground)", stroke: "var(--blue)", "stroke-width": 1.25 };
    if (role) {
      attrs["data-kind"] = role.kind;
      if (role.centre) attrs["data-centre"] = "";
    }
    const c = el("circle", { class: cls, style, cx: f(p.x), cy: f(p.y), r: f(r), ...attrs }, role ? castNodes : bgNodes);
    b.nodes.push({ el: c, i, from: p, r0: r, role });
  });
}

function orbitAndSun(b: Built): void {
  const { G, layout: L, core } = b;
  const P = (i: number): Pt => fieldPoint(L, G.pts[i] ?? G.core);
  const spokes = el("g", { stroke: "var(--accent)", "stroke-opacity": 0.5, "stroke-linecap": "round" }, b.orbit);
  [...G.amber].forEach((i, q) => {
    b.spokes.push(
      el("path", { class: "sema-amber-edge sema-tr", pathLength: 1, style: `--i:${q}`, d: `M${f(P(i).x)} ${f(P(i).y)}L${f(core.x)} ${f(core.y)}` }, spokes),
    );
  });
  [44, 96, 168, 260].forEach((r, q) => {
    b.rings.push(
      el(
        "circle",
        { class: "sema-ring sema-tr", style: `--i:${q}`, cx: f(core.x), cy: f(core.y), r: f(r * L.k), stroke: "var(--accent)", "stroke-opacity": (0.42 - q * 0.1).toFixed(2) },
        b.orbit,
      ),
    );
  });
  const ticks = el("g", { class: "sema-ticks", stroke: "var(--accent)", "stroke-opacity": 0.45 }, b.orbit);
  for (let q = 0; q < 24; q++) {
    const a = (q * Math.PI) / 12;
    const r1 = 260 * L.k;
    const r2 = (q % 6 === 0 ? 284 : 272) * L.k;
    el("path", { d: `M${f(core.x + r1 * Math.cos(a))} ${f(core.y + r1 * Math.sin(a))}L${f(core.x + r2 * Math.cos(a))} ${f(core.y + r2 * Math.sin(a))}` }, ticks);
  }
  el("circle", { class: "sema-glow", cx: f(core.x), cy: f(core.y), r: f(L.glowR), fill: "url(#sema-glow)" }, b.sun);
  el("circle", { class: "sema-core sema-pulse", cx: f(core.x), cy: f(core.y), r: f(Math.max(7, 12 * L.k)), fill: "var(--accent)", "fill-opacity": 0.22, stroke: "var(--accent)", "stroke-width": 1.5 }, b.sun);
  el("circle", { class: "sema-core", cx: f(core.x), cy: f(core.y), r: 3, fill: "var(--accent)" }, b.sun);
}

function landingEdges(b: Built): void {
  const plate = b.layout.plate;
  const add = (kind: G3Edge["kind"], q: number, attrs: Record<string, string | number>): void => {
    b.g3Edges.push({ el: el("path", { class: "sema-g3-edge", pathLength: 1, "stroke-linecap": "round", ...attrs }, b.g3), kind, q });
  };
  plate.graphEdges.forEach((_, q) => add("graph", q, { stroke: "var(--rule-strong)" }));
  plate.sourceCurves.forEach((_, q) => add("source", q, { stroke: "var(--blue)", "stroke-opacity": 0.7 }));
  plate.targetLines.forEach((_, q) => add("target", q, { stroke: "var(--accent)", "stroke-opacity": 0.8 }));
  plate.targetCurves.forEach((_, q) => add("targetCurve", q, { stroke: "var(--accent)", "stroke-opacity": 0.8 }));
}

export function buildPlane(container: HTMLElement, layout: Layout, seed = 11): Built {
  const G = constellationGeometry(seed);
  const svg = document.createElementNS(NS, "svg");
  /* No viewBox: one user unit is one CSS pixel from the top-left, so DOM rects map straight onto the plane. */
  svg.setAttribute("fill", "none");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const core = fieldPoint(layout, G.core);
  defs(svg, layout, core);
  const grid = el("g", { class: "sema-depth-grid" }, svg);
  const gridFill = el("rect", { class: "sema-grid-fill sema-tr", x: 0, y: 0, width: layout.W, height: layout.H, fill: "url(#sema-grid)" }, grid);
  const field = el("g", { class: "sema-depth-field" }, svg);
  const bg = el("g", { class: "sema-bg sema-tr" }, field);
  const cast = el("g", { class: "sema-cast sema-tr" }, field);
  const g3 = el("g", { class: "sema-g3 sema-tr" }, field);
  const glow = el("g", { class: "sema-depth-glow" }, svg);
  const orbit = el("g", { class: "sema-orbit sema-tr" }, glow);
  const sun = el("g", { class: "sema-sun" }, glow);
  const mask = el("rect", { class: "sema-mask sema-tr", x: 0, y: 0, width: layout.W, height: layout.H, fill: "url(#sema-fade)" }, svg);
  const b: Built = { svg, layout, G, core, depth: { grid, field, glow }, gridFill, bg, cast, g3, g3Edges: [], orbit, sun, rings: [], spokes: [], mask, nodes: [], edges: [] };
  nodesAndEdges(b, castFor(G));
  orbitAndSun(b);
  landingEdges(b);
  container.querySelector(":scope > svg")?.remove();
  container.prepend(svg);
  return b;
}
