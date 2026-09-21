import type { Pt } from "@sema/design";
import { clamp, ease, seg, docTop, type Frame } from "./bus";
import { buildPlane, type Built, type PlaneNode } from "./plane-build";
import { G11_CORE, layoutFor, plateScale, plateToScreen, roleRadius, roleTarget, visibleSvg, type Layout } from "./plane-geometry";

/* The plane's state machine. Two writers, never at once: the hero scrub (per frame, transitions off)
   and the chapter states (end values, transitions on). Depth is composed last in every frame. */

export type PlaneHandle = {
  renderHero: (p: number, socket: Element | null) => void;
  setScrub: (on: boolean) => void;
};

type Vals = { bg: number; grid: number; orbit: number; ring: number; spoke: number; sun: number; mask: number; cast: number; g3: number };
const HERO: Vals = { bg: 1, grid: 1, orbit: 1, ring: 1, spoke: 0, sun: 1, mask: 1, cast: 1, g3: 0 };
const PAGE: Vals = { bg: 0.15, grid: 0.15, orbit: 0, ring: 0.6, spoke: 1, sun: 0, mask: 0, cast: 0, g3: 0 };
/* The orbit stays as a carrier while the cast travels (fix 1, finish review): 1 -> 0.6 by the fade, then out over 0.7-0.95. */
const f = (n: number): string => n.toFixed(2);
const DEPTH = { grid: 0.2, field: 0.6, glow: 0.9 };

export class PlaneController implements PlaneHandle {
  private b: Built;
  private chapter = "hero";
  private scrub = false;
  private lastP = 0;
  private socket: Element | null = null;
  private rectKey = "";
  private heroEnd = 0;
  private wallStart = 0;
  private wallTravel = 0;
  private pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  private depthScale = 1;
  private offsets = { grid: { x: 0, y: 0 }, field: { x: 0, y: 0 }, glow: { x: 0, y: 0 } };
  private readonly touch = window.matchMedia("(hover: none)").matches;

  constructor(private readonly host: HTMLElement) {
    this.b = buildPlane(host, layoutFor(window.innerWidth, window.innerHeight));
    this.size();
    this.write(HERO);
  }

  destroy(): void {
    this.host.querySelector(":scope > svg")?.remove();
  }

  /* ---- layout ---- */
  private size(): void {
    const { svg, layout: L, gridFill, mask } = this.b;
    svg.setAttribute("width", String(L.W));
    svg.setAttribute("height", String(L.H + 200));
    gridFill.setAttribute("height", String(L.H + 200));
    mask.setAttribute("height", String(L.H + 200));
  }

  relayout(): void {
    const L = this.b.layout;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    if (vw !== L.W || Math.abs(vh - L.H) > 120) this.rebuild(layoutFor(vw, vh));
    const hero = document.querySelector<HTMLElement>(".sema-stage-hero");
    this.heroEnd = hero ? docTop(hero) + Math.max(0, hero.offsetHeight - vh) : 0;
    const wall = document.querySelector<HTMLElement>(".sema-stage-wall");
    this.wallStart = wall ? docTop(wall) : 0;
    this.wallTravel = wall && vw >= 901 ? Math.max(0, wall.offsetHeight - vh) : 0;
    this.rectKey = "";
  }

  private rebuild(layout: Layout): void {
    this.b = buildPlane(this.host, layout);
    this.size();
    this.rectKey = "";
    if (this.scrub) this.renderHero(this.lastP, this.socket);
    else this.applyChapter();
  }

  /* ---- writers ---- */
  private write(v: Vals): void {
    const { bg, gridFill, orbit, rings, spokes, sun, mask, cast, g3 } = this.b;
    bg.style.opacity = f(v.bg);
    gridFill.style.opacity = f(v.grid);
    orbit.style.opacity = f(v.orbit);
    rings.forEach((r) => {
      r.style.transformBox = "fill-box";
      r.style.transformOrigin = "center";
      r.style.transform = `scale(${f(v.ring)})`;
    });
    spokes.forEach((s) => {
      s.style.strokeDasharray = "1";
      s.style.strokeDashoffset = f(v.spoke);
    });
    sun.style.opacity = f(v.sun);
    mask.style.opacity = f(v.mask);
    cast.style.opacity = f(v.cast);
    g3.style.opacity = f(v.g3);
  }

  setScrub(on: boolean): void {
    if (on === this.scrub) return;
    this.scrub = on;
    this.host.classList.toggle("is-scrub", on);
    if (!on) this.applyChapter();
  }

  /* Progress p of the hero stage (0 = hero, 1 = idea). The cast travels to the plate in the socket;
     the plate crossfades in over 0.78–1 while the plane's copy fades out over 0.88–1. */
  renderHero(p: number, socket: Element | null): void {
    this.lastP = p;
    this.socket = socket;
    const fade = ease(seg(p, 0, 0.35));
    const travel = ease(seg(p, 0.2, 0.8));
    const draw = seg(p, 0.4, 1);
    const out = 1 - seg(p, 0.88, 1);
    const orbit = (1 - 0.4 * fade) * (1 - ease(seg(p, 0.7, 0.95)));
    this.write({ bg: 1 - 0.85 * fade, grid: 1 - 0.85 * fade, orbit, ring: 1 - 0.4 * fade, spoke: fade, sun: 1 - fade, mask: 1 - fade, cast: out, g3: out });
    const svg = visibleSvg(socket);
    const rect = svg?.getBoundingClientRect() ?? null;
    if (rect) this.layLanding(rect);
    const { plate } = this.b.layout;
    const off = this.offsets.field;
    for (const n of this.b.nodes) {
      if (!n.role) continue;
      const to = rect ? plateToScreen(rect, plate.crop, roleTarget(plate, n.role)) : n.from;
      const r1 = rect ? roleRadius(plate, n.role) * plateScale(rect, plate.crop) : n.r0;
      n.el.style.transform = "";
      n.el.classList.toggle("is-landed", travel > 0.5);
      n.el.setAttribute("cx", f(n.from.x + (to.x - off.x - n.from.x) * travel));
      n.el.setAttribute("cy", f(n.from.y + (to.y - off.y - n.from.y) * travel));
      n.el.setAttribute("r", f(n.r0 + (r1 - n.r0) * travel));
    }
    this.b.g3Edges.forEach((e, k) => {
      e.el.style.strokeDasharray = "1";
      e.el.style.strokeDashoffset = f(1 - ease(seg(draw, k * 0.05, 0.6 + k * 0.05)));
    });
    (socket as HTMLElement | null)?.style.setProperty("--g3-in", f(seg(p, 0.78, 1)));
  }

  private layLanding(rect: DOMRect): void {
    const key = `${rect.left | 0},${rect.top | 0},${rect.width | 0},${this.offsets.field.x | 0},${this.offsets.field.y | 0}`;
    if (key === this.rectKey) return;
    this.rectKey = key;
    const { plate } = this.b.layout;
    const off = this.offsets.field;
    const g = (p: Pt): Pt => {
      const s = plateToScreen(rect, plate.crop, p);
      return { x: s.x - off.x, y: s.y - off.y };
    };
    const line = (a: Pt, b: Pt): string => `M${f(a.x)} ${f(a.y)}L${f(b.x)} ${f(b.y)}`;
    const curve = (a: Pt, c: Pt, b: Pt): string => `M${f(a.x)} ${f(a.y)}Q${f(c.x)} ${f(c.y)} ${f(b.x)} ${f(b.y)}`;
    for (const e of this.b.g3Edges) {
      if (e.kind === "graph") {
        const [a, b] = plate.graphEdges[e.q]!;
        e.el.setAttribute("d", line(g(plate.graph[a]!), g(plate.graph[b]!)));
      } else if (e.kind === "source") {
        const [a, c, b] = plate.sourceCurves[e.q]!;
        e.el.setAttribute("d", curve(g(a), g(c), g(b)));
      } else if (e.kind === "target") {
        const [a, b] = plate.targetLines[e.q]!;
        e.el.setAttribute("d", line(g(a), g(b)));
      } else {
        const [a, c, b] = plate.targetCurves[e.q]!;
        e.el.setAttribute("d", curve(g(a), g(c), g(b)));
      }
    }
  }

  /* ---- chapters ---- */
  setChapter(name: string): void {
    if (name === this.chapter) return;
    const prev = this.chapter;
    this.chapter = name;
    this.host.dataset.chapter = name;
    if (!this.scrub) this.applyChapter(prev);
  }

  private applyChapter(prev = ""): void {
    const c = this.chapter;
    if (c === "hero") this.write(HERO);
    else if (c === "pilot") this.write(HERO);
    else if (c === "quote") this.write({ ...PAGE, bg: 0.45, grid: 0.3 });
    else if (c === "why") this.write({ ...PAGE, sun: 1 });
    else this.write(PAGE);
    if (c === "pilot") this.reformCast();
    if (c === "proof") this.cluster();
    else if (prev === "proof" || c === "hero") this.uncluster();
    if (c === "why") this.trackSun();
    else if (prev === "why") this.b.sun.style.transition = "";
    if (c === "pilot") this.b.sun.style.transform = "";
  }

  private castNodes(): PlaneNode[] {
    return this.b.nodes.filter((n) => n.role);
  }

  /* S9: the ten protagonists come back from the plate's direction to their hero positions. */
  private reformCast(): void {
    const { plate } = this.b.layout;
    const svg = visibleSvg(this.socket ?? document.querySelector(".sema-socket"));
    const rect = svg?.getBoundingClientRect() ?? null;
    for (const n of this.castNodes()) {
      let dx = 0;
      let dy = -420;
      if (rect && n.role) {
        const to = plateToScreen(rect, plate.crop, roleTarget(plate, n.role));
        dx = to.x - n.from.x;
        dy = to.y - n.from.y;
        const d = Math.hypot(dx, dy);
        if (d > 600) {
          dx *= 600 / d;
          dy *= 600 / d;
        }
      }
      n.el.style.transition = "none";
      n.el.classList.remove("is-landed");
      n.el.setAttribute("cx", f(n.from.x));
      n.el.setAttribute("cy", f(n.from.y));
      n.el.setAttribute("r", f(n.r0));
      n.el.style.transform = `translate(${f(dx)}px, ${f(dy)}px)`;
      void n.el.getBoundingClientRect();
      n.el.style.transition = "";
      n.el.style.transform = "";
    }
  }

  /* S5 (M7): the background gathers into three clusters behind the three studies of G6. */
  private cluster(): void {
    const svg = visibleSvg(document.querySelector('[data-plane-anchor="g6"]'));
    const r = svg?.getBoundingClientRect();
    const off = this.offsets.field;
    const L = this.b.layout;
    const cy = L.H * 0.34 - off.y;
    const centers: Pt[] = r
      ? [1, 3, 5].map((k) => ({ x: r.left + (r.width * k) / 6 - off.x, y: cy }))
      : [0.25, 0.5, 0.75].map((k) => ({ x: L.W * k, y: cy }));
    for (const n of this.b.nodes) {
      if (n.role) continue;
      const c = centers.reduce((best, p) => (Math.hypot(p.x - n.from.x, p.y - n.from.y) < Math.hypot(best.x - n.from.x, best.y - n.from.y) ? p : best));
      const tx = c.x + (n.from.x - c.x) * 0.22 - n.from.x;
      const ty = c.y + (n.from.y - c.y) * 0.22 - n.from.y;
      n.el.style.setProperty("--td", `${((this.b.G.rank[n.i] ?? 0) % 7) * 40}ms`);
      n.el.style.transform = `translate(${f(tx)}px, ${f(ty)}px)`;
    }
  }

  private uncluster(): void {
    for (const n of this.b.nodes) {
      if (n.role) continue;
      n.el.style.transform = "";
    }
  }

  /* ---- per frame: depth, and the sun behind G11 ---- */
  pointerAt(nx: number, ny: number): void {
    this.pointer.tx = nx;
    this.pointer.ty = ny;
  }

  frame(fr: Frame): boolean {
    const pt = this.pointer;
    pt.x += (pt.tx - pt.x) * 0.08;
    pt.y += (pt.ty - pt.y) * 0.08;
    const target = this.chapter === "pilot" ? 0 : 1;
    this.depthScale += (target - this.depthScale) * 0.08;
    const half = this.touch ? 0.5 : 1;
    const base = -0.12 * clamp(fr.y - this.heroEnd, 0, 1600) * half;
    let wall = 0;
    if (this.wallTravel > 0) {
      wall = -0.45 * clamp(fr.y - this.wallStart, 0, this.wallTravel);
      wall *= 1 - clamp((fr.y - this.wallStart - this.wallTravel) / 1200, 0, 1);
    }
    const s = this.depthScale;
    const place = (g: SVGGElement, o: Pt, factor: number, extra = 0): void => {
      const x = pt.x * 8 * factor * s;
      const y = (base * factor + extra + pt.y * 8 * factor) * s;
      if (Math.abs(x - o.x) < 0.02 && Math.abs(y - o.y) < 0.02) return;
      o.x = x;
      o.y = y;
      g.style.transform = `translate(${f(x)}px, ${f(y)}px)`;
    };
    place(this.b.depth.grid, this.offsets.grid, DEPTH.grid);
    place(this.b.depth.field, this.offsets.field, DEPTH.field, wall);
    place(this.b.depth.glow, this.offsets.glow, DEPTH.glow);
    if (this.chapter === "why") this.trackSun();
    const settling = Math.abs(pt.tx - pt.x) + Math.abs(pt.ty - pt.y) > 0.002 || Math.abs(target - this.depthScale) > 0.002;
    return settling;
  }

  private trackSun(): void {
    const key = this.b.layout.key;
    const svg = visibleSvg(document.querySelector('[data-plane-anchor="g11"]'));
    if (!svg) return;
    const spec = G11_CORE[key];
    const p = plateToScreen(svg.getBoundingClientRect(), spec.crop, spec.at);
    const o = this.offsets.glow;
    this.b.sun.style.transition = "opacity 900ms var(--ease-signal)";
    this.b.sun.style.transform = `translate(${f(p.x - this.b.core.x - o.x)}px, ${f(p.y - this.b.core.y - o.y)}px)`;
  }
}
