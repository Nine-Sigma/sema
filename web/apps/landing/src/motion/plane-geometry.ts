import type { ConstellationGeometry, Pt } from "@sema/design";

/* Where the graph plane places the constellation per breakpoint, and where its ten protagonist
   nodes land: the node positions of the G3 / G3m plates (Figures v5 45:2 and 117:2) in plate
   coordinates, mapped onto the plate's rendered box through the crop that build-figures.mjs applies. */

export type Crop = { x: number; y: number; w: number; h: number };
export type Curve = [Pt, Pt, Pt];
export type Plate = {
  crop: Crop;
  source: Pt[];
  graph: Pt[];
  target: Pt[];
  graphEdges: [number, number][];
  sourceCurves: Curve[];
  targetLines: [Pt, Pt][];
  targetCurves: Curve[];
  radius: { centre: number; graph: number; source: number; target: number };
};

const pt = (x: number, y: number): Pt => ({ x, y });
const crv = (a: number[], c: number[], b: number[]): Curve => [pt(a[0]!, a[1]!), pt(c[0]!, c[1]!), pt(b[0]!, b[1]!)];

export const G3D: Plate = {
  crop: { x: 0, y: 90, w: 800, h: 430 },
  source: [pt(120, 190), pt(120, 300), pt(120, 410)],
  graph: [pt(400, 300), pt(340, 240), pt(468, 248), pt(352, 372), pt(458, 360)],
  target: [pt(620, 245), pt(620, 365)],
  graphEdges: [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]],
  sourceCurves: [crv([128, 190], [209, 190], [332, 240]), crv([128, 300], [209, 300], [392, 300]), crv([128, 410], [209, 410], [344, 372])],
  targetLines: [[pt(476, 245), pt(612, 245)], [pt(466, 360), pt(612, 360)]],
  targetCurves: [],
  radius: { centre: 11.25, graph: 5.375, source: 6.375, target: 6.375 },
};

export const G3M: Plate = {
  crop: { x: 0, y: 36, w: 358, h: 564 },
  source: [pt(60, 110), pt(179, 110), pt(298, 110)],
  graph: [pt(179, 290), pt(120, 245), pt(238, 245), pt(120, 335), pt(238, 335)],
  target: [pt(100, 470), pt(258, 470)],
  graphEdges: [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]],
  sourceCurves: [crv([60, 116], [60, 200], [116, 240]), crv([179, 116], [179, 200], [179, 278]), crv([298, 116], [298, 200], [242, 240])],
  targetLines: [],
  targetCurves: [crv([120, 341], [100, 420], [100, 464]), crv([238, 341], [258, 420], [258, 464])],
  radius: { centre: 11.25, graph: 4.375, source: 5.375, target: 5.375 },
};

/* G11's core, for the "why" chapter: the plane's amber point sits behind it. */
export const G11_CORE = {
  desktop: { crop: { x: 20, y: 40, w: 440, h: 400 }, at: pt(240, 256) },
  portrait: { crop: { x: 0, y: 36, w: 358, h: 356 }, at: pt(179, 212) },
};

export type Mask = { x1: number; y1: number; x2: number; y2: number; a: number; b: number };
export type Layout = {
  key: "desktop" | "portrait";
  W: number;
  H: number;
  /** Scale of the 2400×800 field to CSS px, and its offset. */
  k: number;
  ox: number;
  oy: number;
  glowR: number;
  mask: Mask;
  plate: Plate;
};

/* Desktop: the 1440-wide composition scaled to the viewport width, vertically centred as a 900-tall
   frame (core at (1160, 450) on 1440×900, (1160, 400) on 1440×800). Portrait: re-composed at 0.55,
   core at (300, 250) on 390 wide; never a crop of the desktop frame. */
export function layoutFor(vw: number, vh: number): Layout {
  if (vw < 901) {
    const s = vw / 390;
    return {
      key: "portrait",
      W: vw,
      H: vh,
      k: 0.55 * s,
      ox: -602 * s,
      oy: 30 * s,
      glowR: 230 * s,
      mask: { x1: 0, y1: 0, x2: 0, y2: 1, a: 0.04, b: 0.34 },
      plate: G3M,
    };
  }
  const k = Math.min(1.6, Math.max(0.8, vw / 1440));
  return {
    key: "desktop",
    W: vw,
    H: vh,
    k,
    ox: -480 * k,
    oy: 50 * k + (vh - 900 * k) / 2,
    glowR: 420 * k,
    mask: { x1: 0, y1: 0, x2: 1, y2: 0, a: 0.22, b: 0.52 },
    plate: G3D,
  };
}

export const fieldPoint = (L: Layout, p: Pt): Pt => ({ x: p.x * L.k + L.ox, y: p.y * L.k + L.oy });

/* Plate coordinates → viewport coordinates, given the rendered box of the cropped SVG. */
export function plateToScreen(rect: DOMRect, crop: Crop, p: Pt): Pt {
  return { x: rect.left + ((p.x - crop.x) / crop.w) * rect.width, y: rect.top + ((p.y - crop.y) / crop.h) * rect.height };
}
export const plateScale = (rect: DOMRect, crop: Crop): number => rect.width / crop.w;

export type Role = { kind: "graph" | "source" | "target"; q: number; centre: boolean };

/* The five amber (nearest) nodes become the graph; the next three the sources; the next two the targets. */
export function castFor(G: ConstellationGeometry): Map<number, Role> {
  const cast = new Map<number, Role>();
  G.byDist.slice(0, 5).forEach((i, q) => cast.set(i, { kind: "graph", q, centre: q === 0 }));
  G.byDist.slice(5, 8).forEach((i, q) => cast.set(i, { kind: "source", q, centre: false }));
  G.byDist.slice(8, 10).forEach((i, q) => cast.set(i, { kind: "target", q, centre: false }));
  return cast;
}

export function roleTarget(plate: Plate, role: Role): Pt {
  const list = role.kind === "graph" ? plate.graph : role.kind === "source" ? plate.source : plate.target;
  return list[role.q] ?? plate.graph[0]!;
}
export function roleRadius(plate: Plate, role: Role): number {
  if (role.centre) return plate.radius.centre;
  return plate.radius[role.kind];
}

/* The SVG element of whichever plate copy is on screen (desktop or mobile twin). */
export function visibleSvg(socket: Element | null): SVGSVGElement | null {
  if (!socket) return null;
  for (const svg of socket.querySelectorAll("svg")) {
    if (svg.getClientRects().length) return svg;
  }
  return null;
}
