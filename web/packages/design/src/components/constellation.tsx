import * as React from "react";
import { cn } from "../lib/utils";
import { constellationGeometry } from "../lib/constellation-geometry";

/* The hero ground: a deterministic Signal Cartography constellation. No text. Left third empty.
   Motion hooks: --i is the stagger index (rank by distance to the core), --pd the pulse phase.
   The SVG is generated once per seed at module scope; the markup is program output, not input. */
function constellation(seed = 11): string {
  const G = constellationGeometry(seed);
  const { core, pts } = G;
  const at = (i: number) => pts[i] ?? core;
  const f = (n: number): string => n.toFixed(1);
  let grid = "";
  for (let x = 120; x < 2400; x += 120) grid += `<path d="M${x} 0V800"/>`;
  for (let y = 80; y < 800; y += 120) grid += `<path d="M0 ${y}H2400"/>`;
  let lines = "";
  [...G.edges]
    .sort((u, v) => Math.min(G.rank[u[0]] ?? 0, G.rank[u[1]] ?? 0) - Math.min(G.rank[v[0]] ?? 0, G.rank[v[1]] ?? 0))
    .forEach(([a, b], k) => {
      lines += `<path class="sema-edge" pathLength="1" style="--i:${k}" d="M${f(at(a).x)} ${f(at(a).y)}L${f(at(b).x)} ${f(at(b).y)}"/>`;
    });
  let amberLines = "";
  [...G.amber].forEach((i, k) => {
    amberLines += `<path class="sema-amber-edge" pathLength="1" style="--i:${k}" d="M${f(at(i).x)} ${f(at(i).y)}L${core.x} ${core.y}"/>`;
  });
  let nodes = "";
  G.byDist.forEach((i) => {
    const p = at(i);
    const r = G.radii[i] ?? 4;
    const cls = G.pulse.has(i) ? "sema-node sema-pulse" : "sema-node";
    const style = `--i:${G.rank[i]}` + (G.pulse.has(i) ? `;--pd:${f(G.phase[i] ?? 0)}s` : "");
    nodes += G.amber.has(i)
      ? `<circle class="${cls}" style="${style}" cx="${f(p.x)}" cy="${f(p.y)}" r="${r}" fill="var(--accent)" fill-opacity="0.18" stroke="var(--accent)" stroke-width="1.5"/>`
      : `<circle class="${cls}" style="${style}" cx="${f(p.x)}" cy="${f(p.y)}" r="${f(r)}" fill="var(--ground)" stroke="var(--blue)" stroke-width="1.25"/>`;
  });
  const rings = [44, 96, 168, 260]
    .map((r, k) => `<circle class="sema-ring" style="--i:${k}" cx="${core.x}" cy="${core.y}" r="${r}" stroke="var(--accent)" stroke-opacity="${(0.42 - k * 0.1).toFixed(2)}"/>`)
    .join("");
  const ticks = Array.from({ length: 24 }, (_, k) => {
    const a = (k * Math.PI) / 12;
    const r1 = 260;
    const r2 = k % 6 === 0 ? 284 : 272;
    return `<path d="M${f(core.x + r1 * Math.cos(a))} ${f(core.y + r1 * Math.sin(a))}L${f(core.x + r2 * Math.cos(a))} ${f(core.y + r2 * Math.sin(a))}"/>`;
  }).join("");
  return `<svg viewBox="0 0 2400 800" preserveAspectRatio="xMidYMid slice" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
<g stroke="var(--rule)" stroke-opacity="0.55">${grid}</g>
<g stroke="var(--blue)" stroke-opacity="0.32" stroke-linecap="round">${lines}</g>
<g stroke="var(--accent)" stroke-opacity="0.5" stroke-linecap="round">${amberLines}</g>
${rings}
<g class="sema-ticks" stroke="var(--accent)" stroke-opacity="0.45">${ticks}</g>
${nodes}
<circle class="sema-core sema-pulse" cx="${core.x}" cy="${core.y}" r="12" fill="var(--accent)" fill-opacity="0.22" stroke="var(--accent)" stroke-width="1.5"/>
<circle class="sema-core" cx="${core.x}" cy="${core.y}" r="3" fill="var(--accent)"/>
</svg>`;
}

const cache = new Map<number, string>();
function constellationSvg(seed: number): string {
  let svg = cache.get(seed);
  if (!svg) {
    svg = constellation(seed);
    cache.set(seed, svg);
  }
  return svg;
}

type ConstellationProps = React.ComponentProps<"div"> & { seed?: number };

/* Wraps the art; pass className for placement and masking. The pulse loop pauses offscreen. */
function Constellation({ seed = 11, className, ...props }: ConstellationProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const el = ref.current;
    if (!el || !("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) delete el.dataset.paused;
        else el.dataset.paused = "";
      }
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div
      ref={ref}
      data-slot="constellation"
      className={cn("sema-art [&_svg]:block [&_svg]:h-full [&_svg]:w-full", className)}
      dangerouslySetInnerHTML={{ __html: constellationSvg(seed) }}
      {...props}
    />
  );
}

export { Constellation, constellation, type ConstellationProps };
