import type { FigureOverrides, Plates } from "./index";
import type { FigureSource } from "@sema/design";

/* Copy-only overrides of the existing plates. Geometry, colors, and layout stay shared.
   Both variants use the same figures so the experiment tests positioning, not artwork.
   Never edit the generated design-package SVGs or the control's figure strings here.
   A function of the plates, not a module-level value: vite.config.ts imports the variants for
   their metadata and cannot execute @sema/design; useCopyFigures() and render-copy.ts apply it. */
function relabel(
  source: FigureSource,
  title: string,
  desc: string,
  replacements: Record<string, string>,
): FigureSource {
  let svg = source.svg.replace(/<title\b[^>]*>.*?<\/title>/, `<title id="${source.id}-t">${title}</title>`)
    .replace(/<desc\b[^>]*>.*?<\/desc>/, `<desc id="${source.id}-d">${desc}</desc>`);
  for (const [from, to] of Object.entries(replacements)) {
    const token = `>${from}</tspan>`;
    if (!svg.includes(token)) throw new Error(`Missing figure copy: ${source.id}: ${from}`);
    // Replacement strings below are repository-authored SVG text, never user input.
    svg = svg.replaceAll(token, `>${to}</tspan>`);
  }
  return { ...source, title, desc, svg };
}

/* Figma exports position text by its left edge, so a longer replacement drifts against its node.
   Source labels are re-anchored: G6 right-aligned 8px off the node (x 160, r 5.4); G6m centred
   under each node (x 70, 179, 288). */
type Anchor = { x: number; anchor: "end" | "middle" };
const G6_ANCHORS: Record<string, Anchor> = {
  "Code lookup": { x: 146, anchor: "end" },
  "Target rules": { x: 146, anchor: "end" },
  "Plan checks": { x: 146, anchor: "end" },
};
const G6M_ANCHORS: Record<string, Anchor> = {
  "Code lookup": { x: 70, anchor: "middle" },
  "Target rules": { x: 179, anchor: "middle" },
  "Plan checks": { x: 288, anchor: "middle" },
};
function reanchor(plate: FigureSource, anchors: Record<string, Anchor>): FigureSource {
  let svg = plate.svg;
  for (const [label, { x, anchor }] of Object.entries(anchors)) {
    svg = svg.replace(new RegExp(`<tspan x="[0-9.]+" y="([0-9.]+)">${label}</tspan>`), `<tspan x="${x}" y="$1" text-anchor="${anchor}">${label}</tspan>`);
  }
  return { ...plate, svg };
}

const flowTitle = "Illustrative fitting workflow";
const flowDesc = "Code lookup, target rules, and plan checks produce accepted, unresolved, or review-pending decisions. This is not a measured study run.";
const flowLabels = {
  "Study one": "Code lookup",
  "Study two": "Target rules",
  "Study three": "Plan checks",
  "PUBLIC CANCER STUDIES": "ILLUSTRATIVE FITTING FLOW",
  "People": "Accepted",
  "[N] patients": "valid target",
  "Diagnoses": "Unresolved",
  "[N] diagnoses": "NO_MAP",
};

export const experimentFigures: FigureOverrides = (figures: Plates) => ({
  g11: relabel(figures.g11, "A proposed expansion path", "OMOP first, then fitting organization-defined models, then exploratory model proposal. Broader fitting remains to be proved.", {
    "Emergent models": "Organization models",
    "from public and internal sources": "evaluate the fit to your target",
    "Any model your organization defines": "Exploration: models proposed from sources",
  }),
  g3: relabel(figures.g3, "Source-to-target decisions", "Illustrative source fields and target properties connected through persisted semantic records.", {
    "The graph is what persists.": "Keep the mapping and its basis.",
    "Sources change. Targets change. The resolved meaning stays, with its source and confidence.": "Source references and confidence make a decision inspectable. Rebuilds need a preservation workflow.",
  }),
  g5: relabel(figures.g5, "Filtered semantic context", "Illustrative context for an agent: entities and joins filtered by status and confidence. Not a guarantee of correct answers.", {
    "joins &#xb7; 2, verified": "joins &#xb7; 2, included",
    "The agent never guesses what a column ": "Context is filtered by status",
    "means.": "and confidence.",
  }),
  g6: reanchor(relabel(figures.g6, flowTitle, flowDesc, {
    ...flowLabels,
    "Merged across studies": "Review pending",
    "[N] patients recognized twice": "operator decision",
    "run dated [date] &#xb7; nothing cancer-specific in the code": "Illustrative workflow. See the test fixtures below.",
  }), G6_ANCHORS),
  g6m: reanchor(relabel(figures.g6m, flowTitle, flowDesc, {
    ...flowLabels,
    "Merged across ": "Review",
    "studies": "pending",
    "[N] patients ": "operator",
    "recognized twice": "decision",
    "run dated [date] &#xb7; nothing cancer-specific in ": "Illustrative workflow. See the test",
    "the code": "fixtures below.",
  }), G6M_ANCHORS),
  g7: relabel(figures.g7, "An illustrative integration project", "An illustrative timeline: source reconciliation, a mapping sheet, a new source, and a downstream join error. Not customer results.", {
    "a 1,400-row mapping sheet": "a growing mapping sheet",
    "new source, half the sheet redone": "new source, mappings reopened",
  }),
  g7m: relabel(figures.g7m, "An illustrative integration project", "An illustrative timeline: source reconciliation, a mapping sheet, a new source, and a downstream join error. Not customer results.", {
    "a 1,400-row mapping ": "a growing mapping",
    "new source, half the ": "new source, old",
    "sheet redone": "mappings reopened",
  }),
});
