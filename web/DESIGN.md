---
name: Sema
description: Signal Cartography. One graph the reader scrolls through; amber only where there is signal.
colors:
  ground: "#0a0d19"
  ink: "#e8e6df"
  muted: "#8a8f9c"
  rule: "#2a3040"
  rule-strong: "#5c6478"
  accent: "#e5a23a"
  accent-fill: "#e5a23a"
  on-accent: "#0a0d19"
  blue: "#8fb0d6"
  light-ground: "#f2f1ec"
  light-ink: "#0a0d19"
  light-muted: "#5c6270"
  light-rule: "#c9cbd2"
  light-rule-strong: "#7f8490"
  light-accent: "#8f5e12"
  light-accent-fill: "#b87a1e"
  light-blue: "#3e6a99"
typography:
  display:
    fontFamily: "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.5rem, 5.2vw, 4.75rem)"
    fontWeight: 700
    lineHeight: 0.98
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(1.9rem, 3.2vw, 3rem)"
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "-0.03em"
  title:
    fontFamily: "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.35rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  wall:
    fontFamily: "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.5rem, 6.67vw, 6rem)"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "-0.03em"
  chapter:
    fontFamily: "IBM Plex Mono, SFMono-Regular, Menlo, monospace"
    fontSize: "clamp(7.5rem, 15.3vw, 13.75rem)"
    fontWeight: 400
    lineHeight: 0.85
    letterSpacing: "-0.04em"
  lede:
    fontFamily: "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 400
    lineHeight: 1.5
  body:
    fontFamily: "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "IBM Plex Mono, SFMono-Regular, Menlo, monospace"
    fontSize: "12px"
    fontWeight: 400
    letterSpacing: "0.08em"
  wordmark:
    fontFamily: "Cormorant Garamond, Georgia, serif"
    fontSize: "28px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "0.01em"
rounded:
  none: "0px"
spacing:
  gutter: "clamp(16px, 3vw, 40px)"
  column-gap: "24px"
  column-gap-mobile: "16px"
  section: "clamp(64px, 8vw, 112px)"
  chapter-heavy: "240px"
  chapter-light: "160px"
  chapter-connective: "200px"
  chapter-heavy-mobile: "120px"
  chapter-light-mobile: "96px"
  chapter-connective-mobile: "112px"
components:
  button-primary:
    backgroundColor: "{colors.accent-fill}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.none}"
    padding: "0 22px"
    height: "48px"
  button-primary-lg:
    backgroundColor: "{colors.accent-fill}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.none}"
    padding: "0 32px"
    height: "64px"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 22px"
    height: "48px"
  input:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    height: "48px"
  tag:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "4px 8px"
  record:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "24px 24px 28px"
---

# Design System: Sema

## Overview

**Creative North Star: "Signal Cartography"**

The page is a chart of one graph. A near-black ground carries a faint grid plane and a field of small nodes; silver-blue draws structure (edges, rings, figure strokes) and amber marks signal only (the core, the one call to action, a blocked decision, the break in a timeline, the second sentence of the quote, the lit sub-graph). Type is IBM Plex Sans for reading and Plex Mono for annotation, with the Cormorant wordmark as the single serif. Nothing is rounded and nothing floats: surfaces are flat, edges are hairlines, and the only elevation is the border on the two trace records.

The landing site (`apps/landing`) is built as one graph the reader scrolls through. Every section is a chapter in which the same fixed graph plane does one thing (forms, dissolves into the idea plate, drifts behind the quote wall, lights a sub-graph, gathers into clusters, glows behind the wedge, re-forms behind the pilot). The page refuses the stacked-blocks arrangement (hero, feature grid, cards, CTA): sections are ledger rows, tables, one-line rows and numbered theses, with cards only where elevation means a decision record.

**Key Characteristics:**
- Dark-first; light is a full second mode with the same geometry and a darker, quieter amber.
- Amber is rationed: the CTA fill, the graph core, "Blocked", the break, the quote's second sentence. Never decoration.
- Hairline rules (`--rule`, `--rule-strong`) do the work borders and shadows do elsewhere.
- Figures are authored SVG plates with live text, themed through tokens; no raster art.
- One motion vocabulary: one scroll clock, one easing (`--ease-signal`), end state under reduced motion.

## Colors

A three-role palette on a near-black ground: ink for reading, silver-blue for structure, amber for signal.

### Primary
- **Amber Signal** (`--accent`, #e5a23a dark / #8f5e12 light): the graph core, the Blocked status word and record border, the timeline break, the quote wall's second sentence, focus rings. In light mode the text/stroke amber darkens to #8f5e12 for contrast and the fill amber is `--accent-fill` #b87a1e.
- **Amber Fill** (`--accent-fill`, #e5a23a dark / #b87a1e light): the one filled control, "Join the waitlist", and text selection. Ink on it is `--on-accent` #0a0d19 in both modes.

### Secondary
- **Silver-Blue Structure** (`--blue`, #8fb0d6 dark / #3e6a99 light): figure strokes, graph edges and rings, the Accepted record's trace bar. Structure, never emphasis.

### Neutral
- **Ground** (`--ground`, #0a0d19 dark / #f2f1ec light): page and every surface; the nav when scrolled tints it 94% over the plane (opaque under 901px).
- **Ink** (`--ink`, #e8e6df dark / #0a0d19 light): all reading text.
- **Muted** (`--muted`, #8a8f9c dark / #5c6270 light): labels, captions, help lines, tag text.
- **Rule** (`--rule`, #2a3040 dark / #c9cbd2 light): default border colour, compartment lines, section rules, the nav's bottom edge.
- **Rule Strong** (`--rule-strong`, #5c6478 dark / #7f8490 light): inputs, outline buttons, tags, record borders, the scrolled nav's bottom edge. Meets a 3:1 non-text contrast floor in both modes.

### Named Rules
**The Signal Rule.** Amber marks a decision or a call, never ornament. If a screen has more than one amber fill, one of them is wrong.
**The One Ground Rule.** Every surface is `--ground`. There are no cards with their own background; a container is a border, not a tint.

## Typography

**Display Font:** IBM Plex Sans (with Helvetica Neue, Arial)
**Body Font:** IBM Plex Sans
**Label/Mono Font:** IBM Plex Mono (with SFMono-Regular, Menlo)
**Wordmark:** Cormorant Garamond Medium, 28px, used for the brand mark only.

**Character:** A grotesque set tight and heavy for headings, loose and quiet for reading; the mono voice annotates like a chart. Weights loaded: Sans 400/500/700, Mono 400, Cormorant 500.

### Hierarchy
- **Display** (`type-h1`, 700, clamp(2.5rem, 5.2vw, 4.75rem), 0.98, -0.035em): the hero H1, three lines at 15ch.
- **Wall** (`type-wall`, 500, clamp(2.5rem, 6.67vw, 6rem), 1.0, -0.03em, balanced): the full-bleed quote wall; the second sentence in amber.
- **Chapter numeral** (`type-chapter`, Mono 400, clamp(7.5rem, 15.3vw, 13.75rem), 0.85, tabular): the 01 04 09 10 beats, stroked outlines behind the copy.
- **Headline** (`type-h2`, 700, clamp(1.9rem, 3.2vw, 3rem), 1.02, -0.03em): one per chapter.
- **Title** (`type-h3`, 700, 1.35rem, 1.2): sub-heads ("Two ways in", "For investors").
- **Claim** (`type-claim`, 700, 1.25rem, 1.25): the left column of ledger rows.
- **Turn** (`type-turn`, 400, clamp(1.25rem, 2vw, 1.6rem), 1.35): the closing paragraph of the story chapter, 62ch.
- **Lede** (`type-lede`, 400, 1.15rem, 1.5): under the H1, 34ch.
- **Body** (400, 17px, 1.55): paragraphs at 46 to 58ch; bold lead-ins ("You bring.", "Month one.") carry structure inside a paragraph.
- **Label** (`type-label`, Mono 400, 12px, 0.08em, uppercase): form labels, record status lines, table heads, tags.

### Named Rules
**The Lead-In Rule.** Structure inside running text is a bold lead-in on the paragraph, not a label above it.

## Layout

A 12-column grid (`page-grid`) inside a 1280px container (`page-wrap`) with a fluid gutter `clamp(16px, 3vw, 40px)`; column gap 24px, 16px under 901px. Breakpoints: 901px (desktop above) and 561px (forms stack below). Copy sits in columns 1 to 5 or 1 to 8; figures in 6 to 12 or full width.

Chapters set their own rhythm by weight: heavy 240px above / 120px below, light 160 / 80, connective 200 / 100; mobile 120 / 60, 96 / 48, 112 / 56. Less below a heading than above it. Ledger sections use `compartments` (a 1px `--rule` grid made of gap and background) for rows and columns.

Pinned stages: the hero-to-idea stage travels 160vh (120vh under 901px) with the pin at `100svh`; the quote wall pins for 36vh on desktop and arrives on entry on mobile; the record stack is 140vh with the Blocked record sliding over Accepted. Under `prefers-reduced-motion` nothing pins and every stage renders its end state.

## Elevation & Depth

No shadows. Depth is the fixed graph plane behind the content: grid at 0.2, field at 0.6, glow at 0.9 parallax against scroll, plus 8px pointer displacement on fine pointers. Surfaces are flat; the two trace records are the only bordered containers, and the Blocked record's border is amber. The nav is the one layered chrome: sticky, `--ground` at rest, 94% ground with a 6px blur when scrolled (opaque under 901px), closed by a `--rule-strong` hairline.

### Named Rules
**The Flat Rule.** No box-shadow, no drop shadow, no gradient fills. The Tailwind radius and shadow scales are removed at the theme level.

## Shapes

Every corner is square (radius 0). Borders are 1px hairlines in `--rule` or `--rule-strong`; the Blocked record and the focus ring are the only amber edges. Figures are stroke drawings: circles for nodes, hairline edges, dashed amber for the break. Controls are rectangles at 48px (64px in the pilot bar).

## Components

### Buttons
- **Shape:** square (0px), 48px tall, 22px side padding; `lg` 64px tall, 32px padding. Sans 500, 15px, 0.01em.
- **Primary:** `--accent-fill` with `--on-accent` ink. Hover brightens 8%; active moves 1px down; busy desaturates and blocks the cursor.
- **Outline:** transparent with a 1px inset `--rule-strong` ring; hover ring in `--ink`. "Done" is the outline at rest with no pointer.
- **Focus:** 2px `--accent` outline, 3px offset, everywhere.

### Chips (Tags)
- **Style:** transparent, 1px `--rule-strong` border, `type-label` in `--muted`, 4px 8px padding. Positive wording only ("Showcase", "Exploration").

### Cards (Records)
- **Corner Style:** square.
- **Background:** `--ground`.
- **Border:** 1px `--rule-strong`; `--accent` when the status is Blocked.
- **Internal Padding:** 24px sides, 24px top, 28px bottom; 16px gap between the status line, the mono trace line (rules above and below), and the reason.

### Inputs / Fields
- **Style:** transparent, 1px `--rule-strong` stroke, square, 48px (64px `lg`), placeholder in `--muted`, `type-label` label above.
- **Focus:** the accent focus outline.
- **States:** loading blinks the marker and locks the button; success outlines the button and locks the form; invalid, error and timeout mark the state line in amber with "Try again". The state line is reserved so nothing shifts.

### Navigation
- 56px sticky bar: Cormorant wordmark, five text anchors (hidden under 901px), a bordered theme toggle, the filled CTA. Anchors into a pinned stage land on the stage's resolved end. Scrolled state described under Elevation.

### Graph Plane (signature)
A fixed SVG behind the page, one CSS pixel per unit, sized to the viewport plus 200px so mobile URL-bar resizes never rebuild it. It holds the grid, the node field, the constellation (rings, spokes, ten cast nodes), the amber sun and the G3 edge set. Chapter state on `html[data-chapter]` drives its end states through one transition spec; the hero scrub writes every frame directly with transitions off. The ten cast nodes travel to the idea plate's node positions and take the plate's colours as they land.

## Do's and Don'ts

### Do:
- **Do** put every colour through the tokens; components never hard-code a hex.
- **Do** keep one filled amber control per viewport and let the rest be text and hairlines.
- **Do** build sections as ledger rows, tables, one-line rows or numbered theses; use a bordered record only for a decision.
- **Do** draw figures in Figma with live text and export as SVG; theme them through `--blue`, `--accent`, `--ink`, `--muted`.
- **Do** render the end state when motion is off: no pins, plane static, text in place.

### Don't:
- **Don't** add radius, shadows, gradients or tinted surfaces.
- **Don't** add mono labels above paragraphs; fold structure into a bold lead-in.
- **Don't** write "not shipped" anywhere but the roadmap's eyebrow; tags stay positive.
- **Don't** name mechanisms on the page (databases, models, providers); sell by result.
- **Don't** let a second scroll clock or a second easing into the motion; the bus and `--ease-signal` are the only ones.
