---
target: web/apps/gallery
total_score: 23
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 2
timestamp: 2026-09-18T01-45-29Z
slug: web-apps-gallery
---
Method: dual-agent (A: critA3 · B: critB3). Prior pairs (critique-a/b, critique-A-2/B-2) stalled on a headless-Chrome virtual-time hang (bug-672) and were killed; no findings from them were used.

## Design Health Score

Mode: Operate/Read (team verification surface). All 10 heuristics scored.

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | Two `ThemeToggle` instances with independent `useTheme` state; the second shows a stale `pressed`/`aria-label` after the first is clicked. No current-section indicator in `Nav`. |
| 2 | Match System / Real World | 3 | camelCase token names under `text-transform: uppercase` read "DARK RULESTRONG", "LIGHT ACCENTFILL". |
| 3 | User Control and Freedom | 2 | First toggle writes `localStorage["sema-theme"]` with no "follow system" reset; `EmailForm` locks on success/duplicate with no reset. |
| 4 | Consistency and Standards | 3 | Toggles disagree; `Wordmark` `aria-label="Sema, home"` but its click replays the hero animation. |
| 5 | Error Prevention | 3 | Empty `<a href="mailto:">Contact</a>`; `#privacy` links to itself. |
| 6 | Recognition Rather Than Recall | 1 | No specimen carries its component/utility/prop name except the `type-label` line. |
| 7 | Flexibility and Efficiency | 2 | `?theme=` pin and anchor nav are good; nav anchors `max-lg:hidden` under 901px with no replacement; no copy-hex; no per-component anchors. |
| 8 | Aesthetic and Minimalist Design | 3 | Restraint is real; hero, waitlist CTA, and marketing footer are noise on a tool surface. |
| 9 | Error Recovery | 3 | `EmailForm` messages specific and retryable; `aria-invalid` `Input` specimen shows amber border with no message. |
| 10 | Help and Documentation | 1 | One sentence on demo hooks, one on motion durations; no props, usage, or do/don't. |
| **Total** | | **23/40** | **Acceptable (58%)** |

## Design Specificity Verdict

**LLM assessment.** Authored for Sema, unmistakably: Plex/Cormorant pairing, Signal Cartography palette, mono eyebrow voice on `Label`/`Tag`/`Mark`, the `Record` amber-border rule, Sema's own specimen copy. The problem is what it was authored *as*: `gallery.tsx` is a compressed landing page (hero with `Constellation`, "Join the waitlist" in `Nav`, marketing `Footer`) on a surface whose mode is Operate/Read. Consequences: chrome is indistinguishable from specimen (section h2 uses `type-h3`, then a `type-h3` specimen sits 40px below it; the `Eyebrow` specimen reads as a caption for the `PullQuote`); every section uses the same `border-t + h2 + grid gap-8` envelope regardless of content; the landing hero costs the whole first viewport; no specimen is named; no light/dark side-by-side despite the brief being "see every component in both themes".

**Deterministic scan.** `detect.mjs --json web/apps/gallery` and `web/packages/design`: 0 findings, exit 0, also with `--no-config`. URL mode not run (puppeteer not installed). The detector caught nothing the review missed; there are no false positives to flag. Source-level facts from B that the review confirms: unlabeled inputs `in2`/`in3` (`gallery.tsx:127-128`, placeholder only; `in2` has `aria-invalid` with no error text); `SkipLink href="#tokens"` targets a `<section>` without `tabindex="-1"`; no `theme-color` meta; zero `prefers-reduced-motion` rules in CSS, reduced motion is JS-only via `useMotionGate`, and two `infinite` loops (`sema-pulse`, `sema-blink`) are ungated; saved theme is applied in a post-mount `useEffect` so first paint can flash the wrong theme; bundle JS 318 KB raw / 94 KB gzip, CSS 24 KB / 6 KB gzip, 9 figure SVGs (42 KB) inlined in JS.

**Visual overlays.** Not available: no mutable browser tab in this session, so no `[Human]` overlay was injected. Fallback signal: headless captures at 1440 (dark and `?theme=light`), 1440×7500 full page, and 500 wide, all under `scratchpad/critA/`. The 390px iframe capture failed twice (React had not mounted when Chrome fired); the 500px capture stands in.

Where A and B agree: the desynchronised toggles (A: P0; B: "0 listeners, independent useState"), the missing nav under 901px, the empty `mailto:`, the unlabeled inputs, the motion gate holding the hero blank. Where B added what A missed: no CSS `prefers-reduced-motion` and the ungated infinite loops; the post-mount theme apply; the `SkipLink` target. Where A added what B cannot see: invisible swatches (`dark ground`, `dark onAccent`, `light ink`, `light onAccent` render as empty boxes, confirmed in `top-dark.png`), heading-vs-specimen size inversion, blank first viewport.

## Overall Impression

The library underneath is disciplined and the gallery proves it: every component re-themes with zero drift, motion is authored and gated, `EmailForm` is a model of state design. The gallery itself is the landing page wearing a lab coat. It shows the components but never names them, never shows both themes at once, and opens on 420px of nothing because the hero waits for fonts and two animation frames. Single biggest opportunity: turn it from a second landing page into a ledger, one row per component, dark left and light right, with the prop caption the `type-label` specimen already models.

## What's Working

1. **Token discipline is total.** `theme.css` routes every color through `--ground/--ink/--muted/--rule/--accent/--blue`, maps the shadcn vocabulary onto the same variables, and deletes the radius/shadow scales. The `?theme=light` capture shows every component re-theming with zero drift, including SVG figures (`stroke="var(--blue)"`).
2. **Motion is authored, gated, and legible.** One moment, two responses, one loop, all behind `html[data-motion]`; `motion.durations` in `tokens.ts` matches its CSS to the millisecond. (Caveat from B: the gate is JS-only; see P2 below.)
3. **`EmailForm` state design.** Six states, reserved `min-h-[1.4em]` status line so nothing shifts, `role="status" aria-live="polite"`, distinct glyph per state, retry clears on input. This is the bar for the rest of the library.

## Priority Issues

**[P0] Two `ThemeToggle`s desynchronise.** `gallery.tsx` renders `ThemeToggle` in `Nav` (`:71`) and again in Controls (`:122`); `use-theme.ts` keeps `useState` per instance with no storage or event sync. *Why it matters:* after one click the second toggle shows the wrong pressed state and announces the wrong `aria-label`; the gallery's stated job (verify both themes) is undermined by its own theme control. *Fix:* make `useTheme` subscribe to `data-theme` on `document.documentElement` (a `MutationObserver` or a module-level emitter) and derive state from `resolve()`; also apply the saved theme before first paint (inline script or `useLayoutEffect`). *Suggested command:* /impeccable harden.

**[P1] No specimen is named.** Type scale, Controls, and Tokens carry no component/utility/prop labels; only the `type-label` line self-describes. *Why it matters:* heuristics 6 and 10 both score 1; a verifier cannot tell `variant="outline"` from `variant="done"` without reading `gallery.tsx`. *Fix:* give `Section` a `Specimen` child using the existing `compartments` utility: rendered component left, `type-label` mono caption right (`Button · variant="done" · disabled`, `type-turn · clamp(1.25rem, 2vw, 1.6rem)`), copying the form of "Label · IBM Plex Mono 12 / 0.08em". Also label `in2`/`in3` and pair the `aria-invalid` specimen with its message. *Suggested command:* /impeccable document.

**[P1] Gallery chrome is the landing page.** Hero (`sema-hero` + `Constellation`), `Nav` CTA "Join the waitlist", marketing `Footer`. *Why it matters:* the hero costs the whole first viewport (`top-dark.png` is an empty grid above the fold because the hero is motion-gated on `load` + `fonts.ready` + two rAFs, up to 4 s on slow fonts); chrome and specimen share one type; the page ends on its weakest, least tool-like moment. *Fix:* replace the hero with a 64px strip under `Nav` (`type-label` "Sema design library · 14 components · dark/light" left, light/dark segmented control right); move `Constellation` into Figures as a named specimen with its replay button; drop the CTA from `Nav`; render `Footer` as a labelled specimen, not page chrome. *Suggested command:* /impeccable distill.

**[P2] Token section does not show tokens in context.** `Swatch` is a static hex fill; `dark ground` / `dark onAccent` are invisible in dark mode, `light ink` / `light onAccent` invisible in light mode (confirmed in `top-dark.png`). *Why it matters:* the brief is "both themes"; the grid shows hex values, not the pairings the library depends on (ink on ground, on-accent on accent-fill). *Fix:* render each token as a pairing cell with its contrast ratio in the mono caption; show both themes as two columns via a container-scoped `[data-theme="light"]` block in `theme.css`; fix the uppercase run-on names. *Suggested command:* /impeccable colorize.

**[P2] Reduced motion and narrow nav.** (a) Zero `prefers-reduced-motion` rules in CSS; `sema-pulse` (6 s infinite alternate) and `sema-blink` (1 s infinite) run under reduced motion, and `html { scroll-behavior: smooth }` has no override. (b) `nav.tsx` sets `max-lg:hidden` on anchors with no fallback; under 901px the reader has wordmark + toggle + CTA and no way to reach Figures short of scrolling 6000px. *Why it matters:* (a) is a WCAG 2.3.3 gap in the library, not only the gallery; (b) removes orientation on any half-width window. *Fix:* add `@media (prefers-reduced-motion: reduce)` in `motion.css` that sets `animation: none` on `.sema-pulse` and `.sema-state::before` and `scroll-behavior: auto`; in the gallery, add a sticky `type-label` anchor row (`overflow-x: auto`, `whitespace-nowrap`) under the bar at `max-lg`. *Suggested command:* /impeccable adapt.

## Persona Red Flags

**Alex (power user):** cannot copy a hex (`Swatch` renders a plain `<p class="font-mono">`); cannot reset `EmailForm` after "On the list" without reload; `?theme=` pin is one-shot, the toggle then writes `localStorage` and the pin silently loses on the next visit; `#controls` covers 7 components with no per-component anchor.

**Sam (screen reader/keyboard):** second `ThemeToggle` announces the opposite of the real state; `Wordmark` `aria-label="Sema, home"` triggers `replay()` (label does not describe the action); `SkipLink href="#tokens"` targets a `<section>` without `tabindex="-1"` (focus does not move in Safari); `<Button aria-busy disabled>Adding…</Button>` is announced with no context; `<a href="mailto:">Contact</a>` is an empty link. Figures pass: `role="img"` + `aria-labelledby` title/desc on all 9; `Constellation` is `aria-hidden`.

**Riley (stress tester):** load the tab in the background and the hero stays blank until foregrounded (`useMotionGate` needs two rAFs; the 4 s fallback also ends in rAF, so there is no non-rAF path); toggle theme 10 times fast and the two icons' `rotate-180` go out of phase (same root cause as P0); `Figure variant="wide"` under 720px gives a horizontal scroll with `scrollbar-width: thin` and no hint, so the right half of `figures.g5` is found by accident. Passes: `timeout@x` demo, 254-char `maxLength`, 900px `Record` stacking.

## Minor Observations

- `Section` `gap-8` between h2 and content equals the gap between specimens, so the heading does not own its block; h2 at `type-h3` (1.35rem) is smaller than most specimens beneath it.
- Placeholders "[contact address]" (`MESSAGES.error`), "[N]"/"[study]" (Records) should be marked as placeholders so they cannot ship.
- Controls `ThemeToggle` specimen has no `aria-label` override, so two controls share one accessible name.
- `PullQuote` `lg:pr-[28%]` is a landing-page measure; in the gallery it leaves an unexplained empty right margin.
- `index.html` loads Google Fonts with `preconnect` to `fonts.googleapis.com` only, none to `fonts.gstatic.com`; no `theme-color` meta.
- `Footer` "Join the waitlist" links to `#forms`, a demo specimen.
- 9 figure SVGs (42 KB) are inlined in the JS bundle; fine for a gallery, worth a note for the landing app.

## Questions to Consider

1. If a headless verifier cannot screenshot the gallery without a blank hero, is it a verification surface at all, or a second landing page with the same motion gate?
2. Why does the gallery need a `Nav` CTA, a `Footer`, and a hero when the only reader is the person who wrote the library? What would a single `compartments` ledger look like: one row per component, dark left, light right?
3. `Record` teaches its rule by behaving. Which other rules in the source comments (Eyebrow budget, "amber is spent on the one filled control", "positive tag words only") could the gallery display or enforce, so the gallery becomes where the rules live?
