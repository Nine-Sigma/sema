---
target: web/apps/gallery
total_score: 28
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 2
timestamp: 2026-09-18T03-44-44Z
slug: web-apps-gallery
---
Method: dual-agent (A: critA4 · B: critB4). Re-run after the ledger rebuild; captures used `--force-prefers-reduced-motion` so the gated hero art renders its end state.

## Design Health Score

Mode: Operate/Read. All 10 heuristics scored.

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | `IdentityStrip` shows theme and source; no current-section indicator in `Nav`/`AnchorRow`; Reset/Replay give no acknowledgement. |
| 2 | Match System / Real World | 3 | Caption vocabulary fits the engineer audience; "Rule / Ground 1.47:1" with no qualifier reads like a failure. |
| 3 | User Control and Freedom | 3 | Reset, Replay, "Follow system" exist; the `ThemeToggle` specimens mutate and persist the global theme. |
| 4 | Consistency and Standards | 2 | `CaptionButton` is styled identically to the static specimen name; anchors "Tokens"/"Forms" vs headings "Color tokens"/"Email form"; 14 of ~31 specimens break the "dark left, light right" contract (`single`/`stack`). |
| 5 | Error Prevention | 3 | Suffixed ids, motion gate, reduced-motion fallbacks solid; nothing stops a specimen toggle from re-theming the frame. |
| 6 | Recognition Rather Than Recall | 3 | Pairing cells fully labelled; the five Button variants have no per-item label. |
| 7 | Flexibility and Efficiency | 2 | No per-specimen deep links, no keyboard theme shortcut, `?theme=` undocumented on the page. |
| 8 | Aesthetic and Minimalist Design | 3 | Restraint holds; page is ~9,800px at 1440. |
| 9 | Error Recovery | 3 | Failing ratios turn amber with "below 4.5" text; library error copy still carries `[contact address]` (placeholder by rule). |
| 10 | Help and Documentation | 3 | Per-row notes from the source comments; `theme.css`/`tokens.ts` named but not linked. |
| **Total** | | **28/40** | **Good (70%)** |

## Design Specificity Verdict

**LLM assessment.** Authored for Sema as a verification tool, not interchangeable: `compartments` hairline rows, `type-label` caption column, one amber fill per row, product copy as specimen copy. Two mechanisms make it a verifier rather than a showcase: `PairingCell` paints from the live CSS variable and computes the ratio from `tokens.ts` (drift is visible); `TypeRow` reads `getComputedStyle` under a `ResizeObserver` (captions cannot lie). Coherence holds in both themes. Structural gap: the header promises "dark left, light right" but the 10 type roles, Motion, Eyebrow, PullQuote and SkipLink are `single` (live theme only), and lg EmailForm, Constellation, three wide figures, Nav and Footer are `stack`.

**Deterministic scan.** `detect.mjs` on `web/apps/gallery` (11 files) and `web/packages/design` (26 files): 0 findings, also with `--no-config`; no ignore config or inline suppressions exist. DOM (3 dumps, byte-identical): h1 ×1, h2 ×7, 24 buttons + 10 inputs with 0 unnamed, 73 ids with 0 duplicates across 40 themed cells, 27 SVGs all titled or `aria-hidden`, no `tabindex>0`, skip link → `main#ledger[tabindex=-1]`, `theme-color` ×2, 2 reduced-motion blocks covering all 4 `infinite` loops, `theme-boot.js` pre-paint, `useTheme` via `useSyncExternalStore` on `html[data-theme]`. Bundle JS 330,856 B (98,920 gz), CSS 26,470 B (6,318 gz).

**Visual overlays.** Not available (no mutable browser tab). Fallback: captures at 1440 dark/light (top + 10,000px full), 500 wide, and 1000/1040 wide under `scratchpad/critA4/`.

Where A and B agree: no duplicate ids, all controls named, reduced motion covered, skip target correct. B adds nothing A missed this round; the detector fired on nothing. A adds what the DOM cannot show: the 901–1100px pairing-grid collision, the missing affordance on caption controls, the single/stack contract gap.

## Overall Impression

The ledger works as a verification surface: measured ratios on the first fold, computed type captions, both themes for the components that vary most. What remains is finish: two P1s (a responsive collision at the first supported lg viewport, and caption controls that look like labels), and a contract gap where the type roles are the one theme-sensitive section shown in a single theme. Biggest opportunity: let the ledger say pass or fail as a rollup.

## What's Working

1. **`PairingCell` is a drift detector.** Fill from `var(--ink)`, ratio from `colors[theme]` in `tokens.ts`, explained in the intro; a mismatch between `theme.css` and `tokens.ts` shows on the first fold.
2. **`useComputedType` cannot lie.** `getComputedStyle` + `fonts.ready` + `ResizeObserver`; if Google Fonts fails, the caption reports the fallback family.
3. **Identity discipline in both themes.** One amber fill per row, records the only elevated boxes, mono captions never mixed with sans.

## Priority Issues

**[P1] Pairing grid collides at 901–~1100px.** `tokens-ledger.tsx` `lg:grid-cols-4` inside a `1fr 1fr 18rem` cell: at 1000/1040px "On accent" abuts "Rule strong" and the hex line overflows (`w1000.png`). `--breakpoint-lg` is 901px, so this is the first supported lg viewport. *Fix:* `repeat(auto-fill, minmax(7.5rem, 1fr))`, `min-w-0`, `[overflow-wrap:anywhere]` on the hex line. *Command:* /impeccable layout.

**[P1] Interactive captions have no affordance.** `CaptionButton` (Reset, Replay, Follow system) and "Top" are `type-label text-ink`, identical to the static specimen name; only hover underlines them; "Top" is ~28px tall. *Fix:* persistent underline with `decoration-rule-strong underline-offset-3`, 44px targets. *Command:* /impeccable clarify.

**[P2] Dark/light contract broken for 14 specimens.** Type roles are the most theme-sensitive section and the one shown in a single theme. *Fix:* type rows as `pair` (caption computed once), and a caption tag on `single`/`stack` rows ("live theme only", "stacked"). *Command:* /impeccable harden.

**[P2] Specimen `ThemeToggle` mutates and persists the global theme.** Clicking it re-themes the frame and writes `localStorage`; its cell stays dark, so the icon contradicts its surroundings. *Fix:* derive `pressed` from the cell theme with a no-op handler, restore the state-carrying `aria-label`. *Command:* /impeccable harden.

**[P3] No control-state coverage, no verdict rollup; naming/navigation drift.** Hover/focus/active never shown for Button/Input; no "7/7 pairings AA" line; anchors vs headings differ; no `aria-current`; no per-specimen `id`. *Command:* /impeccable document, then /impeccable onboard.

## Persona Red Flags

**Alex:** no deep link per specimen; `?theme=` undocumented on the page; no copy-to-clipboard for hex or computed type; one specimen-toggle click pins the theme until "Follow system" is found.

**Sam:** `ThemeToggle` specimens override the state-carrying label with "Switch theme (specimen dark-0)"; `Record` `<article>`s have no accessible name; sticky `AnchorRow` under 901px has no `scroll-padding-top`, so tabbed content can land under it; the SkipLink specimen is a second skip link mid-tab-order. Focus ring and `aria-labelledby` sections are correct.

**Riley:** 901–1100px collision; at 500px the anchor row masks "Shell" to "SHE" with the fade as the only cue; `error@x.com` surfaces `[contact address]`; Replay under reduced motion is a silent no-op; specimen toggle persists theme across reloads.

## Minor Observations

- Records intro says "the only bordered boxes on the page"; in the gallery, swatches, inputs, tags, toggles and outline buttons are also bordered.
- `IdentityStrip` wraps to three lines at 500px; drop "dark left, light right" under `sm`.
- Cormorant Garamond 500 reads lighter on dark; worth a note in the Wordmark caption.
- Google Fonts is a runtime dependency of a verification page; self-host under `public/`.
- Motion `<dl>` shows 8 durations in one row; scans as a choice.

## Questions to Consider

1. If this page exists to verify, why can it never say "fail"? The measurements exist; a rollup would make the ledger a gate.
2. Should a specimen ever have side effects on its host page? Is the `ThemeToggle` specimen a gallery bug or a missing theme scope in `useTheme`?
3. Type roles are the most theme-sensitive thing in this palette. Why are they the one section shown in a single theme?
