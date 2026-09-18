# @sema/design

Sema's design library. Tailwind CSS v4 tokens plus React components, shared by the landing page
and, later, the Sema product UI. Identity: "Signal Cartography" (`docs/brand/philosophy.md`,
not in the repo): near-black ground, amber signal, silver-blue structure.

## Use

```css
/* app.css */
@import "@sema/design/theme.css";
@import "@sema/design/motion.css";
@source "../../../packages/design/src"; /* workspace packages sit under node_modules, which Tailwind skips */
```

```tsx
import { Button, EmailForm, Record, Figure, figures } from "@sema/design";
import { colors, motion } from "@sema/design/tokens";
```

Load the fonts in the page head (`fontsHref` in `tokens.ts` has the exact request).

## What is in it

| Path | Contents |
|---|---|
| `src/theme.css` | Tokens (`:root` dark, light by `data-theme="light"` or system), `@theme` (fonts, ease, breakpoints, radius and shadow scales removed), shadcn aliases, base layer, type utilities (`type-h1` … `type-wordmark`), layout utilities (`page-wrap`, `page-grid`, `compartments`) |
| `src/motion.css` | Keyframes and the `html[data-motion]` rules: hero moment, form state line, record trace, hero pulse |
| `src/tokens.ts` | Typed mirror of the CSS tokens for code that cannot read CSS variables |
| `src/components/ui/` | shadcn primitives restyled: `Button` (default, outline, done; default, lg), `Input`, `Label`, `Toggle` |
| `src/components/` | `Constellation`, `EmailForm`, `Eyebrow`, `Figure`, `Footer`, `Mark`, `Nav`, `PullQuote`, `Record`, `SkipLink`, `Tag`, `ThemeToggle`, `Wordmark` |
| `src/hooks/` | `useMotionGate` (owns `html[data-motion]`), `useTheme` (owns `html[data-theme]`), `useSeen` (first-view trigger) |
| `figures/*.svg` | Figma exports with live text. `npm run figures` themes them into `src/figures/index.ts`; never edit the generated file |

Gallery: `npm run gallery` from `web/` renders everything in the live theme. `?theme=dark|light`
pins the theme for screenshots.

## Rules the components enforce

These come from the landing page brief and review rounds. Change them there first.

- Sharp corners, flat surfaces. The radius and shadow scales are removed from the theme; `rounded-*` and `shadow-*` utilities do not exist.
- One accent. Amber is spent on: the filled button, the pull-quote rule, the Blocked record border and status word, invalid/error/timeout form markers, focus rings. Nothing else.
- Elevation means "a record was written". `Record` is the only bordered box; do not add cards.
- Eyebrows: at most one per three sections.
- Positive tags only: Exploration, Showcase, Shipped. Never "not shipped".
- Targets are 44px or larger: `Toggle` is 44px, nav anchors and footer links are padded to it.
- Motion is gated on `html[data-motion]`. Under `prefers-reduced-motion` the attribute is never set and the end state renders. Only `useMotionGate` writes the attribute.
- Every color comes through a token. No hex in components.
- Figures with words are drawn in Figma with real text and exported as SVG. No raster art carrying text.
- Components never use a `dark:` variant; the tokens carry the theme.
