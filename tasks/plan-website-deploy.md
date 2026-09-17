# Plan: design library, React landing page, Cloudflare deploy

Status: draft 2026-09-17. Owner: Dean. Source of the page today: `docs/brand/design-v5.tpl.html`
(gitignored; `docs/` is excluded by `.gitignore` line 42). Artifact v4:
https://claude.ai/artifact/5ckwrepU1Pq3aGnCiN84AK.

Order of work: design library first, then the React page built from it, then the deploy.

## State of Figma

The `Design v5` frames (`61:2` dark, `60:2` light, `67:2` mobile) are stale. They were captured
before review round 1 (artifact v3) and the motion fix (artifact v4, bug-666). Figma is a record
of the design, not a build input: nothing below reads from it. Re-capture is in phase C and does
not block the deploy.

## Decisions (recommended; change here before starting)

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Folder | `web/` at repo root, npm workspaces: `web/packages/design`, `web/apps/landing` | `docs/` is gitignored and `docs/brand/` holds drafts that must not go public (brief, Codex copy, sign-off page, generated comps). The repo is PUBLIC (`Nine-Sigma/sema`). A workspace lets a later product app consume the same design package. |
| D2 | Branch | `dean/feat/website` from `main` | CLAUDE.md naming. |
| D3 | Stack | Vite + React 19 + TypeScript, Tailwind CSS v4 (CSS-first `@theme` tokens) + shadcn/ui primitives on Radix, copied into `packages/design` (Dean, 2026-09-17: a CSS framework, reused for the Sema product UI) | Tokens stay CSS custom properties, so the port maps 1:1 from `design-v5.tpl.html`. shadcn owns no look; radius and shadow set to 0 in the theme. Product UI later takes Table, Dialog, Command, Select, Tabs, Tooltip from the same package. Rejected: Mantine/Chakra/MUI (opinionated defaults fight brutalist minimalist), Panda/vanilla-extract (no Radix-grade primitive ecosystem). |
| D4 | Cloudflare product | Pages with Git integration (Dean's call, 2026-09-17): Cloudflare's GitHub app builds and deploys on push; no GitHub Action, no API token in repo secrets | Per-branch previews for free; the waitlist endpoint later goes in `apps/landing/functions/api/waitlist.ts` (Pages Functions). Deploy gating moves to branch protection on `main` requiring `ci.yml`. |
| D5 | Fonts | Keep Google Fonts for v1 | Self-hosting IBM Plex + Cormorant is a follow-up tied to the analytics/consent decision. |
| D6 | Waitlist backend | Out of scope for the first deploy | The demo hook stays off in production (gate G2). Backend is its own task. |
| D7 | Figma library | Generate from the code package with `/figma-generate-library` after phase A | Code is the source of truth; Figma mirrors it. No hand-built Figma components. |

## Phase A: design library (`web/packages/design`)

Inputs: `docs/brand/philosophy.md`, `design-v5.tpl.html` (tokens, components, motion), the
`impeccable` critique snapshot, the review round 1 decisions in the `landing-page-handoff` skill,
and artifact v4 as rendered.

Deliverables:

1. `src/theme.css`: Tailwind v4 `@theme` block (single source of tokens; also exported as a
   typed TS object generated from the CSS), `@custom-variant dark`, base `@layer` with the
   resets the design needs (sharp corners, focus rings, `::selection`):
   - Color: `--bg`, `--ink`, `--muted`, `--rule`, `--rule-strong`, `--accent`, `--accent-fill`,
     `--on-accent`, `--blue`, for dark (default, `#0A0D19`) and light, with the light split
     (`--accent` `#8F5E12` for text/strokes, `--accent-fill` for buttons).
   - Type: IBM Plex Sans (400/500/700), IBM Plex Mono (400), Cormorant Garamond (500); the
     scale used by H1, lede, body, eyebrow, tag, mono labels.
   - Space, rule widths, `--ease`, durations (120/180/220/260/320/700 ms), the hero moment
     timeline, `prefers-reduced-motion` policy.
2. `src/components/`: shadcn CLI initialised in the package (`components.json`, radius 0);
   primitives `Button`, `Input`, `Label`, `Toggle` from shadcn, restyled to the theme; the rest
   authored: `EmailForm` with the six states (`loading`, `success`, `duplicate`, `invalid`,
   `error`, `timeout`), `Eyebrow`, `Tag`, `PullQuote`, `Record` (accepted, blocked; trace bar), `Mark` (typographic
   product marks), `Figure` (themed inline SVG with `<title>`/`<desc>`, mobile swap under 900px),
   `Constellation` (the hero, `constellation(seed)` ported from `inline-figures.mjs`, motion
   hooks `.node/.edge/.amber-edge/.ring/.ticks/.core`, `data-motion` gate), `SkipLink`,
   `Nav`, `Footer`. Motion keyframes live in `src/motion.css` (plain `@keyframes`, gated on
   `html[data-motion]`), not in component files.
3. `docs/`: a rules page carrying the constraints from the brief that shape components
   (amber budget, one eyebrow per three sections, cards only for the two trace records, 44px
   targets, sharp corners, motion gated on `html[data-motion]`, no raster with words).
4. Storybook or a Vite "gallery" page rendering every component in both themes. Gallery is
   cheaper and is enough for one consumer; Storybook if Dean wants it for the product app.
5. Figma: run `/figma-generate-library` from the package into a new page `Design library` in
   `yIRpVGpV7dOgrZlJ1N0se1`. Record node ids in the handoff skill.

Exit: every visual on the artifact v4 page maps to a token or a component; `npm run build` in
the package succeeds; both themes render in the gallery; `/impeccable critique` on the gallery.

## Phase B: React landing page (`web/apps/landing`)

1. Scaffold Vite + React + TS; depend on `@sema/design` via the workspace.
2. Port `design-v5.tpl.html` section by section into components from the library. Sections:
   hero, story, idea, ways, proof (records), changes, investors, pilot, roadmap, footer. Copy
   stays verbatim from v4 plus the four listed deviations until the picks land (gate G4).
3. Figures: `g3 g4 g5 g6 g6m g7 g7m g11 g12` as `Figure` instances; the inliner's crop and
   hex→token mapping moves into a build-time script or a one-time conversion to `.tsx`.
4. Motion: hero `pending` → release after `load` + `fonts.ready` + two frames, 4 s fallback;
   wordmark replays; `.pulse` pauses offscreen; reduced motion renders the end state.
5. Production hygiene: `<title>`, description, Open Graph and Twitter tags, canonical, favicon
   (G2, open), `og.png`, `robots.txt`, `404.html`. Gate the demo hook behind
   `import.meta.env.DEV`. `#privacy` and the `mailto:` contact need real targets from Dean.
6. Verify: build, `vite preview`, screenshot 1440 and 390 in both themes, diff against
   artifact v4.

## Phase C: Figma re-capture

After Dean approves artifact v4 and the React build matches it: serve `apps/landing/dist`,
`generate_figma_design` at 1440 dark, 1440 light, 390 dark; replace `61:2`, `60:2`, `67:2`.
Never hand-edit frames.

## Phase D: Cloudflare deploy (Pages, Git integration)

1. Dashboard (Dean): Workers & Pages → Create → Pages → Connect to Git → `Nine-Sigma/sema`.
   Settings: root directory `web`; build command `npm ci && npm run build --workspaces`;
   output directory `apps/landing/dist`; production branch `main`; build watch paths include
   `web/**`; Node version via `NODE_VERSION=22` env var or `web/.nvmrc`.
2. Security headers: `web/apps/landing/public/_headers` (copied to `dist` by Vite) with
   `Content-Security-Policy` allowing `fonts.googleapis.com` and `fonts.gstatic.com`,
   `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`.
   `404.html` in `public/` is served by Pages for unknown paths.
3. No deploy gating for the landing page (Dean, 2026-09-17). Pages deploys `main` on push;
   previews per branch. CI/CD-gated deployment starts when the Sema product UI is built.
4. `.github/workflows/ci.yml`: add `paths-ignore: [web/**]` to both triggers so website PRs do
   not run the Python suite. No `web` job for now.
5. Domain: Pages project → Custom domains. Dean owns DNS.

## Go-live gates

| Gate | Check | Owner |
|---|---|---|
| G1 | React build matches artifact v4 at 1440 and 390, both themes, except meta and paths | Claude |
| G2 | Demo hook cannot fire in the production build | Claude |
| G3 | No draft material under `web/` (brief, Codex copy, sign-off page, comps, Figma kits) | Claude, Dean confirms |
| G4 | Final copy applied from the sign-off picks (`copy-final.md`) | Dean picks, Claude applies |
| G5 | Waitlist backend live; six form states wired to real responses | separate task |
| G6 | Privacy page, contact address, analytics/consent decision, official Databricks and PostgreSQL marks | Dean |

None of the gates block a deploy (Dean, 2026-09-17). G1–G3 are checked by hand before merge;
G4–G6 can land after the first deploy. The site is unlinked until Dean shares it.

## Not in this plan

- Waitlist backend (Worker route, storage, notification). Design when Dean chooses the
  destination.
- Repo README exposure (the page withholds the mechanism; the README does not). Dean's call.
- G1 2k raster hero. The SVG constellation ships unless Dean commissions the raster.
