# web

The Sema website workspace. npm workspaces, Node 22.

| Package | What |
|---|---|
| `packages/design` | `@sema/design`: Tailwind v4 tokens, shadcn-based components, figures, motion. See its README. |
| `apps/gallery` | Renders every component in the library. `npm run gallery`. |
| `apps/landing` | The landing page: design v5 ported section by section onto the library. `npm run landing`. Waitlist posts to `/api/waitlist` (below); dev builds use the demo hook `error@` / `timeout@` / `duplicate@` instead. |

```sh
npm install
npm run typecheck
npm run build
npm run gallery
```

Deploy: Cloudflare Pages Git integration, root directory `web`, build `npm ci && npm run build --workspaces`,
output `apps/landing/dist`. See `tasks/plan-website-deploy.md`.

## Pages Functions (`functions/`) and the mail Worker (`workers/waitlist-mail`)

Pages reads `functions/` from this directory (the Pages root). `apps/landing/public/_routes.json`
limits them to `/` and `/api/*`.

- `functions/_middleware.ts` — A/B assignment for `/` (`WEIGHTS`, cookie `sema-ab`, `?ab=<arm>`)
  and the pages.dev → custom-domain redirect (env var `CANONICAL_HOST`).
- `functions/api/waitlist.ts` — `POST /api/waitlist` from both landing forms. Stores the signup in
  D1 (`DB`, table `waitlist`, created on first use) with the A/B arm, then notifies through the
  `MAIL` service binding. 200 / 409 duplicate / 400 invalid / 500 when `DB` is missing.
- `workers/waitlist-mail` — the only place an email binding can live (Pages Functions have none).
  Sends to `waitlist@withsema.ai` from `waitlist@withsema.ai`; no route, no `workers.dev` URL.

One-time setup (`npx wrangler login` once; the Cloudflare MCP server from the `cloudflare` plugin
covers the Pages steps that wrangler has no command for):

1. `npx wrangler email sending enable withsema.ai` (or Dashboard → Email → Email Service). Until
   the domain is onboarded, sends fail with `E_SENDER_NOT_VERIFIED`. Check: `npx wrangler email sending list`.
2. `cd workers/waitlist-mail && npx wrangler deploy`. Redeploy on change; the Pages Git build does
   not deploy Workers.
3. `npx wrangler d1 create sema-waitlist`. Schema is created by the first request.
4. Pages project `sema` → Bindings (Production and Preview): D1 `DB` → `sema-waitlist`; Service
   binding `MAIL` → `waitlist-mail`; optional Analytics Engine `AB` for A/B assignments.
   (Dashboard, or the Cloudflare API MCP server.)
5. Pages project `sema` → Custom domains → `withsema.ai` (and `www.withsema.ai` if wanted). Then
   env var `CANONICAL_HOST=withsema.ai` (Production only) so `sema-10w.pages.dev` 301s to the
   domain. Leave it unset on Preview.

Local check: `npm run pages:dev` serves `apps/landing/dist` with the Functions, a local D1, and the
`MAIL` binding; run `npx wrangler dev --port 8789 --inspector-port 9230` in `workers/waitlist-mail`
first (the local email binding writes the message under
`workers/waitlist-mail/.wrangler/tmp/email/`). Read signups with
`npx wrangler d1 execute sema-waitlist --remote --command "select email, variant, created_at from waitlist order by id"`.
