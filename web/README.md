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

Deploy: one Cloudflare Worker, `sema`, serves `apps/landing/dist` as static assets and handles
`/` and `/api/waitlist` in `worker/` (`wrangler.toml` at this directory). `npm run deploy` builds
every workspace and runs `wrangler deploy`. See `tasks/plan-website-deploy.md`.

## The Worker (`worker/`, `wrangler.toml`)

`assets.run_worker_first = ["/", "/api/*"]` sends only those paths through the code; everything
else is served by the asset layer, where `apps/landing/public/_headers` applies.

- `worker/index.ts` — `www.withsema.ai` → `withsema.ai` 301 (`CANONICAL_HOST`), A/B assignment for
  `/`, and the `/api/waitlist` route. Preview URLs (`*.workers.dev`) are served as-is.
- `worker/ab.ts` — arm weights (`WEIGHTS`), cookie `sema-ab`, `?ab=<arm>` override.
- `worker/waitlist.ts` — `POST /api/waitlist` from both landing forms. Stores the signup in D1
  (`DB`, table `waitlist`, created on first use) with the A/B arm, then emails
  `waitlist@withsema.ai` with Reply-To set to the signup. 200 / 409 duplicate / 400 invalid /
  405 / 500 when `DB` is missing. A mail failure is logged, never surfaced.
- `worker/mail.ts` — SMTP through Titan (`smtp.titan.email:465`, the mailbox behind the
  Squarespace email plan) using `worker-mailer` over `cloudflare:sockets`. Needs the
  `SMTP_PASSWORD` secret; without it every send fails and is logged.
- Tests: `npm test` (vitest, `worker/*.test.ts`, mocked bindings). Types: `npm run typecheck:worker`.

Bindings are declared in `wrangler.toml`: D1 `sema-waitlist`, vars `CANONICAL_HOST`, `SMTP_*`,
`FROM_ADDRESS`, `TO_ADDRESS`, the secret `SMTP_PASSWORD`, and the custom
domains `withsema.ai` and `www.withsema.ai` (`wrangler deploy` creates their DNS records). Workers
cannot vary bindings between production and preview, so previews share the production D1.

One-time setup (done 2026-09-18; here for rebuilding the account):

1. `npx wrangler login`. Two accounts are visible; `account_id` in `wrangler.toml` picks Dean's.
2. `npx wrangler secret put SMTP_PASSWORD` with the Titan password of `SMTP_USER`. `SMTP_USER`
   must be a mailbox login, not an alias; `FROM_ADDRESS` may be one of its aliases.
3. `npx wrangler d1 create sema-waitlist`; put the id in `wrangler.toml`.
4. Delete any DNS record on `withsema.ai` and `www` (a custom domain refuses a hostname that has
   one), then `npm run deploy`.
5. Optional: Workers Builds (dashboard → Worker `sema` → Settings → Build) connected to
   `Nine-Sigma/sema`, root `web`, build `npm ci && npm run build --workspaces`, deploy
   `npx wrangler deploy`, non-production branch builds on for preview URLs.

Local check: `npm run build && npm run dev:worker` serves the built site with a local D1
(`http://localhost:8787`). Mail is skipped locally unless `SMTP_PASSWORD` is in `.dev.vars`, in
which case it really sends.
`?ab=a` forces an arm. Read signups with
`npx wrangler d1 execute sema-waitlist --remote --command "select email, variant, created_at from waitlist order by id"`.
