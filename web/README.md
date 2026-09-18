# web

The Sema website workspace. npm workspaces, Node 22.

| Package | What |
|---|---|
| `packages/design` | `@sema/design`: Tailwind v4 tokens, shadcn-based components, figures, motion. See its README. |
| `apps/gallery` | Renders every component in the library. `npm run gallery`. |
| `apps/landing` | The landing page: design v5 ported section by section onto the library. `npm run landing`. Waitlist posts to `/api/waitlist` (not built yet; dev builds use the demo hook `error@` / `timeout@` / `duplicate@`). |

```sh
npm install
npm run typecheck
npm run build
npm run gallery
```

Deploy: Cloudflare Pages Git integration, root directory `web`, build `npm ci && npm run build --workspaces`,
output `apps/landing/dist`. See `tasks/plan-website-deploy.md`.
