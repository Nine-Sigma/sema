/* A/B assignment for the landing page (tasks/plan-landing-copy.md, phase 4, D10/D11).

   Every visitor requests `/`. This middleware picks an arm once, stores it in the `sema-ab` cookie,
   and for a variant arm returns that arm's document from the static build in place of `/`. The
   address bar never changes. Arm paths must match `apps/landing/src/copy/variants.ts`.

   Cloudflare Pages reads `functions/` from the project root directory (`web`). `_routes.json` in
   the landing's `public/` limits invocation to `/` and `/api/*`. Pattern:
   developers.cloudflare.com/pages/how-to/use-worker-for-ab-testing-in-pages */

type ArmId = "control" | "a" | "b";

/* Relative weights. `control: 1, a: 0, b: 0` = test off: `/` is served untouched and no cookie is
   set. Change and redeploy to start the test; fold the winner into base.ts and reset to off. */
const WEIGHTS: Record<ArmId, number> = { control: 1, a: 0, b: 0 };

const PATHS: Record<ArmId, string> = { control: "/", a: "/ab/a", b: "/ab/b" };
const COOKIE = "sema-ab";
const MAX_AGE = 60 * 60 * 24 * 30;

type AnalyticsEngine = { writeDataPoint(point: { blobs?: string[]; doubles?: number[]; indexes?: string[] }): void };
/* CANONICAL_HOST (Pages env var, e.g. `withsema.ai`): once set, the production pages.dev host
   redirects there. Unset until the custom domain is attached; previews never match. */
type Env = { ASSETS: { fetch(request: Request): Promise<Response> }; AB?: AnalyticsEngine; CANONICAL_HOST?: string };
type Context = { request: Request; env: Env; next(): Promise<Response> };

const PAGES_HOST = "sema-10w.pages.dev";

export const onRequest = async ({ request, env, next }: Context): Promise<Response> => {
  const url = new URL(request.url);
  if (env.CANONICAL_HOST && url.hostname === PAGES_HOST) {
    url.hostname = env.CANONICAL_HOST;
    url.protocol = "https:";
    return Response.redirect(url.toString(), 301);
  }
  if (url.pathname !== "/" || (request.method !== "GET" && request.method !== "HEAD")) return next();

  const forced = parseArm(url.searchParams.get("ab"));
  if (!forced && activeArms().length < 2) return next();

  const existing = parseArm(readCookie(request.headers.get("cookie"), COOKIE));
  const arm = forced ?? existing ?? draw();
  if (!existing && !forced) env.AB?.writeDataPoint({ blobs: [arm], doubles: [1], indexes: [arm] });

  const upstream = arm === "control" ? await next() : await env.ASSETS.fetch(new Request(url.origin + PATHS[arm], request));
  const response = new Response(upstream.body, upstream);
  response.headers.append("set-cookie", `${COOKIE}=${arm}; Path=/; Max-Age=${MAX_AGE}; SameSite=Lax; Secure`);
  response.headers.append("vary", "cookie");
  response.headers.set("cache-control", "no-store");
  return response;
};

function activeArms(): ArmId[] {
  return (Object.keys(WEIGHTS) as ArmId[]).filter((id) => WEIGHTS[id] > 0);
}

/* Weighted draw over the active arms. */
function draw(): ArmId {
  const active = activeArms();
  let roll = Math.random() * active.reduce((sum, id) => sum + WEIGHTS[id], 0);
  for (const id of active) {
    roll -= WEIGHTS[id];
    if (roll < 0) return id;
  }
  return active[active.length - 1] ?? "control";
}

function parseArm(value: string | null | undefined): ArmId | undefined {
  return value !== null && value !== undefined && value in WEIGHTS ? (value as ArmId) : undefined;
}

function readCookie(header: string | null, name: string): string | undefined {
  if (!header) return undefined;
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return undefined;
}
