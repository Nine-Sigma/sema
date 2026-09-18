/* The landing site Worker: static assets from apps/landing/dist plus two dynamic paths.
   `run_worker_first` in wrangler.toml sends only `/` and `/api/*` here; every other request is
   served by the asset layer directly (`_headers` applies there). */

import { AB_COOKIE, AB_MAX_AGE, ARM_PATHS, activeArms, draw, parseArm, readCookie } from "./ab";
import type { Env } from "./env";
import { handleWaitlist } from "./waitlist";

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url);
    const redirect = canonicalRedirect(url, env.CANONICAL_HOST);
    if (redirect) return redirect;
    if (url.pathname === "/api/waitlist") return handleWaitlist(request, env);
    if (url.pathname === "/" && (request.method === "GET" || request.method === "HEAD")) return serveLanding(request, url, env);
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;

function canonicalRedirect(url: URL, canonicalHost: string | undefined): Response | undefined {
  if (!canonicalHost || url.hostname !== `www.${canonicalHost}`) return undefined;
  url.hostname = canonicalHost;
  url.protocol = "https:";
  return Response.redirect(url.toString(), 301);
}

async function serveLanding(request: Request, url: URL, env: Env): Promise<Response> {
  const forced = parseArm(url.searchParams.get("ab"));
  if (!forced && activeArms().length < 2) return env.ASSETS.fetch(request);

  const existing = parseArm(readCookie(request.headers.get("cookie"), AB_COOKIE));
  const arm = forced ?? existing ?? draw();
  if (!existing && !forced) env.AB?.writeDataPoint({ blobs: [arm], doubles: [1], indexes: [arm] });

  const upstream = await env.ASSETS.fetch(new Request(url.origin + ARM_PATHS[arm], request));
  const response = new Response(upstream.body, upstream);
  response.headers.append("set-cookie", `${AB_COOKIE}=${arm}; Path=/; Max-Age=${AB_MAX_AGE}; SameSite=Lax; Secure`);
  response.headers.append("vary", "cookie");
  response.headers.set("cache-control", "no-store");
  return response;
}
