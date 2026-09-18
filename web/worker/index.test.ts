import { describe, expect, it, vi } from "vitest";
import worker from "./index";
import type { Env } from "./env";

/* worker-mailer imports cloudflare:sockets, which Node cannot resolve. */
vi.mock("./mail", () => ({ sendMail: vi.fn(async () => undefined) }));

function fakeEnv(overrides: Partial<Env> = {}) {
  const fetched: Request[] = [];
  const env = {
    ASSETS: { fetch: async (r: Request) => { fetched.push(r); return new Response(`asset:${new URL(r.url).pathname}`, { headers: { "content-type": "text/html" } }); } },
    DB: undefined,
    SMTP_HOST: "smtp.titan.email",
    SMTP_PORT: "465",
    SMTP_USER: "waitlist@withsema.ai",
    FROM_ADDRESS: "waitlist@withsema.ai",
    TO_ADDRESS: "waitlist@withsema.ai",
    ...overrides,
  } as unknown as Env;
  return { env, fetched };
}

type Incoming = Parameters<typeof worker.fetch>[0];
const get = (url: string, headers: Record<string, string> = {}): Incoming => new Request(url, { headers }) as unknown as Incoming;

describe("worker fetch", () => {
  it("301s www to the canonical host, keeping path and query", async () => {
    const { env, fetched } = fakeEnv({ CANONICAL_HOST: "withsema.ai" });
    const res = await worker.fetch(get("https://www.withsema.ai/ab/a?x=1"), env);
    expect(res.status).toBe(301);
    expect(res.headers.get("location")).toBe("https://withsema.ai/ab/a?x=1");
    expect(fetched).toHaveLength(0);
  });

  it("serves preview and canonical hosts without redirecting", async () => {
    const { env } = fakeEnv({ CANONICAL_HOST: "withsema.ai" });
    expect((await worker.fetch(get("https://withsema.ai/"), env)).status).toBe(200);
    expect((await worker.fetch(get("https://abc-sema.dean.workers.dev/"), env)).status).toBe(200);
  });

  it("passes non-root paths straight to assets", async () => {
    const { env, fetched } = fakeEnv();
    const res = await worker.fetch(get("https://withsema.ai/robots.txt"), env);
    expect(await res.text()).toBe("asset:/robots.txt");
    expect(fetched).toHaveLength(1);
  });

  it("serves / untouched with no cookie when the test is off", async () => {
    const { env } = fakeEnv();
    const res = await worker.fetch(get("https://withsema.ai/"), env);
    expect(await res.text()).toBe("asset:/");
    expect(res.headers.get("set-cookie")).toBeNull();
  });

  it("serves a forced arm via ?ab= and pins it in the cookie", async () => {
    const { env } = fakeEnv();
    const res = await worker.fetch(get("https://withsema.ai/?ab=a"), env);
    expect(await res.text()).toBe("asset:/ab/a");
    expect(res.headers.get("set-cookie")).toContain("sema-ab=a;");
    expect(res.headers.get("cache-control")).toBe("no-store");
    expect(res.headers.get("vary")).toContain("cookie");
  });

  it("records only fresh assignments in Analytics Engine", async () => {
    const writeDataPoint = vi.fn();
    const { env } = fakeEnv({ AB: { writeDataPoint } as unknown as Env["AB"] });
    await worker.fetch(get("https://withsema.ai/?ab=b", { cookie: "sema-ab=a" }), env);
    expect(writeDataPoint).not.toHaveBeenCalled();
  });

  it("routes /api/waitlist to the handler", async () => {
    const { env } = fakeEnv();
    const res = await worker.fetch(get("https://withsema.ai/api/waitlist"), env);
    expect(res.status).toBe(405);
  });
});
