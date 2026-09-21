import { describe, expect, it, vi } from "vitest";
import worker from "./index";
import type { Env } from "./env";

/* worker-mailer imports cloudflare:sockets, which Node cannot resolve. */
vi.mock("./mail", () => ({ sendMail: vi.fn(async () => undefined) }));

type Call = { sql: string; values: unknown[] };
function fakeDb() {
  const calls: Call[] = [];
  const statement = (sql: string) => {
    const s = { sql, values: [] as unknown[], bind: (...v: unknown[]) => { s.values = v; return s; }, run: async () => { calls.push({ sql, values: s.values }); return { success: true }; } };
    return s;
  };
  const db = { prepare: statement, batch: async (stmts: ReturnType<typeof statement>[]) => stmts.map(() => ({ success: true })) };
  return { db: db as unknown as D1Database, calls };
}

function fakeCtx() {
  const pending: Promise<unknown>[] = [];
  const ctx = { waitUntil: (p: Promise<unknown>) => { pending.push(p); }, passThroughOnException: () => undefined, settled: () => Promise.all(pending) };
  return ctx as unknown as ExecutionContext & { settled: () => Promise<unknown> };
}

function fakeEnv(overrides: Partial<Env> = {}) {
  const fetched: Request[] = [];
  const db = fakeDb();
  const env = {
    ASSETS: { fetch: async (r: Request) => { fetched.push(r); return new Response(`asset:${new URL(r.url).pathname}`, { headers: { "content-type": "text/html", "x-robots-tag": new URL(r.url).pathname.startsWith("/ab/") ? "noindex" : "" } }); } },
    DB: db.db,
    SMTP_HOST: "smtp.titan.email",
    SMTP_PORT: "465",
    SMTP_USER: "waitlist@withsema.ai",
    FROM_ADDRESS: "waitlist@withsema.ai",
    TO_ADDRESS: "waitlist@withsema.ai",
    ...overrides,
  } as unknown as Env;
  return { env, fetched, db };
}

type Incoming = Parameters<typeof worker.fetch>[0];
const get = (url: string, headers: Record<string, string> = {}): Incoming => new Request(url, { headers }) as unknown as Incoming;

describe("worker fetch", () => {
  it("301s www to the canonical host, keeping path and query", async () => {
    const { env, fetched } = fakeEnv({ CANONICAL_HOST: "withsema.ai" });
    const res = await worker.fetch(get("https://www.withsema.ai/ab/a?x=1"), env, fakeCtx());
    expect(res.status).toBe(301);
    expect(res.headers.get("location")).toBe("https://withsema.ai/ab/a?x=1");
    expect(fetched).toHaveLength(0);
  });

  it("serves preview and canonical hosts without redirecting", async () => {
    const { env } = fakeEnv({ CANONICAL_HOST: "withsema.ai" });
    expect((await worker.fetch(get("https://withsema.ai/"), env, fakeCtx())).status).toBe(200);
    expect((await worker.fetch(get("https://abc-sema.dean.workers.dev/"), env, fakeCtx())).status).toBe(200);
  });

  it("passes non-root paths straight to assets", async () => {
    const { env, fetched } = fakeEnv();
    const res = await worker.fetch(get("https://withsema.ai/robots.txt"), env, fakeCtx());
    expect(await res.text()).toBe("asset:/robots.txt");
    expect(fetched).toHaveLength(1);
  });

  it("assigns an arm on / with no cookie while the test is on (equal split since 2026-09-21)", async () => {
    const { env } = fakeEnv();
    const res = await worker.fetch(get("https://withsema.ai/"), env, fakeCtx());
    const body = await res.text();
    expect(["asset:/", "asset:/ab/a", "asset:/ab/b"]).toContain(body);
    expect(res.headers.get("set-cookie")).toMatch(/^sema-ab=(control|a|b);/);
    expect(res.headers.get("cache-control")).toBe("no-store");
  });

  it("serves / untouched with no cookie when the test is off", async () => {
    const ab = await import("./ab");
    const spy = vi.spyOn(ab, "activeArms").mockReturnValue(["control"]);
    try {
      const { env } = fakeEnv();
      const res = await worker.fetch(get("https://withsema.ai/"), env, fakeCtx());
      expect(await res.text()).toBe("asset:/");
      expect(res.headers.get("set-cookie")).toBeNull();
    } finally {
      spy.mockRestore();
    }
  });

  it("serves a forced arm via ?ab= and pins it in the cookie", async () => {
    const { env } = fakeEnv();
    const res = await worker.fetch(get("https://withsema.ai/?ab=a"), env, fakeCtx());
    expect(await res.text()).toBe("asset:/ab/a");
    expect(res.headers.get("set-cookie")).toContain("sema-ab=a;");
    expect(res.headers.get("cache-control")).toBe("no-store");
    expect(res.headers.get("vary")).toContain("cookie");
  });

  it("records a fresh browser assignment in D1, in the background", async () => {
    const { env, db } = fakeEnv();
    const ctx = fakeCtx();
    await worker.fetch(get("https://withsema.ai/", { "user-agent": "Mozilla/5.0" }), env, ctx);
    await ctx.settled();
    const insert = db.calls.find((c) => c.sql.startsWith("INSERT INTO ab_assignment"));
    expect(insert?.values[0]).toMatch(/^(control|a|b)$/);
  });

  it("does not record returning, forced, HEAD or bot visits", async () => {
    const { env, db } = fakeEnv();
    const ctx = fakeCtx();
    await worker.fetch(get("https://withsema.ai/", { cookie: "sema-ab=a", "user-agent": "Mozilla/5.0" }), env, ctx);
    await worker.fetch(get("https://withsema.ai/?ab=b", { "user-agent": "Mozilla/5.0" }), env, ctx);
    await worker.fetch(get("https://withsema.ai/", { "user-agent": "Mozilla/5.0 (compatible; Googlebot/2.1)" }), env, ctx);
    await worker.fetch(new Request("https://withsema.ai/", { method: "HEAD", headers: { "user-agent": "Mozilla/5.0" } }) as unknown as Incoming, env, ctx);
    await ctx.settled();
    expect(db.calls.filter((c) => c.sql.startsWith("INSERT INTO ab_assignment"))).toHaveLength(0);
  });

  it("routes /api/waitlist to the handler", async () => {
    const { env } = fakeEnv();
    const res = await worker.fetch(get("https://withsema.ai/api/waitlist"), env, fakeCtx());
    expect(res.status).toBe(405);
  });
});
