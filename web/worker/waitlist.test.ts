import { beforeEach, describe, expect, it, vi } from "vitest";
import { handleWaitlist } from "./waitlist";
import type { Env } from "./env";
import { sendMail } from "./mail";

vi.mock("./mail", () => ({ sendMail: vi.fn(async () => undefined) }));
const sendMailMock = vi.mocked(sendMail);

type Call = { sql: string; values: unknown[] };

function fakeDb(runError?: Error) {
  const calls: Call[] = [];
  const statement = (sql: string) => {
    const s = { sql, values: [] as unknown[], bind: (...v: unknown[]) => { s.values = v; return s; }, run: async () => { calls.push({ sql, values: s.values }); if (runError) throw runError; return { success: true }; } };
    return s;
  };
  const db = { prepare: statement, batch: async (stmts: ReturnType<typeof statement>[]) => { for (const s of stmts) calls.push({ sql: s.sql, values: [] }); return stmts.map(() => ({ success: true })); } };
  return { db: db as unknown as Env["DB"], calls };
}

function fakeEnv(overrides: Partial<Env> = {}): { env: Env; calls: Call[] } {
  const { db, calls } = fakeDb();
  const env = {
    ASSETS: { fetch: async () => new Response("asset") },
    DB: db,
    SMTP_HOST: "smtp.titan.email",
    SMTP_PORT: "465",
    SMTP_USER: "waitlist@withsema.ai",
    SMTP_PASSWORD: "pw",
    FROM_ADDRESS: "waitlist@withsema.ai",
    TO_ADDRESS: "waitlist@withsema.ai",
    ...overrides,
  } as unknown as Env;
  return { env, calls };
}

const post = (body: unknown, headers: Record<string, string> = {}) =>
  new Request("https://withsema.ai/api/waitlist", { method: "POST", headers: { "content-type": "application/json", ...headers }, body: typeof body === "string" ? body : JSON.stringify(body) });

describe("handleWaitlist", () => {
  beforeEach(() => sendMailMock.mockReset().mockResolvedValue(undefined));

  it("rejects non-POST with 405 and an allow header", async () => {
    const { env } = fakeEnv();
    const res = await handleWaitlist(new Request("https://withsema.ai/api/waitlist"), env);
    expect(res.status).toBe(405);
    expect(res.headers.get("allow")).toBe("POST");
  });

  it("returns 400 for a malformed body or invalid email", async () => {
    const { env } = fakeEnv();
    expect((await handleWaitlist(post("not json"), env)).status).toBe(400);
    expect((await handleWaitlist(post({ email: "nope" }), env)).status).toBe(400);
    expect((await handleWaitlist(post({ email: `${"a".repeat(250)}@x.io` }), env)).status).toBe(400);
  });

  it("returns 500 when the DB binding is missing", async () => {
    const { env } = fakeEnv({ DB: undefined });
    const res = await handleWaitlist(post({ email: "a@b.co" }), env);
    expect(res.status).toBe(500);
  });

  it("stores the signup with its arm and emails a notification with reply-to", async () => {
    const { env, calls } = fakeEnv();
    const res = await handleWaitlist(post({ email: " a@b.co " }, { cookie: "sema-ab=b", referer: "https://x.io/", "user-agent": "UA" }), env);
    expect(res.status).toBe(200);
    expect(res.headers.get("cache-control")).toBe("no-store");
    const insert = calls.find((c) => c.sql.startsWith("INSERT"));
    expect(insert?.values.slice(0, 2)).toEqual(["a@b.co", "b"]);
    expect(insert?.values[3]).toBe("UA");
    expect(insert?.values[4]).toBe("https://x.io/");
    expect(sendMailMock).toHaveBeenCalledTimes(1);
    const [mailEnv, mail] = sendMailMock.mock.calls[0]!;
    expect(mailEnv).toBe(env);
    expect(mail.replyTo).toBe("a@b.co");
    expect(mail.subject).toBe("Waitlist: a@b.co (arm b)");
    expect(mail.text).toContain("Arm:      b");
  });

  it("creates the schema before the first insert", async () => {
    const { env, calls } = fakeEnv();
    await handleWaitlist(post({ email: "a@b.co" }), env);
    expect(calls[0]?.sql).toMatch(/CREATE TABLE IF NOT EXISTS waitlist/);
    expect(calls.at(-1)?.sql).toMatch(/^INSERT/);
  });

  it("maps a unique violation to 409 and sends no mail", async () => {
    const { db } = fakeDb(new Error("D1_ERROR: UNIQUE constraint failed: waitlist.email"));
    const { env } = fakeEnv({ DB: db });
    const res = await handleWaitlist(post({ email: "a@b.co" }), env);
    expect(res.status).toBe(409);
    expect(sendMailMock).not.toHaveBeenCalled();
  });

  it("rethrows other DB errors", async () => {
    const { db } = fakeDb(new Error("D1_ERROR: disk full"));
    const { env } = fakeEnv({ DB: db });
    await expect(handleWaitlist(post({ email: "a@b.co" }), env)).rejects.toThrow("disk full");
  });

  it("keeps the signup and returns 200 when the mail send fails", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    sendMailMock.mockRejectedValueOnce(new Error("535 Authentication failed"));
    const { env } = fakeEnv();
    const res = await handleWaitlist(post({ email: "a@b.co" }), env);
    expect(res.status).toBe(200);
    expect(error).toHaveBeenCalledWith(expect.stringContaining("535 Authentication failed"));
    error.mockRestore();
  });
});
