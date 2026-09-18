/* POST /api/waitlist — the two landing forms post { email } here (apps/landing/src/waitlist.ts).

   Stores the signup in D1 (binding `DB`, table `waitlist`; the A/B arm from the `sema-ab` cookie
   sits beside it, which is how conversions per arm are counted) and notifies dean@withsema.ai
   through the `MAIL` service binding (workers/waitlist-mail: Pages Functions cannot hold a
   send_email binding, a Worker can). Responses the client maps: 200 success, 409 duplicate,
   anything else → the form's error state. A missing binding is a 500, never a silent success. */

type D1Result = { success: boolean };
type D1Statement = { bind(...values: unknown[]): D1Statement; run(): Promise<D1Result> };
type D1 = { prepare(sql: string): D1Statement; batch(statements: D1Statement[]): Promise<D1Result[]> };
type Env = { DB?: D1; MAIL?: { fetch(request: Request): Promise<Response> } };
type Context = { request: Request; env: Env };

const MAX_EMAIL = 254;
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const COOKIE = "sema-ab";

const SCHEMA = [
  `CREATE TABLE IF NOT EXISTS waitlist (
     id INTEGER PRIMARY KEY,
     email TEXT NOT NULL,
     variant TEXT,
     created_at TEXT NOT NULL,
     user_agent TEXT,
     referer TEXT
   )`,
  "CREATE UNIQUE INDEX IF NOT EXISTS waitlist_email ON waitlist (lower(email))",
];

export const onRequestPost = async ({ request, env }: Context): Promise<Response> => {
  if (!env.DB) return json(500, { error: "waitlist store not configured" });
  const email = await readEmail(request);
  if (!email) return json(400, { error: "invalid email" });

  const row = {
    email,
    variant: readCookie(request.headers.get("cookie"), COOKIE) ?? null,
    created_at: new Date().toISOString(),
    user_agent: request.headers.get("user-agent"),
    referer: request.headers.get("referer"),
  };

  await ensureSchema(env.DB);
  try {
    await env.DB.prepare("INSERT INTO waitlist (email, variant, created_at, user_agent, referer) VALUES (?, ?, ?, ?, ?)")
      .bind(row.email, row.variant, row.created_at, row.user_agent, row.referer)
      .run();
  } catch (err) {
    if (isUniqueViolation(err)) return json(409, { error: "already on the list" });
    throw err;
  }

  await notify(env, row);
  return json(200, { ok: true });
};

export const onRequest = async ({ request }: Context): Promise<Response> =>
  request.method === "POST" ? json(405, { error: "method not allowed" }) : new Response(null, { status: 405, headers: { allow: "POST" } });

let schemaReady: Promise<unknown> | undefined;
function ensureSchema(db: D1): Promise<unknown> {
  schemaReady ??= db.batch(SCHEMA.map((sql) => db.prepare(sql)));
  return schemaReady;
}

async function readEmail(request: Request): Promise<string | undefined> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return undefined;
  }
  const email = typeof body === "object" && body !== null ? (body as { email?: unknown }).email : undefined;
  if (typeof email !== "string") return undefined;
  const trimmed = email.trim();
  return trimmed.length <= MAX_EMAIL && EMAIL.test(trimmed) ? trimmed : undefined;
}

/* The signup is stored either way; a mail failure is logged, not surfaced to the visitor. */
async function notify(env: Env, row: Record<string, string | null>): Promise<void> {
  if (!env.MAIL) {
    console.error("waitlist: MAIL binding missing, signup stored without notification");
    return;
  }
  try {
    const res = await env.MAIL.fetch(new Request("https://waitlist-mail/notify", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(row) }));
    if (!res.ok) console.error(`waitlist: notify failed ${res.status} ${await res.text()}`);
  } catch (err) {
    console.error(`waitlist: notify threw ${String(err)}`);
  }
}

function isUniqueViolation(err: unknown): boolean {
  return err instanceof Error && /UNIQUE constraint failed/i.test(err.message);
}

function readCookie(header: string | null, name: string): string | undefined {
  if (!header) return undefined;
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return undefined;
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
}
