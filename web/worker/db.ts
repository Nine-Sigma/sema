/* The D1 store (binding `DB`): waitlist signups and A/B assignments side by side, so one query
   gives conversions per arm (`npm run ab:report`). Schema is created on first use. */

export const SCHEMA = [
  `CREATE TABLE IF NOT EXISTS waitlist (
     id INTEGER PRIMARY KEY,
     email TEXT NOT NULL,
     variant TEXT,
     created_at TEXT NOT NULL,
     user_agent TEXT,
     referer TEXT
   )`,
  "CREATE UNIQUE INDEX IF NOT EXISTS waitlist_email ON waitlist (lower(email))",
  `CREATE TABLE IF NOT EXISTS ab_assignment (
     id INTEGER PRIMARY KEY,
     arm TEXT NOT NULL,
     created_at TEXT NOT NULL
   )`,
];

/* One schema run per database per isolate. */
const schemaReady = new WeakMap<D1Database, Promise<unknown>>();
export function ensureSchema(db: D1Database): Promise<unknown> {
  let ready = schemaReady.get(db);
  if (!ready) {
    ready = db.batch(SCHEMA.map((sql) => db.prepare(sql)));
    schemaReady.set(db, ready);
  }
  return ready;
}

/* One row per first visit to `/` while the test is on. Crawlers and uptime monitors would inflate
   the denominator, so obvious bots and HEAD requests are not counted. */
const BOT_UA = /bot|crawl|spider|slurp|preview|fetch|headless|monitor|curl|wget|python-requests|facebookexternalhit/i;

export function countable(request: Request): boolean {
  if (request.method !== "GET") return false;
  const ua = request.headers.get("user-agent") ?? "";
  return ua.length > 0 && !BOT_UA.test(ua);
}

export async function recordAssignment(db: D1Database, arm: string): Promise<void> {
  try {
    await ensureSchema(db);
    await db.prepare("INSERT INTO ab_assignment (arm, created_at) VALUES (?, ?)").bind(arm, new Date().toISOString()).run();
  } catch (err) {
    console.error(`ab: assignment not recorded ${String(err)}`);
  }
}
