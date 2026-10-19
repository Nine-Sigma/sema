/* POST /api/waitlist — the two landing forms post { email } here (apps/landing/src/waitlist.ts).

   Stores the signup in D1 (worker/db.ts, table `waitlist`; the A/B arm sits beside it, which is
   how conversions per arm are counted: the `sema-ab` cookie when the visitor allowed it, else the
   `variant` the page sends from <html data-arm>), then emails TO_ADDRESS over
   Titan SMTP (worker/mail.ts) with Reply-To set to the signup. Responses the client maps: 200 success,
   409 duplicate, anything else → the form's error state. A missing DB binding is a 500, never a
   silent success. A mail failure is logged, not surfaced: the signup is stored either way. */

import { AB_COOKIE, parseArm, readCookie } from "./ab";
import { ensureSchema } from "./db";
import type { Env } from "./env";
import { sendMail } from "./mail";

type Signup = { email: string; variant: string | null; created_at: string; user_agent: string | null; referer: string | null };

const MAX_EMAIL = 254;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export async function handleWaitlist(request: Request, env: Env): Promise<Response> {
  if (request.method !== "POST") return new Response(null, { status: 405, headers: { allow: "POST" } });
  if (!env.DB) return json(500, { error: "waitlist store not configured" });
  const body = await readBody(request);
  if (!body) return json(400, { error: "invalid email" });

  const signup: Signup = {
    email: body.email,
    variant: parseArm(readCookie(request.headers.get("cookie"), AB_COOKIE)) ?? parseArm(body.variant) ?? null,
    created_at: new Date().toISOString(),
    user_agent: request.headers.get("user-agent"),
    referer: request.headers.get("referer"),
  };

  await ensureSchema(env.DB);
  try {
    await env.DB.prepare("INSERT INTO waitlist (email, variant, created_at, user_agent, referer) VALUES (?, ?, ?, ?, ?)")
      .bind(signup.email, signup.variant, signup.created_at, signup.user_agent, signup.referer)
      .run();
  } catch (err) {
    if (isUniqueViolation(err)) return json(409, { error: "already on the list" });
    throw err;
  }

  await notify(env, signup);
  return json(200, { ok: true });
}

type Body = { email: string; variant: string | undefined };

async function readBody(request: Request): Promise<Body | undefined> {
  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return undefined;
  }
  if (typeof raw !== "object" || raw === null) return undefined;
  const { email, variant } = raw as { email?: unknown; variant?: unknown };
  if (typeof email !== "string") return undefined;
  const trimmed = email.trim();
  if (trimmed.length > MAX_EMAIL || !EMAIL_RE.test(trimmed)) return undefined;
  return { email: trimmed, variant: typeof variant === "string" ? variant : undefined };
}

async function notify(env: Env, signup: Signup): Promise<void> {
  try {
    await sendMail(env, {
      subject: `Waitlist: ${signup.email}${signup.variant ? ` (arm ${signup.variant})` : ""}`,
      text: mailBody(signup),
      replyTo: signup.email,
    });
  } catch (err) {
    console.error(`waitlist: notify failed ${String(err)}`);
  }
}

function mailBody(s: Signup): string {
  return [
    `Email:    ${s.email}`,
    `Arm:      ${s.variant ?? "none (test off)"}`,
    `Time:     ${s.created_at}`,
    `Referer:  ${s.referer ?? "-"}`,
    `Browser:  ${s.user_agent ?? "-"}`,
    "",
    "Stored in D1 `waitlist`. Reply to this message to answer them.",
  ].join("\n");
}

function isUniqueViolation(err: unknown): boolean {
  return err instanceof Error && /UNIQUE constraint failed/i.test(err.message);
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
}
