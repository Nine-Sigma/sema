/* POST /notify { email, variant, created_at, user_agent, referer } → one email to the address the
   EMAIL binding is restricted to (wrangler.toml). Reply-To is the signup, so replying from the
   inbox answers the person directly. */

type EmailBinding = {
  send(message: { to: string; from: string; subject: string; text: string; replyTo?: string }): Promise<{ messageId: string }>;
};
type Env = { EMAIL: EmailBinding; FROM_ADDRESS: string; TO_ADDRESS: string };
type Signup = { email: string; variant: string | null; created_at: string; user_agent: string | null; referer: string | null };

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== "POST" || new URL(request.url).pathname !== "/notify") return new Response(null, { status: 404 });
    const signup = await readSignup(request);
    if (!signup) return new Response("bad signup payload", { status: 400 });

    const { messageId } = await env.EMAIL.send({
      to: env.TO_ADDRESS,
      from: env.FROM_ADDRESS,
      subject: `Waitlist: ${signup.email}${signup.variant ? ` (arm ${signup.variant})` : ""}`,
      text: body(signup),
      replyTo: signup.email,
    });
    return Response.json({ messageId });
  },
};

async function readSignup(request: Request): Promise<Signup | undefined> {
  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return undefined;
  }
  if (typeof raw !== "object" || raw === null) return undefined;
  const s = raw as Partial<Signup>;
  if (typeof s.email !== "string" || typeof s.created_at !== "string") return undefined;
  return { email: s.email, variant: s.variant ?? null, created_at: s.created_at, user_agent: s.user_agent ?? null, referer: s.referer ?? null };
}

function body(s: Signup): string {
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
