import type { FormOutcome } from "@sema/design";

const TIMEOUT_MS = 8000;
/* Demo only, and only in dev builds (gate G2): error@, timeout@, duplicate@ show those states. */
const DEMO = /^(error|timeout|duplicate)@/i;

function demo(email: string): Promise<FormOutcome> {
  const m = DEMO.exec(email);
  const outcome = (m?.[1]?.toLowerCase() as FormOutcome | undefined) ?? "success";
  return new Promise((resolve) => setTimeout(() => resolve(outcome), outcome === "timeout" ? 2400 : 800));
}

/* Posts to the Pages Function at /api/waitlist (web/functions/api/waitlist.ts). The sema-ab cookie
   travels with the same-origin request, which is how a signup is attributed to an A/B arm. */
export async function submitWaitlist(email: string): Promise<FormOutcome> {
  if (import.meta.env.DEV) return demo(email);
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch("/api/waitlist", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email }),
      signal: ctl.signal,
    });
    if (res.status === 409) return "duplicate";
    return res.ok ? "success" : "error";
  } catch (err) {
    return err instanceof DOMException && err.name === "AbortError" ? "timeout" : "error";
  } finally {
    clearTimeout(timer);
  }
}
