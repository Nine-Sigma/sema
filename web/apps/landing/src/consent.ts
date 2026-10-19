/* Consent for the one cookie this site sets, `sema-ab` (the A/B arm, 30 days).

   The Worker never sets the cookie itself (worker/index.ts): it only reads one a visitor already
   holds. The arm of the served document is in <html data-arm>; when the visitor allows the cookie,
   this module writes it from there. Until then, and after a decline, each visit draws a fresh arm.
   The decision lives in localStorage, which is the one storage a choice about cookies may use
   without asking. Cloudflare Web Analytics is cookieless and is not part of this consent. */

export type Consent = "granted" | "denied";

const KEY = "sema-consent";
const COOKIE = "sema-ab";
const MAX_AGE = 60 * 60 * 24 * 30;

export function readConsent(): Consent | undefined {
  try {
    const value = localStorage.getItem(KEY);
    return value === "granted" || value === "denied" ? value : undefined;
  } catch {
    return undefined;
  }
}

/* The A/B arm of the served document, stamped by vite.config.ts (`%arm.id%`). */
export function currentArm(): string | undefined {
  return document.documentElement.dataset.arm || undefined;
}

export function grant(): void {
  store("granted");
  writeArmCookie();
}

export function deny(): void {
  store("denied");
  document.cookie = `${COOKIE}=; Path=/; Max-Age=0; SameSite=Lax; Secure`;
}

/* Consent given on an earlier visit: keep the cookie on this document's arm (a ?ab= visit re-pins it). */
export function syncArmCookie(): void {
  if (readConsent() === "granted") writeArmCookie();
}

function writeArmCookie(): void {
  const arm = currentArm();
  if (!arm) return;
  document.cookie = `${COOKIE}=${arm}; Path=/; Max-Age=${MAX_AGE}; SameSite=Lax; Secure`;
}

function store(value: Consent): void {
  try {
    localStorage.setItem(KEY, value);
  } catch {
    /* Private mode or blocked storage: the notice returns next visit, nothing else breaks. */
  }
}
