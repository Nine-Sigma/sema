/* A/B arm selection for `/` (tasks/plan-landing-copy.md, phase 4, D10/D11).

   Every visitor requests `/`. The Worker serves the arm named in the `sema-ab` cookie when there is
   one, else draws an arm for this response; for a variant arm it returns that arm's document from
   the static build in place of `/`. The address bar never changes. The cookie is written by the page
   after consent (apps/landing/src/consent.ts, 30 days), never by the Worker, so a visitor who has not
   allowed it is redrawn on every visit and counted in `ab_assignment` each time. Arm paths must
   match `apps/landing/src/copy/variants.ts`. */

export type ArmId = "control" | "a" | "b";
export type Weights = Record<ArmId, number>;

/* Relative weights. `control: 1, a: 0, b: 0` = test off: `/` is served untouched and no cookie is
   set. Test ON since 2026-09-21 (equal split); fold the winner into base.ts and reset to off. */
export const WEIGHTS: Weights = { control: 1, a: 1, b: 1 };

export const ARM_PATHS: Record<ArmId, string> = { control: "/", a: "/ab/a", b: "/ab/b" };
export const AB_COOKIE = "sema-ab";

export function activeArms(weights: Weights = WEIGHTS): ArmId[] {
  return (Object.keys(weights) as ArmId[]).filter((id) => weights[id] > 0);
}

/* Weighted draw over the active arms. */
export function draw(weights: Weights = WEIGHTS): ArmId {
  const active = activeArms(weights);
  let roll = Math.random() * active.reduce((sum, id) => sum + weights[id], 0);
  for (const id of active) {
    roll -= weights[id];
    if (roll < 0) return id;
  }
  return active[active.length - 1] ?? "control";
}

export function parseArm(value: string | null | undefined): ArmId | undefined {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(WEIGHTS, value) ? (value as ArmId) : undefined;
}

export function readCookie(header: string | null, name: string): string | undefined {
  if (!header) return undefined;
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return undefined;
}
