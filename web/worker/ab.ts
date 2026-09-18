/* A/B arm selection for `/` (tasks/plan-landing-copy.md, phase 4, D10/D11).

   Every visitor requests `/`. The Worker picks an arm once, stores it in the `sema-ab` cookie, and
   for a variant arm returns that arm's document from the static build in place of `/`. The address
   bar never changes. Arm paths must match `apps/landing/src/copy/variants.ts`. */

export type ArmId = "control" | "a" | "b";
export type Weights = Record<ArmId, number>;

/* Relative weights. `control: 1, a: 0, b: 0` = test off: `/` is served untouched and no cookie is
   set. Change and redeploy to start the test; fold the winner into base.ts and reset to off. */
export const WEIGHTS: Weights = { control: 1, a: 0, b: 0 };

export const ARM_PATHS: Record<ArmId, string> = { control: "/", a: "/ab/a", b: "/ab/b" };
export const AB_COOKIE = "sema-ab";
export const AB_MAX_AGE = 60 * 60 * 24 * 30;

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
