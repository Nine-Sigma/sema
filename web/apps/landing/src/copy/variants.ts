import type { Copy } from "./index";
import { base } from "./base";
import { variantA } from "./variant-a";
import { variantB } from "./variant-b";

/* The A/B arms. One Vite entry per arm; the control is `/`, the variants sit under `/ab/`.
   `web/functions/_middleware.ts` picks an arm per visitor and serves its entry at `/`, so
   `path` here must match the `ARMS` table there. Split weights live in the middleware. */
export type Arm = { id: ArmId; path: string; copy: Copy };
export type ArmId = "control" | "a" | "b";

export const arms: readonly Arm[] = [
  { id: "control", path: "/index.html", copy: base },
  { id: "a", path: "/ab/a.html", copy: variantA },
  { id: "b", path: "/ab/b.html", copy: variantB },
];

export function armByPath(path: string): Arm | undefined {
  return arms.find((arm) => arm.path === path);
}
