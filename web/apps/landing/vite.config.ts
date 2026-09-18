import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { armByPath, arms } from "./src/copy/variants";

const HERE = dirname(fileURLToPath(import.meta.url));
const escapeHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/* One HTML template (index.html) for every arm. A variant entry (ab/a.html) is a stub: its markup is
   index.html with the entry script swapped, plus a canonical back to `/` because the middleware
   serves it at `/`. %copy.meta.<slot>% tokens read from the arm's copy, so metadata switches too. */
const copyHtml = () => ({
  name: "sema-copy-html",
  transformIndexHtml: {
    order: "pre" as const,
    handler(html: string, ctx: { path: string }) {
      const arm = armByPath(ctx.path);
      if (!arm) throw new Error(`${ctx.path}: not an A/B arm; add it to src/copy/variants.ts`);
      const template = arm.id === "control" ? html : variantHtml(arm.id);
      const filled = template.replace(/%copy\.meta\.(\w+)%/g, (token, slot: string) => {
        const value = (arm.copy.meta as Record<string, string | undefined>)[slot];
        if (value === undefined) throw new Error(`${ctx.path}: unknown copy slot ${token}`);
        return escapeHtml(value);
      });
      return filled;
    },
  },
});

/* Variant entries point search engines at `/`. Relative on purpose (the domain will change), and
   injected after Vite's asset pass ("post"), which would otherwise try to read `/` as a file. */
const variantCanonical = () => ({
  name: "sema-variant-canonical",
  transformIndexHtml: {
    order: "post" as const,
    handler(_html: string, ctx: { path: string }) {
      if (armByPath(ctx.path)?.id === "control") return [];
      return [{ tag: "link", attrs: { rel: "canonical", href: "/" }, injectTo: "head" as const }];
    },
  },
});

function variantHtml(id: string): string {
  const html = readFileSync(resolve(HERE, "index.html"), "utf8");
  const swapped = html.replace('src="/src/main.tsx"', `src="/src/main-${id}.tsx"`);
  if (swapped === html) throw new Error("index.html: entry script /src/main.tsx not found");
  return swapped;
}

export default defineConfig({
  plugins: [react(), tailwindcss(), copyHtml(), variantCanonical()],
  build: {
    rollupOptions: {
      input: Object.fromEntries(arms.map((arm) => [arm.id, resolve(HERE, arm.path.slice(1))])),
    },
  },
});
