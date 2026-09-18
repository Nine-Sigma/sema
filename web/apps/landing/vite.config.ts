import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { armByPath, arms } from "./src/copy/variants";

const HERE = dirname(fileURLToPath(import.meta.url));
/* Production origin. Social scrapers need absolute og:url/og:image, and every arm's canonical is the
   home page on this host (the middleware serves variants at `/`). */
const SITE_URL = "https://withsema.ai";
const escapeHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/* One HTML template (index.html) for every arm. A variant entry (ab/a.html) is a stub: its markup is
   index.html with the entry script swapped. %copy.meta.<slot>% tokens read from the arm's copy, so
   metadata switches too; %site.url% is SITE_URL. */
const copyHtml = () => ({
  name: "sema-copy-html",
  transformIndexHtml: {
    order: "pre" as const,
    handler(html: string, ctx: { path: string }) {
      const arm = armByPath(ctx.path);
      if (!arm) throw new Error(`${ctx.path}: not an A/B arm; add it to src/copy/variants.ts`);
      const template = arm.id === "control" ? html : variantHtml(arm.id);
      return template.replaceAll("%site.url%", SITE_URL).replace(/%copy\.meta\.(\w+)%/g, (token, slot: string) => {
        const value = (arm.copy.meta as Record<string, string | undefined>)[slot];
        if (value === undefined) throw new Error(`${ctx.path}: unknown copy slot ${token}`);
        return escapeHtml(value);
      });
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
  plugins: [react(), tailwindcss(), copyHtml()],
  build: {
    rollupOptions: {
      input: Object.fromEntries(arms.map((arm) => [arm.id, resolve(HERE, arm.path.slice(1))])),
    },
  },
});
