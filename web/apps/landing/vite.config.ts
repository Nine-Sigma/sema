import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { base } from "./src/copy/base";

const escapeHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/* %copy.meta.<slot>% in index.html reads from src/copy/base.ts, so page metadata has one source too. */
const copyHtml = () => ({
  name: "sema-copy-html",
  transformIndexHtml(html: string) {
    return html.replace(/%copy\.meta\.(\w+)%/g, (token, slot: string) => {
      const value = (base.meta as Record<string, string | undefined>)[slot];
      if (value === undefined) throw new Error(`index.html: unknown copy slot ${token}`);
      return escapeHtml(value);
    });
  },
});

export default defineConfig({
  plugins: [react(), tailwindcss(), copyHtml()],
});
