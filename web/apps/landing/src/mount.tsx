import React from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import "./index.css";
import { Landing } from "./landing";
import { CopyProvider, type Copy } from "./copy";

/* One entry per A/B arm (main.tsx, main-a.tsx, main-b.tsx); each calls this with its copy.
   Production documents are prerendered (scripts/prerender.ts) and hydrated; `vite dev` mounts fresh. */
export function mount(copy: Copy): void {
  /* QA convenience: ?theme=dark|light pins the theme before first paint. */
  const pinned = new URLSearchParams(location.search).get("theme");
  if (pinned === "dark" || pinned === "light") document.documentElement.setAttribute("data-theme", pinned);

  const root = document.getElementById("root");
  if (!root) throw new Error("#root missing");
  const tree = (
    <React.StrictMode>
      <CopyProvider value={copy}>
        <Landing />
      </CopyProvider>
    </React.StrictMode>
  );
  if (root.hasChildNodes()) hydrateRoot(root, tree);
  else createRoot(root).render(tree);
}
