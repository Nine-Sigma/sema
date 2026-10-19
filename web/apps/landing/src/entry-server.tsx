import React from "react";
import { renderToString } from "react-dom/server";
import { Landing } from "./landing";
import { CopyProvider, type Copy } from "./copy";
import { arms } from "./copy/variants";

/* Build-time prerender (scripts/prerender.ts): the markup the client hydrates in mount.tsx. */
export function render(copy: Copy): string {
  return renderToString(
    <React.StrictMode>
      <CopyProvider value={copy}>
        <Landing />
      </CopyProvider>
    </React.StrictMode>,
  );
}

export { arms };
