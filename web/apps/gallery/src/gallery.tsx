import * as React from "react";
import { Nav, SkipLink, ThemeToggle, Wordmark, useTheme } from "@sema/design";
import { AnchorRow, CaptionButton, type Anchor } from "./ledger";
import { ControlsSection, FiguresSection, FormsSection, RecordsSection, ShellSection, TokensSection, TypeSection } from "./sections";
import { TYPE_ROLES } from "./type-ledger";

/* Verification ledger for @sema/design: one row per specimen, dark left, light right, prop caption.
   The frame around the page (skip link, nav, identity strip, end strip) is the live shell; the same
   components appear again as specimens under #shell. */

const COMPONENTS = [
  "Button", "Input", "Label", "Toggle", "Constellation", "EmailForm", "Eyebrow", "Figure", "Footer",
  "Mark", "Nav", "PullQuote", "Record", "SkipLink", "Tag", "ThemeToggle", "Wordmark",
] as const;
const FIGURE_COUNT = 7;

const ANCHORS: Anchor[] = [
  { href: "#tokens", label: "Tokens" },
  { href: "#type", label: "Type" },
  { href: "#controls", label: "Controls" },
  { href: "#forms", label: "Forms" },
  { href: "#records", label: "Records" },
  { href: "#figures", label: "Figures" },
  { href: "#shell", label: "Shell" },
];

const INVENTORY = `${COMPONENTS.length} components · ${TYPE_ROLES.length} type roles · ${FIGURE_COUNT} figures`;

/* 64px strip under the nav: what this page is, what it holds, and where the theme comes from. */
function IdentityStrip() {
  const { theme, source, setTheme } = useTheme();
  return (
    <div className="border-b border-rule">
      <div className="page-wrap flex min-h-16 flex-wrap items-center justify-between gap-x-8 gap-y-2 py-3">
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
          <h1 className="type-h3">Sema design library</h1>
          <p className="type-label text-muted">
            {INVENTORY}
            <span className="max-sm:hidden"> · dark left, light right</span>
          </p>
        </div>
        <p className="type-label flex items-center gap-4 text-muted">
          <span>
            Theme {theme} · {source}
          </span>
          {source === "pinned" ? <CaptionButton onClick={() => setTheme("system")}>Follow system</CaptionButton> : null}
        </p>
      </div>
    </div>
  );
}

function EndStrip() {
  return (
    <div className="mt-14 border-t border-rule">
      <div className="page-wrap flex min-h-16 flex-wrap items-center justify-between gap-4 py-3">
        <p className="type-label text-muted">End of ledger · {INVENTORY}</p>
        <a href="#top" className="type-label inline-flex min-h-11 items-center text-ink underline decoration-rule-strong underline-offset-4 hover:decoration-ink">
          Top
        </a>
      </div>
    </div>
  );
}

export function Gallery() {
  return (
    <>
      <SkipLink href="#ledger" />
      <Nav id="top" brand={<Wordmark href="#top" />} anchors={ANCHORS} actions={<ThemeToggle />} />
      <AnchorRow anchors={ANCHORS} />
      <IdentityStrip />
      <main id="ledger" tabIndex={-1} className="outline-none">
        <TokensSection />
        <TypeSection />
        <ControlsSection />
        <FormsSection />
        <RecordsSection />
        <FiguresSection />
        <ShellSection />
      </main>
      <EndStrip />
    </>
  );
}
