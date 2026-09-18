import * as React from "react";
import { cn, type Theme } from "@sema/design";

/* The ledger: one compartments grid per section, one row per specimen.
   Columns at lg: dark render | light render | caption. Under lg everything stacks in that order. */

type Layout = "pair" | "stack" | "single";

type SpecimenProps = {
  /** Component or utility name, set in the caption voice. */
  name: string;
  /** Prop values or measurements; mono, never uppercased. */
  props?: React.ReactNode;
  /** The rule this specimen demonstrates, taken from the source comment. */
  note?: React.ReactNode;
  /** pair: dark | light side by side. stack: dark over light, full width. single: live theme only; the caption says which. */
  layout?: Layout;
  /** Adds a Reset control to the caption that remounts the render cells. */
  resettable?: boolean;
  /** Extra caption control, e.g. Replay. */
  action?: React.ReactNode;
  /** Theme of the cell and a suffix that keeps ids unique across cells. */
  render: (theme: Theme, suffix: string) => React.ReactNode;
  cellClassName?: string;
};

const THEMES: Theme[] = ["dark", "light"];

const GRID: Record<Layout, string> = {
  pair: "lg:grid-cols-[1fr_1fr_18rem]",
  stack: "lg:grid-cols-[1fr_18rem]",
  single: "lg:grid-cols-[1fr_18rem]",
};

function Cell({ theme, row, className, children }: { theme?: Theme; row: number; className?: string; children: React.ReactNode }) {
  return (
    <div
      data-theme={theme}
      data-slot="specimen-cell"
      className={cn("min-w-0 bg-ground p-6 text-ink", row === 2 && "lg:col-start-1 lg:row-start-2", className)}
    >
      {children}
    </div>
  );
}

/* Each specimen is its own compartments grid with explicit placement, so dark always sits first. */
function Specimen({ name, props, note, layout = "pair", resettable, action, render, cellClassName }: SpecimenProps) {
  const [run, setRun] = React.useState(0);
  const cells = layout === "single" ? [undefined] : THEMES;
  return (
    <div data-slot="specimen" className={cn("compartments grid-cols-1", GRID[layout])}>
      {cells.map((t, i) => (
        <Cell key={`${t ?? "live"}-${run}`} theme={t} row={layout === "stack" ? i + 1 : 1} className={cellClassName}>
          {render(t ?? "dark", `${t ?? "live"}-${run}`)}
        </Cell>
      ))}
      <div
        className={cn(
          "grid content-start gap-2 bg-ground p-6",
          layout === "stack" && "lg:col-start-2 lg:row-start-1 lg:row-span-2",
        )}
      >
        <p className="type-label flex flex-wrap justify-between gap-x-4 text-ink">
          <span>{name}</span>
          {layout !== "pair" ? <span className="text-muted">{layout === "stack" ? "stacked" : "live theme"}</span> : null}
        </p>
        {props ? <p className="font-mono text-xs/relaxed text-muted [overflow-wrap:anywhere]">{props}</p> : null}
        {note ? <p className="text-[13px]/normal text-muted">{note}</p> : null}
        {resettable || action ? (
          <p className="flex flex-wrap gap-x-6">
            {action}
            {resettable ? <CaptionButton onClick={() => setRun((n) => n + 1)}>Reset</CaptionButton> : null}
          </p>
        ) : null}
      </div>
    </div>
  );
}

function CaptionButton(props: React.ComponentProps<"button">) {
  return (
    <button
      type="button"
      className="type-label inline-flex min-h-11 cursor-pointer items-center border-0 bg-transparent p-0 text-ink underline decoration-rule-strong underline-offset-4 hover:decoration-ink"
      {...props}
    />
  );
}

type SectionProps = { id: string; title: string; intro?: React.ReactNode; children: React.ReactNode };

function Section({ id, title, intro, children }: SectionProps) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="pt-14 pb-6">
      <div className="page-wrap grid gap-5">
        <div className="grid gap-2">
          <h2 id={`${id}-h`} className="type-h3">
            {title}
          </h2>
          {intro ? <p className="max-w-[60ch] text-[15px] text-muted">{intro}</p> : null}
        </div>
        <div className="grid [&>*+*]:-mt-px">{children}</div>
      </div>
    </section>
  );
}

type Anchor = { href: string; label: string };

/* Sections stay reachable under lg, where the library Nav hides its anchors by contract. */
function AnchorRow({ anchors }: { anchors: Anchor[] }) {
  return (
    <nav aria-label="Sections" className="sticky top-0 z-10 border-b border-rule bg-ground lg:hidden">
      <div className="flex gap-1 overflow-x-auto px-gutter whitespace-nowrap [scrollbar-width:none] [mask-image:linear-gradient(90deg,#000_calc(100%-40px),transparent)]">
        {anchors.map((a) => (
          <a key={a.href} href={a.href} className="type-label px-2.5 py-4 text-muted hover:text-ink">
            {a.label}
          </a>
        ))}
      </div>
    </nav>
  );
}

export { AnchorRow, CaptionButton, Section, Specimen, type Anchor, type Layout, type SpecimenProps };
