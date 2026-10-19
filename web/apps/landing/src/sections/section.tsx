import * as React from "react";
import { cn } from "@sema/design";

/* Chapter rhythm (site plan v7 §3.4): heavy 240, light 160, connective 200 above; half below. */
export type Weight = "heavy" | "light" | "connective";
const WEIGHT: Record<Weight, string> = {
  heavy: "chapter-heavy",
  light: "chapter-light",
  connective: "chapter-connective",
};

type SectionProps = React.ComponentProps<"section"> & {
  /** id of the heading that names the section (aria-labelledby). */
  heading: string;
  weight?: Weight;
  /** Chapter name for the graph plane (html[data-chapter]). */
  chapter?: string;
};

/* Every section after the hero: 1px rule above, the chapter's rhythm, a 12-column grid inside. */
function Section({ heading, weight = "light", chapter, className, children, ...props }: SectionProps) {
  return (
    <section
      aria-labelledby={heading}
      data-chapter={chapter}
      className={cn("relative border-t border-rule", WEIGHT[weight], className)}
      {...props}
    >
      <div className="page-wrap page-grid">{children}</div>
    </section>
  );
}

/* Bold claim over a sentence: ledger rows, table rows, theses. */
function Claim({ className, ...props }: React.ComponentProps<"p">) {
  return <p className={cn("type-claim", className)} {...props} />;
}

export { Section, Claim };
