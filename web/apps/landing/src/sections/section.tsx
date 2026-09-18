import * as React from "react";
import { cn } from "@sema/design";

type SectionProps = React.ComponentProps<"section"> & {
  /** id of the heading that names the section (aria-labelledby). */
  heading: string;
};

/* Every section after the hero: 1px rule above, the page's vertical rhythm, a 12-column grid inside. */
function Section({ heading, className, children, ...props }: SectionProps) {
  return (
    <section aria-labelledby={heading} className={cn("border-t border-rule py-section", className)} {...props}>
      <div className="page-wrap page-grid">{children}</div>
    </section>
  );
}

/* Bold claim over a sentence: ledger rows, compartment cells, theses. */
function Claim({ className, ...props }: React.ComponentProps<"p">) {
  return <p className={cn("type-claim", className)} {...props} />;
}

export { Section, Claim };
