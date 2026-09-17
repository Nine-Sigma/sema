import * as React from "react";
import { cn } from "../lib/utils";

type FooterProps = React.ComponentProps<"footer"> & {
  /** Up to four columns; each stacks its children, links padded to ~44px. */
  columns: React.ReactNode[];
};

function Footer({ columns, className, ...props }: FooterProps) {
  return (
    <footer data-slot="footer" className={cn("border-t border-rule pt-10 pb-14", className)} {...props}>
      <div className="page-wrap grid grid-cols-4 gap-6 text-sm max-lg:grid-cols-1">
        {columns.map((col, i) => (
          <div key={i} className="grid content-start justify-items-start [&_a]:py-[9px]">
            {col}
          </div>
        ))}
      </div>
    </footer>
  );
}

export { Footer, type FooterProps };
