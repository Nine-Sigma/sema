import * as React from "react";
import { cn } from "../lib/utils";

/* Typographic product mark, used until official marks arrive. */
function Mark({ className, ...props }: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="mark"
      className={cn("type-label border border-rule-strong px-3 py-2 text-ink", className)}
      {...props}
    />
  );
}

export { Mark };
