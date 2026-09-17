import * as React from "react";
import { cn } from "../lib/utils";

/* Status tag. Positive words only: Exploration, Showcase, Shipped. */
function Tag({ className, ...props }: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="tag"
      className={cn("type-label whitespace-nowrap border border-rule-strong px-2 py-1 text-muted", className)}
      {...props}
    />
  );
}

export { Tag };
