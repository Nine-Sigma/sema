import * as React from "react";
import { cn } from "../lib/utils";

/* Mono caption above a heading. Budget: at most one per three sections. */
function Eyebrow({ className, ...props }: React.ComponentProps<"p">) {
  return <p data-slot="eyebrow" className={cn("type-label text-muted", className)} {...props} />;
}

export { Eyebrow };
