import * as React from "react";
import { cn } from "../lib/utils";

/* Serif "Sema". Clicking it replays the hero moment (see useMotionGate). */
function Wordmark({ className, children = "Sema", ...props }: React.ComponentProps<"a">) {
  return (
    <a data-slot="wordmark" className={cn("type-wordmark", className)} aria-label="Sema, home" {...props}>
      {children}
    </a>
  );
}

export { Wordmark };
