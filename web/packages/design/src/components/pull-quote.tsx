import * as React from "react";
import { cn } from "../lib/utils";

/* The one amber rule on the page that is not a control. */
function PullQuote({ className, ...props }: React.ComponentProps<"p">) {
  return (
    <p
      data-slot="pull-quote"
      className={cn("type-pull border-t-2 border-accent pt-7 lg:pr-[28%]", className)}
      {...props}
    />
  );
}

export { PullQuote };
