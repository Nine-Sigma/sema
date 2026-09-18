import * as React from "react";
import { cn } from "../lib/utils";

function SkipLink({ className, children = "Skip to content", ...props }: React.ComponentProps<"a">) {
  return (
    <a
      data-slot="skip-link"
      className={cn(
        "absolute left-gutter -top-[100px] z-10 bg-ink px-4 py-3 text-sm/none font-medium text-ground focus-visible:top-3 focus-visible:outline-offset-0",
        className,
      )}
      {...props}
    >
      {children}
    </a>
  );
}

export { SkipLink };
