import * as React from "react";
import { Toggle as TogglePrimitive } from "radix-ui";
import { cn } from "../../lib/utils";

/* 44px square, outlined, no fill in either state; the icon inside carries the state. */
function Toggle({ className, ...props }: React.ComponentProps<typeof TogglePrimitive.Root>) {
  return (
    <TogglePrimitive.Root
      data-slot="toggle"
      className={cn(
        "grid size-11 cursor-pointer place-items-center border border-rule-strong bg-transparent p-0 text-ink transition-colors duration-150 ease-signal hover:border-ink [&_svg]:block [&_svg]:size-4 [&_svg]:transition-transform [&_svg]:duration-200 [&_svg]:ease-signal data-[state=on]:[&_svg]:rotate-180",
        className,
      )}
      {...props}
    />
  );
}

export { Toggle };
