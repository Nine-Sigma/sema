import * as React from "react";
import { Label as LabelPrimitive } from "radix-ui";
import { cn } from "../../lib/utils";

/* Field labels are mono eyebrows, the same voice as every other small caption on the page. */
function Label({ className, ...props }: React.ComponentProps<typeof LabelPrimitive.Root>) {
  return (
    <LabelPrimitive.Root
      data-slot="label"
      className={cn("type-label select-none text-muted", className)}
      {...props}
    />
  );
}

export { Label };
