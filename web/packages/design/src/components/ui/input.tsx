import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

/* Transparent field on the ground, one rule around it; amber only on focus and invalid. */
const inputVariants = cva(
  "min-w-0 flex-auto border border-rule-strong bg-transparent font-sans text-ink transition-colors duration-150 ease-signal placeholder:text-muted hover:border-ink focus-visible:border-accent focus-visible:shadow-[inset_0_0_0_1px_var(--accent)] focus-visible:outline-none aria-invalid:border-accent read-only:text-muted",
  {
    variants: {
      size: {
        default: "h-12 px-3.5 text-base",
        lg: "h-16 px-3.5 text-lg",
      },
    },
    defaultVariants: { size: "default" },
  },
);

type InputProps = Omit<React.ComponentProps<"input">, "size"> & VariantProps<typeof inputVariants>;

function Input({ className, size, type = "text", ...props }: InputProps) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(inputVariants({ size, className }))}
      {...props}
    />
  );
}

export { Input, inputVariants, type InputProps };
