import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "radix-ui";
import { cn } from "../../lib/utils";

/* The page's one filled control. Amber is spent here; everything else is ink on ground. */
const buttonVariants = cva(
  "inline-flex shrink-0 cursor-pointer items-center justify-center gap-2.5 whitespace-nowrap border-0 font-sans font-medium tracking-[0.01em] transition-[filter,transform] duration-150 ease-signal hover:no-underline active:translate-y-px aria-busy:cursor-progress aria-busy:saturate-[0.4] aria-busy:brightness-90 disabled:pointer-events-none",
  {
    variants: {
      variant: {
        default: "bg-accent-fill text-on-accent hover:brightness-[1.08]",
        outline:
          "bg-transparent text-ink shadow-[inset_0_0_0_1px_var(--rule-strong)] hover:shadow-[inset_0_0_0_1px_var(--ink)]",
        done: "cursor-default bg-transparent text-ink shadow-[inset_0_0_0_1px_var(--rule-strong)]",
      },
      size: {
        default: "h-12 px-[22px] text-[15px]",
        lg: "h-16 px-8 text-base",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

type ButtonProps = React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & { asChild?: boolean };

function Button({ className, variant, size, asChild = false, ...props }: ButtonProps) {
  const Comp = asChild ? Slot.Root : "button";
  return (
    <Comp
      data-slot="button"
      data-variant={variant ?? "default"}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants, type ButtonProps };
