import * as React from "react";
import { cn } from "../lib/utils";

type NavProps = React.ComponentProps<"nav"> & {
  /** Left: the wordmark. */
  brand: React.ReactNode;
  /** Middle: anchor links; hidden under 900px. */
  anchors?: { href: string; label: string }[];
  /** Right: theme toggle and the call to action. */
  actions?: React.ReactNode;
};

/* 64px bar under a 1px rule. Anchor targets are padded to ~44px. */
function Nav({ brand, anchors = [], actions, className, ...props }: NavProps) {
  return (
    <nav aria-label="Primary" data-slot="nav" className={cn("border-b border-rule", className)} {...props}>
      <div className="page-wrap flex h-16 items-center gap-8">
        {brand}
        {anchors.length ? (
          <div className="ml-4 flex gap-2 text-[15px] max-lg:hidden">
            {anchors.map((a) => (
              <a key={a.href} href={a.href} className="px-2.5 py-[11px]">
                {a.label}
              </a>
            ))}
          </div>
        ) : null}
        {actions ? <div className="ml-auto flex items-center gap-3">{actions}</div> : null}
      </div>
    </nav>
  );
}

export { Nav, type NavProps };
