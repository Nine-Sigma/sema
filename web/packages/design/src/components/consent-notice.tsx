import * as React from "react";
import { cn } from "../lib/utils";
import { Button } from "./ui/button";

type ConsentNoticeProps = Omit<React.ComponentProps<"section">, "children"> & {
  /** Accessible name for the region, e.g. "Cookie notice". */
  label: string;
  /** What the one cookie does, in plain words. */
  body: React.ReactNode;
  allowText: string;
  declineText: string;
  onAllow: () => void;
  onDecline: () => void;
};

/* Notice for one cookie. Bottom-left, bordered like a record (a choice gets written down), never
   covers the page and has no close control: the two buttons are the whole choice, with equal weight.
   Arrival is gated on html[data-motion] like the form state line (motion.css `.sema-consent`). */
function ConsentNotice({ label, body, allowText, declineText, onAllow, onDecline, className, ...props }: ConsentNoticeProps) {
  const id = React.useId();
  return (
    <section
      data-slot="consent-notice"
      role="region"
      aria-label={label}
      aria-describedby={`${id}-body`}
      className={cn(
        "sema-consent fixed bottom-4 left-4 z-30 grid max-w-[400px] gap-4 border border-rule-strong bg-ground px-5 pt-5 pb-5",
        "max-sm:right-4 max-sm:max-w-none",
        className,
      )}
      {...props}
    >
      <p id={`${id}-body`} className="text-sm text-ink [&_a]:underline [&_a]:underline-offset-3">
        {body}
      </p>
      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="outline" onClick={onAllow}>
          {allowText}
        </Button>
        <Button type="button" variant="outline" onClick={onDecline}>
          {declineText}
        </Button>
      </div>
    </section>
  );
}

export { ConsentNotice, type ConsentNoticeProps };
