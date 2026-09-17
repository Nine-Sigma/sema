import * as React from "react";
import { cn } from "../lib/utils";
import { useSeen } from "../hooks/use-seen";

type RecordStatus = "accepted" | "blocked";

type RecordProps = Omit<React.ComponentProps<"article">, "children"> & {
  status: RecordStatus;
  /** Status word(s), e.g. "Accepted" or "Blocked · sent to a person". */
  label: string;
  /** Right-hand caption, e.g. "Decision 3 of 41". */
  meta?: string;
  /** Mono trace line, e.g. "study.table.column → target.column". */
  trace: string;
  /** One or two sentences on why. */
  why: React.ReactNode;
};

/* The only bordered boxes on the page: elevation means "a record was written".
   Blocked carries the amber border; its trace stops at 60% and the status word turns amber on first view. */
function Record({ status, label, meta, trace, why, className, ...props }: RecordProps) {
  const ref = useSeen<HTMLElement>();
  return (
    <article
      ref={ref}
      data-slot="record"
      data-status={status}
      className={cn(
        "sema-record grid content-start gap-4 border border-rule-strong px-6 pt-6 pb-7",
        status === "blocked" && "border-accent",
        className,
      )}
      {...props}
    >
      <p className="type-label flex justify-between gap-3 text-ink max-lg:flex-col max-lg:gap-1">
        <span className={cn("sema-status-word", status === "blocked" && "text-accent")}>{label}</span>
        {meta ? <span className="text-muted">{meta}</span> : null}
      </p>
      <p className="sema-trace relative border-y border-rule py-3.5 font-mono text-sm/normal [overflow-wrap:anywhere]">
        {trace}
      </p>
      <p className="text-base">{why}</p>
    </article>
  );
}

export { Record, type RecordProps, type RecordStatus };
