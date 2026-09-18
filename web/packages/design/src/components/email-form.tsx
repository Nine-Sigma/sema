import * as React from "react";
import { cn } from "../lib/utils";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

type FormOutcome = "success" | "duplicate" | "error" | "timeout";
type FormState = "" | "loading" | "invalid" | FormOutcome;

const MESSAGES: Record<Exclude<FormState, "">, string> = {
  loading: "Adding you…",
  success: "You're on the list. We'll email when a pilot slot opens.",
  invalid: "That doesn't look like an email address. Check it and try again.",
  duplicate: "You're already on the list.",
  error: "We couldn't save that. Try again in a minute, or email [contact address].",
  timeout: "This is taking longer than it should. Trying again is safe.",
};
const BUTTON: Partial<Record<FormState, string>> = {
  loading: "Adding…",
  success: "On the list",
  duplicate: "On the list",
  error: "Try again",
  timeout: "Try again",
};
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

type EmailFormProps = Omit<React.ComponentProps<"form">, "onSubmit"> & {
  id: string;
  /** Resolves to the outcome; reject or throw counts as "error". */
  onSubmit: (email: string) => Promise<FormOutcome>;
  label?: string;
  placeholder?: string;
  buttonText?: string;
  help?: React.ReactNode;
  /** lg: the pilot form, a full-width bar with a rule above. */
  size?: "default" | "lg";
  messages?: Partial<typeof MESSAGES>;
  /** Button text per state; falls back to buttonText. */
  buttons?: Partial<typeof BUTTON>;
};

/* Waitlist form. Six designed states after idle: loading, success, duplicate, invalid, error, timeout.
   The state line is reserved so nothing shifts; done states lock the form. */
function EmailForm({
  id,
  onSubmit,
  label = "Work email",
  placeholder = "you@company.com",
  buttonText = "Join the waitlist",
  help,
  size = "default",
  messages,
  buttons,
  className,
  ...props
}: EmailFormProps) {
  const [state, setState] = React.useState<FormState>("");
  const inputRef = React.useRef<HTMLInputElement>(null);
  const text = { ...MESSAGES, ...messages };
  const button = { ...BUTTON, ...buttons };
  const done = state === "success" || state === "duplicate";
  const busy = state === "loading";
  const retryable = state === "invalid" || state === "error" || state === "timeout";

  async function submit(ev: React.FormEvent<HTMLFormElement>): Promise<void> {
    ev.preventDefault();
    if (done || busy) return;
    const input = inputRef.current;
    const value = input?.value ?? "";
    if (!input?.validity.valid || !EMAIL.test(value)) {
      setState("invalid");
      input?.focus();
      return;
    }
    setState("loading");
    try {
      setState(await onSubmit(value));
    } catch {
      setState("error");
    }
  }

  return (
    <form
      id={id}
      noValidate
      data-slot="email-form"
      data-done={done ? "" : undefined}
      className={cn("grid gap-2", className)}
      onSubmit={(ev) => void submit(ev)}
      {...props}
    >
      <Label htmlFor={`${id}-email`}>{label}</Label>
      <div
        className={cn(
          "flex max-sm:flex-col",
          size === "lg" && "border-t border-rule-strong pt-6 max-lg:flex-col",
        )}
      >
        <Input
          ref={inputRef}
          type="email"
          id={`${id}-email`}
          name="email"
          size={size}
          autoComplete="email"
          inputMode="email"
          enterKeyHint="send"
          autoCapitalize="off"
          spellCheck={false}
          maxLength={254}
          placeholder={placeholder}
          required
          readOnly={done}
          aria-invalid={state === "invalid"}
          aria-describedby={help ? `${id}-help` : undefined}
          className={cn(
            "sm:border-r-0",
            size === "lg" && "max-lg:border-r max-lg:border-b-0 lg:border-r-0",
            size === "default" && "max-sm:border-b-0",
          )}
          onInput={() => {
            if (retryable) setState("");
          }}
        />
        <Button
          type="submit"
          size={size}
          variant={done ? "done" : "default"}
          disabled={busy || done}
          aria-busy={busy}
        >
          {button[state] ?? buttonText}
        </Button>
      </div>
      {help ? (
        <p id={`${id}-help`} className="text-[13px] text-muted [&_a]:underline [&_a]:underline-offset-3">
          {help}
        </p>
      ) : null}
      <p
        role="status"
        aria-live="polite"
        data-state={state}
        className={cn(
          "sema-state flex min-h-[1.4em] items-baseline gap-2.5 text-sm text-ink",
          "before:size-2 before:flex-none before:-translate-y-px before:bg-muted before:content-['']",
          state === "" && "before:invisible",
          state === "loading" && "text-muted",
          state === "success" && "before:bg-blue",
          state === "duplicate" && "before:bg-transparent before:shadow-[inset_0_0_0_1px_var(--ink)]",
          retryable && "before:bg-accent",
          (state === "error" || state === "timeout") && "flex-wrap",
        )}
      >
        {state ? text[state] : ""}
      </p>
    </form>
  );
}

export { EmailForm, type EmailFormProps, type FormOutcome, type FormState };
