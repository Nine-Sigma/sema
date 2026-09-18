import * as React from "react";
import { figures, type FigureId, type FigureSource } from "@sema/design";

export type Link = { text: string; href: string };
export type Lead = { lead: string; text: string };
export type ClaimRow = { claim: string; text: string };

/* Every string on the page, keyed by section in page order. Slot names follow COPY.md, so a
   sign-off row id `S5__card_blocked_why` reads as `copy.proof.cards.blocked.why`. Text slots are plain text; components own the markup.
   Optional figure overrides reuse repository-authored SVG plates with revised labels. */
export type Plates = Record<FigureId, FigureSource>;
export type FigureOverrides = (plates: Plates) => Partial<Plates>;

export type Copy = {
  /** Optional copy-only plate overrides (a function of the design package's plates); absent for the control. */
  figures?: FigureOverrides;
  meta: { title: string; description: string; ogImageAlt: string };
  nav: {
    wordmark: string;
    wordmarkLabel: string;
    wordmarkHref: string;
    anchors: { href: string; label: string }[];
    cta: Link;
    skip: string;
    theme: { toDark: string; toLight: string };
  };
  hero: { h1: string; subhead: string; help: { text: string; link: Link }; secondary: Link };
  story: { h2: string; beats: { num: string; label: string; text: string }[]; turn: string };
  idea: {
    h2: string;
    paras: string[];
    pullQuote: string;
    h3: string;
    ways: { have: Way; dont: Way };
  };
  changes: { h2: string; rows: ClaimRow[] };
  proof: {
    eyebrow: string;
    h2: string;
    body: string;
    cards: { accepted: RecordCopy; blocked: RecordCopy };
    after: string;
    cta: Link;
  };
  stays: {
    h2: string;
    cells: { sources: ClaimRow & { marksLabel: string; marks: string[] }; leaves: ClaimRow; back: ClaimRow; unsure: ClaimRow };
  };
  why: { h2: string; theses: ClaimRow[]; steps: ClaimRow[] };
  roadmap: { eyebrow: string; h2: string; items: { head: string; text: string }[] };
  investors: { h3: string; body: string; link: Link };
  pilot: { h2: string; deal: Lead[]; help: { text: string; link: Link } };
  footer: { origin: { pre: string; word: string; post: string }; cta: Link; links: Link[]; copyright: string };
  form: {
    label: string;
    placeholder: string;
    button: string;
    messages: Record<FormMessageKey, string>;
    buttons: Record<FormButtonKey, string>;
  };
  notFound: { title: string; numeral: string; h1: string; body: string; link: Link };
};

type Way = { title: string; tag: string; body: string };
type RecordCopy = { label: string; meta: string; trace: string; why: string };
type FormMessageKey = "loading" | "success" | "invalid" | "duplicate" | "error" | "timeout";
type FormButtonKey = Exclude<FormMessageKey, "invalid">;

export type DeepPartial<T> = T extends (infer U)[]
  ? U[]
  : T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;

/* Deep merge for a variant: objects recurse, arrays and strings replace. */
export function resolve<T extends object>(base: T, overrides: DeepPartial<T>): T {
  const out: Record<string, unknown> = { ...(base as Record<string, unknown>) };
  for (const [key, value] of Object.entries(overrides as Record<string, unknown>)) {
    if (value === undefined) continue;
    const current = out[key];
    out[key] = isPlainObject(current) && isPlainObject(value) ? resolve(current, value) : value;
  }
  return out as T;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/* EmailForm props from the form copy; the prop names differ from the slot names. */
export function formProps({ label, placeholder, button, messages, buttons }: Copy["form"]) {
  return { label, placeholder, buttonText: button, messages, buttons };
}

const CopyContext = React.createContext<Copy | undefined>(undefined);

export function CopyProvider({ value, children }: { value: Copy; children: React.ReactNode }) {
  return <CopyContext.Provider value={value}>{children}</CopyContext.Provider>;
}

export function useCopy(): Copy {
  const copy = React.useContext(CopyContext);
  if (!copy) throw new Error("useCopy() called outside <CopyProvider>");
  return copy;
}

/* Existing plates are the default; a variant can relabel some without a redesign. */
export function resolveFigures(copy: Copy): Plates {
  return { ...figures, ...copy.figures?.(figures) };
}

export function useCopyFigures(): Plates {
  const copy = useCopy();
  return React.useMemo(() => resolveFigures(copy), [copy]);
}
