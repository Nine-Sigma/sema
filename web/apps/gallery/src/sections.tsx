import * as React from "react";
import {
  Button,
  Constellation,
  EmailForm,
  Eyebrow,
  Figure,
  Footer,
  Input,
  Label,
  Mark,
  Nav,
  PullQuote,
  Record,
  SkipLink,
  Tag,
  Toggle,
  Wordmark,
  figures,
  useMotionGate,
  type FigureSource,
  type FormOutcome,
  type Theme,
} from "@sema/design";
import { motion } from "@sema/design/tokens";
import { CaptionButton, Section, Specimen } from "./ledger";
import { TokenPairings } from "./tokens-ledger";
import { TypeLedger } from "./type-ledger";

/* DEMO HOOK: emails starting error@, timeout@, duplicate@ show those states. Gallery only. */
const DEMO = /^(error|timeout|duplicate)@/i;
function fakeSubmit(email: string): Promise<FormOutcome> {
  const m = DEMO.exec(email);
  const outcome = (m?.[1]?.toLowerCase() as FormOutcome | undefined) ?? "success";
  return new Promise((resolve) => setTimeout(() => resolve(outcome), outcome === "timeout" ? 2400 : 800));
}

/* Figma exports carry fixed title/desc ids; a second render of the same plate needs its own. */
function suffixed(src: FigureSource, sfx: string): FigureSource {
  const re = new RegExp(`${src.id}-(t|d)\\b`, "g");
  return { ...src, svg: src.svg.replace(re, `${src.id}-$1-${sfx}`) };
}

function TokensSection() {
  return (
    <Section id="tokens" title="Tokens" intro="Each cell is a pairing a component depends on. The fill reads the live CSS variable; the ratio is computed from tokens.ts, so a drift between the two is visible here.">
      <Specimen
        name="Pairings"
        props="theme.css :root / [data-theme] · tokens.ts colors"
        note="Text pairs need 4.5:1 (AA). Rule strong is a border: 3:1. Rule is a hairline on ground and has no floor."
        render={(t) => <TokenPairings theme={t} />}
      />
      <Specimen
        name="Motion"
        layout="single"
        props={`ease ${motion.ease} · heroFallbackMs ${motion.heroFallbackMs}`}
        note="One authored moment (the constellation forms, copy arrives), two responses (record trace, form state line), one loop (pulse). All gated on html[data-motion]; reduced motion gets the end state and no loop."
        render={() => (
          <dl className="grid grid-cols-4 gap-x-4 gap-y-5 sm:grid-cols-8">
            {Object.entries(motion.durations).map(([k, v]) => (
              <div key={k} className="grid gap-1">
                <dt className="type-label text-muted">{k}</dt>
                <dd className="m-0 font-mono text-sm">{v}ms</dd>
              </div>
            ))}
          </dl>
        )}
      />
    </Section>
  );
}

function TypeSection() {
  return (
    <Section id="type" title="Type" intro="One utility per role, so the page never restates sizes. Captions are read from the rendered element at the current viewport.">
      <TypeLedger />
    </Section>
  );
}

/* The library ThemeToggle icon, pressed by the cell's theme; no side effects on the page frame. */
function ThemeToggleSpecimen({ theme }: { theme: Theme }) {
  const light = theme === "light";
  return (
    <Toggle pressed={light} aria-label={light ? "Switch to dark theme" : "Switch to light theme"} aria-disabled tabIndex={-1}>
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.25" />
        <path d="M8 1.75A6.25 6.25 0 0 1 8 14.25Z" fill="currentColor" />
      </svg>
    </Toggle>
  );
}

function ToggleSpecimen() {
  const [on, setOn] = React.useState(false);
  return (
    <Toggle pressed={on} onPressedChange={setOn} aria-label="Toggle specimen">
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <rect x="2" y="2" width="12" height="12" fill={on ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.25" />
      </svg>
    </Toggle>
  );
}

function InputsSpecimen() {
  return (
    <Specimen
      name="Input · Label"
      props={'size="default" | "lg" · aria-invalid + aria-describedby · placeholder'}
      note="Transparent field on the ground, one rule around it; amber only on focus and invalid. An invalid field always carries its message."
      render={(_, s) => (
        <div className="grid max-w-md gap-4">
          <div className="grid gap-2">
            <Label htmlFor={`in1-${s}`}>Work email</Label>
            <Input id={`in1-${s}`} type="email" placeholder="you@company.com" />
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`in2-${s}`}>Work email</Label>
            <Input id={`in2-${s}`} type="email" defaultValue="dean@" aria-invalid aria-describedby={`in2-${s}-msg`} />
            <p id={`in2-${s}-msg`} className="text-[13px] text-accent">That doesn't look like an email address. Check it and try again.</p>
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`in3-${s}`}>Work email</Label>
            <Input id={`in3-${s}`} type="email" size="lg" placeholder="you@company.com" />
          </div>
        </div>
      )}
    />
  );
}

function ControlsSection() {
  return (
    <Section id="controls" title="Controls">
      <Specimen
        name="Button"
        props={'variant="default" | "outline" | "done" · size="default" | "lg" · aria-busy · disabled'}
        note="The page's one filled control. Amber is spent here; everything else is ink on ground."
        render={() => (
          <div className="flex flex-wrap items-center gap-4">
            <Button>Join the waitlist</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="done" disabled>On the list</Button>
            <Button aria-busy disabled>Adding…</Button>
            <Button size="lg">Join the waitlist</Button>
          </div>
        )}
      />
      <InputsSpecimen />
      <Specimen
        name="Toggle · ThemeToggle"
        props="Toggle: pressed · onPressedChange · aria-label · ThemeToggle: useTheme(), aria-label switches with state"
        note="44px square, outlined, no fill in either state; the icon carries the state and rotates 180°. The live ThemeToggle is in the nav; every instance on a page shares html[data-theme]. This one is inert."
        render={(t) => (
          <div className="flex flex-wrap items-center gap-4">
            <ToggleSpecimen />
            <ThemeToggleSpecimen theme={t} />
          </div>
        )}
      />
      <Specimen
        name="Tag · Mark"
        props="Tag: status word · Mark: product name"
        note="Tags use positive words only: Exploration, Showcase, Shipped. Marks are typographic until official marks arrive."
        render={() => (
          <div className="flex flex-wrap gap-3">
            <Tag>Exploration</Tag>
            <Tag>Showcase</Tag>
            <Tag>Shipped</Tag>
            <Mark>Databricks</Mark>
            <Mark>PostgreSQL</Mark>
          </div>
        )}
      />
      <Specimen
        name="Eyebrow"
        layout="single"
        note="Mono caption above a heading. Budget: at most one per three sections."
        render={() => <Eyebrow>Roadmap</Eyebrow>}
      />
      <Specimen
        name="PullQuote"
        layout="single"
        props="lg:pr-[28%] measure"
        note="The one amber rule on the page that is not a control."
        render={() => <PullQuote>The model is the product. The sheet was a symptom.</PullQuote>}
      />
    </Section>
  );
}

function FormsSection() {
  return (
    <Section id="forms" title="Forms" intro="Six designed states after idle: loading, success, duplicate, invalid, error, timeout. Demo hook: emails starting error@, timeout@, duplicate@ show those states.">
      <Specimen
        name="EmailForm"
        props={'size="default" · help · onSubmit → "success" | "duplicate" | "error" | "timeout"'}
        note="The state line is reserved so nothing shifts; done states lock the form. Reset remounts it."
        resettable
        render={(_, s) => (
          <div className="max-w-md">
            <EmailForm
              id={`form-a-${s}`}
              onSubmit={fakeSubmit}
              help={
                <>
                  We email when pilot slots open. Nothing else. <a href="#shell">Privacy</a>
                </>
              }
            />
          </div>
        )}
      />
      <Specimen
        name="EmailForm"
        props={'size="lg" · help'}
        layout="stack"
        resettable
        render={(_, s) => <EmailForm id={`form-b-${s}`} size="lg" onSubmit={fakeSubmit} help="A few pilots at a time. We email when a slot opens." />}
      />
    </Section>
  );
}

function RecordsSection() {
  return (
    <Section id="records" title="Records" intro="On the landing page these are the only bordered boxes: elevation means a record was written.">
      <Specimen
        name="Record"
        props={'status="accepted" · label · meta · trace · why'}
        note="Blue trace draws through on first view."
        render={() => (
          <Record status="accepted" label="Accepted" meta="Decision [n] of [N]" trace="[study].[table].[column] → [target table].[column]" why="Same concept. The values already use the model's codes. One per patient." />
        )}
      />
      <Specimen
        name="Record"
        props={'status="blocked"'}
        note="Amber border. The trace stops at 60% and the status word turns amber on first view."
        render={() => (
          <Record status="blocked" label="Blocked · sent to a person" meta="Decision [n] of [N]" trace="[study].[table].[column] → [target table].[column]" why="Many rows per patient where the model allows one. That's a decision, not a rename." />
        )}
      />
    </Section>
  );
}

const PLATES: { id: keyof typeof figures; variant?: "wide" | "plate"; mobile?: keyof typeof figures }[] = [
  { id: "g7", variant: "wide", mobile: "g7m" },
  { id: "g3" },
  { id: "g4", variant: "plate" },
  { id: "g11" },
  { id: "g12", variant: "plate" },
  { id: "g5", variant: "wide" },
  { id: "g6", variant: "wide", mobile: "g6m" },
];

function FiguresSection() {
  const { replay } = useMotionGate();
  return (
    <Section id="figures" title="Figures" intro="Inline SVG from Figma, themed through the tokens at build time. Plates are drawn for 640–1200px; wide ones scroll sideways under 720px.">
      <Specimen
        name="Constellation"
        layout="stack"
        props={`seed=11 · viewBox 2400×800 · ${motion.durations.arrive}ms arrive · pulse ${motion.durations.pulse}ms`}
        note="The hero ground. Held at its first frame until the page is painted in front of the reader, then forms a fit. The pulse pauses offscreen."
        action={<CaptionButton onClick={replay}>Replay</CaptionButton>}
        cellClassName="sema-hero relative h-[300px] p-0"
        render={() => <Constellation className="absolute inset-0" />}
      />
      {PLATES.map((p) => {
        const src = figures[p.id];
        const mobile = p.mobile ? figures[p.mobile] : undefined;
        return (
          <Specimen
            key={p.id}
            name={`Figure · ${src.title}`}
            layout={p.variant === "wide" ? "stack" : "pair"}
            props={`figures.${p.id}${p.variant ? ` · variant="${p.variant}"` : ""}${p.mobile ? ` · mobile=figures.${p.mobile}` : ""}`}
            note={src.desc}
            render={(t) => <Figure source={t === "light" ? suffixed(src, "l") : src} mobile={mobile && t === "light" ? suffixed(mobile, "l") : mobile} variant={p.variant} />}
          />
        );
      })}
    </Section>
  );
}

const SHELL_ANCHORS = [
  { href: "#tokens", label: "Tokens" },
  { href: "#type", label: "Type" },
  { href: "#controls", label: "Controls" },
];

function NavSpecimen() {
  return (
    <Specimen
      name="Nav"
      layout="stack"
      props="brand · anchors[] · actions"
      note="64px bar under a 1px rule. Anchors hide under 901px by contract; this gallery adds its own anchor row there. Targets are padded to ~44px."
      cellClassName="p-0"
      render={(t) => (
        <Nav
          brand={<Wordmark href="#top" />}
          anchors={SHELL_ANCHORS}
          actions={
            <>
              <ThemeToggleSpecimen theme={t} />
              <Button asChild>
                <a href="#forms">Join the waitlist</a>
              </Button>
            </>
          }
        />
      )}
    />
  );
}

function FooterSpecimen() {
  return (
    <Specimen
      name="Footer"
      layout="stack"
      props="columns[] (up to four)"
      note="Each column stacks its children; links padded to ~44px. Hrefs here are placeholders."
      cellClassName="p-0"
      render={() => (
        <Footer
          columns={[
            <>
              <span className="type-wordmark">Sema</span>
              <span className="text-muted">
                from Greek <i className="font-serif not-italic text-base">σῆμα</i>, "sign"
              </span>
            </>,
            <a href="#forms">Join the waitlist</a>,
            <>
              <a href="#shell">Privacy</a>
              <a href="#shell">Contact</a>
            </>,
            <span className="text-muted">© 2026</span>,
          ]}
        />
      )}
    />
  );
}

function ShellSection() {
  return (
    <Section id="shell" title="Shell" intro="The page frame: skip link, nav, wordmark, footer. Rendered here as specimens; the frame around this page is the live instance.">
      <Specimen
        name="SkipLink"
        layout="single"
        props='href · children="Skip to content"'
        note="Offscreen until focused (Tab from the top of the page). Shown in place here. The target needs tabindex=-1 so focus moves in every browser."
        render={() => <SkipLink href="#ledger" className="static inline-block" />}
      />
      <Specimen
        name="Wordmark"
        props='href · aria-label="Sema, home"'
        note="Serif Sema. On the landing page its click replays the hero moment."
        render={() => <Wordmark href="#top" />}
      />
      <NavSpecimen />
      <FooterSpecimen />
    </Section>
  );
}

export { ControlsSection, FiguresSection, FormsSection, RecordsSection, ShellSection, TokensSection, TypeSection };
