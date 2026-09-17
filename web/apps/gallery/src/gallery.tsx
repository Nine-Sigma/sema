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
  ThemeToggle,
  Wordmark,
  figures,
  useMotionGate,
  type FormOutcome,
} from "@sema/design";
import { colors, motion } from "@sema/design/tokens";

/* Every component in the library, rendered in the live theme. Toggle in the nav switches both themes. */
const DEMO = /^(error|timeout|duplicate)@/i;
function fakeSubmit(email: string): Promise<FormOutcome> {
  const m = DEMO.exec(email);
  const outcome = (m?.[1]?.toLowerCase() as FormOutcome | undefined) ?? "success";
  return new Promise((resolve) => setTimeout(() => resolve(outcome), outcome === "timeout" ? 2400 : 800));
}

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="border-t border-rule py-16">
      <div className="page-wrap grid gap-8">
        <h2 className="type-h3">{title}</h2>
        {children}
      </div>
    </section>
  );
}

function Swatch({ name, value }: { name: string; value: string }) {
  return (
    <div className="grid gap-2">
      <div className="h-16 border border-rule" style={{ background: value }} />
      <p className="type-label text-muted">{name}</p>
      <p className="font-mono text-sm">{value}</p>
    </div>
  );
}

export function Gallery() {
  const { replay } = useMotionGate();
  return (
    <>
      <SkipLink href="#tokens" />
      <Nav
        brand={<Wordmark href="#top" onClick={replay} />}
        anchors={[
          { href: "#tokens", label: "Tokens" },
          { href: "#type", label: "Type" },
          { href: "#controls", label: "Controls" },
          { href: "#forms", label: "Forms" },
          { href: "#records", label: "Records" },
          { href: "#figures", label: "Figures" },
        ]}
        actions={
          <>
            <ThemeToggle />
            <Button asChild>
              <a href="#forms">Join the waitlist</a>
            </Button>
          </>
        }
      />
      <main id="top">
        <section className="sema-hero relative overflow-hidden pt-18 pb-24">
          <Constellation className="absolute inset-0 z-0 pointer-events-none [mask-image:linear-gradient(90deg,transparent_0_22%,#000_52%,#000_100%)]" />
          <div className="page-wrap page-grid relative z-1">
            <h1 className="sema-arrive type-h1 col-span-10 max-w-[15ch]">Sema design library</h1>
            <p className="sema-arrive type-lede col-span-6 mt-7 max-w-[34ch]" style={{ "--arrive": "420ms" } as React.CSSProperties}>
              Tokens, type, controls, forms, records, figures. Click the wordmark to replay the hero moment.
            </p>
          </div>
        </section>

        <Section id="tokens" title="Color tokens">
          <div className="grid grid-cols-3 gap-6 lg:grid-cols-9">
            {Object.entries(colors.dark).map(([k, v]) => (
              <Swatch key={k} name={`dark ${k}`} value={v} />
            ))}
            {Object.entries(colors.light).map(([k, v]) => (
              <Swatch key={k} name={`light ${k}`} value={v} />
            ))}
          </div>
          <p className="text-muted text-sm">
            Motion ease <code className="font-mono">{motion.ease}</code>. Durations: {Object.entries(motion.durations).map(([k, v]) => `${k} ${v}ms`).join(", ")}.
          </p>
        </Section>

        <Section id="type" title="Type scale">
          <p className="type-h1">Unify your data by meaning.</p>
          <p className="type-h2">You've run this project.</p>
          <p className="type-h3">Two ways to start</p>
          <p className="type-claim">A claim in a ledger row</p>
          <p className="type-lede max-w-[34ch]">Lede. Sema fits your sources into one data model and keeps them there.</p>
          <p className="type-turn max-w-[40ch]">Turn. The sheet was never the problem. The sheet was the model, written down in the wrong place.</p>
          <p className="type-label text-muted">Label · IBM Plex Mono 12 / 0.08em</p>
          <p className="type-num">09</p>
          <p className="type-wordmark">Sema</p>
        </Section>

        <Section id="controls" title="Controls">
          <div className="flex flex-wrap items-center gap-4">
            <Button>Join the waitlist</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="done" disabled>On the list</Button>
            <Button aria-busy disabled>Adding…</Button>
            <Button size="lg">Join the waitlist</Button>
            <ThemeToggle />
          </div>
          <div className="grid max-w-md gap-2">
            <Label htmlFor="in1">Work email</Label>
            <Input id="in1" type="email" placeholder="you@company.com" />
            <Input id="in2" type="email" placeholder="invalid" aria-invalid />
            <Input id="in3" type="email" size="lg" placeholder="large" />
          </div>
          <div className="flex flex-wrap gap-3">
            <Tag>Exploration</Tag>
            <Tag>Showcase</Tag>
            <Tag>Shipped</Tag>
            <Mark>Databricks</Mark>
            <Mark>PostgreSQL</Mark>
          </div>
          <Eyebrow>Roadmap</Eyebrow>
          <PullQuote>The model is the product. The sheet was a symptom.</PullQuote>
        </Section>

        <Section id="forms" title="Email form">
          <p className="text-muted text-sm">Demo hook: emails starting with error@, timeout@, duplicate@ show those states.</p>
          <div className="max-w-md">
            <EmailForm
              id="form-a"
              onSubmit={fakeSubmit}
              help={
                <>
                  We email when pilot slots open. Nothing else. <a href="#privacy">Privacy</a>
                </>
              }
            />
          </div>
          <EmailForm id="form-b" size="lg" onSubmit={fakeSubmit} help="A few pilots at a time. We email when a slot opens." />
        </Section>

        <Section id="records" title="Records">
          <div className="grid grid-cols-2 gap-6 max-lg:grid-cols-1">
            <Record
              status="accepted"
              label="Accepted"
              meta="Decision [n] of [N]"
              trace="[study].[table].[column] → [target table].[column]"
              why="Same concept. The values already use the model's codes. One per patient."
            />
            <Record
              status="blocked"
              label="Blocked · sent to a person"
              meta="Decision [n] of [N]"
              trace="[study].[table].[column] → [target table].[column]"
              why="Many rows per patient where the model allows one. That's a decision, not a rename."
            />
          </div>
        </Section>

        <Section id="figures" title="Figures">
          <Figure source={figures.g7} mobile={figures.g7m} variant="wide" />
          <div className="grid grid-cols-2 gap-6 max-lg:grid-cols-1">
            <Figure source={figures.g3} />
            <Figure source={figures.g4} variant="plate" />
            <Figure source={figures.g11} />
            <Figure source={figures.g12} variant="plate" />
          </div>
          <Figure source={figures.g5} variant="wide" />
          <Figure source={figures.g6} mobile={figures.g6m} variant="wide" />
        </Section>
      </main>
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
            <a href="#privacy" id="privacy">Privacy</a>
            <a href="mailto:">Contact</a>
          </>,
          <span className="text-muted">© 2026</span>,
        ]}
      />
    </>
  );
}
