import { Button, Eyebrow, Figure, Record } from "@sema/design";
import { Section } from "./section";
import { useCopy, useCopyFigures } from "../copy";

/* S5. The showcase. The two records are the only bordered boxes on the page. */
export function Proof() {
  const c = useCopy().proof;
  const figures = useCopyFigures();
  return (
    <Section id="proof" heading="h2-proof">
      <Eyebrow className="col-span-12">{c.eyebrow}</Eyebrow>
      <h2 id="h2-proof" className="type-h2 col-span-12 mt-4">
        {c.h2}
      </h2>
      <p className="col-span-7 mt-6 max-lg:col-span-12">
        {c.body}
      </p>
      <Figure source={figures.g6} mobile={figures.g6m} variant="wide" className="col-span-12 mt-12 max-lg:mt-10" />
      <div className="col-span-12 mt-12 grid grid-cols-2 gap-6 max-lg:mt-8 max-lg:grid-cols-1">
        <Record status="accepted" {...c.cards.accepted} />
        <Record status="blocked" {...c.cards.blocked} />
      </div>
      <p className="col-span-8 mt-6 text-[15px] text-muted max-lg:col-span-12">
        {c.after}
      </p>
      <p className="col-span-12 mt-9">
        <Button asChild>
          <a href={c.cta.href}>{c.cta.text}</a>
        </Button>
      </p>
    </Section>
  );
}
