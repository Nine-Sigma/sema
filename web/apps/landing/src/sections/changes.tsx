import { Figure, useSeen } from "@sema/design";
import { Claim, Section } from "./section";
import { useCopy, useCopyFigures } from "../copy";

/* S4 (M5). Ledger rows: claim left, sentence right, rules between. G5 reveals top to bottom on first
   view (question, lit sub-graph, context block) while the plane lights five nodes amber. */
export function Changes() {
  const c = useCopy().changes;
  const figures = useCopyFigures();
  const g5 = useSeen<HTMLDivElement>(0.3);
  return (
    <Section id="changes" heading="h2-changes" weight="light" chapter="changes">
      <h2 id="h2-changes" className="type-h2 col-span-12 mb-14 max-lg:mb-8">
        {c.h2}
      </h2>
      <div className="col-span-12">
        {c.rows.map(({ claim, text }) => (
          <div
            key={claim}
            className="grid grid-cols-[4fr_8fr] gap-x-6 border-t border-rule py-7 last:border-b max-lg:grid-cols-1 max-lg:gap-y-2"
          >
            <Claim className="max-w-[22ch]">{claim}</Claim>
            <p className="max-w-[52ch]">{text}</p>
          </div>
        ))}
      </div>
      <div ref={g5} className="sema-reveal-y col-span-12 mt-16 max-lg:mt-10">
        <Figure source={figures.g5} mobile={figures.g5m} variant="wide" />
      </div>
    </Section>
  );
}
