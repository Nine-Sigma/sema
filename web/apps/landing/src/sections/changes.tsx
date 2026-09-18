import { Figure, figures } from "@sema/design";
import { Claim, Section } from "./section";
import { useCopy } from "../copy";

/* S4. Ledger rows: claim left, sentence right, rules between. */
export function Changes() {
  const c = useCopy().changes;
  return (
    <Section id="changes" heading="h2-changes">
      <h2 id="h2-changes" className="type-h2 col-span-12 mb-10 max-lg:mb-6">
        {c.h2}
      </h2>
      <div className="col-span-12">
        {c.rows.map(({ claim, text }) => (
          <div
            key={claim}
            className="grid grid-cols-[5fr_7fr] gap-x-6 border-t border-rule py-7 last:border-b max-lg:grid-cols-1 max-lg:gap-y-2"
          >
            <Claim className="max-w-[22ch]">{claim}</Claim>
            <p className="max-w-[52ch]">{text}</p>
          </div>
        ))}
      </div>
      <Figure source={figures.g5} variant="wide" className="col-span-12 mt-12 max-lg:hidden" />
    </Section>
  );
}
