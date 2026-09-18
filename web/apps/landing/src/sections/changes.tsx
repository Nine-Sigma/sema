import { Figure, figures } from "@sema/design";
import { Claim, Section } from "./section";

const ROWS = [
  ["A new source is an increment, not a restart.", "Sema fits it to what's already there."],
  ["Agents answer from meaning.", "Ask a question. The agent gets the part of the model that matters and never guesses what a column means."],
  ["Corrections stick.", "Fix something once. The fix is kept, attributed, and survives every rebuild."],
  ["You can see why.", "Every decision shows where it came from and how sure Sema was."],
] as const;

/* S4. Ledger rows: claim left, sentence right, rules between. */
export function Changes() {
  return (
    <Section id="changes" heading="h2-changes">
      <h2 id="h2-changes" className="type-h2 col-span-12 mb-10 max-lg:mb-6">
        What changes for you.
      </h2>
      <div className="col-span-12">
        {ROWS.map(([claim, text]) => (
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
