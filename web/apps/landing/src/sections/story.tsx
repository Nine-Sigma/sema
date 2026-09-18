import { Figure, figures } from "@sema/design";
import { Section } from "./section";

const BEATS = [
  ["01", "Month one", "Six systems. Six names for the same customer. Everyone agrees it's fixable."],
  ["04", "Month four", "A 1,400-row mapping sheet. The best document in the company, and it lives in one person's head."],
  ["09", "Month nine", "A new source. Half the sheet has to be redone, and nobody wrote down why the first half was right."],
  ["10", "Month ten", "The agent pilot joins on the wrong key and answers with total confidence."],
] as const;

/* S2. Four compartments with month numerals, then the turn. */
export function Story() {
  return (
    <Section id="story" heading="h2-story">
      <h2 id="h2-story" className="type-h2 col-span-12 mb-10 max-lg:mb-8">
        You've run this project.
      </h2>
      <Figure source={figures.g7} mobile={figures.g7m} variant="wide" className="col-span-12 mb-10 max-lg:mb-8" />
      <div className="compartments col-span-12 grid-cols-4 max-lg:grid-cols-1">
        {BEATS.map(([num, label, text]) => (
          <div
            key={num}
            className="grid content-start gap-5 px-6 pt-7 pb-8 max-lg:grid-cols-[80px_1fr] max-lg:grid-rows-[auto_auto] max-lg:gap-x-5"
          >
            <span aria-hidden="true" className="type-num max-lg:row-span-2 max-lg:text-5xl">
              {num}
            </span>
            <span className="type-label text-muted">{label}</span>
            <p className="text-base/normal">{text}</p>
          </div>
        ))}
      </div>
      <p className="type-turn col-span-8 mt-14 max-lg:col-span-12">
        The work was right. Where it lived was wrong. Meaning sat in a spreadsheet and in people's heads. It needs
        to live somewhere a machine can read, keep, and build on.
      </p>
    </Section>
  );
}
