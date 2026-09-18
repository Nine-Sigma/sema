import { Figure, figures } from "@sema/design";
import { Section } from "./section";
import { useCopy } from "../copy";

/* S2. Four compartments with month numerals, then the turn. */
export function Story() {
  const c = useCopy().story;
  return (
    <Section id="story" heading="h2-story">
      <h2 id="h2-story" className="type-h2 col-span-12 mb-10 max-lg:mb-8">
        {c.h2}
      </h2>
      <Figure source={figures.g7} mobile={figures.g7m} variant="wide" className="col-span-12 mb-10 max-lg:mb-8" />
      <div className="compartments col-span-12 grid-cols-4 max-lg:grid-cols-1">
        {c.beats.map(({ num, label, text }) => (
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
        {c.turn}
      </p>
    </Section>
  );
}
