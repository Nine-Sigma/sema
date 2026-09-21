import { Figure, useSeen } from "@sema/design";
import { Section } from "./section";
import { useCopy, useCopyFigures } from "../copy";
import { useReveal } from "../motion/use-stage";

/* S2 (M3). G7 draws left to right on first view. The beats are a 4-up text row with no dividers;
   each carries a chapter numeral at 220px that draws to 12% ink as the reader passes it. */
export function Story() {
  const c = useCopy().story;
  const figures = useCopyFigures();
  const beats = useReveal<HTMLDivElement>(".sema-numeral");
  const g7 = useSeen<HTMLDivElement>(0.4);
  return (
    <Section id="story" heading="h2-story" weight="connective" chapter="story">
      <h2 id="h2-story" className="type-h2 col-span-8 mb-14 max-lg:col-span-12 max-lg:mb-8">
        {c.h2}
      </h2>
      <div ref={g7} className="sema-reveal-x col-span-12 mb-16 max-lg:mb-10">
        <Figure source={figures.g7} mobile={figures.g7m} variant="wide" />
      </div>
      <div ref={beats} className="col-span-12 grid grid-cols-4 gap-x-6 max-lg:grid-cols-1 max-lg:gap-y-12">
        {c.beats.map(({ num, label, text }) => (
          <div key={num} className="relative grid content-start gap-4 max-lg:pr-24">
            <span aria-hidden="true" className="sema-numeral type-chapter">
              {num}
            </span>
            <p className="relative text-base/normal">
              <strong>{label}.</strong> {text}
            </p>
          </div>
        ))}
      </div>
      <p className="type-turn col-span-8 mt-20 max-lg:col-span-12 max-lg:mt-14">
        {c.turn}
      </p>
    </Section>
  );
}
