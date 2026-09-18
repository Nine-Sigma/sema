import { Figure } from "@sema/design";
import { Claim, Section } from "./section";
import { useCopy, useCopyFigures } from "../copy";

/* S7. Essay left, figure rail right. */
export function Why() {
  const c = useCopy().why;
  const figures = useCopyFigures();
  return (
    <Section id="why" heading="h2-why">
      <h2 id="h2-why" className="type-h2 col-span-12 mb-10">
        {c.h2}
      </h2>
      <div className="col-span-7 grid content-start gap-9 max-lg:col-span-12">
        {c.theses.map(({ claim, text }) => (
          <div key={claim}>
            <Claim className="mb-2">{claim}</Claim>
            <p className="max-w-[52ch]">{text}</p>
          </div>
        ))}
      </div>
      <div className="col-span-4 col-start-9 grid content-start gap-7 max-lg:col-span-12 max-lg:col-start-1 max-lg:mt-8">
        <Figure source={figures.g11} />
        {c.steps.map(({ claim, text }) => (
          <div key={claim} className="grid gap-1.5">
            <p className="font-bold">{claim}</p>
            <p className="text-[15px]">{text}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}
