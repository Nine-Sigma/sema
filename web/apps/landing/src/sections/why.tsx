import { Figure, useSeen } from "@sema/design";
import { Claim, Section } from "./section";
import { useCopy, useCopyFigures } from "../copy";

/* S7. Three theses as a numbered column (mono numerals, not oversized), G11 to the right. The rings
   draw outside in on first view, core last; the plane's amber point settles behind G11's core. */
export function Why() {
  const c = useCopy().why;
  const figures = useCopyFigures();
  const g11 = useSeen<HTMLDivElement>(0.5);
  return (
    <Section id="why" heading="h2-why" weight="light" chapter="why">
      <h2 id="h2-why" className="type-h2 col-span-12 mb-14 max-lg:mb-8">
        {c.h2}
      </h2>
      <ol className="col-span-6 m-0 grid list-none content-start gap-10 p-0 max-lg:col-span-12">
        {c.theses.map(({ claim, text }, k) => (
          <li key={claim} className="grid grid-cols-[3rem_1fr] gap-x-4">
            <span aria-hidden="true" className="type-label text-muted pt-1.5">
              {String(k + 1).padStart(2, "0")}
            </span>
            <div>
              <Claim className="mb-2">{claim}</Claim>
              <p className="max-w-[46ch]">{text}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="col-span-5 col-start-8 grid content-start gap-7 max-lg:col-span-12 max-lg:col-start-1 max-lg:mt-10">
        <div ref={g11} data-plane-anchor="g11" className="sema-reveal-rings">
          <Figure source={figures.g11} mobile={figures.g11m} />
        </div>
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
