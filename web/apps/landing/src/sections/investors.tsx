import { Section } from "./section";

/* S10. Offset block, shorter rhythm than the other sections. */
export function Investors() {
  return (
    <Section id="investors" heading="h3-inv" className="py-16">
      <div className="col-span-6 col-start-7 grid gap-3.5 max-lg:col-span-12 max-lg:col-start-1 [&_p]:max-w-[52ch]">
        <h3 id="h3-inv" className="type-h3">
          For investors.
        </h3>
        <p>
          We're raising a [stage] round to take Sema from showcase to paid pilots, and to ship the second way in:
          models proposed from the sources. Deck and the showcase run on request.
        </p>
        <p className="font-mono text-sm">
          <a href="mailto:">[investor address] →</a>
        </p>
      </div>
    </Section>
  );
}
