import { Figure, figures } from "@sema/design";
import { Claim, Section } from "./section";

const THESES = [
  ["Agents need legible data.", "Every company is putting agents on its data. Agents fail on data with no shared meaning and no trusted joins. The bottleneck moved from the model to the data."],
  ["A services business waiting to become software.", "Consultants do the fit once and it decays. Software that keeps it fitted changes who can afford a unified model."],
  ["Understanding got cheap. Trust didn't.", "The durable asset is the checked, corrected record of what your data means. That compounds. A prompt doesn't."],
] as const;

const STEPS = [
  ["The wedge.", "Domains with a standard model, where every conversion into it is still a project. Healthcare first. The same shape exists in finance, retail, and manufacturing."],
  ["The expansion.", "Domains with no standard model, where Sema proposes one. Property records are the first exploration. Then the model your organization defines."],
] as const;

/* S7. Essay left, figure rail right. */
export function Why() {
  return (
    <Section id="why" heading="h2-why">
      <h2 id="h2-why" className="type-h2 col-span-12 mb-10">
        Why now, and why this.
      </h2>
      <div className="col-span-7 grid content-start gap-9 max-lg:col-span-12">
        {THESES.map(([claim, text]) => (
          <div key={claim}>
            <Claim className="mb-2">{claim}</Claim>
            <p className="max-w-[52ch]">{text}</p>
          </div>
        ))}
      </div>
      <div className="col-span-4 col-start-9 grid content-start gap-7 max-lg:col-span-12 max-lg:col-start-1 max-lg:mt-8">
        <Figure source={figures.g11} />
        {STEPS.map(([claim, text]) => (
          <div key={claim} className="grid gap-1.5">
            <p className="font-bold">{claim}</p>
            <p className="text-[15px]">{text}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}
