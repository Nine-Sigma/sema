import { Eyebrow } from "@sema/design";
import { Section } from "./section";

const ITEMS = [
  ["More warehouses.", "Chosen with pilot partners."],
  ["Documents and notes.", "Brought into the same model as your tables."],
  ["Review in the product.", "Answer Sema's questions in a UI, not an export."],
] as const;

/* S8. Compact strip. The only place "not yet shipped" is allowed. */
export function Roadmap() {
  return (
    <Section id="roadmap" heading="h2-roadmap">
      <div className="col-span-4 max-lg:col-span-12">
        <Eyebrow>Roadmap · not yet shipped</Eyebrow>
        <h2 id="h2-roadmap" className="type-h2 mt-4">
          What's next.
        </h2>
      </div>
      <ul className="col-span-8 col-start-5 m-0 grid list-none grid-cols-3 gap-6 p-0 max-lg:col-span-12 max-lg:col-start-1 max-lg:mt-8 max-lg:grid-cols-1">
        {ITEMS.map(([head, text]) => (
          <li key={head} className="border-t border-rule pt-4 text-base">
            <strong className="mb-1 block">{head}</strong>
            {text}
          </li>
        ))}
      </ul>
    </Section>
  );
}
