import { Eyebrow } from "@sema/design";
import { Section } from "./section";
import { useCopy } from "../copy";

/* S8. Compact strip. The only place "not yet shipped" is allowed. */
export function Roadmap() {
  const c = useCopy().roadmap;
  return (
    <Section id="roadmap" heading="h2-roadmap">
      <div className="col-span-4 max-lg:col-span-12">
        <Eyebrow>{c.eyebrow}</Eyebrow>
        <h2 id="h2-roadmap" className="type-h2 mt-4">
          {c.h2}
        </h2>
      </div>
      <ul className="col-span-8 col-start-5 m-0 grid list-none grid-cols-3 gap-6 p-0 max-lg:col-span-12 max-lg:col-start-1 max-lg:mt-8 max-lg:grid-cols-1">
        {c.items.map(({ head, text }) => (
          <li key={head} className="border-t border-rule pt-4 text-base">
            <strong className="mb-1 block">{head}</strong>
            {text}
          </li>
        ))}
      </ul>
    </Section>
  );
}
