import { Eyebrow } from "@sema/design";
import { Section } from "./section";
import { useCopy } from "../copy";

/* S8. The second eyebrow, and three items on one line separated by hairlines, no boxes. */
export function Roadmap() {
  const c = useCopy().roadmap;
  return (
    <Section id="roadmap" heading="h2-roadmap" weight="connective" chapter="roadmap">
      <div className="col-span-12">
        <Eyebrow>{c.eyebrow}</Eyebrow>
        <h2 id="h2-roadmap" className="type-h2 mt-4">
          {c.h2}
        </h2>
      </div>
      <ul className="col-span-12 m-0 mt-12 grid list-none grid-cols-3 border-y border-rule p-0 max-lg:mt-8 max-lg:grid-cols-1">
        {c.items.map(({ head, text }, k) => (
          <li
            key={head}
            className={`py-6 pr-8 text-base max-lg:border-t max-lg:first:border-t-0 max-lg:pr-0 ${k > 0 ? "border-l border-rule pl-8 max-lg:border-l-0 max-lg:pl-0" : ""}`}
          >
            <strong className="mb-1 block">{head}</strong>
            {text}
          </li>
        ))}
      </ul>
    </Section>
  );
}
