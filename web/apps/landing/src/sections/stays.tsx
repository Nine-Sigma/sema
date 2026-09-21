import * as React from "react";
import { Mark } from "@sema/design";
import { Claim, Section } from "./section";
import { useCopy } from "../copy";

/* S6. An annotated table, not cells: four rows, hairline rules, claim left, text right. A quiet passage. */
export function Stays() {
  const c = useCopy().stays;
  const { sources, leaves, back, unsure } = c.cells;
  return (
    <Section id="runs" heading="h2-stays" weight="light" chapter="stays">
      <h2 id="h2-stays" className="type-h2 col-span-12 mb-14 max-lg:mb-8">
        {c.h2}
      </h2>
      <div className="col-span-12">
        <Row claim={sources.claim}>
          <p>{sources.text}</p>
          <div className="mt-3 flex flex-wrap gap-3" aria-label={sources.marksLabel}>
            {sources.marks.map((m) => (
              <Mark key={m}>{m}</Mark>
            ))}
          </div>
        </Row>
        <Row claim={leaves.claim}>
          <p>{leaves.text}</p>
        </Row>
        <Row claim={back.claim}>
          <p>{back.text}</p>
        </Row>
        <Row claim={unsure.claim}>
          <p>{unsure.text}</p>
        </Row>
      </div>
    </Section>
  );
}

function Row({ claim, children }: { claim: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[4fr_8fr] gap-x-6 border-t border-rule py-8 last:border-b max-lg:grid-cols-1 max-lg:gap-y-3 [&_p]:max-w-[52ch]">
      <Claim className="max-w-[20ch]">{claim}</Claim>
      <div>{children}</div>
    </div>
  );
}
