import * as React from "react";
import { Mark } from "@sema/design";
import { Claim, Section } from "./section";
import { useCopy } from "../copy";

/* S6. What we need, and what stays where: a 2×2 compartment grid. */
export function Stays() {
  const c = useCopy().stays;
  const { sources, leaves, back, unsure } = c.cells;
  return (
    <Section id="runs" heading="h2-stays">
      <h2 id="h2-stays" className="type-h2 col-span-12 mb-10">
        {c.h2}
      </h2>
      <div className="compartments col-span-12 grid-cols-2 border border-rule max-lg:grid-cols-1">
        <Cell claim={sources.claim}>
          <p>{sources.text}</p>
          <div className="mt-2 flex flex-wrap gap-3" aria-label={sources.marksLabel}>
            {sources.marks.map((m) => (
              <Mark key={m}>{m}</Mark>
            ))}
          </div>
        </Cell>
        <Cell claim={leaves.claim}>
          <p>{leaves.text}</p>
        </Cell>
        <Cell claim={back.claim}>
          <p>{back.text}</p>
        </Cell>
        <Cell claim={unsure.claim}>
          <p>{unsure.text}</p>
        </Cell>
      </div>
    </Section>
  );
}

function Cell({ claim, children }: { claim: string; children: React.ReactNode }) {
  return (
    <div className="grid content-start gap-3.5 px-7 pt-8 pb-9 [&_p]:max-w-[46ch]">
      <Claim className="text-[1.2rem]">{claim}</Claim>
      {children}
    </div>
  );
}
