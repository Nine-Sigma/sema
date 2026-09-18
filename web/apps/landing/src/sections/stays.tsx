import * as React from "react";
import { Mark } from "@sema/design";
import { Claim, Section } from "./section";

/* S6. What we need, and what stays where: a 2×2 compartment grid. */
export function Stays() {
  return (
    <Section id="runs" heading="h2-stays">
      <h2 id="h2-stays" className="type-h2 col-span-12 mb-10">
        What we need, and what stays where.
      </h2>
      <div className="compartments col-span-12 grid-cols-2 border border-rule max-lg:grid-cols-1">
        <Cell claim="Your sources.">
          <p>
            Databricks or PostgreSQL today. Sema reads; it never writes to your source tables. Bring an industry
            standard model (OMOP is included) or your own.
          </p>
          <div className="mt-2 flex flex-wrap gap-3" aria-label="Supported sources">
            <Mark>Databricks</Mark>
            <Mark>PostgreSQL</Mark>
          </div>
        </Cell>
        <Cell claim="What leaves your warehouse.">
          <p>Column names, descriptions, and small value samples go to the AI provider you choose. Row data stays where it is.</p>
        </Cell>
        <Cell claim="What comes back.">
          <p>
            Unified tables in your warehouse, and a model any agent or query tool can ask for context. Every decision
            carries its source and confidence.
          </p>
        </Cell>
        <Cell claim="When Sema isn't sure.">
          <p>It asks a person. The answer is kept.</p>
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
