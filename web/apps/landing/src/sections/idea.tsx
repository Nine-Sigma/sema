import * as React from "react";
import { Figure, PullQuote, Tag, figures } from "@sema/design";
import { Section } from "./section";

/* S3. Prose left, G3 right, the pull quote, then the two ways in. */
export function Idea() {
  return (
    <Section id="idea" heading="h2-idea">
      <h2 id="h2-idea" className="type-h2 col-span-12">
        Integrate by meaning.
      </h2>
      <div className="col-span-5 mt-10 max-w-[66ch] [&_p+p]:mt-[1em] max-lg:col-span-12">
        <p>
          Sema reads your sources and works out what they mean: which columns are the same thing, which codes mean
          what, which tables join. Then it fits them to one model and checks its own work. When a fit can't hold,
          Sema blocks it and says why.
        </p>
        <p>
          Every decision keeps its source, how sure Sema was, and who changed it. Corrections stay. Add a source,
          and only the new part gets fitted.
        </p>
      </div>
      <Figure source={figures.g3} className="col-span-7 col-start-6 mt-10 max-lg:hidden" />
      <PullQuote className="col-span-12 mt-18">ETL moves data between schemas. Sema moves it between meanings.</PullQuote>
      <div className="col-span-12 mt-18">
        <h3 className="type-h3 mb-7">Two ways in.</h3>
        <div className="compartments grid-cols-2 max-lg:grid-cols-1">
          <Way title="You have a model." tag="Showcase below ↓" figure={figures.g4}>
            An industry standard or your own. Sema fits your sources to it and shows what fit, what didn't, and why.
          </Way>
          <Way title="You don't have one." tag="Exploration" figure={figures.g12}>
            Sema reads the sources and proposes the model: the things, the roles, the links between them. We're
            exploring this with New York City property records, where a deed, a housing registration, and a company
            filing each hold one link from a building to the people who run it. Sema connects the links the record
            supports and shows where the record stops.
          </Way>
        </div>
      </div>
    </Section>
  );
}

function Way({
  title,
  tag,
  figure,
  children,
}: {
  title: string;
  tag: string;
  figure: (typeof figures)[keyof typeof figures];
  children: React.ReactNode;
}) {
  return (
    <div className="grid content-start gap-[18px] px-7 pt-8 pb-9">
      <div className="flex items-baseline justify-between gap-4">
        <h4 className="type-h3">{title}</h4>
        <Tag>{tag}</Tag>
      </div>
      <p>{children}</p>
      <Figure source={figure} className="mt-2 max-lg:hidden" />
    </div>
  );
}
