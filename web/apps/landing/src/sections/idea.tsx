import { Figure, PullQuote, Tag, type FigureSource } from "@sema/design";
import { Section } from "./section";
import { useCopy, useCopyFigures } from "../copy";

/* S3. Prose left, G3 right, the pull quote, then the two ways in. */
export function Idea() {
  const c = useCopy().idea;
  const figures = useCopyFigures();
  return (
    <Section id="idea" heading="h2-idea">
      <h2 id="h2-idea" className="type-h2 col-span-12">
        {c.h2}
      </h2>
      <div className="col-span-5 mt-10 max-w-[66ch] [&_p+p]:mt-[1em] max-lg:col-span-12">
        {c.paras.map((text) => (
          <p key={text}>{text}</p>
        ))}
      </div>
      <Figure source={figures.g3} className="col-span-7 col-start-6 mt-10 max-lg:hidden" />
      <PullQuote className="col-span-12 mt-18">{c.pullQuote}</PullQuote>
      <div className="col-span-12 mt-18">
        <h3 className="type-h3 mb-7">{c.h3}</h3>
        <div className="compartments grid-cols-2 max-lg:grid-cols-1">
          <Way {...c.ways.have} figure={figures.g4} />
          <Way {...c.ways.dont} figure={figures.g12} />
        </div>
      </div>
    </Section>
  );
}

function Way({
  title,
  tag,
  body,
  figure,
}: {
  title: string;
  tag: string;
  body: string;
  figure: FigureSource;
}) {
  return (
    <div className="grid content-start gap-[18px] px-7 pt-8 pb-9">
      <div className="flex items-baseline justify-between gap-4">
        <h4 className="type-h3">{title}</h4>
        <Tag>{tag}</Tag>
      </div>
      <p>{body}</p>
      <Figure source={figure} className="mt-2 max-lg:hidden" />
    </div>
  );
}
