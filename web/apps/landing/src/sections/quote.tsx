import * as React from "react";
import { Figure, Tag, useSeen, type FigureSource } from "@sema/design";
import { Section } from "./section";
import { useCopy, useCopyFigures } from "../copy";
import { ease, seg } from "../motion/bus";
import { useStage } from "../motion/use-stage";

/* S3b (M4). Desktop: the wall is pinned for 60vh of travel; the rule draws, the two sentences arrive,
   and the plane passes behind (the plane reads the stage itself). Under 901px the wall arrives on
   entry. Then the two ways in, split by one hairline. */
export function Quote() {
  const c = useCopy().quote;
  const figures = useCopyFigures();
  const wall = useSeen<HTMLQuoteElement>(0.4);
  const s1 = React.useRef<HTMLSpanElement>(null);
  const s2 = React.useRef<HTMLSpanElement>(null);

  const stage = useStage<HTMLDivElement>((p, f) => {
    const w = wall.current;
    if (!w) return;
    if (f.vw < 901) {
      w.style.removeProperty("--rule-p");
      for (const s of [s1.current, s2.current]) {
        s?.style.removeProperty("--o");
        s?.style.removeProperty("--y");
      }
      return;
    }
    w.style.setProperty("--rule-p", ease(seg(p, 0, 0.2)).toFixed(3));
    const set = (el: HTMLElement | null, a: number): void => {
      el?.style.setProperty("--o", a.toFixed(3));
      el?.style.setProperty("--y", `${((1 - a) * 12).toFixed(1)}px`);
    };
    set(s1.current, ease(seg(p, 0.2, 0.5)));
    set(s2.current, ease(seg(p, 0.45, 0.9)));
  });

  return (
    <Section id="quote" heading="h3-ways" weight="heavy" chapter="quote">
      <div ref={stage} className="sema-stage sema-stage-wall col-span-12">
        <div className="sema-pin">
          <blockquote ref={wall} className="sema-wall type-wall m-0">
            <span ref={s1} className="sema-sentence">
              {c.pullQuote}
            </span>
            {c.pullQuoteAccent ? (
              <>
                {" "}
                <span ref={s2} className="sema-sentence text-accent">
                  {c.pullQuoteAccent}
                </span>
              </>
            ) : null}
          </blockquote>
        </div>
        <div className="sema-spacer" aria-hidden="true" />
      </div>
      <div className="col-span-12 mt-40 max-lg:mt-24">
        <h3 id="h3-ways" className="type-h3 mb-10">
          {c.h3}
        </h3>
        <div className="grid grid-cols-[1fr_1px_1fr] gap-x-10 max-lg:grid-cols-1 max-lg:gap-y-12">
          <Way {...c.ways.have} figure={figures.g4} mobile={figures.g4m} />
          <div aria-hidden="true" className="bg-rule max-lg:hidden" />
          <Way {...c.ways.dont} figure={figures.g12} mobile={figures.g12m} />
        </div>
      </div>
    </Section>
  );
}

function Way({ title, tag, body, figure, mobile }: { title: string; tag: string; body: string; figure: FigureSource; mobile: FigureSource }) {
  return (
    <div className="grid content-start gap-[18px]">
      <div className="flex items-baseline justify-between gap-4 max-lg:flex-col max-lg:items-start max-lg:gap-2.5">
        <h4 className="type-claim">{title}</h4>
        <Tag>{tag}</Tag>
      </div>
      <p className="max-w-[46ch]">{body}</p>
      <Figure source={figure} mobile={mobile} className="mt-2" />
    </div>
  );
}
