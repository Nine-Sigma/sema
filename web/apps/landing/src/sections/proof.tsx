import * as React from "react";
import { Button, Eyebrow, Figure, Record, useSeen } from "@sema/design";
import { Section } from "./section";
import { useCopy, useCopyFigures } from "../copy";
import { clamp } from "../motion/bus";
import { useMotion } from "../motion/context";

/* S5 (M6 + M7). G6 draws left to right while the plane gathers into three clusters behind it. The two
   records are a sticky stack: Accepted pins, Blocked slides over it and pins 24px lower; Accepted
   scales to 0.96 and fades to 60% under the cover. Traces draw once on view (Record). */
export function Proof() {
  const c = useCopy().proof;
  const figures = useCopyFigures();
  const g6 = useSeen<HTMLDivElement>(0.4);
  const stack = React.useRef<HTMLDivElement>(null);
  const { bus } = useMotion();

  React.useEffect(() => {
    const host = stack.current;
    if (!host) return;
    const a = host.querySelector<HTMLElement>('[data-status="accepted"]');
    const b = host.querySelector<HTMLElement>('[data-status="blocked"]');
    if (!a || !b) return;
    let last = -1;
    return bus.subscribe(({ vh }) => {
      const ra = a.getBoundingClientRect();
      if (ra.bottom < -vh || ra.top > vh * 2) return;
      const rb = b.getBoundingClientRect();
      const p = clamp(1 - (rb.top - ra.top - 24) / (ra.height + 16), 0, 1);
      if (p === last) return;
      last = p;
      a.style.transform = `scale(${(1 - 0.04 * p).toFixed(4)})`;
      a.style.opacity = (1 - 0.4 * p).toFixed(3);
    });
  }, [bus]);

  return (
    <Section id="proof" heading="h2-proof" weight="heavy" chapter="proof">
      <Eyebrow className="col-span-12">{c.eyebrow}</Eyebrow>
      <h2 id="h2-proof" className="type-h2 col-span-12 mt-4">
        {c.h2}
      </h2>
      <p className="col-span-7 mt-6 max-lg:col-span-12">
        {c.body}
      </p>
      <div ref={g6} data-plane-anchor="g6" className="sema-reveal-x col-span-12 mt-14 max-lg:mt-10">
        <Figure source={figures.g6} mobile={figures.g6m} variant="wide" />
      </div>
      <div ref={stack} className="sema-stack col-span-8 mt-18 max-lg:col-span-12 max-lg:mt-12">
        <Record status="accepted" {...c.cards.accepted} />
        <Record status="blocked" {...c.cards.blocked} />
      </div>
      <p className="col-span-8 mt-9 text-[15px] text-muted max-lg:col-span-12">
        {c.after}
      </p>
      <p className="col-span-12 mt-9">
        <Button asChild>
          <a href={c.cta.href}>{c.cta.text}</a>
        </Button>
      </p>
    </Section>
  );
}
