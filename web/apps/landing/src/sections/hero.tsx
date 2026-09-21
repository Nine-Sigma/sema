import * as React from "react";
import { EmailForm, Figure } from "@sema/design";
import { submitWaitlist } from "../waitlist";
import { formProps, useCopy, useCopyFigures } from "../copy";
import { ease, seg } from "../motion/bus";
import { useMotion } from "../motion/context";
import { useStage } from "../motion/use-stage";

/* S1 → S3a as one pinned stage (M1). The idea copy is in flow and gives the pin its height; the hero
   copy floats over it and fades up as the reader scrolls. The plane's ten protagonist nodes travel
   into the G3 socket; at the end of the travel the real plate has crossfaded in and the pin releases
   with the idea already laid out, so nothing jumps when the page resumes scrolling. */
export function Hero() {
  const { hero: c, idea, form } = useCopy();
  const figures = useCopyFigures();
  const { plane } = useMotion();
  const heroCopy = React.useRef<HTMLElement>(null);
  const ideaCopy = React.useRef<HTMLElement>(null);
  const socket = React.useRef<HTMLDivElement>(null);

  const stage = useStage<HTMLDivElement>((p) => {
    const root = document.documentElement;
    if (p > 0 && !root.hasAttribute("data-scrub")) root.setAttribute("data-scrub", "");
    const out = ease(seg(p, 0, 0.25));
    const h = heroCopy.current;
    if (h) {
      h.style.opacity = (1 - out).toFixed(3);
      h.style.transform = `translateY(${(-24 * out).toFixed(1)}px)`;
      h.style.pointerEvents = out > 0.5 ? "none" : "";
      h.inert = out > 0.5;
    }
    const inn = ease(seg(p, 0.66, 0.92));
    const i = ideaCopy.current;
    if (i) {
      i.style.opacity = inn.toFixed(3);
      i.style.transform = `translateY(${((1 - inn) * 12).toFixed(1)}px)`;
      i.inert = inn < 0.5;
    }
    /* After the copy transforms, so the socket is measured where the plate will be. */
    const pl = plane.current;
    if (pl) {
      pl.setScrub(p < 1);
      pl.renderHero(p, socket.current);
    }
  });

  return (
    <>
    <div ref={stage} className="sema-stage sema-stage-hero" data-chapter="hero" data-chapter-next="idea" data-chapter-at="0.7">
      <div className="sema-pin">
        <section ref={heroCopy} className="sema-hero sema-hero-copy page-wrap page-grid pb-24 max-lg:pb-16" aria-labelledby="h1">
          <h1 id="h1" tabIndex={-1} className="sema-arrive type-h1 col-span-10 max-w-[15ch] outline-none max-lg:col-span-12">
            {c.h1}
          </h1>
          <p
            className="sema-arrive type-lede col-span-6 col-start-1 mt-7 max-w-[34ch] max-lg:col-span-12"
            style={{ "--arrive": "420ms" } as React.CSSProperties}
          >
            {c.subhead}
          </p>
          <EmailForm
            id="form-hero"
            onSubmit={submitWaitlist}
            {...formProps(form)}
            className="sema-arrive col-span-5 col-start-1 mt-9 max-lg:col-span-12"
            style={{ "--arrive": "500ms" } as React.CSSProperties}
            help={
              <>
                {c.help.text} <a href={c.help.link.href}>{c.help.link.text}</a>
              </>
            }
          />
          <p
            className="sema-arrive col-span-5 col-start-1 mt-5 text-[15px] max-lg:col-span-12"
            style={{ "--arrive": "560ms" } as React.CSSProperties}
          >
            <a href={c.secondary.href}>{c.secondary.text}</a>
          </p>
        </section>
        <section ref={ideaCopy} id="idea" className="sema-idea-copy page-wrap page-grid pb-20 max-lg:pb-12" aria-labelledby="h2-idea">
          <h2 id="h2-idea" className="type-h2 col-span-12">
            {idea.h2}
          </h2>
          <div className="col-span-5 mt-10 max-w-[46ch] [&_p+p]:mt-[1em] max-lg:hidden">
            {idea.paras.map((text) => (
              <p key={text}>{text}</p>
            ))}
          </div>
          <div ref={socket} className="sema-socket col-span-7 col-start-6 mt-10 max-lg:col-span-12 max-lg:col-start-1 max-lg:mt-8">
            <Figure source={figures.g3} mobile={figures.g3m} />
          </div>
        </section>
      </div>
      <div className="sema-spacer" aria-hidden="true" />
    </div>
    {/* Under 1024px the paragraphs follow the stage, outside the pin, so the pinned block (H2 + plate)
        fits one viewport and the copy scrolls in after the plate has landed (finish review, fix 2). */}
    <div className="page-wrap pb-12 pt-6 [&_p+p]:mt-[1em] lg:hidden">
      {idea.paras.map((text) => (
        <p key={text}>{text}</p>
      ))}
    </div>
    </>
  );
}
