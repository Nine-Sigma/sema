import * as React from "react";
import { Button, Footer, Nav, SkipLink, ThemeToggle, Wordmark, useMotionGate } from "@sema/design";
import { Hero } from "./sections/hero";
import { Story } from "./sections/story";
import { Quote } from "./sections/quote";
import { Changes } from "./sections/changes";
import { Proof } from "./sections/proof";
import { Stays } from "./sections/stays";
import { Why } from "./sections/why";
import { Roadmap } from "./sections/roadmap";
import { Investors } from "./sections/investors";
import { Pilot } from "./sections/pilot";
import { useCopy } from "./copy";
import { MotionProvider, useMotion } from "./motion/context";
import { GraphPlane } from "./motion/plane";
import { routeStageAnchors, trackChapters } from "./motion/chapters";

/* Nav button: focus the hero form while it is on screen and still open; otherwise go to the pilot form. */
function focusHeroForm(ev: React.MouseEvent<HTMLAnchorElement>): void {
  const form = document.getElementById("form-hero");
  if (!form || form.hasAttribute("data-done")) return;
  const r = form.getBoundingClientRect();
  if (r.top < 0 || r.bottom > window.innerHeight) return;
  ev.preventDefault();
  form.querySelector("input")?.focus();
}

export function Landing() {
  return (
    <MotionProvider>
      <Page />
    </MotionProvider>
  );
}

function Page() {
  const { replay } = useMotionGate();
  const { bus } = useMotion();
  const { nav, footer } = useCopy();

  React.useEffect(() => {
    const offChapters = trackChapters(bus);
    const offAnchors = routeStageAnchors();
    return () => {
      offChapters();
      offAnchors();
    };
  }, [bus]);

  /* The wordmark replays the load moment: back to the top at once, the scroll's hold on the plane released. */
  const replayFromTop = React.useCallback((ev: React.MouseEvent) => {
    ev.preventDefault();
    window.scrollTo({ top: 0, behavior: "instant" });
    document.documentElement.removeAttribute("data-scrub");
    replay();
    bus.request();
  }, [replay, bus]);

  return (
    <>
      <SkipLink href="#h1">{nav.skip}</SkipLink>
      <GraphPlane />
      <Nav
        className="sema-nav"
        brand={
          <Wordmark href={nav.wordmarkHref} aria-label={nav.wordmarkLabel} onClick={replayFromTop}>
            {nav.wordmark}
          </Wordmark>
        }
        anchors={nav.anchors}
        actions={
          <>
            <ThemeToggle labels={nav.theme} />
            <Button asChild>
              <a id="nav-cta" href={nav.cta.href} onClick={focusHeroForm}>
                {nav.cta.text}
              </a>
            </Button>
          </>
        }
      />
      <main id="top" className="relative z-1">
        <Hero />
        <Story />
        <Quote />
        <Changes />
        <Proof />
        <Stays />
        <Why />
        <Roadmap />
        <Investors />
        <Pilot />
      </main>
      <Footer
        className="relative z-1"
        columns={[
          <>
            <span className="type-wordmark">{nav.wordmark}</span>
            <span className="text-muted">
              {footer.origin.pre} <i className="font-serif not-italic text-base">{footer.origin.word}</i>
              {footer.origin.post}
            </span>
          </>,
          <a href={footer.cta.href}>{footer.cta.text}</a>,
          <>
            {footer.links.map((l) => (
              <a key={l.href} href={l.href} id={l.href === "#privacy" ? "privacy" : undefined}>
                {l.text}
              </a>
            ))}
          </>,
          <span className="text-muted">{footer.copyright}</span>,
        ]}
      />
    </>
  );
}
