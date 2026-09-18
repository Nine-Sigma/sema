import * as React from "react";
import { Button, Footer, Nav, SkipLink, ThemeToggle, Wordmark, useMotionGate } from "@sema/design";
import { Hero } from "./sections/hero";
import { Story } from "./sections/story";
import { Idea } from "./sections/idea";
import { Changes } from "./sections/changes";
import { Proof } from "./sections/proof";
import { Stays } from "./sections/stays";
import { Why } from "./sections/why";
import { Roadmap } from "./sections/roadmap";
import { Investors } from "./sections/investors";
import { Pilot } from "./sections/pilot";

const ANCHORS = [
  { href: "#idea", label: "The idea" },
  { href: "#proof", label: "Proof" },
  { href: "#runs", label: "What stays where" },
  { href: "#why", label: "Why now" },
  { href: "#pilot", label: "Pilot" },
];

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
  const { replay } = useMotionGate();
  return (
    <>
      <SkipLink href="#h1" />
      <Nav
        brand={<Wordmark href="#top" onClick={replay} />}
        anchors={ANCHORS}
        actions={
          <>
            <ThemeToggle />
            <Button asChild>
              <a id="nav-cta" href="#pilot" onClick={focusHeroForm}>
                Join the waitlist
              </a>
            </Button>
          </>
        }
      />
      <main id="top">
        <Hero />
        <Story />
        <Idea />
        <Changes />
        <Proof />
        <Stays />
        <Why />
        <Roadmap />
        <Investors />
        <Pilot />
      </main>
      <Footer
        columns={[
          <>
            <span className="type-wordmark">Sema</span>
            <span className="text-muted">
              from Greek <i className="font-serif not-italic text-base">σῆμα</i>, "sign"
            </span>
          </>,
          <a href="#pilot">Join the waitlist</a>,
          <>
            <a href="#privacy" id="privacy">
              Privacy
            </a>
            <a href="mailto:">Contact</a>
          </>,
          <span className="text-muted">© 2026</span>,
        ]}
      />
    </>
  );
}
