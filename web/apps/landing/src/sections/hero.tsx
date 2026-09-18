import * as React from "react";
import { Constellation, EmailForm } from "@sema/design";
import { submitWaitlist } from "../waitlist";

const MASK_DESKTOP = "[mask-image:linear-gradient(90deg,transparent_0_22%,#000_52%,#000_100%)]";
const MASK_MOBILE = "[mask-image:linear-gradient(180deg,transparent_0,#000_30%,#000_100%)]";

/* S1. The constellation is the ground; the copy sits in the empty left third.
   Under 901px the art moves below the copy as a masked 280px band. */
export function Hero() {
  return (
    <section className="sema-hero relative overflow-hidden pt-18 pb-24 max-lg:pt-12 max-lg:pb-0" aria-labelledby="h1">
      <Constellation className={`absolute inset-0 z-0 pointer-events-none max-lg:hidden ${MASK_DESKTOP}`} />
      <div className="page-wrap page-grid relative z-1">
        <h1 id="h1" tabIndex={-1} className="sema-arrive type-h1 col-span-10 max-w-[15ch] outline-none max-lg:col-span-12">
          Unify your data by meaning, not by spreadsheet.
        </h1>
        <p
          className="sema-arrive type-lede col-span-6 col-start-1 mt-7 max-w-[34ch] max-lg:col-span-12"
          style={{ "--arrive": "420ms" } as React.CSSProperties}
        >
          Sema fits your sources into one data model, keeps them there as new sources arrive, and gives AI
          agents the meaning behind every table.
        </p>
        <EmailForm
          id="form-hero"
          onSubmit={submitWaitlist}
          className="sema-arrive col-span-5 col-start-1 mt-9 max-lg:col-span-12"
          style={{ "--arrive": "500ms" } as React.CSSProperties}
          help={
            <>
              We email when pilot slots open. Nothing else. <a href="#privacy">Privacy</a>
            </>
          }
        />
        <p
          className="sema-arrive col-span-5 col-start-1 mt-5 text-[15px] max-lg:col-span-12"
          style={{ "--arrive": "560ms" } as React.CSSProperties}
        >
          <a href="#investors">Investor? Deck on request →</a>
        </p>
        <Constellation
          aria-hidden="true"
          className={`col-span-12 mt-10 -mx-gutter h-[280px] overflow-hidden lg:hidden ${MASK_MOBILE}`}
        />
      </div>
    </section>
  );
}
