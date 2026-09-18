import { EmailForm } from "@sema/design";
import { submitWaitlist } from "../waitlist";
import { Section } from "./section";

/* S9. The last beat: the deal, then the full-width form bar. */
export function Pilot() {
  return (
    <Section id="pilot" heading="h2-pilot">
      <h2 id="h2-pilot" className="type-h2 col-span-10 max-lg:col-span-12">
        Early access is a pilot, not a mailing list.
      </h2>
      <div className="col-span-8 mt-10 grid gap-[22px] max-lg:col-span-12 [&_p]:max-w-[58ch]">
        <p>
          <strong>You bring.</strong> Two or three sources in Databricks or PostgreSQL, and a target model, or
          willingness to start from a standard one.
        </p>
        <p>
          <strong>We deliver, in [six] weeks.</strong> Unified tables in your warehouse, the model, and the decision
          log: every fit, its confidence, and what we asked a person to decide.
        </p>
        <p>
          <strong>You tell us what was wrong.</strong> That's the deal.
        </p>
      </div>
      <EmailForm
        id="form-pilot"
        size="lg"
        onSubmit={submitWaitlist}
        className="col-span-12 mt-14"
        help={
          <>
            A few pilots at a time. We email when a slot opens. <a href="#privacy">Privacy</a>
          </>
        }
      />
    </Section>
  );
}
