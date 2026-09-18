import { Button, Eyebrow, Figure, Record, figures } from "@sema/design";
import { Section } from "./section";

/* S5. The showcase. The two records are the only bordered boxes on the page. */
export function Proof() {
  return (
    <Section id="proof" heading="h2-proof">
      <Eyebrow className="col-span-12">Showcase · public cancer data → one standard model</Eyebrow>
      <h2 id="h2-proof" className="type-h2 col-span-12 mt-4">
        Three cancer studies, one model.
      </h2>
      <p className="col-span-7 mt-6 max-lg:col-span-12">
        We pointed Sema at three public cancer studies and fitted them to OMOP, the standard model for health
        research. [N] patients, [N] diagnoses, and [N] patients recognized across studies and merged. Nothing
        cancer-specific in the code. Run dated [date].
      </p>
      <Figure source={figures.g6} mobile={figures.g6m} variant="wide" className="col-span-12 mt-12 max-lg:mt-10" />
      <div className="col-span-12 mt-12 grid grid-cols-2 gap-6 max-lg:mt-8 max-lg:grid-cols-1">
        <Record
          status="accepted"
          label="Accepted"
          meta="Decision [n] of [N]"
          trace="[study].[table].[column] → [target table].[column]"
          why="Same concept. The values already use the model's codes. One per patient."
        />
        <Record
          status="blocked"
          label="Blocked · sent to a person"
          meta="Decision [n] of [N]"
          trace="[study].[table].[column] → [target table].[column]"
          why="Many rows per patient where the model allows one. That's a decision, not a rename."
        />
      </div>
      <p className="col-span-8 mt-6 text-[15px] text-muted max-lg:col-span-12">
        Two of [N] decisions from the run. The blocked one is the point: Sema doesn't guess when it shouldn't.
      </p>
      <p className="col-span-12 mt-9">
        <Button asChild>
          <a href="#pilot">Join the waitlist</a>
        </Button>
      </p>
    </Section>
  );
}
