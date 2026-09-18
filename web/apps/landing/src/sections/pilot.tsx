import { EmailForm } from "@sema/design";
import { submitWaitlist } from "../waitlist";
import { Section } from "./section";
import { formProps, useCopy } from "../copy";

/* S9. The last beat: the deal, then the full-width form bar. */
export function Pilot() {
  const { pilot: c, form } = useCopy();
  return (
    <Section id="pilot" heading="h2-pilot">
      <h2 id="h2-pilot" className="type-h2 col-span-10 max-lg:col-span-12">
        {c.h2}
      </h2>
      <div className="col-span-8 mt-10 grid gap-[22px] max-lg:col-span-12 [&_p]:max-w-[58ch]">
        {c.deal.map(({ lead, text }) => (
          <p key={lead}>
            <strong>{lead}</strong> {text}
          </p>
        ))}
      </div>
      <EmailForm
        id="form-pilot"
        size="lg"
        onSubmit={submitWaitlist}
        {...formProps(form)}
        className="col-span-12 mt-14"
        help={
          <>
            {c.help.text} <a href={c.help.link.href}>{c.help.link.text}</a>
          </>
        }
      />
    </Section>
  );
}
