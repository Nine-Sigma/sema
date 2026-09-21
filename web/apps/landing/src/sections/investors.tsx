import { Section } from "./section";
import { useCopy } from "../copy";

/* S10. Right half only; the left half is empty on purpose so the plane shows. */
export function Investors() {
  const c = useCopy().investors;
  return (
    <Section id="investors" heading="h3-inv" weight="connective" chapter="investors">
      <div className="col-span-6 col-start-7 grid gap-3.5 max-lg:col-span-12 max-lg:col-start-1 [&_p]:max-w-[52ch]">
        <h3 id="h3-inv" className="type-h3">
          {c.h3}
        </h3>
        <p>{c.body}</p>
        <p className="font-mono text-sm">
          <a href={c.link.href}>{c.link.text}</a>
        </p>
      </div>
    </Section>
  );
}
