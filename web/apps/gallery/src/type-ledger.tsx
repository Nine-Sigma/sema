import * as React from "react";
import { Specimen } from "./ledger";

/* Type roles. The caption reads the dark cell's computed style, so it cannot drift from theme.css. */
type TypeRole = { utility: string; sample: string; measure?: string };

const TYPE_ROLES: TypeRole[] = [
  { utility: "type-h1", sample: "Unify your data by meaning." },
  { utility: "type-h2", sample: "You've run this project." },
  { utility: "type-h3", sample: "Two ways to start" },
  { utility: "type-claim", sample: "A claim in a ledger row" },
  { utility: "type-lede", sample: "Lede. Sema fits your sources into one data model and keeps them there.", measure: "max-w-[34ch]" },
  { utility: "type-pull", sample: "The model is the product. The sheet was a symptom.", measure: "max-w-[24ch]" },
  { utility: "type-turn", sample: "Turn. The sheet was never the problem. The sheet was the model, written down in the wrong place.", measure: "max-w-[40ch]" },
  { utility: "type-label", sample: "Label · mono caption voice" },
  { utility: "type-num", sample: "09" },
  { utility: "type-wordmark", sample: "Sema" },
];

function describe(el: HTMLElement): string {
  const cs = getComputedStyle(el);
  const family = cs.fontFamily.split(",")[0]?.replace(/"/g, "") ?? "";
  const size = parseFloat(cs.fontSize);
  const lh = cs.lineHeight === "normal" ? "normal" : (parseFloat(cs.lineHeight) / size).toFixed(2);
  const ls = cs.letterSpacing === "normal" ? "0" : `${(parseFloat(cs.letterSpacing) / size).toFixed(3)}em`;
  const upper = cs.textTransform === "uppercase" ? " · uppercase" : "";
  return `${family} · ${size.toFixed(0)}px / ${lh} · ${ls} · ${cs.fontWeight}${upper}`;
}

function useComputedType(): [React.RefObject<HTMLParagraphElement | null>, string] {
  const ref = React.useRef<HTMLParagraphElement>(null);
  const [text, setText] = React.useState("");
  React.useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = (): void => setText(describe(el));
    read();
    void document.fonts?.ready.then(read);
    const ro = new ResizeObserver(read);
    ro.observe(document.documentElement);
    return () => ro.disconnect();
  }, []);
  return [ref, text];
}

function TypeRow({ role }: { role: TypeRole }) {
  const [ref, computed] = useComputedType();
  return (
    <Specimen
      name={role.utility}
      props={computed || "measuring…"}
      render={(theme) => (
        <p ref={theme === "dark" ? ref : undefined} className={`${role.utility} ${role.measure ?? ""}`}>
          {role.sample}
        </p>
      )}
    />
  );
}

function TypeLedger() {
  return (
    <>
      {TYPE_ROLES.map((r) => (
        <TypeRow key={r.utility} role={r} />
      ))}
    </>
  );
}

export { TypeLedger, TYPE_ROLES };
