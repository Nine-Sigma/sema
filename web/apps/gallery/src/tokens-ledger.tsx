import * as React from "react";
import type { Theme } from "@sema/design";
import { colors, type ColorToken } from "@sema/design/tokens";
import { contrast, tokenName } from "./contrast";

/* Tokens are shown as the pairings the components rely on, with the measured ratio, not as bare fills. */
type Pairing = { fg: ColorToken; bg: ColorToken; sample: string; floor: number };

const PAIRINGS: Pairing[] = [
  { fg: "ink", bg: "ground", sample: "Ink", floor: 4.5 },
  { fg: "muted", bg: "ground", sample: "Muted", floor: 4.5 },
  { fg: "accent", bg: "ground", sample: "Accent", floor: 4.5 },
  { fg: "blue", bg: "ground", sample: "Blue", floor: 4.5 },
  { fg: "onAccent", bg: "accentFill", sample: "On accent", floor: 4.5 },
  { fg: "ruleStrong", bg: "ground", sample: "Rule strong", floor: 3 },
  { fg: "rule", bg: "ground", sample: "Rule", floor: 1 },
];

function cssVar(token: ColorToken): string {
  return `var(--${tokenName(token).replace(" ", "-")})`;
}

function PairingCell({ theme, pairing }: { theme: Theme; pairing: Pairing }) {
  const set = colors[theme];
  const fg = set[pairing.fg];
  const bg = set[pairing.bg];
  const ratio = contrast(fg, bg);
  const rule = pairing.fg === "rule" || pairing.fg === "ruleStrong";
  const passes = ratio >= pairing.floor;
  /* The fill reads the live CSS variable; the ratio reads tokens.ts. A drift between them shows here. */
  const live = { background: cssVar(pairing.bg), color: cssVar(pairing.fg) };
  return (
    <div className="grid min-w-0 gap-2">
      <div className="grid h-20 content-center border border-rule px-3" style={live}>
        {rule ? <div className="h-px w-full" style={{ background: live.color }} /> : <span className="text-base font-medium">{pairing.sample}</span>}
      </div>
      <p className="type-label text-muted">
        {tokenName(pairing.fg)} / {tokenName(pairing.bg)}
      </p>
      <p className="font-mono text-xs text-muted [overflow-wrap:anywhere]">
        {fg} on {bg}
      </p>
      <p className="font-mono text-xs" style={{ color: passes ? undefined : set.accent }}>
        {ratio.toFixed(2)}:1{pairing.floor > 1 ? (passes ? ` · AA ≥ ${pairing.floor}` : ` · below ${pairing.floor}`) : " · no floor"}
      </p>
    </div>
  );
}

function TokenPairings({ theme }: { theme: Theme }) {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(7.5rem,1fr))] gap-x-4 gap-y-6">
      {PAIRINGS.map((p) => (
        <PairingCell key={`${p.fg}-${p.bg}`} theme={theme} pairing={p} />
      ))}
    </div>
  );
}

export { TokenPairings, PAIRINGS };
