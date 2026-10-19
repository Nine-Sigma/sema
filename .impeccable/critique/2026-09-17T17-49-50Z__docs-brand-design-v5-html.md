---
target: design v5 landing page
total_score: 24
max_score: 32
na_heuristics: 7,10
p0_count: 0
p1_count: 3
timestamp: 2026-09-17T17-49-50Z
slug: docs-brand-design-v5-html
---
Method: dual-agent (A: critique-A design review with headless-Chrome screenshots light/dark 1440 + mobile 390 · B: critique-B detector + mechanical checks)

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | 3 | Six form states with `aria-live`/`aria-busy`; no active-anchor state in `.anchors`; theme toggle flashes with no transition |
| 2 | Match system / real world | 3 | `.record .trace` grammar `[study].[table].[column] →` reads as code with no key; "How it runs" anchor lands on "What we need, and what stays where." |
| 3 | User control and freedom | 3 | Theme persists; no modals; after `success` the form stays in success, button re-enabled, no clear |
| 4 | Consistency and standards | 3 | Footer "Join the waitlist" is a plain link, the other three are `.btn`; `.tag` (S3) and `.mark` (S6) share a style but mean status vs platform; `section.investors` labelled by an h3, every other section by an h2 |
| 5 | Error prevention | 3 | `type=email` + `autocomplete` + regex; validation only on submit; no `maxlength`; resubmission after success allowed |
| 6 | Recognition rather than recall | 3 | "Proof" anchor lands on `[N]` placeholders; G11 ring labels decoded only by the `.step` blocks below |
| 7 | Flexibility and efficiency | n/a | Single-action Persuade surface |
| 8 | Aesthetic and minimalist design | 3 | Desktop disciplined; S6/S7/S8 are three consecutive "claim + line" families; mobile is ~4,000px of that one family |
| 9 | Error recovery | 3 | Messages name problem and next step; `error` points at `[contact address]`; `duplicate`/`error`/`timeout` unreachable until a backend exists |
| 10 | Help and documentation | n/a | The page is the explanation; the one help link (`#privacy`) is circular |
| **Total** | | **24/32 (75%)** | **Good** |

## Design Specificity Verdict

**LLM assessment (A):** authored, not interchangeable. Constellation hero with the left third empty, mono month numerals, ledger rows, two bordered trace records, wedge/expansion rings, and the "ETL moves data between schemas. Sema moves it between meanings." rule-quote all come from this product's argument. Caveat: S6 `.cells` → S7 `.theses` → S8 `.roadmap` repeat one layout family, and the mobile page (figures hidden) is closer to category-generic than desktop.

**Deterministic scan (B):** `detect.mjs` exit 0 on both `design-v5.html` and `design-v5.tpl.html`, zero findings. Mechanical checks B added: light `--accent` on ground 3.18:1 and `--on-accent` on `--accent` 3.18:1 (fail AA); `--rule-strong` on ground 2.44 dark / 2.23 light (fails 3:1 for input, toggle, tag, mark, record borders); 2 empty `mailto:` hrefs; `#privacy` ×3 resolves to the footer link itself; `.toggle` 40px and every inline link under 44px; no horizontal overflow at 390px; unused webfont weight IBM Plex Mono 500; built file 81,347 B of which 49,888 B inline SVG across 10 `<svg>`; `.pulse` loop never pauses offscreen.

**B false positive:** "7 titled SVGs lack `role="img"`" is wrong; the built file has `role="img" aria-labelledby="gN-t gN-d"` on all seven figures.

**Visual overlays:** none. No mutable browser automation in this session; headless Chrome CLI only.

## Overall Impression

The desktop page reads as one system and the S2→S3 arc (four months → "The work was right. Where it lived was wrong." → "Integrate by meaning.") is the page. The biggest opportunity is the Proof section: it is the trust beat and currently the weakest-looking section, and in light theme the one word it exists to show ("Blocked") is under AA contrast. The charset gap is a build bug that will ship unless fixed at the template.

## What's Working

1. `.hero` composition: masked constellation right two-thirds, copy in the empty left third, one `.pulse` as the page's only motion. Specific to Signal Cartography, no stock illustration.
2. `.story .beats` + `.turn`: 6.5rem mono numerals, `.label` months, then the 1.6rem turn sentence. Brutalist, not cold, and the strongest narrative device on the page.
3. Form state design: six named states, `aria-live="polite"`, `aria-busy`, `aria-invalid`, disabled button while loading, invalid clears on input. More complete than most shipped waitlists.

## Priority Issues

**[P1] No `<meta charset="utf-8">`; served standalone, every `→ · © σῆμα` renders as mojibake.** A saw it in the headless captures (`.secondary`, `.proof .eyebrow`, `.record .trace`, `.record.blocked .status`, `.roadmap .eyebrow`, `.investors .link`, `.foot .etym`, copyright). The artifact wrapper supplies the charset, which is why the live artifact looks right; any raw host ships the bug on the trace cards. Fix: first line of `design-v5.tpl.html` = `<meta charset="utf-8">`. Command: `/impeccable harden`.

**[P1] Light-theme amber fails text contrast where amber carries meaning.** `#B87A1E` on `#F2F1EC` = 3.18:1 (A and B agree). Used as 12px text on `.record.blocked .status`, on amber figure labels (G7 "breaks here"/"wrong key", G6 counts, G11 "THE WEDGE"), and as the `.btn` label (`#F2F1EC` on `#B87A1E`, 3.18:1) on all four CTAs. Dark theme is 8.82:1. Also `--rule-strong` borders 2.2–2.4:1 in both themes. Fix: light `--accent` ≈ `#8F5E12` for text/strokes, light `--on-accent` = `#0A0D19` (dark ink on amber ≈ 7:1), and `--rule-strong` ≥ 3:1. Command: `/impeccable colorize`, then `/impeccable audit`.

**[P1] Proof carries the most placeholders and the least visual authority.** `#proof`: `[N]` ×5, `[date]` ×2, `[study].[table].[column]` ×2, `[n] of [N]` ×2, inside the only bordered boxes on the page. Brackets stay by Dean's rule; the design has to hold until they are filled and after. A real trace (~70 chars) will wrap in `.trace` at 14px mono. Fix: amber left rule on `.record.blocked` so the point is visible before reading; make `Decision [n] of [N]` subordinate to the status; test `.trace` with a 70-char path at 1440 and 390. Command: `/impeccable layout` on `#proof`.

**[P2] Mobile loses the page's identity.** Below 900px only G1 and G11 remain; S3/S4/S6/S7/S8 collapse to one "bold claim / sentence / hairline" column for ~4,000 of 9,740px. Fix within the no-scaling rule: draw G7 as a vertical timeline and G6 as a stacked three-into-one in Figma, swap in under `@media (max-width: 900px)`. Command: `/impeccable adapt`.

**[P2] Three of four "Join the waitlist" instances are links to the fourth, and the last beat is for the other reader.** Nav `.btn`, hero `.btn` (in-form, fine) and S5 `.cta` → `#pilot` at ~8,300px; `#investors` follows `#pilot`. A convinced product lead at the fold is scrolled past the whole argument, then sees the investor ask. Fix: nav `.btn` targets `#email-hero` while the hero form is in view; move `#investors` above `#pilot` so the pilot form is the terminal beat. Command: `/impeccable layout` + `/impeccable clarify`.

## Persona Red Flags

**Jordan (first-timer):** "How it runs" anchor → heading "What we need, and what stays where."; `.trace` grammar unexplained; `.tag` "Showcase below ↓" does not say Showcase = built; the only way to ask a question is the footer `mailto:` with an empty href.

**Riley (stress tester):** `a@b.c` and a 200-char local part both pass; after `success` the button re-enables and resubmit shows `success` again (`duplicate` unreachable client-side); hero and pilot forms have no cross-awareness; empty `mailto:` ×2 opens a blank composer; `error`/`timeout`/`duplicate` never render in the demo.

**Casey (mobile):** `.toggle` 40px and `.help a` ~10px tall; S5 `.cta` at ~5,700px and `#pilot` at ~8,300px; S3 second `.way` is 11 lines with no figure; no `enterkeyhint="send"` on the email input.

## Minor Observations

- `.hero .secondary` competes with `.help` for the fold's last stop.
- `.turn` (`grid-column: 1/9`) and `.pull` (`padding-right: 28%`) are two ways to say "wide quote".
- `section.story`, `.ledger`, `.roadmap` have no `id`; `#privacy` ×3 is circular; `mailto:` ×2 empty.
- No `scroll-margin-top` on `section[id]`; harmless until the nav becomes sticky.
- Cormorant Garamond loaded for 8 glyphs; IBM Plex Mono 500 requested, unused.
- `::selection` in light theme is the same 3.18:1 pair.
- `.pulse` runs offscreen; 22 desktop + 11 mobile nodes.

## Questions to Consider

1. What if Proof were the fold: real numbers, two real trace records, then the headline?
2. If mobile can only carry two figures, are G1 (atmosphere) and G11 the right two, or should G7 and G6 (the argument) take their place?
3. Does the nav need a button when the hero form is 60px below it, or is that button for the returning visitor and should say so?
