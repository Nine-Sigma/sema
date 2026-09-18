# Sema landing page copy · as shipped

Date 2026-09-18. Extracted verbatim from the deployed build at https://sema-10w.pages.dev/
(`web/apps/landing/dist`, commit c9a0d97 on `dean/feat/website`, assets `index-BDsW3yOG.css` /
`index-DdKEGrpR.js`). Slot names follow `copy-v4-articulate.md` so the sign-off picker keys still apply.

Source of each string:

- Sections: `web/apps/landing/src/sections/*.tsx` and `landing.tsx` (nav, footer).
- Forms, theme toggle, skip link, wordmark: `web/packages/design/src/components/`.
- Figure titles, descriptions, and in-figure labels: `web/packages/design/src/figures/index.ts`
  (generated from `figures/*.svg`).
- Page metadata and 404: `web/apps/landing/index.html`, `public/404.html`.

Bracketed values (`[N]`, `[date]`, `[n]`, `[six]`, `[stage]`, `[investor address]`, `[contact address]`)
are still placeholders in production.

## Deviations from copy v4

| Slot | v4 | Shipped |
|---|---|---|
| S1 · subhead | "…into one model…" | "…into one data model…" |
| S0 · anchors | How it runs | What stays where |
| S3 · way 2 · title | You don't. | You don't have one. |
| S5 · card · accepted · why | Values already in the required vocabulary. | The values already use the model's codes. |
| S1 · eyebrow | SEMA | (not rendered) |
| S3 · figure caption (G12) | separate caption | inside the figure: "Example entities. Not real companies." |

---

## Page metadata

| Slot | Copy |
|---|---|
| title | Sema — Unify your data by meaning |
| description | Sema fits your sources into one data model, keeps them there as new sources arrive, and gives AI agents the meaning behind every table. |
| og:title | Sema — Unify your data by meaning |
| og:description | (same as description) |
| og:image alt | Sema. Unify your data by meaning, not by spreadsheet. |
| skip link | Skip to content |

## S0 · Navigation

| Slot | Copy |
|---|---|
| wordmark | Sema (`aria-label` "Sema, home"; click replays the hero moment) |
| anchors | The idea · Proof · What stays where · Why now · Pilot |
| button | Join the waitlist |
| theme toggle | icon only; `aria-label` "Switch to light theme" / "Switch to dark theme" |

## S1 · Hero

| Slot | Copy |
|---|---|
| h1 | Unify your data by meaning, not by spreadsheet. |
| subhead | Sema fits your sources into one data model, keeps them there as new sources arrive, and gives AI agents the meaning behind every table. |
| form label | Work email |
| placeholder | you@company.com |
| button | Join the waitlist |
| under form | We email when pilot slots open. Nothing else. Privacy |
| secondary | Investor? Deck on request → |

## S2 · You've run this project

| Slot | Copy |
|---|---|
| h2 | You've run this project. |
| beat 1 | 01 · Month one · Six systems. Six names for the same customer. Everyone agrees it's fixable. |
| beat 2 | 04 · Month four · A 1,400-row mapping sheet. The best document in the company, and it lives in one person's head. |
| beat 3 | 09 · Month nine · A new source. Half the sheet has to be redone, and nobody wrote down why the first half was right. |
| beat 4 | 10 · Month ten · The agent pilot joins on the wrong key and answers with total confidence. |
| turn | The work was right. Where it lived was wrong. Meaning sat in a spreadsheet and in people's heads. It needs to live somewhere a machine can read, keep, and build on. |

Figure G7 (desktop) / G7m (mobile), title "The project you have run": the mapping sheet · breaks here · agent · wrong key · Month one · six names for one customer · Month four · a 1,400-row mapping sheet · Month nine · new source, half the sheet redone · Month ten · agent joins on the wrong key.

## S3 · The idea

| Slot | Copy |
|---|---|
| h2 | Integrate by meaning. |
| para 1 | Sema reads your sources and works out what they mean: which columns are the same thing, which codes mean what, which tables join. Then it fits them to one model and checks its own work. When a fit can't hold, Sema blocks it and says why. |
| para 2 | Every decision keeps its source, how sure Sema was, and who changed it. Corrections stay. Add a source, and only the new part gets fitted. |
| pull quote | ETL moves data between schemas. Sema moves it between meanings. |
| h3 | Two ways in. |
| way 1 · title | You have a model. |
| way 1 · tag | Showcase below ↓ |
| way 1 · body | An industry standard or your own. Sema fits your sources to it and shows what fit, what didn't, and why. |
| way 2 · title | You don't have one. |
| way 2 · tag | Exploration |
| way 2 · body | Sema reads the sources and proposes the model: the things, the roles, the links between them. We're exploring this with New York City property records, where a deed, a housing registration, and a company filing each hold one link from a building to the people who run it. Sema connects the links the record supports and shows where the record stops. |

Figures (desktop only):

- G3 "The idea": SOURCE MEANINGS · ONE GRAPH · TARGET MEANINGS · Sales / cust_no · Billing / customer_id · Support / client_ref · Customer · Customer / id · name · region · Order / customer · placed_on · The graph is what persists. · Sources change. Targets change. The resolved meaning stays, with its source and confidence.
- G4 "Many to one": BIRTH_YR · yob · birth_year · year_of_birth · THREE SOURCES · ONE TARGET PROPERTY · Year to year. The figure never implies precision the sources lack.
- G12 "Ownership chain": deed · housing registration · corporate filing · Parcel / 123 Example St · Example Property LLC / owner of record · Example Holdings LLC / sole member · A. Manager / signs the filing · beneficial owner / not in the public record · Example entities. Not real companies.

## S4 · What changes for you

| Slot | Copy |
|---|---|
| h2 | What changes for you. |
| item 1 | A new source is an increment, not a restart. — Sema fits it to what's already there. |
| item 2 | Agents answer from meaning. — Ask a question. The agent gets the part of the model that matters and never guesses what a column means. |
| item 3 | Corrections stick. — Fix something once. The fix is kept, attributed, and survives every rebuild. |
| item 4 | You can see why. — Every decision shows where it came from and how sure Sema was. |

Figure G5 "What the agent gets" (desktop only): A QUESTION · Which customers churned after the price change? · customer · churn · price change · THE MODEL · lit: the part that matters · CONTEXT · customer · 3 tables · joins · 2, verified · churn · closed < 90 days · price · changed 2026-03 · agent · The agent never guesses what a column means.

## S5 · Proof

| Slot | Copy |
|---|---|
| eyebrow | Showcase · public cancer data → one standard model |
| h2 | Three cancer studies, one model. |
| body | We pointed Sema at three public cancer studies and fitted them to OMOP, the standard model for health research. [N] patients, [N] diagnoses, and [N] patients recognized across studies and merged. Nothing cancer-specific in the code. Run dated [date]. |
| card · accepted · head | Accepted · Decision [n] of [N] |
| card · accepted · trace | [study].[table].[column] → [target table].[column] |
| card · accepted · why | Same concept. The values already use the model's codes. One per patient. |
| card · blocked · head | Blocked · sent to a person · Decision [n] of [N] |
| card · blocked · trace | [study].[table].[column] → [target table].[column] |
| card · blocked · why | Many rows per patient where the model allows one. That's a decision, not a rename. |
| caption | Two of [N] decisions from the run. The blocked one is the point: Sema doesn't guess when it shouldn't. |
| cta | Join the waitlist |

Figure G6 / G6m "Three studies, one model": PUBLIC CANCER STUDIES · Study one · Study two · Study three · OMOP / the standard model for health research · People / [N] patients · Diagnoses / [N] diagnoses · Merged across studies / [N] patients recognized twice · run dated [date] · nothing cancer-specific in the code.

## S6 · What we need, and what stays where

| Slot | Copy |
|---|---|
| h2 | What we need, and what stays where. |
| sources | Your sources. — Databricks or PostgreSQL today. Sema reads; it never writes to your source tables. Bring an industry standard model (OMOP is included) or your own. Marks: DATABRICKS · POSTGRESQL |
| what leaves | What leaves your warehouse. — Column names, descriptions, and small value samples go to the AI provider you choose. Row data stays where it is. |
| what comes back | What comes back. — Unified tables in your warehouse, and a model any agent or query tool can ask for context. Every decision carries its source and confidence. |
| when unsure | When Sema isn't sure. — It asks a person. The answer is kept. |

## S7 · Why now

| Slot | Copy |
|---|---|
| h2 | Why now, and why this. |
| thesis 1 | Agents need legible data. — Every company is putting agents on its data. Agents fail on data with no shared meaning and no trusted joins. The bottleneck moved from the model to the data. |
| thesis 2 | A services business waiting to become software. — Consultants do the fit once and it decays. Software that keeps it fitted changes who can afford a unified model. |
| thesis 3 | Understanding got cheap. Trust didn't. — The durable asset is the checked, corrected record of what your data means. That compounds. A prompt doesn't. |
| wedge | The wedge. — Domains with a standard model, where every conversion into it is still a project. Healthcare first. The same shape exists in finance, retail, and manufacturing. |
| expansion | The expansion. — Domains with no standard model, where Sema proposes one. Property records are the first exploration. Then the model your organization defines. |

Figure G11 "Wedge to expansion": Supplied standard models / OMOP first · Emergent models / from public and internal sources · Any model your organization defines · THE WEDGE · THE EXPANSION.

## S8 · Roadmap

| Slot | Copy |
|---|---|
| eyebrow | Roadmap · not yet shipped |
| h2 | What's next. |
| item 1 | More warehouses. — Chosen with pilot partners. |
| item 2 | Documents and notes. — Brought into the same model as your tables. |
| item 3 | Review in the product. — Answer Sema's questions in a UI, not an export. |

## S10 · Investors (rendered before S9)

| Slot | Copy |
|---|---|
| h3 | For investors. |
| body | We're raising a [stage] round to take Sema from showcase to paid pilots, and to ship the second way in: models proposed from the sources. Deck and the showcase run on request. |
| link | [investor address] → (`mailto:` with no address yet) |

## S9 · The pilot

| Slot | Copy |
|---|---|
| h2 | Early access is a pilot, not a mailing list. |
| you bring | **You bring.** Two or three sources in Databricks or PostgreSQL, and a target model, or willingness to start from a standard one. |
| we deliver | **We deliver, in [six] weeks.** Unified tables in your warehouse, the model, and the decision log: every fit, its confidence, and what we asked a person to decide. |
| you tell us | **You tell us what was wrong.** That's the deal. |
| form label | Work email |
| placeholder | you@company.com |
| button | Join the waitlist |
| under form | A few pilots at a time. We email when a slot opens. Privacy |

## S11 · Footer

| Slot | Copy |
|---|---|
| column 1 | Sema · from Greek σῆμα, "sign" |
| column 2 | Join the waitlist |
| column 3 | Privacy · Contact (`mailto:` with no address yet; Privacy links to itself) |
| column 4 | © 2026 |

## Form states (both forms)

| State | Message | Button |
|---|---|---|
| loading | Adding you… | Adding… |
| success | You're on the list. We'll email when a pilot slot opens. | On the list |
| invalid | That doesn't look like an email address. Check it and try again. | Join the waitlist |
| duplicate | You're already on the list. | On the list |
| server error | We couldn't save that. Try again in a minute, or email [contact address]. | Try again |
| timeout | This is taking longer than it should. Trying again is safe. | Try again |

## 404

| Slot | Copy |
|---|---|
| title | Not found — Sema |
| numeral | 404 |
| h1 | Nothing lives at this address. |
| body | The page moved or never existed. The record stops here. |
| link | Back to Sema |

## Still open on the page

- Placeholders: `[N]` ×4 in S5 body, `[N]` ×3 and `[date]` in G6, `[n] of [N]` ×2 on the record cards, `[six]` weeks, `[stage]`, `[investor address]`, `[contact address]`.
- `Privacy` (`#privacy`) points at itself; `Contact` and the investor link are empty `mailto:`.
- Production waitlist `POST /api/waitlist` has no backend; every real submit shows the server-error state.
