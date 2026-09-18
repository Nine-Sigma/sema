import type { Copy } from "./index";

/* The master copy. COPY.md is generated from this file; edit here, never there. */
export const base: Copy = {
  /* Page metadata */
  meta: {
    title: "Sema — Unify your data by meaning",
    description:
      "Sema fits your sources into one data model, keeps them there as new sources arrive, and gives AI agents the meaning behind every table.",
    ogImageAlt: "Sema. Unify your data by meaning, not by spreadsheet.",
  },

  /* S0 · Navigation */
  nav: {
    wordmark: "Sema",
    wordmarkLabel: "Sema, home",
    wordmarkHref: "#top",
    anchors: [
      { href: "#idea", label: "The idea" },
      { href: "#proof", label: "Proof" },
      { href: "#runs", label: "What stays where" },
      { href: "#why", label: "Why now" },
      { href: "#pilot", label: "Pilot" },
    ],
    cta: { text: "Join the waitlist", href: "#pilot" },
    skip: "Skip to content",
    theme: { toDark: "Switch to dark theme", toLight: "Switch to light theme" },
  },

  /* S1 · Hero */
  hero: {
    h1: "Unify your data by meaning, not by spreadsheet.",
    subhead:
      "Sema fits your sources into one data model, keeps them there as new sources arrive, and gives AI agents the meaning behind every table.",
    help: { text: "We email when pilot slots open. Nothing else.", link: { text: "Privacy", href: "#privacy" } },
    secondary: { text: "Investor? Deck on request →", href: "#investors" },
  },

  /* S2 · You've run this project */
  story: {
    h2: "You've run this project.",
    beats: [
      { num: "01", label: "Month one", text: "Six systems. Six names for the same customer. Everyone agrees it's fixable." },
      {
        num: "04",
        label: "Month four",
        text: "A 1,400-row mapping sheet. The best document in the company, and it lives in one person's head.",
      },
      {
        num: "09",
        label: "Month nine",
        text: "A new source. Half the sheet has to be redone, and nobody wrote down why the first half was right.",
      },
      { num: "10", label: "Month ten", text: "The agent pilot joins on the wrong key and answers with total confidence." },
    ],
    turn: "The work was right. Where it lived was wrong. Meaning sat in a spreadsheet and in people's heads. It needs to live somewhere a machine can read, keep, and build on.",
  },

  /* S3 · The idea */
  idea: {
    h2: "Integrate by meaning.",
    paras: [
      "Sema reads your sources and works out what they mean: which columns are the same thing, which codes mean what, which tables join. Then it fits them to one model and checks its own work. When a fit can't hold, Sema blocks it and says why.",
      "Every decision keeps its source, how sure Sema was, and who changed it. Corrections stay. Add a source, and only the new part gets fitted.",
    ],
    pullQuote: "ETL moves data between schemas. Sema moves it between meanings.",
    h3: "Two ways in.",
    ways: {
      have: {
        title: "You have a model.",
        tag: "Showcase below ↓",
        body: "An industry standard or your own. Sema fits your sources to it and shows what fit, what didn't, and why.",
      },
      dont: {
        title: "You don't have one.",
        tag: "Exploration",
        body: "Sema reads the sources and proposes the model: the things, the roles, the links between them. We're exploring this with New York City property records, where a deed, a housing registration, and a company filing each hold one link from a building to the people who run it. Sema connects the links the record supports and shows where the record stops.",
      },
    },
  },

  /* S4 · What changes for you */
  changes: {
    h2: "What changes for you.",
    rows: [
      { claim: "A new source is an increment, not a restart.", text: "Sema fits it to what's already there." },
      {
        claim: "Agents answer from meaning.",
        text: "Ask a question. The agent gets the part of the model that matters and never guesses what a column means.",
      },
      { claim: "Corrections stick.", text: "Fix something once. The fix is kept, attributed, and survives every rebuild." },
      { claim: "You can see why.", text: "Every decision shows where it came from and how sure Sema was." },
    ],
  },

  /* S5 · Proof */
  proof: {
    eyebrow: "Showcase · public cancer data → one standard model",
    h2: "Three cancer studies, one model.",
    body: "We pointed Sema at three public cancer studies and fitted them to OMOP, the standard model for health research. [N] patients, [N] diagnoses, and [N] patients recognized across studies and merged. Nothing cancer-specific in the code. Run dated [date].",
    cards: {
      accepted: {
        label: "Accepted",
        meta: "Decision [n] of [N]",
        trace: "[study].[table].[column] → [target table].[column]",
        why: "Same concept. The values already use the model's codes. One per patient.",
      },
      blocked: {
        label: "Blocked · sent to a person",
        meta: "Decision [n] of [N]",
        trace: "[study].[table].[column] → [target table].[column]",
        why: "Many rows per patient where the model allows one. That's a decision, not a rename.",
      },
    },
    after: "Two of [N] decisions from the run. The blocked one is the point: Sema doesn't guess when it shouldn't.",
    cta: { text: "Join the waitlist", href: "#pilot" },
  },

  /* S6 · What we need, and what stays where */
  stays: {
    h2: "What we need, and what stays where.",
    cells: {
      sources: {
        claim: "Your sources.",
        text: "Databricks or PostgreSQL today. Sema reads; it never writes to your source tables. Bring an industry standard model (OMOP is included) or your own.",
        marksLabel: "Supported sources",
        marks: ["Databricks", "PostgreSQL"],
      },
      leaves: {
        claim: "What leaves your warehouse.",
        text: "Column names, descriptions, and small value samples go to the AI provider you choose. Row data stays where it is.",
      },
      back: {
        claim: "What comes back.",
        text: "Unified tables in your warehouse, and a model any agent or query tool can ask for context. Every decision carries its source and confidence.",
      },
      unsure: { claim: "When Sema isn't sure.", text: "It asks a person. The answer is kept." },
    },
  },

  /* S7 · Why now */
  why: {
    h2: "Why now, and why this.",
    theses: [
      {
        claim: "Agents need legible data.",
        text: "Every company is putting agents on its data. Agents fail on data with no shared meaning and no trusted joins. The bottleneck moved from the model to the data.",
      },
      {
        claim: "A services business waiting to become software.",
        text: "Consultants do the fit once and it decays. Software that keeps it fitted changes who can afford a unified model.",
      },
      {
        claim: "Understanding got cheap. Trust didn't.",
        text: "The durable asset is the checked, corrected record of what your data means. That compounds. A prompt doesn't.",
      },
    ],
    steps: [
      {
        claim: "The wedge.",
        text: "Domains with a standard model, where every conversion into it is still a project. Healthcare first. The same shape exists in finance, retail, and manufacturing.",
      },
      {
        claim: "The expansion.",
        text: "Domains with no standard model, where Sema proposes one. Property records are the first exploration. Then the model your organization defines.",
      },
    ],
  },

  /* S8 · Roadmap */
  roadmap: {
    eyebrow: "Roadmap · not yet shipped",
    h2: "What's next.",
    items: [
      { head: "More warehouses.", text: "Chosen with pilot partners." },
      { head: "Documents and notes.", text: "Brought into the same model as your tables." },
      { head: "Review in the product.", text: "Answer Sema's questions in a UI, not an export." },
    ],
  },

  /* S10 · Investors (rendered before S9) */
  investors: {
    h3: "For investors.",
    body: "We're raising a [stage] round to take Sema from showcase to paid pilots, and to ship the second way in: models proposed from the sources. Deck and the showcase run on request.",
    link: { text: "[investor address] →", href: "mailto:" },
  },

  /* S9 · The pilot */
  pilot: {
    h2: "Early access is a pilot, not a mailing list.",
    deal: [
      {
        lead: "You bring.",
        text: "Two or three sources in Databricks or PostgreSQL, and a target model, or willingness to start from a standard one.",
      },
      {
        lead: "We deliver, in [six] weeks.",
        text: "Unified tables in your warehouse, the model, and the decision log: every fit, its confidence, and what we asked a person to decide.",
      },
      { lead: "You tell us what was wrong.", text: "That's the deal." },
    ],
    help: { text: "A few pilots at a time. We email when a slot opens.", link: { text: "Privacy", href: "#privacy" } },
  },

  /* S11 · Footer */
  footer: {
    origin: { pre: "from Greek", word: "σῆμα", post: ', "sign"' },
    cta: { text: "Join the waitlist", href: "#pilot" },
    links: [
      { text: "Privacy", href: "#privacy" },
      { text: "Contact", href: "mailto:" },
    ],
    copyright: "© 2026",
  },

  /* Form states (both forms) */
  form: {
    label: "Work email",
    placeholder: "you@company.com",
    button: "Join the waitlist",
    messages: {
      loading: "Adding you…",
      success: "You're on the list. We'll email when a pilot slot opens.",
      invalid: "That doesn't look like an email address. Check it and try again.",
      duplicate: "You're already on the list.",
      error: "We couldn't save that. Try again in a minute, or email [contact address].",
      timeout: "This is taking longer than it should. Trying again is safe.",
    },
    buttons: {
      loading: "Adding…",
      success: "On the list",
      duplicate: "On the list",
      error: "Try again",
      timeout: "Try again",
    },
  },

  /* 404 */
  notFound: {
    title: "Not found — Sema",
    numeral: "404",
    h1: "Nothing lives at this address.",
    body: "The page moved or never existed. The record stops here.",
    link: { text: "Back to Sema", href: "/" },
  },
};
