import type { Copy } from "./index";
import { experimentFigures } from "./variant-figures";

/* Hypothesis A: recurring integration work → durable mappings. See AB_COPY_NOTES.md. */
export const variantA: Copy = {
  meta: {
    title: "Sema — Make the mapping work last",
    description: "Source systems change. Keep the mapping decisions. Sema records source-to-target fits, confidence, and corrections, starting with cBioPortal to OMOP.",
    // The shared OG asset still carries the control headline; describe the actual asset.
    ogImageAlt: "Sema. Unify your data by meaning, not by spreadsheet.",
  },
  nav: {
    wordmark: "Sema",
    wordmarkLabel: "Sema, home",
    wordmarkHref: "#top",
    anchors: [
      { href: "#idea", label: "The idea" },
      { href: "#proof", label: "Decisions" },
      { href: "#runs", label: "What stays where" },
      { href: "#why", label: "Why now" },
      { href: "#pilot", label: "Pilot" },
    ],
    cta: { text: "Join the waitlist", href: "#pilot" },
    skip: "Skip to content",
    theme: { toDark: "Switch to dark theme", toLight: "Switch to light theme" },
  },
  hero: {
    h1: "The next source. The same mapping project?",
    subhead: "Sema makes source-to-target mapping decisions explicit, reviewable, and reusable. Keep the fit and the reasoning behind it, starting with cBioPortal to OMOP.",
    // TODO(launch): publish the waitlist privacy notice and confirm email-use policy.
    help: { text: "Sign up for pilot availability.", link: { text: "Data handling", href: "#runs" } },
    secondary: { text: "For investors →", href: "#investors" },
  },
  story: {
    h2: "You've run this project.",
    beats: [
      { num: "01", label: "Month one", text: "Six systems. Six names for the same customer. You agree on a target model and start mapping." },
      { num: "04", label: "Month four", text: "Codes reconciled. Joins checked. The mapping sheet has the answers; the reasoning lives with its author." },
      { num: "09", label: "Month nine", text: "A new source changes the cardinality. You reopen old mappings because nobody recorded why they held." },
      { num: "10", label: "Month ten", text: "An agent uses the old join. You trace its answer back through SQL, tickets, and someone's memory." },
    ],
    turn: "The work was right. Where it lived was wrong. The next integration needs the decisions behind the mappings, including the ones you rejected.",
  },
  idea: {
    h2: "Keep the reasoning with the mapping.",
    paras: [
      "Sema inspects source schemas, metadata, and samples to infer entities, codes, and joins. Its cBioPortal-to-OMOP fitting workflow resolves codes against a target vocabulary and checks mapping plans for required fields and review status.",
      "Mapping records carry source references, confidence, status, and run provenance. Human corrections are recorded separately from machine assertions, with replay support for matching assertions. Missing fits and unresolved choices remain visible.",
    ],
    pullQuote: "A mapping is more useful when the next engineer can see why it exists.",
    h3: "Two ways in.",
    ways: {
      have: {
        title: "You have a model.",
        tag: "OMOP first",
        body: "Start with a defined destination. The implemented fitting workflow covers cBioPortal to OMOP. Other target models can be loaded; fitting your sources to them needs a scoped evaluation.",
      },
      dont: {
        title: "You need a model.",
        tag: "Exploration",
        body: "Model proposal is an exploration: use source evidence to suggest entities and relationships, then review the proposed model. Property records illustrate the question: a deed, a housing registration, and a filing each establish different relationships. Which belong in the target? The ownership diagram is illustrative, not a completed deployment.",
      },
    },
  },
  changes: {
    h2: "Less reconstruction. More recorded work.",
    rows: [
      { claim: "Build on a source at a time.", text: "Schema-scoped graph builds let you work on one study while retaining other studies' assertions and edges." },
      { claim: "Give downstream tools a starting point.", text: "Context retrieval filters assertions by status and confidence, and includes explicit join paths for consumers such as NL2SQL." },
      { claim: "Keep corrections as records.", text: "Pins and rejections are stored as status events. Replay matches stored corrections to assertions; rebuilds still need a preservation workflow." },
      { claim: "Inspect the reason for a fit.", text: "Source references, resolver policy, vocabulary release, and run provenance make the code-mapping decision traceable." },
    ],
  },
  proof: {
    eyebrow: "Implementation examples · test fixtures",
    h2: "A fit, and a reason to stop.",
    body: "The cBioPortal-to-OMOP workflow has reproducible resolver and planner tests. These two fixtures show a code accepted and a target refused. They demonstrate rules, not patient outcomes or measured mapping accuracy. The diagram shows the fitting flow, not a measured study run.",
    // TODO(proof): replace fixtures only with a reviewed run artifact: source study IDs,
    // target/vocabulary versions, run date, decision IDs, accepted/blocked/review counts,
    // and human-labelled evaluation. Do not substitute synthetic counts for patient totals.
    cards: {
      accepted: {
        label: "Auto-accepted",
        meta: "Resolver fixture",
        trace: "OncoTree LUAD → condition_concept_id: 45768916",
        why: "The fixture's vocabulary crosswalk leaves one valid standard concept in the Condition domain. The resolver records an accepted fit without an LLM call on this path.",
      },
      blocked: {
        label: "Blocked target · NO_MAP",
        meta: "Resolver fixture",
        trace: "Fixture MEAS → Measurement target refused",
        why: "The target field requires the Condition domain. The fixture offers a Measurement concept, so the domain gate removes it. A reviewer must check the source meaning and target choice before supplying a valid fit.",
      },
    },
    after: "The blocked one is the point. A failed fit leaves a reason to investigate. NO_MAP stays in staging without a valid target concept. Planner tests also block an unmapped required field.",
    cta: { text: "Join the waitlist", href: "#pilot" },
  },
  stays: {
    h2: "What goes in. What gets recorded.",
    cells: {
      sources: {
        claim: "Your sources and target.",
        text: "Databricks is available today; PostgreSQL is coming soon. The cBioPortal-to-OMOP workflow is the starting point; bring your target model to assess the fitting work required.",
        marksLabel: "Warehouse connectors and availability",
        marks: ["Databricks", "PostgreSQL · coming soon"],
      },
      leaves: {
        claim: "What reaches the model provider.",
        text: "Prompts can include schema metadata, column values, and sampled rows. Samples may contain sensitive data. Scope the sources and provider configuration before a pilot.",
      },
      back: {
        claim: "What Sema persists.",
        text: "Semantic assertions and joins in Neo4j, code-mapping records in DuckDB, and correction events in local files. The OMOP workflow can also write staging and target tables to Databricks.",
      },
      unsure: {
        claim: "When the fit is unresolved.",
        text: "No valid vocabulary target yields NO_MAP; ambiguous candidates can be marked review_pending. Mapping plans expose missing requirements and pending review for an operator to resolve.",
      },
    },
  },
  why: {
    h2: "Why this work can become software.",
    theses: [
      { claim: "The first draft got cheaper.", text: "LLMs make it practical to propose source meaning at scale. That helps with the mapping backlog. Agents also make an undocumented join easier to reuse far beyond the project that created it." },
      { claim: "The destination still needs a fit.", text: "Semantic layers represent and serve business meaning. Heterogeneous sources still need codes, keys, and relationships reconciled to that meaning. Sema focuses on that source-to-target work." },
      { claim: "A services business waiting to become software.", text: "Mapping, reconciliation, and exception review already consume engineering and consulting time. Sema's thesis is to retain that work as decisions, so each new source can build on what the team has settled." },
    ],
    steps: [
      { claim: "The wedge: an existing standard.", text: "OMOP gives healthcare a defined destination. Codes, required fields, and relationships make a fit inspectable. The first job is to demonstrate that fit on a bounded scope." },
      { claim: "The expansion: other targets.", text: "Evaluate fitting to an organization's own model next. Explore model proposal where no target exists. The decision record is the common foundation; broader fitting remains work to prove." },
    ],
  },
  roadmap: {
    eyebrow: "Roadmap · not yet shipped",
    h2: "More of the integration job.",
    items: [
      { head: "PostgreSQL next.", text: "PostgreSQL support is coming soon. Further connectors depend on pilot requirements." },
      { head: "Documents and notes.", text: "Explore evidence outside tables when fitting sources to a target." },
      { head: "Review in the product.", text: "Bring mapping decisions and exception review into a dedicated UI." },
    ],
  },
  investors: {
    h3: "For investors.",
    body: "The company thesis starts with recurring integration work: mapping sources, reconciling codes, and resolving exceptions. Healthcare and OMOP give it a bounded first test. The asset we aim to compound is the accepted and corrected decision record. Pilots should test how much of that work can be reused for the next source.",
    // TODO(launch): confirm an investor contact or deck URL before offering a deck.
    link: { text: "How pilots test the thesis →", href: "#pilot" },
  },
  pilot: {
    h2: "Bring the integration you keep reopening.",
    deal: [
      // TODO(pilot): confirm capacity, engagement duration, and supported target scope.
      // TODO(postgres): owner expects support in 1–2 days from 2026-09-18; verify the
      // deployed connector before changing "coming soon" to shipped or offering it in pilots.
      { lead: "A starting scope.", text: "Two or three sources in Databricks and a target model. We assess fit against the current OMOP workflow before agreeing on a pilot." },
      { lead: "The proposed output.", text: "An inspectable model, mapping records, and a review of unresolved fits. Agree any table outputs and delivery schedule with the team; this is an evaluation, not a production rollout." },
      { lead: "The question to test.", text: "Which decisions can the next source reuse, and which still need an engineer? Plan for a domain reviewer to inspect both." },
    ],
    help: { text: "Sign up for pilot availability.", link: { text: "Data handling", href: "#runs" } },
  },
  footer: {
    origin: { pre: "from Greek", word: "σῆμα", post: ', "sign"' },
    cta: { text: "Join the waitlist", href: "#pilot" },
    // TODO(launch): add a real privacy-policy URL and confirmed contact address.
    links: [{ text: "Data handling", href: "#runs" }, { text: "Pilot details", href: "#pilot" }],
    copyright: "© 2026",
  },
  form: {
    label: "Work email",
    placeholder: "you@company.com",
    button: "Join the waitlist",
    // TODO(launch): /api/waitlist is not implemented in this checkout. Wire and verify
    // persistence + duplicate handling before collecting experiment conversions.
    messages: {
      loading: "Submitting your email…",
      success: "You're on the list for pilot updates.",
      invalid: "Enter a valid email address and try again.",
      duplicate: "This email is already on the list.",
      error: "We couldn't confirm your signup. Please try again later.",
      timeout: "We haven't received confirmation yet. Please try again later.",
    },
    buttons: { loading: "Submitting…", success: "On the list", duplicate: "On the list", error: "Try again", timeout: "Try again" },
  },
  notFound: {
    title: "Not found — Sema",
    numeral: "404",
    h1: "This page isn't here.",
    body: "Check the address or return to the Sema homepage.",
    link: { text: "Back to Sema", href: "/" },
  },
  figures: experimentFigures,
};
