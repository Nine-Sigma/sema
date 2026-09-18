import type { Copy } from "./index";
import { experimentFigures } from "./variant-figures";

/* Hypothesis B: cheap semantic inference → trustworthy durable decisions. */
export const variantB: Copy = {
  meta: {
    title: "Sema — Keep the evidence behind the fit",
    description: "Plausible mappings need evidence. Sema records source-to-target decisions, confidence, and human corrections, starting with cBioPortal to OMOP.",
    // Describe the existing shared asset until variant social cards are supplied.
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
    h1: "Understanding got cheap. Trust didn't.",
    subhead: "An LLM can propose a mapping. Sema records the evidence, confidence, and review status behind the source-to-target fit, starting with cBioPortal to OMOP.",
    // TODO(launch): publish the waitlist privacy notice and confirm email-use policy.
    help: { text: "Sign up for pilot availability.", link: { text: "Data handling", href: "#runs" } },
    secondary: { text: "For investors →", href: "#investors" },
  },
  story: {
    h2: "You've reviewed this answer.",
    beats: [
      { num: "01", label: "Month one", text: "Six names for one customer. An LLM proposes the matches. The first mapping sheet arrives quickly." },
      { num: "04", label: "Month four", text: "The sheet looks complete. Which joins were checked? Which codes were inferred? The answers are elsewhere." },
      { num: "09", label: "Month nine", text: "A new source changes the grain. Someone's correction survives in a comment, but the next model call never sees it." },
      { num: "10", label: "Month ten", text: "An agent uses the old join. A plausible interpretation has become an answer nobody can adequately justify." },
    ],
    turn: "Plausible is a starting point. A decision needs evidence, a status, and a record of what a person changed. Otherwise uncertainty travels downstream as fact.",
  },
  idea: {
    h2: "Give interpretations a decision record.",
    paras: [
      "Sema separates inferred meaning from the decision to use it. Source assertions carry confidence and provenance. In the cBioPortal-to-OMOP workflow, vocabulary rules filter candidate targets; mapping plans check required fields and flag pending review.",
      "Accepted, rejected, and unresolved are different states. Human pins and rejections become recorded events, with replay support for matching assertions. A correction can become system state instead of another instruction lost in a prompt.",
    ],
    pullQuote: "Confidence describes an inference. The decision record tells you what happened to it.",
    h3: "Two ways in.",
    ways: {
      have: {
        title: "You have a model.",
        tag: "OMOP first",
        body: "Use its vocabulary and requirements to judge candidate fits. The implemented workflow covers cBioPortal to OMOP. Other models can be loaded; their fitting rules need a scoped evaluation.",
      },
      dont: {
        title: "You need a model.",
        tag: "Exploration",
        body: "Model proposal is an exploration, with the proposed model itself subject to review. Property records illustrate the limit: a deed names an owner of record; a filing names a manager. Neither alone proves beneficial ownership. The ownership diagram is illustrative, not evidence of a deployed model-proposal capability.",
      },
    },
  },
  changes: {
    h2: "Make uncertainty inspectable.",
    rows: [
      { claim: "Keep the evidence scoped.", text: "Schema-scoped graph builds retain other studies' assertions and edges, giving each source an identifiable contribution." },
      { claim: "Constrain the context you serve.", text: "Retrieval excludes rejected and superseded assertions. Automatic assertions must meet confidence thresholds; included context still needs appropriate validation." },
      { claim: "Record the human judgment.", text: "Pins and rejections have status events with actor and time. Replay can apply stored corrections to matching assertions; preservation needs an explicit rebuild workflow." },
      { claim: "Trace a mapping to its rules.", text: "Code-mapping records include source references, resolver policy, vocabulary release, and run provenance. Inspect the basis for a decision." },
    ],
  },
  proof: {
    eyebrow: "Implementation examples · test fixtures",
    h2: "The refused target earns its place.",
    body: "A plausible target can still fail a rule. These reproducible fixtures from the cBioPortal-to-OMOP resolver and planner tests show an accepted code and a refused target. They demonstrate decision behavior, not measured accuracy. The diagram is illustrative, not a measured study run.",
    // TODO(proof): confirm a publishable run artifact, study IDs, target/vocabulary
    // versions, run date, decision IDs and counts, and human-labelled evaluation.
    cards: {
      accepted: {
        label: "Auto-accepted",
        meta: "Resolver fixture",
        trace: "OncoTree LUAD → condition_concept_id: 45768916",
        why: "The fixture leaves one valid standard concept after vocabulary standardization and the Condition-domain check. Acceptance follows those rules; the single-candidate path makes no LLM call.",
      },
      blocked: {
        label: "Blocked target · NO_MAP",
        meta: "Resolver fixture",
        trace: "Fixture MEAS → Measurement target refused",
        why: "The offered concept belongs to Measurement; the target requires Condition. The domain gate removes it and records NO_MAP. A reviewer must establish the source meaning and a valid target before treating this as a fit.",
      },
    },
    after: "The blocked one is the point. NO_MAP retains the unresolved result and its reason in staging. Planner tests also block an unmapped required field. Recording uncertainty does not approve a fit.",
    cta: { text: "Join the waitlist", href: "#pilot" },
  },
  stays: {
    h2: "The boundaries of the decision record.",
    cells: {
      sources: {
        claim: "The evidence Sema reads.",
        text: "Databricks is available today; PostgreSQL is coming soon. The implemented fitting workflow starts with cBioPortal and OMOP; other source-to-target combinations need evaluation.",
        marksLabel: "Warehouse connectors and availability",
        marks: ["Databricks", "PostgreSQL · coming soon"],
      },
      leaves: {
        claim: "What reaches the model provider.",
        text: "Prompts can include schema metadata, column values, and sampled rows. Samples may contain sensitive data. Scope the sources and provider configuration before a pilot.",
      },
      back: {
        claim: "Where the record lives.",
        text: "Neo4j holds semantic assertions and joins; DuckDB holds code-mapping records; local files hold correction events. The OMOP workflow can also write staging and target tables to Databricks.",
      },
      unsure: {
        claim: "How uncertainty is represented.",
        text: "No valid vocabulary target yields NO_MAP. Ambiguous candidates can be marked review_pending. Mapping plans distinguish missing requirements from choices awaiting an operator's judgment.",
      },
    },
  },
  why: {
    h2: "Why the decision matters now.",
    theses: [
      { claim: "Interpretation became inexpensive.", text: "LLMs make plausible semantic proposals cheap enough to generate at scale. Agentic consumers can carry a bad assumption into more queries and actions. The cost of a proposal has fallen; the need to justify it has not." },
      { claim: "Meaning still needs evidence of fit.", text: "Semantic layers make governed meaning useful to BI and agents. A heterogeneous source still has to earn its mapping to that meaning. Sema focuses on the evidence, rules, and unresolved decisions at that boundary." },
      { claim: "The corrected record is the asset.", text: "Sema's thesis is that accepted fits, rejected interpretations, and human corrections should accumulate as reusable infrastructure. The next inference can be reconsidered against a record of prior decisions, rather than treated as a fresh answer." },
    ],
    steps: [
      { claim: "The wedge: a testable destination.", text: "OMOP supplies healthcare concepts and requirements against which to inspect a fit. A bounded workflow makes accepted decisions and refusals easier to examine than an open-ended claim of understanding." },
      { claim: "The expansion: other decisions.", text: "Evaluate fitting to organization-defined models next. Explore model proposal only with review of the proposed meaning itself. The broader opportunity depends on proving those decision workflows." },
    ],
  },
  roadmap: {
    eyebrow: "Roadmap · not yet shipped",
    h2: "Extend the evidence and review.",
    items: [
      { head: "PostgreSQL next.", text: "PostgreSQL support is coming soon, extending the source evidence available for review." },
      { head: "Documents and notes.", text: "Explore how textual evidence can support or challenge a proposed fit." },
      { head: "Review in the product.", text: "Put evidence, unresolved choices, and human adjudication in a dedicated UI." },
    ],
  },
  investors: {
    h3: "For investors.",
    body: "The company thesis is the decision record that outlasts the model call. Teams already pay for semantic reconciliation and expert review. Healthcare and OMOP make a bounded test of turning that work into software. Pilots should establish whether corrections become reusable evidence and reduce repeated adjudication as sources change.",
    // TODO(launch): confirm an investor contact or deck URL before offering a deck.
    link: { text: "How pilots test the thesis →", href: "#pilot" },
  },
  pilot: {
    h2: "Bring a fit you need to justify.",
    deal: [
      // TODO(pilot): confirm capacity, engagement duration, and supported target scope.
      // TODO(postgres): owner expects support in 1–2 days from 2026-09-18; verify the
      // deployed connector before changing "coming soon" to shipped or offering it in pilots.
      { lead: "A starting scope.", text: "Two or three sources in Databricks and a target model. We assess fit against the current OMOP workflow before agreeing on a pilot." },
      { lead: "The proposed output.", text: "An inspectable model, mapping records, and a review of unresolved fits. Agree any table outputs and delivery schedule with the team; this is an evaluation, not a production rollout." },
      { lead: "The question to test.", text: "Which interpretations have enough evidence, and which require a human decision? Plan for a domain reviewer to challenge both." },
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
    // TODO(launch): wire /api/waitlist persistence and duplicate handling before the test.
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
