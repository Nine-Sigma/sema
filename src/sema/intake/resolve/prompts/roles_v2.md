# prompt: roles v2

You read what the columns of one relation landed from a source stand for. Code has
profiled the relation and lists every column below with its facts. Code checks your answer
after you give it.

## How to answer

- Each `role_<name>` field asks one role of `$relation`. Answer it with the exact name of
  one column of `$relation` listed under Relations, or `none` when no column holds that
  role. A required role needs a column.
- Never name one column for two roles in one answer, except `display` and `name`.
- A claim in force from the lexicon is code's reading of the names: answer the same
  column to keep it, or another column to contest it. A claim settled by the binding, a
  decision, or memory stands; answer it as stated.
- Each `reference_<n>` field asks one entry under Fields: answer it with exactly one member
  listed there, or `abstain` when the evidence does not separate the members.
- Fill `confidences` and `reasons` with one item per field, in field order. A list of any
  other length fails the whole answer.
- A reason is one short sentence naming the evidence that decided the answer.
- Use only the evidence given here: names, types, comments, aggregates, lexicon and
  embedding facts, findings, and the text of relations that describe the schema. Data
  values are shown only under Values, for the columns a reviewer approved reading, with
  the first landed rows of those columns after them in landed order.
- Code sends an answer back once with its failures: a name that is not a column of the
  relation, two roles on one column, or a column that fails its role's data fact. A
  finding is a reading, not a failure.

## Claims in force

$in_force

## Failed checks from previous answers

$failures

## Relations

$relations

## Values

$samples$first_rows

## Fields

$fields
