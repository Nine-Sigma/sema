# prompt: structure v1

You choose how the relations landed from one source are structured. Code has profiled every
relation and proved, for each entry below, the bindings its predicates accept. Code could not
decide between them, so each entry asks you to choose one.

## How to answer

- Answer field `entry_<n>` with exactly one member listed under entry n, or `abstain` when the
  evidence below does not separate the members.
- Fill `confidences` and `reasons` with one item per entry, in entry order from entry 0. A list
  of any other length fails the whole answer.
- A reason is one short sentence naming the evidence that decided the choice, or why the
  evidence could not decide it.
- A member that spans several relations settles every relation it spans. When it spans the
  subject of another entry, choose it for that entry too, or choose it for neither.
- Every relation is covered by exactly one binding. Never choose two different members that
  span one relation, and never choose a member that spans a relation a binding in force
  already covers.
- Use only the evidence given here. Names, types, comments, and aggregates are shown; data
  values are not.

## Member notation

A member reads `template(slot=value, ...)`; each slot names a relation or a column.

- `concept_list`: one relation with an identifier column and a label column, read as a flat
  list of concepts.
- `concept_and_value_set`: a concept relation and a value relation joined by a key, so each
  concept owns a set of values of its own.
- `concept_with_values_denormalized`: one relation grouped by a column, with codes unique
  within each group.
- `edge_list_hierarchy`: concept identifiers with a parent column or an edge relation that
  resolves into them, with no cycle.
- `level_columns`: ordered columns of one relation, each one level below the previous.
- `path_codes`: codes split by a separator into segments, so a code's prefix is its parent.
- `entity_model`: properties described per entity, either as a dictionary beside a key-value
  header (`per_file`) or within one relation (`single_relation`).

## Bindings in force

$in_force

## Failed checks from previous answers

$failures

## Relations

$relations

## Entries

$entries
