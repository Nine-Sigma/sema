# prompt: layout v1

You propose how one landed relation is read. Code has profiled it and found that no shape
template reads it as landed, or the loader landed it without a header. You may bind it with
one template from the closed set below, over the relation itself or over one view from the
view registry. Code then verifies your proposal against the landed data; it is bound only
when every check passes.

## How to answer

- Set `answer` to `propose`, `abstain`, or `needs_values`.
- `propose`: set `template` to one template listed below. List in `relations` the relations
  that fill the template's relation slots, in the order the template lists them; the
  relation under "Relation" must be one of them. For each column slot of the template, set
  `slot_<slot>` to a column of that relation. For each literal slot, set `literal_<slot>`
  to one of its listed values. A slot marked "when" applies only under that literal.
- Optionally set `view` to one registered view and `view_columns` to the columns it acts
  on; the template then reads the view in place of the relation. Otherwise set `view` to
  `none` and `view_columns` to an empty list.
- `abstain`: no template over any registered view reads this relation. Use `reason` to
  describe the layout you see, so a reviewer can decide what reads it.
- `needs_values`: list in `needs_values` the columns whose values would let you decide.
  Values are shown only after a reviewer approves reading them.
- Set every field you do not use to `none` or to an empty list. Give a `confidence` between
  0 and 1 and a one-sentence `reason` naming the evidence.
- Use only the evidence given here. Names, types, comments, and aggregates are shown; data
  values are shown only under "Values".

## Templates

Each template lists its slots in order: a relation slot is filled from `relations`, a
column slot from `slot_<slot>`, and a literal slot from `literal_<slot>`.

- `concept_list`: one relation with an identifier column and a label column, read as a flat
  list of concepts.
- `concept_and_value_set`: a concept relation and a value relation joined by a key, so each
  concept owns a set of values of its own.
- `concept_with_values_denormalized`: one relation grouped by a column, with codes unique
  within each group.
- `edge_list_hierarchy`: concept identifiers with a parent column, or an edge relation of
  child and parent columns that resolve into them, with no cycle.
- `level_columns`: ordered columns of one relation, each one level below the previous.
- `path_codes`: codes split by a separator into segments, so a code's prefix is its parent.
- `entity_model`: properties described per entity, either as a dictionary beside an
  optional key-value header (`per_file`) or within one relation (`single_relation`).
- `semsql`, `athena_bundle`: relations whose names a known loader fixed.

$templates

## Views

- `fill_down`: each null in the named columns takes the last non-null value above it in
  landed row order. Use it when a column's value is stated once on a first row and left
  empty on the continuation rows below it. It is refused when a named column is empty on
  the first landed row.

Registered views:

$views

## Failed checks from previous answers

$failures

## Relation

$relation

## Other relations in its group

$group

## Values

$samples
