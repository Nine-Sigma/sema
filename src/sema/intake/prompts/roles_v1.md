# prompt: roles v1

You choose what the columns and properties of one relation landed from a source stand for.
Code has profiled the relation and proved, for each entry below, the members its predicates
accept. Code could not decide between them, so each entry asks you to choose one.

## How to answer

- Answer the field named under each entry with exactly one member listed there, or `abstain`
  when the evidence below does not separate the members.
- Fill `confidences` and `reasons` with one item per entry, in entry order from entry 0. A list
  of any other length fails the whole answer.
- A reason is one short sentence naming the evidence that decided the choice, or why the
  evidence could not decide it.
- A role entry names a role and offers the columns eligible for it; a column another role
  already holds and may not share is listed under exclusions and not offered. Never choose
  one column for two roles in one answer. An optional role also offers `none`, which means
  the relation has no column for that role.
- A reference entry names a property and offers the code lists or entities its text may refer
  to; `none` means the property refers to nothing landed.
- Use only the evidence given here. Names, types, comments, aggregates, and the text of
  relations that describe the schema are shown; data values are not.

## Claims in force

$in_force

## Failed checks from previous answers

$failures

## Relations

$relations

## Entries

$entries
