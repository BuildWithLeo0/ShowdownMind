# Planner maintenance, action protocol, and fixed-scenario benchmark

## Goal

Improve the controlled Agent using the evidence from the 19 accepted v2
evaluation battles:

- avoid Planner calls for routine state maintenance;
- recover only protocol mistakes that can be interpreted without changing the
  selected action or inventing tactical intent;
- create a fixed set of model-facing battle decisions that can be rerun without
  starting Pokémon Showdown.

The change must not make illegal actions executable, expose hidden opponent
state, or claim that historical Agent choices are ground-truth tactical labels.

## Decisions

### Maintain completed priority targets locally

Three options were considered:

1. keep replanning for every listed faint;
2. remove every fainted species locally and never replan;
3. remove defeated opponent priority targets locally, but still replan when a
   protected ally faints.

Option 3 preserves the important distinction between completing a subgoal and
losing a win condition. When a listed opponent target faints, the controller
removes it from `priority_targets` and removes `target_fainted` from the trigger
list when no targets remain. No Planner call is made. When a listed protected
ally faints, the name is removed from `preserve`, the maintenance is recorded,
and `preserve_fainted` still calls the Planner because the strategy may now be
invalid.

Local maintenance keeps the plan version and creation turn unchanged: those
identify the latest model-authored strategy. A separate `plan_maintenance`
record explains deterministic edits. Evaluation counts a replan only when a
Planner trigger actually caused a model call.

### Use narrow protocol normalization

Prompt-only changes would leave known provider formatting failures untreated.
General JSON repair would be too permissive and could silently change meaning.
The parser therefore accepts only observed, unambiguous variants:

- a JSON-encoded `opponent_prediction` object string;
- a JSON-encoded `reason_codes` array string;
- an unquoted final `short_rationale` value when the rest of the object is
  valid JSON and the value occupies the final field;
- a prose `reason_codes` string when `short_rationale` is missing, treating the
  prose as the rationale and assigning the auditable `OTHER` code;
- an action ID with exactly one inserted, removed, or substituted character
  when it maps to exactly one legal action of the same kind and has the same
  modifiers. The correction cannot add or remove Tera or change a move into a
  switch.

Every recovery is recorded in `decision_normalizations`. Unknown reason codes,
missing required decision semantics, arbitrary malformed JSON, ambiguous or
larger action-ID changes, and other illegal `action_id` values still receive
one model repair and then the existing fallback.

The action tool changes prediction fields from a nested object to three flat
fields. The parser continues to accept the v2 nested object for old records and
provider compatibility, but new requests use the simpler schema. Rationale and
prediction-detail limits are reduced because the evaluation showed that longer
prose did not improve execution.

## Fixed-scenario benchmark

A scenario stores the exact frozen `controlled-agent-v2` decision context,
legal actions, source battle/turn, historical action, and calculator
recommendations already visible in the original record. It does not store
unrevealed opponent truth.

The build command selects deterministic, diverse records from decision logs and
writes a versioned JSON bank. Protocol-error turns are prioritized, followed by
high-stakes turns such as forced switches, KO opportunities, and plan changes.
Duplicate context hashes are removed.

The evaluate command sends each frozen context directly to
`choose_battle_action`; it does not start Showdown, rerun the Planner, rebuild
memory, or recalculate tactics. It records:

- protocol success, retries, and normalizations;
- model calls, tokens, and latency;
- agreement with the historical action;
- alignment with the calculator's recommendation when one exists.

Historical agreement and calculator alignment are regression signals, not
win-rate or tactical accuracy. A future curated `acceptable_action_ids` field
can provide expert labels without changing the bank schema.

## Acceptance criteria

- Defeating a listed priority target makes no Planner model call.
- Losing a protected ally still replans.
- Local plan edits are visible in decision artifacts and the viewer.
- All observed nested-string, stringified-list, missing-rationale, and unquoted
  final-rationale failures parse in one call and record their normalization.
- A unique one-character action-ID typo is corrected and recorded; ambiguous
  or larger changes remain invalid.
- A scenario bank can be built and validated offline from existing JSONL logs.
- A deterministic model can evaluate the bank without a Showdown server.
- Existing logs, policy modes, reports, and viewers remain readable.
