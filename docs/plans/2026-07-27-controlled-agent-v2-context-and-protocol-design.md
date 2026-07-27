# Controlled Agent v2: context and protocol optimization

## Why this change exists

The first two-battle live smoke completed 41 decisions with full plan,
tactical-tool, and rationale coverage, but used 347,001 battle tokens. The
average action context was 15,979 serialized characters. Four decisions needed
repairs or fallback: three responses contained malformed JSON and one otherwise
legal action was rejected only because the model supplied four known reason
codes instead of the allowed three.

Two battles are not enough to infer a strategic win-rate problem. This change
therefore leaves plan triggers, tactical calculations, fallback selection, and
battle strategy unchanged. It targets only model context and output protocol
reliability.

## Decision

Controlled Agent v2 uses two purpose-built model views:

- the action view contains the current snapshot, current plan, four recent
  public events, the active opponent's hypotheses, and one compact record for
  every legal action;
- the Planner view contains the broader one-battle memory and hypotheses, but
  only aggregate tactical signals such as best damage, safest actions, speed
  relation, and switch candidates.

The complete memory, belief state, and tactical analysis remain in the decision
log. Only the model-facing projections are smaller.

The action tool also asks for a rationale of at most 160 characters and a
prediction detail of at most 60 characters. The prompt explicitly requires
valid JSON, quoted strings, and no more than three reason codes.

## Narrow normalization boundary

When a response contains four or more unique reason codes and every code is in
the known whitelist, the parser deterministically keeps the first three. The
decision remains valid and the record stores a
`reason_codes_truncated:N->3` normalization.

This exception is deliberately narrow. The controller does not guess missing
quotes, repair arbitrary JSON locally, accept unknown reason codes, change an
action ID, or relax legality checks. Those failures still receive one
model-based repair and then use the separately counted deterministic fallback.

## Measurement

Evaluation reports add:

- average and maximum action-context characters;
- action input, output, and total tokens excluding Planner calls;
- average action tokens per decision;
- average Planner tokens per replan;
- normalized-decision count and rate.

The first validation run should repeat the same two opponents with one battle
each. It can test reliability, context reduction, latency, and cost, but its
win rate is diagnostic only. A strategic improvement claim still requires the
larger standard evaluation matrix.
