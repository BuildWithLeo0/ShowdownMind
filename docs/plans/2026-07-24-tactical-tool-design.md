# Tactical calculator tool design

## Goal

The `direct-v0` Agent beat Random 10–0 and MaxBasePower 6–4, but lost to
SimpleHeuristics 1–9. The next experiment adds deterministic battle arithmetic
without adding open-ended planning, memory, or web tools.

The candidate is named `tactical-tool-v1`. It must remain safe, auditable, and
directly comparable with `direct-v0`.

## Tool workflow

Each decision uses a bounded two-stage native tool workflow:

1. The model is forced to call `analyze_battle_options`.
2. The host validates the empty tool arguments and executes `TacticalAdvisor`
   against the current player-visible battle.
3. The assistant tool-call message and matching `tool_call_id` result are added
   to the conversation.
4. The model is forced to call `choose_battle_action`.
5. The host validates the selected action against the live action catalog.

There is no autonomous loop. Exactly one tactical call and one final action call
are expected per decision. Existing retry and deterministic fallback boundaries
remain in place.

## Tactical output

The calculator returns facts, not a final action:

- estimated offensive score for each damaging move;
- base power, accuracy, expected hits, STAB, type multiplier, stat ratio, and
  priority;
- relative damage rank among currently legal moves;
- estimated move-order relation;
- switch matchup scores based on public types, base stats, HP, and speed;
- explicit best-damage and best-switch candidate IDs.

The calculations use only state already visible to the player. They do not read
unrevealed opponent moves, items, abilities, or team members.

## Experiment boundary

`direct` remains available for reproducing the baseline. CLI runs select
`--policy-mode tactical-tool` for the candidate, and manifests and evaluation
plans record this mode.

Decision logs record the tactical tool call, result, model-call count, and both
tool names. Evaluation distinguishes the two expected model calls from genuine
retries.

## Deferred tools

No other auxiliary tool is added in v1. In particular, memory, search, planner,
and simulator tools are deferred until the tactical calculator is evaluated.
Adding one capability at a time keeps the result attributable.

