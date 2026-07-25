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

## Calculator v2

The second calculator version keeps the same bounded two-tool workflow and
improves the deterministic analysis rather than adding another Agent tool.

For damaging moves it now returns:

- effective power and an explanation of where that power came from;
- support for common weight-, speed-, HP-, boost-, status-, item-, and
  turn-order-dependent moves;
- the correct attacking and defending stats for exceptions such as Body Press,
  Foul Play, Psyshock, and Tera Blast;
- an approximate damage interval as a fraction of maximum HP;
- an approximate one-hit KO probability that includes move accuracy;
- explicit best expected-damage and best KO candidate IDs;
- Tera Blast's effective Tera type;
- a type-only defensive Tera comparison against the opponent's visible STAB
  types.

The schema is `tactical-analysis-v2`. Damage approximates the public Pokémon
damage formula and its 16 random rolls, but deliberately does not invent hidden
EVs, items, abilities, or moves. It uses exact player-visible stats when
available; for missing stats it assumes 31 IVs, zero EVs, and a neutral nature,
and clearly labels unsupported variable-power moves as unknown.
Weather, screens, critical hits, ability and item modifiers, and exact
multi-hit distributions remain outside this approximation.

This version still does not simulate the next turn. A two-turn searcher would
be a separate experiment because it changes the Agent from using arithmetic
facts to exploring future game states.

## Survival calculator v2.1

Version 2.1 adds a bounded single-reply estimate without creating an Agent loop
or a general game-tree searcher. Every legal action receives a `counterplay`
object calculated from opponent moves that have already been revealed in the
battle protocol.

For move actions, the calculator:

- compares both moves' priority and the current speed order;
- estimates the worst revealed incoming damage and KO probability;
- discounts the reply when the player's faster move can KO first;
- accounts for healing, drain, and recoil before the estimated reply;
- evaluates the player's post-Tera defensive typing;
- recognizes Protect as blocking the modeled damaging reply.

For switch actions, it:

- applies visible Stealth Rock, Spikes, Toxic Spikes, Sticky Web, and
  G-Max Steelsurge effects;
- respects visible Heavy-Duty Boots, Magic Guard, Levitate, Air Balloon,
  Flying typing, and Gravity;
- estimates the worst revealed attack against the incoming Pokémon after entry
  damage;
- includes entry damage and counter-KO probability in the switch matchup score.

The shared damage modifier now includes visible rain, sun, sand, snow, terrain,
burn, Reflect, Light Screen, and Aurora Veil. The top-level result identifies
the actions with the lowest modeled counter-KO probability.

The schema is `tactical-analysis-v2.1`. This remains a conservative one-reply
calculator: it does not guess unrevealed moves, predict switches, model status
move outcomes, or search multiple future turns. If no revealed damaging move
can be scored, the counterplay result is explicitly marked unavailable.

## Compact model transport

The first v2.1 smoke battle showed that repeating every arithmetic intermediate
for every legal action made the tool result much larger than v1. The calculator
therefore produces two views from the same deterministic analysis:

- the full `tactical-analysis-v2.1` object remains in the decision record,
  tool-execution audit, and replay viewer;
- the second model call receives `model-compact-v1`, containing only action
  ranking, damage and KO estimates, survival risk, entry effects, Tera value,
  and concise uncertainty markers.

The compact view removes repeated estimated stats, raw counterplay
intermediates, full limitation prose, and zero-value self-HP details. No
calculation or action is removed from the audit record. Each tool execution
records both serialized character counts so transport growth can be measured
without relying only on provider token accounting.
