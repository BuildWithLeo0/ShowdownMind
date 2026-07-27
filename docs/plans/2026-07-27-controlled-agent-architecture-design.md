# ShowdownMind controlled Agent architecture

## Goal

Build a research-first, bounded Pokémon Showdown Agent that keeps state and
goals across turns while remaining reproducible, auditable, and safe. The first
delivery ends at an event-triggered battle plan. Multi-turn game-tree search,
cross-battle memory, vector databases, ReAct loops, and general Agent
frameworks remain out of scope.

The existing `direct` and `tactical-tool` policy modes remain unchanged for
reproducing earlier experiments. The new mode is `controlled-agent`.

## Architecture

```text
Pokémon Showdown public protocol
        ↓
PokeEnvEventAdapter → BattleEvent stream
        ↓
BattleMemory → BeliefState
        ↓
TacticalAdvisor (runs automatically every decision)
        ↓
PlanManager ── significant event ──→ update_battle_plan tool
        ↓
DecisionContext
        ↓
choose_battle_action tool
        ↓
legality validation → Showdown order
        ↓
decision trace + synchronized replay viewer
```

The environment loop is controlled by Python. The LLM does not decide when a
turn starts, whether basic arithmetic should run, or when execution stops.
Agent behavior comes from persistent single-battle state, a plan that survives
multiple turns, opponent prediction, and feedback from prediction outcomes.

## Turn lifecycle

For every unique `(battle_id, request_id, turn)`:

1. Build the player-visible snapshot and legal action catalog.
2. Consume public protocol messages not seen on the previous request.
3. Convert supported messages into immutable `BattleEvent` values.
4. Reduce events into `BattleMemory`.
5. Resolve the previous opponent prediction against newly observed events.
6. Rebuild `BeliefState` from public priors, facts, and evidence.
7. Run `TacticalAdvisor` over every legal action without an LLM call.
8. Detect whether the current plan is missing or invalidated.
9. If required, call `update_battle_plan`; a planner failure keeps the previous
   plan or installs a neutral default and never blocks the turn.
10. Compile one bounded `DecisionContext`.
11. Call `choose_battle_action` once, validate it, and resolve it through the
    live action catalog.
12. Record new events, belief changes, the active plan, tactical analysis,
    prediction, model usage, errors, and the chosen action.

Duplicate Showdown requests reuse the cached order and do not consume events,
update state, call a model, or append a second decision.

## Data contracts

### BattleEvent

An immutable, player-visible protocol fact:

```json
{
  "event_id": "battle-gen9randombattle-1:42",
  "battle_id": "battle-gen9randombattle-1",
  "sequence": 42,
  "turn": 7,
  "kind": "move_used",
  "actor": "opponent:gholdengo",
  "target": "own:dragapult",
  "payload": {"move_id": "makeitrain"}
}
```

Supported v1 kinds are `turn_started`, `move_used`, `damage`, `heal`,
`switch`, `faint`, `item_revealed`, `item_consumed`, `ability_revealed`,
`status`, `boost`, `tera`, and `unknown_public_event`. Unknown messages are
preserved for audit but cannot create facts or hypotheses.

The adapter consumes `poke-env 0.15.0` replay data behind a single compatibility
boundary. The repository pins both `poke-env` and the Showdown commit, and
adapter contract tests fail loudly if either protocol shape changes.

### BattleMemory

Program-maintained single-battle state:

- historical union of revealed moves, items, and abilities per Pokémon;
- resource events such as Tera use and fainting;
- ordered speed evidence when priority and field conditions make the
  observation usable;
- damage observations without inventing hidden EVs or items;
- recent opponent action classes and aggregate behavior counts;
- the most recent opponent prediction and its resolution;
- an event cursor and evidence IDs needed to reproduce every update.

The audit view retains all events. The model view contains only facts that add
information beyond the current snapshot, up to six recent events, four speed or
damage evidence items, and the latest prediction resolution.

### BeliefState

Replaceable hypotheses, never facts:

```json
{
  "subject": "opponent:gholdengo",
  "kind": "possible_move",
  "value": "recover",
  "confidence": "possible",
  "evidence_ids": ["prior:gen9randombattle:gholdengo"],
  "contradiction_ids": []
}
```

Confidence is `likely`, `possible`, or `unsupported`; v1 does not present
fabricated numeric probabilities. Candidate roles, moves, abilities, and Tera
types come from the public Gen 9 Random Battle set data at the pinned Showdown
commit. Revealed facts filter incompatible candidates. Items are treated as
unknown until public evidence exists because the static set file does not fully
encode item selection. Each subject/kind contributes at most three hypotheses,
the active opponent is prioritized, and the complete model view is capped at
sixteen hypotheses. Repeated evidence lists keep the public prior plus the two
newest event IDs.

Belief v1 may record speed and damage evidence, but only promotes conclusions
supported by deterministic rules. Ambiguous evidence remains possible rather
than being converted into a hidden stat claim.

### BattlePlan

The single-battle plan returned by `update_battle_plan`:

```json
{
  "schema": "battle-plan-v1",
  "version": 2,
  "created_turn": 8,
  "win_condition": "Preserve Dragapult for a late clean.",
  "preserve": ["dragapult"],
  "priority_targets": ["kingambit"],
  "tera_policy": "Hold Tera unless it preserves the win condition.",
  "risk_posture": "balanced",
  "replan_triggers": ["preserve_fainted", "target_fainted", "opponent_tera"]
}
```

Text fields are public summaries, not chain-of-thought. Species references must
exist in the player-visible teams. `risk_posture` is `conservative`,
`balanced`, or `aggressive`. The controller automatically adds
`preserve_fainted` and `target_fainted` whenever the corresponding plan lists
are non-empty, so a model cannot disable basic plan invalidation.

The controller triggers planning on the first decision, the loss of a preserved
ally, the removal of a priority target, either side's Tera when configured, a
newly revealed item or ability that changes a likely belief when configured, or
a Policy request from the previous turn. An unrelated faint does not invalidate
the plan. Repeated requests in the same turn never replan.

If planning fails, the old plan remains active. If no plan exists, a deterministic
neutral plan preserves no named Pokémon, identifies no target, uses balanced
risk, and recommends holding Tera unless needed for immediate survival.

### DecisionContext and PolicyDecision

`DecisionContext` is the only payload sent to the action Policy:

```text
compact current snapshot
+ compact memory
+ compact beliefs
+ current battle plan
+ compact tactical analysis
```

The full object must be deterministic, hashable, and capped at 24,000
characters. Trimming order is old recent events, low-confidence hypotheses,
then repeated prose; legal actions, current state, plan, and tactical facts are
never removed.

`choose_battle_action` returns:

```json
{
  "action_id": "switch:greattusk",
  "confidence": 0.78,
  "opponent_prediction": {
    "kind": "attack",
    "detail": "suckerpunch",
    "confidence": 0.65
  },
  "request_replan": false,
  "reason_codes": ["PLAN_ALIGNMENT", "SURVIVAL"],
  "short_rationale": "Preserve the cleaner and move to the safer check."
}
```

Prediction kinds are `attack`, `switch`, `setup`, `status`, `recovery`,
`protect`, and `unknown`. `detail` is an optional public move or species hint,
not a hidden assertion. The next observed opponent action resolves the
prediction by kind; the final unresolved prediction is excluded from accuracy
metrics.

## Tools and model calls

Internal Python services run without model orchestration:

- event extraction and memory reduction;
- public Random Battle prior loading and belief updates;
- tactical analysis for all legal actions;
- plan-trigger detection and input compilation.

Only two native tools are exposed to the model:

- `update_battle_plan`, called only on a replan trigger;
- `choose_battle_action`, called once per normal decision.

The current forced, no-argument `analyze_battle_options` LLM call is not used by
`controlled-agent`. The existing `tactical-tool` mode retains it for historical
reproducibility.

The planner and Policy use the same configured model by default. A future
planner-specific model override may be added only as a recorded experimental
variable.

## Failure and safety boundaries

- All live input is derived from the player's websocket protocol and
  player-visible snapshots. Public candidate-set priors may enumerate possible
  configurations but never inspect the generated opponent team.
- Unknown protocol messages are logged and ignored by inference.
- Contradictory beliefs are downgraded or removed; immutable events and facts
  are never rewritten.
- Memory, belief, or tactical enrichment failures degrade to the last valid
  state and are recorded; they do not stop a legal turn.
- Planner transport or validation failures retain the previous/default plan and
  proceed to Policy.
- Policy transport or validation failure receives one repair attempt, then uses
  the existing deterministic legal fallback. Fallback remains deliberately
  weak and separately counted so research results cannot hide model failures.
- Duplicate requests are idempotent, and stale plan/action outputs cannot be
  executed against a new request ID.
- No component writes private chain-of-thought to logs.

## Observability

`DecisionRecord` gains additive fields for:

- `new_events` and compact `memory`;
- `belief_state` and `belief_changes`;
- `battle_plan`, `plan_update`, and `plan_trigger`;
- `opponent_prediction` and `previous_prediction_resolution`;
- planner calls, tokens, latency, and errors separately from Policy metrics;
- the automatically executed tactical analysis.

The replay viewer adds an Agent State panel with Memory, Beliefs, Plan, and
Prediction sections. For each synchronized turn it shows which facts arrived,
why hypotheses changed, whether planning ran, the current win condition and
preserved resources, the prediction from the previous decision versus the
observed action, and the final action rationale. Older logs remain viewable
with empty sections.

Experiment manifests record the controlled capability set, schema versions,
prior source, pinned Showdown commit, model, prompt version, and planning
trigger policy. Evaluation continues to read old records and adds prediction
accuracy, replan frequency, planner cost, and context-size metrics when the new
fields exist.

## Delivery sequence

1. Add immutable domain contracts, event adapter, memory reducer, public-prior
   loader, and rule belief updater with no live model calls.
2. Add deterministic compact views and prediction resolution.
3. Add automatic tactical execution and a one-call action Policy in
   `controlled-agent`.
4. Add `update_battle_plan`, trigger detection, neutral fallback plan, and
   planner accounting.
5. Extend logs, manifests, replay viewer, CLI, and evaluation summaries.
6. Run unit tests, deterministic integration tests, the full offline suite, and
   one local deterministic-model smoke battle. Live paid-model evaluation is
   explicitly deferred until separately requested.

## Acceptance criteria

- A normal controlled turn makes exactly one model call.
- A triggered replan turn makes one planner call plus one action call.
- Tactical analysis runs every unique decision without an LLM tool request.
- Duplicate requests produce no state or model side effects.
- Every belief shown to the model cites public prior or event evidence.
- No unrevealed opponent truth enters snapshots, memory, beliefs, plans, or
  tactical analysis.
- Planner failure cannot prevent a legal action.
- Existing modes, logs, reports, and viewers remain compatible.
- The viewer reconstructs the Agent loop and plan changes from artifacts.
- All tests pass, and implementation artifacts identify the exact code and
  Showdown versions.
