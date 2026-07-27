# ShowdownMind

ShowdownMind is a research project for building and evaluating an LLM agent that
plays Pokémon Showdown without access to hidden opponent information.

The project currently has these completed foundations:

1. run a pinned Pokémon Showdown server locally;
2. connect with `poke-env`;
3. run built-in baseline players;
4. give a Policy-first agent a whitelist-only battle snapshot and legal actions;
5. force one native `choose_battle_action` tool call with a short public reason;
6. validate the tool arguments, allow one repair, and fall back safely;
7. maintain single-battle memory, evidence-backed opponent beliefs, automatic
   tactical estimates, and an event-triggered battle plan;
8. record every visible snapshot, plan, tool call, action, and fallback in JSONL.

The current version also supports a real OpenAI-compatible model endpoint.
The older `direct` and `tactical-tool` modes remain available as experimental
controls. The research architecture is the `controlled-agent` mode.

## Requirements

- macOS or Linux
- Python 3.12
- Node.js 16 or newer
- Git
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
uv run showdown-mind doctor
uv run showdown-mind showdown setup
```

## Run a baseline smoke test

The command starts the pinned local server when necessary and stops the process
when the battles finish:

```bash
uv run showdown-mind smoke --battles 1
```

Choose other built-in players:

```bash
uv run showdown-mind smoke \
  --player simple-heuristics \
  --opponent max-base-power \
  --battles 5
```

Run the server manually:

```bash
uv run showdown-mind showdown start
```

The local server binds only to `127.0.0.1:8765`; port 8000 is intentionally
avoided because it is commonly used by other local development services.

## Run the Policy-first agent

```bash
uv run showdown-mind agent-smoke --battles 1
```

This command tests the complete Agent path against a built-in opponent:

```text
visible battle state
  -> legal action catalog
  -> forced choose_battle_action tool
  -> argument and whitelist validation
  -> Showdown action
  -> per-turn JSONL log
```

`agent-smoke` deliberately uses a deterministic test double that selects the
highest-base-power legal move. It verifies the model boundary but is not an
LLM and should not be reported as an LLM result. The output prints the decision
log path under `.runtime/decisions/`.

## Connect a live model

Copy the environment template and add your own API key:

```bash
cp .env.example .env
chmod 600 .env
```

`.env` is ignored by Git. ShowdownMind uses project-specific environment names
so it does not overwrite an official OpenAI configuration:

```dotenv
SHOWDOWN_MIND_API_KEY=replace-with-your-api-key
SHOWDOWN_MIND_BASE_URL=https://www.codexapis.com/v1
SHOWDOWN_MIND_MODEL=gpt-5.6-luna
SHOWDOWN_MIND_THINKING=
```

Verify the provider, credentials, model name, and native tool-call contract with
one small request:

```bash
uv run --env-file .env showdown-mind model-check
```

Then run one real LLM battle:

```bash
uv run --env-file .env showdown-mind llm-smoke \
  --opponent max-base-power \
  --battles 1
```

Live battles default to a 300-second batch timeout because each turn makes a
network request. Override it with `--battle-timeout`. The final result reports
model calls and input, output, and total tokens; per-turn response IDs, native
tool-call IDs, short rationales, and usage are stored in the JSONL decision log.

The model must call:

```text
choose_battle_action(
  action_id,
  confidence,
  opponent_prediction,
  request_replan,
  reason_codes,
  short_rationale
)
```

`action_id` is restricted to the current legal-action enum. The rationale is a
required public sentence, not private chain-of-thought. Controlled-Agent limits
it to 160 characters and the prediction detail to 60 characters. The program
validates every argument again before resolving the ID into a real `poke-env`
battle order.

### Run the controlled Agent

This is the main research architecture:

```text
public Showdown events
  -> one-battle memory
  -> evidence-backed opponent hypotheses
  -> automatic tactical calculator
  -> update the battle plan only after important events
  -> choose one legal action
  -> validate, execute, and record
```

Run it locally with the deterministic model boundary:

```bash
uv run showdown-mind agent-smoke \
  --policy-mode controlled-agent \
  --opponent max-base-power \
  --battles 1
```

Or use the configured live model:

```bash
uv run --env-file .env showdown-mind llm-smoke \
  --policy-mode controlled-agent \
  --opponent simple-heuristics \
  --battles 1
```

The tactical calculator runs inside Python every decision; the LLM does not
choose whether to invoke it. Only two native model tools exist:

- `update_battle_plan`, called on the first decision or after a meaningful
  change such as a faint, Tera, important belief change, or requested replan;
- `choose_battle_action`, called on every decision with a legal action ID,
  confidence, short reason, next-opponent-action prediction, and replan flag.

A normal turn therefore makes one model call. A turn that needs a new plan
makes one Planner call followed by one action call. If the Planner fails, the
Agent keeps the previous plan or installs a neutral plan and still chooses a
legal action. Memory lasts for one battle only; there is no vector database,
cross-battle memory, open-ended ReAct loop, or web search.

The action model and Planner receive different bounded views. The action model
gets the active opponent's hypotheses, four recent events, and per-action
tactical estimates. The Planner gets the broader one-battle evidence and a
strategic tactical summary without repeated per-action damage details. If a
provider returns more than three known reason codes, the controller keeps the
first three and records that normalization; unknown codes and malformed JSON
still require the single repair attempt.

### Use the tactical calculator tool

The older `tactical-tool` policy runs a bounded two-stage native tool workflow:

```text
analyze_battle_options()
  -> host calculates player-visible tactical facts
  -> tool result is returned with the matching tool_call_id
  -> choose_battle_action(...)
  -> whitelist validation and Showdown action
```

Its v2.1 result estimates effective power, damage ranges, KO probability, STAB,
type effectiveness, speed order, defensive Tera value, priority, and switch
matchups. It also checks the worst revealed opponent reply, accounts for
visible weather, terrain, screens, burn, and entry hazards, and identifies
actions with the lowest modeled counter-KO risk. Supported variable-power moves
use the current visible state; unknown dynamic moves remain explicitly
unranked. The calculation does not inspect unrevealed opponent data and does
not pretend hidden EVs, items, abilities, or moves are known.
The model receives a compact decision view of these facts, while decision logs
and the replay viewer retain the complete calculator output for auditing.
Try it with the deterministic model boundary or the configured live model:

```bash
uv run showdown-mind agent-smoke \
  --policy-mode tactical-tool \
  --battles 1

uv run --env-file .env showdown-mind llm-smoke \
  --policy-mode tactical-tool \
  --opponent simple-heuristics \
  --battles 1
```

`direct` remains the CLI default so earlier experiments stay reproducible.
Research runs should select `controlled-agent` explicitly.

For the official DeepSeek V4 API, use:

```dotenv
SHOWDOWN_MIND_BASE_URL=https://api.deepseek.com
SHOWDOWN_MIND_MODEL=deepseek-v4-flash
SHOWDOWN_MIND_THINKING=disabled
```

DeepSeek V4 enables thinking mode by default, but its thinking mode does not
accept ShowdownMind's forced named `tool_choice`. The explicit `disabled` value
is recorded in experiment manifests; leaving the variable empty preserves the
provider's default behavior.

### Choose and benchmark the model input

Live and deterministic Agent commands use `pruned-v1` by default. It keeps
readable field names but removes empty values and execution-only metadata.
`full-v1` and the more aggressive `compact-v1` remain available for controlled
experiments:

```bash
uv run --env-file .env showdown-mind model-check --prompt-format pruned
uv run --env-file .env showdown-mind llm-smoke --prompt-format full
```

Compare all three formats offline using any decision log:

```bash
uv run showdown-mind prompt-benchmark \
  .runtime/decisions/gpt-5.6-luna-final.jsonl
```

Each decision record contains the authoritative full snapshot plus the exact
compiled model input, its version, character count, and SHA-256 hash. In the
first 23-decision live log, `pruned-v1` reduced serialized characters by 31.16%.
On repeated live connectivity checks it used 214 provider-reported input
tokens, compared with 275 for `full-v1`. `compact-v1` used fewer characters but
more reported tokens on this provider, so it is not the default.

### Experiment files

Every Agent run uses a new decision-log path and writes related files beside it:

```text
run.jsonl          what the Agent saw and chose on every turn
run.manifest.json  battle, model, prompt, and software configuration
run.summary.json   result, timing, fallback count, and token usage
run.failure.json   sanitized error details, only when the run fails
```

These files make two Agent versions comparable later. ShowdownMind refuses to
reuse an existing artifact path so separate experiments cannot be mixed
accidentally. API keys, request headers, raw environment variables, and hidden
opponent state are never written to the manifest.

## Evaluate an Agent version

Preview the standard evaluation matrix without loading credentials, starting
Showdown, creating files, or calling the model:

```bash
uv run showdown-mind evaluate \
  --name controlled-agent-v1 \
  --output-dir .runtime/evaluations/controlled-agent-v1 \
  --policy-mode controlled-agent
```

Exact token cost depends on battle length and is not known in advance. Add
`--run` only after reviewing the printed plan:

```bash
uv run --env-file .env showdown-mind evaluate \
  --name controlled-agent-v1 \
  --output-dir .runtime/evaluations/controlled-agent-v1 \
  --policy-mode controlled-agent \
  --run
```

The default matrix is 20 live battles: ten against `max-base-power` and ten
against `simple-heuristics`. Random is still available explicitly as a debug
opponent, but is not informative enough for the default research comparison.
For a cheap pipeline check, add `--battles-per-opponent 1`. A real comparison
should contain at least 20 battles per version.

Each battle is an independent checkpoint. If balance, network, or the process
fails after some battles, replenish the account and repeat the exact command
with `--resume`:

```bash
uv run --env-file .env showdown-mind evaluate \
  --name controlled-agent-v2 \
  --output-dir .runtime/evaluations/controlled-agent-v2 \
  --policy-mode controlled-agent \
  --run \
  --resume
```

Accepted battles are never rerun. Failed attempts remain under `runs/` and a
retry gets the next `a01`, `a02`, and so on suffix. Resume requires the same
evaluation plan, model ID, Git commit, and clean/dirty state, preventing
different Agent versions from being mixed. `report.json` is updated after
every accepted battle and reports remaining battles plus token usage from
excluded attempts.

The evaluation directory contains every underlying run plus `report.json` and
`report.md`. Reports aggregate win/score rates and Wilson intervals by
opponent, retries, fallbacks, decision errors, tool-call and rationale
coverage, confidence, tokens, and model latency. Controlled-Agent reports also
include plan coverage, replan frequency, Planner cost and errors, enrichment
errors, opponent-prediction coverage and accuracy, protocol normalizations,
action-context size, and action versus Planner token cost.

Research validity is stricter than merely finishing battles. Reports require
fallback rate ≤5%, decision-error rate ≤10%, and tool-call/rationale coverage
≥95% before `compare` accepts them. Controlled-Agent reports additionally
require ≥95% plan/prediction coverage, ≤10% final Planner-failure rate, and
≤5% enrichment-error rate. Recovered Planner retries remain visible as a
separate rate but do not invalidate an otherwise successful plan. A severe
checkpoint quality failure stops the remaining live matrix early to protect API
cost while preserving an incomplete diagnostic report.

Compare a completed candidate against a completed baseline:

```bash
uv run showdown-mind compare \
  .runtime/evaluations/tactical-tool-v1/report.json \
  .runtime/evaluations/controlled-agent-v1/report.json \
  --output .runtime/evaluations/controlled-vs-tactical.json
```

The comparison uses a reproducible stratified bootstrap and reports
`improved`, `regressed`, `inconclusive`, or `insufficient_data`. Reliability
and cost remain explicit trade-offs rather than being hidden in one arbitrary
score.

## Review a battle visually

Generate a local review page from any single-battle decision log:

```bash
uv run showdown-mind visualize \
  .runtime/decisions/gpt-5.6-luna-pruned.jsonl
```

The command finds the matching `poke-env` replay by battle ID, creates a sibling
`.viewer.html` file, and opens it in the default browser. The left side uses the
native Pokémon Showdown animated replay. The right side explains the Agent's
visible state, legal actions, selected action, public rationale, retries,
fallbacks, latency, and token usage. Controlled-Agent records add an
**Agent 状态** tab for the current battle plan, opponent prediction and previous
prediction result, evidence-backed beliefs, and newly remembered events.

The Agent timeline follows the native replay automatically, including play,
pause, reset, previous turn, next turn, and go-to-turn. If one turn contains a
second decision after a Pokémon faints, the inspector advances when the
replacement switch appears in the replay. Manually choosing a decision pauses
follow mode; click **跟随回放** to catch up again.

Use `--no-open` in headless environments, `--force` to replace an existing
viewer, or `--battle-id` when one JSONL file contains several battles.

The viewer deliberately excludes raw model responses, environment variables,
credentials, and hidden opponent information. It shows an auditable decision
trace, not hidden chain-of-thought. Future records also show the native tool
name and provider tool-call ID in **执行记录**.

## Learn the Agent loop

If you are using this project to learn Agent development, start with the
[observe → decide → act → review walkthrough](docs/learning/01-agent-loop-and-replay-viewer.md).
It maps each step to the relevant source file and uses the viewer to make the
otherwise invisible model boundary concrete.

Then read [how to evaluate an Agent change](docs/learning/02-evaluating-agent-improvements.md)
before adding the first analysis tool.

## Tests

```bash
uv run pytest
```

Run real local battle integration tests explicitly:

```bash
SHOWDOWN_MIND_RUN_INTEGRATION=1 uv run pytest tests/test_integration_smoke.py
```

The design is documented in the
[plain-language plan](docs/plans/2026-07-24-pokemon-showdown-agent-design.md)
and the
[technical plan](docs/plans/2026-07-24-pokemon-showdown-agent-technical-design.md).
Experiment provenance is described in the
[artifact design](docs/plans/2026-07-24-experiment-artifacts-design.md).
The native replay interface is described in the
[viewer design](docs/plans/2026-07-24-replay-decision-viewer-design.md).
Its protocol-step synchronization is described in the
[replay sync design](docs/plans/2026-07-24-replay-sync-design.md).
The model action boundary is described in the
[native action tool design](docs/plans/2026-07-24-native-action-tool-design.md).
The experiment matrix and comparison rules are described in the
[evaluation system design](docs/plans/2026-07-24-agent-evaluation-system-design.md).
The current research architecture is specified in the
[controlled Agent design](docs/plans/2026-07-27-controlled-agent-architecture-design.md).
