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
7. record every visible snapshot, tool call, action, and fallback in JSONL.

The current version also supports a real OpenAI-compatible model endpoint.
Belief tracking, damage tools, and planning stay separate so their effects can
be measured later.

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
  reason_codes,
  short_rationale
)
```

`action_id` is restricted to the current legal-action enum. The rationale is a
required public sentence capped at 240 characters, not private chain-of-thought.
The program validates every argument again before resolving the ID into a real
`poke-env` battle order.

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
  --name direct-v0 \
  --output-dir .runtime/evaluations/direct-v0
```

The default matrix is 30 live battles: ten against each built-in opponent.
Exact token cost depends on battle length and is not known in advance. Add
`--run` only after reviewing the printed plan:

```bash
uv run --env-file .env showdown-mind evaluate \
  --name direct-v0 \
  --output-dir .runtime/evaluations/direct-v0 \
  --run
```

For a cheap pipeline check, add `--battles-per-opponent 1`. A real comparison
should contain at least 20 battles per version.

The evaluation directory contains every underlying run plus `report.json` and
`report.md`. Reports aggregate win/score rates and Wilson intervals by
opponent, retries, fallbacks, decision errors, tool-call and rationale
coverage, confidence, tokens, and model latency.

Compare a completed candidate against a completed baseline:

```bash
uv run showdown-mind compare \
  .runtime/evaluations/direct-v0/report.json \
  .runtime/evaluations/damage-tool-v1/report.json \
  --output .runtime/evaluations/damage-vs-direct.json
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
fallbacks, latency, and token usage.

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
