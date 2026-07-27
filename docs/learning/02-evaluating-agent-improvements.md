# Learning 02: Evaluate an Agent Change

An Agent change is not an improvement until it performs better in a controlled
comparison.

## 1. Freeze the baseline

Before adding a damage tool, evaluate the current direct-choice Agent:

```bash
uv run --env-file .env showdown-mind evaluate \
  --name direct-v0 \
  --output-dir .runtime/evaluations/direct-v0
```

This command is a dry run. Read the matrix and total battle count first. Add
`--run` only when the plan and expected API usage are acceptable.

The default matrix contains 20 battles: ten against max-base-power and ten
against simple-heuristics. Random remains available for debugging but is too
weak to be a useful default research opponent. For a cheap end-to-end check,
explicitly request fewer battles:

```bash
uv run --env-file .env showdown-mind evaluate \
  --name direct-v0-smoke \
  --output-dir .runtime/evaluations/direct-v0-smoke \
  --battles-per-opponent 1 \
  --run
```

## 2. Change one capability

Implement one meaningful change, such as an `estimate_damage` tool. Keep the
model, prompt format, opponent matrix, and battle counts unchanged where
possible. Name the new evaluation clearly:

```bash
uv run --env-file .env showdown-mind evaluate \
  --name damage-tool-v1 \
  --output-dir .runtime/evaluations/damage-tool-v1 \
  --run
```

Each evaluation records its Git commit and dirty-worktree state. Commit before
running a real benchmark so the code behind a result can be recovered later.

The runner also separates "the battles finished" from "the data is trustworthy."
Fallback-heavy or error-heavy runs are marked invalid, and severe failures stop
the remaining matrix to avoid wasting API calls. An invalid report is useful
for debugging but cannot be used in `compare`.

## 3. Compare reports

```bash
uv run showdown-mind compare \
  .runtime/evaluations/direct-v0/report.json \
  .runtime/evaluations/damage-tool-v1/report.json \
  --output .runtime/evaluations/damage-vs-direct.json
```

The primary metric is score rate: win = 1, draw = 0.5, loss = 0. The comparison
uses a deterministic stratified bootstrap within each opponent and reports a
95% interval for the candidate-minus-baseline difference.

Interpret the conclusion carefully:

- `improved`: the entire interval is above zero;
- `regressed`: the entire interval is below zero;
- `inconclusive`: the observed difference is still compatible with noise;
- `insufficient_data`: either evaluation contains fewer than 20 battles.

## 4. Read the trade-offs

Win rate is not the only concern. The comparison also shows changes in:

- fallback and retry rates;
- decisions containing errors;
- native tool-call and rationale coverage;
- tokens per battle and per decision;
- average model latency per decision.

For `controlled-agent`, also compare plan coverage, replan frequency, Planner
cost and errors, enrichment failures, and opponent-prediction accuracy. These
diagnostics help distinguish a genuinely better architecture from one that
merely spent more tokens or replanned more often.

A candidate that wins slightly more but doubles cost or becomes unreliable may
not be the better engineering choice.

Random Battle cannot currently replay identical random teams for two diverging
Agents. The online evaluation therefore controls the opponent mix and relies on
repetition, not perfectly paired games. A future fixed-scenario benchmark can
add paired tactical evaluation once the project has trustworthy reference
answers.
