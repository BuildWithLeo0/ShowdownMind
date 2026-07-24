# Agent evaluation system design

## Purpose

The evaluation system answers a narrow question: did one ShowdownMind version
perform better than another under a declared experiment matrix?

It must not treat a single win as evidence. A report combines end-to-end battle
outcomes with reliability, latency, and token usage so an apparent strength
gain cannot hide a large operational regression.

## Commands and cost protection

The system adds two commands:

```text
showdown-mind evaluate [plan options] [--run]
showdown-mind compare BASELINE_REPORT CANDIDATE_REPORT --output PATH
```

`evaluate` is dry-run by default. It prints the opponent matrix, repetitions,
total live battles, output directory, and a warning that exact token cost is
unknown. It does not read credentials, start Showdown, create files, or call a
model until `--run` is present.

The standard default matrix is three fixed baselines—random,
max-base-power, and simple-heuristics—with ten battles against each. The caller
can reduce this for a cheap smoke evaluation or add repetitions. Every real
evaluation requires a new output directory so experiments cannot be mixed.

Before the first battle, a small model connectivity check validates the native
tool-call contract. The full matrix then runs sequentially with one concurrent
battle, keeping provider load and experiment behavior predictable.

Passing preflight is not enough because a provider can fail during a long
matrix. After every opponent/repeat batch, a hard cost-protection gate stops
the remaining live calls when fallback rate exceeds 20%, decision-error rate
exceeds 30%, or tool-call/rationale coverage falls below 70%.

## Artifacts

An evaluation directory contains:

```text
plan.json
report.json
report.md
runs/<opponent>-r<repeat>.jsonl
runs/<opponent>-r<repeat>.manifest.json
runs/<opponent>-r<repeat>.summary.json
```

Existing per-run artifacts remain authoritative. `report.json` contains the
evaluation schema version, creation time, model and Git provenance, declared
matrix, each run result, per-opponent metrics, and overall metrics.

The Markdown report is a human-readable view of the same data. It contains no
raw model arguments, API keys, environment variables, hidden opponent state, or
private chain-of-thought.

## Metrics

Outcome metrics:

- wins, losses, draws, win rate, and chess-style score rate where a draw is 0.5;
- a 95% Wilson interval around the score rate;
- the same metrics broken down by opponent.

Decision reliability metrics:

- decisions, retries, fallbacks, decisions with errors;
- tool-call and public-rationale coverage;
- average confidence and reason-code counts.

Efficiency metrics:

- input, output, and total tokens;
- tokens per battle and per decision;
- total model decision latency and average latency per decision.

Dollar cost is deliberately excluded because the configured third-party model
has no authoritative pricing source in the project.

## Comparison

`compare` requires completed reports with the same battle format and opponent
set. It records other configuration differences rather than hiding them.

The primary outcome is the change in score rate. A deterministic stratified
bootstrap resamples outcomes within each opponent and reports:

- candidate-minus-baseline score delta;
- a 95% bootstrap interval;
- estimated probability that the candidate is better.

The conclusion is:

- `improved` when the entire delta interval is above zero;
- `regressed` when the entire interval is below zero;
- `inconclusive` otherwise;
- `insufficient_data` when either side has fewer than 20 battles.

Reliability and efficiency deltas are reported as trade-offs, not folded into
an arbitrary single score. The comparison does not claim causality when model,
prompt, Git commit, or other declared configuration differs.

## Research-quality gate

Execution completion and research validity are separate. A completed report is
marked `valid` only when:

- fallback rate is at most 5%;
- decision-error rate is at most 10%;
- native tool-call coverage is at least 95%;
- public-rationale coverage is at least 95%.

Invalid reports preserve their artifacts and metrics for diagnosis but cannot
be passed to `compare`. This prevents provider outages or fallback-heavy games
from being mistaken for evidence about Agent strength.

## Failures and testing

Invalid plans fail before any external action. A failed connectivity check
aborts before the battle matrix. If a later run fails, the evaluation writes an
incomplete report with completed-run data and the sanitized failure, then
stops; rerunning must use a new output directory.

Unit tests cover plan validation, dry-run behavior, aggregation, Wilson
intervals, stratified comparison, incompatibility checks, Markdown rendering,
and CLI routing. Integration tests use deterministic model and local Showdown
battles; automated tests never call the paid endpoint.
