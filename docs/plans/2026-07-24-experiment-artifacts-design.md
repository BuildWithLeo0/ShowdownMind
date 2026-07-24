# ShowdownMind Experiment Artifacts

## Goal

Make every Agent battle self-describing and reproducible enough to compare
later architecture changes.

Each run writes four sibling files derived from the decision-log name:

- `run.jsonl`: immutable per-decision records;
- `run.manifest.json`: configuration and software provenance written before the
  battle;
- `run.summary.json`: final battle and usage metrics;
- `run.failure.json`: a sanitized failure record, written only if the run does
  not finish.

## Manifest contents

The manifest records the run ID, UTC start time, battle format, opponent,
requested battle count, prompt format, timeout, model ID, sanitized provider
base URL, model-client type, Showdown commit, ShowdownMind Git commit and dirty
state, Python version, and relevant installed package versions.

It never records environment variables, request headers, API keys, provider
query strings, URL credentials, model responses, or hidden opponent state.
Provider URLs are reduced to scheme, host, port, and path.

## Write behavior

Decision logs are append-only at the storage layer, but an experiment run must
start with a new empty path. The runner rejects a non-empty target instead of
silently mixing two runs.

Manifest, summary, and failure JSON files are written atomically through a
temporary sibling followed by `replace`. A failed run keeps its partial
decision log and manifest for diagnosis.

Failure messages are bounded and redact common bearer-token and `sk-...`
patterns. They are diagnostic metadata, not raw provider traces.

## Validation

Tests cover artifact naming, URL sanitization, credential redaction, rejection
of reused logs, successful summary writing, and failure artifact writing.
Existing hidden-information and decision-log tests continue to own the battle
state boundary.
