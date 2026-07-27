# Resumable live evaluation design

## Goal

Allow a paid live evaluation to stop because of insufficient balance, network
failure, process termination, or a quality stop and later continue without
rerunning accepted battles. Recovery must preserve research provenance and
must not silently delete failed attempts.

## Atomic unit

One battle is one checkpoint. A 20-battle evaluation produces twenty matrix
cells:

```text
max-base-power / repeat 1 / battle 1..10
simple-heuristics / repeat 1 / battle 1..10
```

Each attempt has an immutable suffix:

```text
max-base-power-r01-b001-a01.jsonl
max-base-power-r01-b001-a01.manifest.json
max-base-power-r01-b001-a01.summary.json
max-base-power-r01-b001-a01.failure.json
max-base-power-r01-b001-a01.attempt.json
```

An accepted attempt satisfies the existing hard quality gate and becomes the
checkpoint for that matrix cell. Interrupted or hard-invalid attempts remain
on disk and are listed in the evaluation report, but do not mark the cell
complete. A resume uses `a02`, `a03`, and so on; it never overwrites an earlier
attempt.

## Evaluation state

`evaluation-state.json` is written atomically and locks:

- the complete evaluation plan;
- model ID;
- Git commit and dirty state;
- creation time;
- the successful preflight result.

`--resume` requires this file and rejects a changed plan, model, Git commit, or
dirty state. API credentials are never stored and may be replenished between
attempts. A saved successful preflight is reused so resuming does not spend
another model call merely to rediscover completed work.

Every accepted battle rewrites `report.json` and `report.md` with status
`in_progress`. A controlled stop writes status `incomplete`; completion writes
status `complete`. Progress reports include target, accepted, remaining,
attempt count, excluded attempt count, and excluded token usage.

## Crash recovery

The attempt record is written as `running` before model calls begin. On normal
success it becomes `accepted`; an invalid completed battle becomes `rejected`;
an exception becomes `interrupted`.

If the process dies after the Agent summary is written but before the attempt
record is finalized, resume reconstructs the result from the summary and
applies the hard quality gate. If there is no summary, the old attempt remains
abandoned and the cell receives a new attempt number.

## CLI

Start:

```bash
showdown-mind evaluate ... --run
```

Continue the exact evaluation:

```bash
showdown-mind evaluate ... --run --resume
```

The second command may be repeated until all cells are accepted. Resuming an
already complete evaluation is read-only and returns the existing report.

## Research boundary

Rejected attempts are not erased or silently treated as wins/losses. Their
errors, decision metrics, and token usage remain visible. This makes
infrastructure interruptions recoverable without pretending they never cost
anything. The final accepted-battle metrics remain separate from excluded
attempt cost.
