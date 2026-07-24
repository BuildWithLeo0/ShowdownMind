# Native action tool design

## Goal

Replace the model's free-form JSON response with one forced native function call:

```text
choose_battle_action(action_id, confidence, reason_codes, short_rationale)
```

The tool call is the model-to-agent boundary. The application still resolves the
selected `action_id` through `ActionCatalog`; the model never executes a
Pokémon Showdown command directly.

## Contract

Every model request exposes exactly one tool and forces that tool with
`tool_choice`. Parallel tool calls are disabled.

The JSON Schema is built for each battle request:

- `action_id` is an enum containing only the currently legal action IDs.
- `confidence` is required and must be between 0 and 1. It is a self-report,
  not a calibrated probability.
- `reason_codes` is required, contains one to three values from a small fixed
  vocabulary, and supports later quantitative analysis.
- `short_rationale` is required, non-empty, and at most 240 characters. It is a
  short public explanation of the decision, not hidden chain-of-thought.
- Extra properties are rejected and strict schema mode is requested.

The program validates all arguments again after receiving the tool call.
Schema enforcement improves reliability but does not replace the action
whitelist.

## Failure behavior

A response is invalid when it contains no tool call, more than one tool call,
the wrong tool name, malformed arguments, an illegal action, or missing/invalid
required fields.

The policy makes at most one repair attempt using the same forced tool. Network
and timeout failures also receive one retry. If both attempts fail, the existing
deterministic fallback selects a legal action and records a public fallback
reason so the viewer never presents an unexplained choice.

## Audit data and viewer

Decision records retain the provider response ID and native tool-call ID for
each successful transport response. The viewer shows that the decision came
through `choose_battle_action`, displays the short rationale and confidence,
and includes tool-call IDs in the execution trace.

Raw tool-argument strings remain in local JSONL logs for research auditing but
are not embedded into the standalone viewer.

## Compatibility

The transport remains the Chat Completions endpoint because the configured
third-party provider already supports it. The implementation uses the current
OpenAI Python SDK shape:

- `tools=[{"type": "function", "function": {..., "strict": true}}]`
- a named `tool_choice`
- `parallel_tool_calls=False`
- arguments read from `message.tool_calls[0].function.arguments`

A live `model-check` request must pass against the configured
`gpt-5.6-luna` endpoint before the migration is considered complete.

Provider-specific thinking configuration is explicit rather than inferred from
the URL. `SHOWDOWN_MIND_THINKING=disabled` adds
`{"thinking":{"type":"disabled"}}` to the request and is recorded in the run
manifest. This supports DeepSeek V4's forced function calling without weakening
the named `tool_choice` contract for every provider.
