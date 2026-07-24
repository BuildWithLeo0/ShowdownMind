# ShowdownMind Replay Decision Viewer

## What the first version does

The viewer explains one completed Agent battle. It places the native Pokémon
Showdown replay beside an inspectable record of every Agent decision.

The left side is the existing `poke-env` replay, so Showdown remains responsible
for battle animation, sprites, effects, and playback controls. The right side
shows only recorded, verifiable Agent data:

- the visible player and opponent state;
- legal moves and switches;
- the selected action;
- confidence, reason codes, and short rationale;
- retry, fallback, latency, and token metadata.

It does not claim to expose hidden chain-of-thought.

## User flow

```bash
uv run showdown-mind visualize .runtime/decisions/run.jsonl
```

The command reads the decision log, finds its battle ID, locates the matching
HTML replay under `.runtime/replays`, generates `run.viewer.html`, and opens it.
`--no-open` supports tests and headless environments. `--replay` and
`--battle-id` resolve unusual or multi-battle logs explicitly.

The generated file is self-contained except for the official remote assets
already required by `poke-env` replay HTML. Replay HTML and viewer data are
base64-encoded inside the file, avoiding local file-fetch restrictions and
preventing logged text from becoming executable page markup.

## Data boundary

A viewer payload contains the selected battle's snapshots, legal actions,
choice, public rationale, validation errors after credential redaction, usage,
and timing. It excludes raw model responses, environment variables, headers,
API keys, hidden opponent state, and decisions from other battles.

The first version does not depend on an undocumented Showdown playback API.
The replay retains its own controls while the decision inspector has its own
previous/next controls and turn timeline. Full automatic playback
synchronization is a later enhancement.

## Visual direction

The interface is a dark competitive-analysis desk: graphite panels, warm
ivory text, signal-green success, amber decisions, and red fallbacks. Large
turn numerals and a dense event rail evoke a match broadcast rather than a
generic dashboard. It uses plain HTML, CSS, and JavaScript so the UI remains
approachable for someone learning Agent development.

## Validation

Unit tests cover JSONL parsing, battle selection, replay discovery, sensitive
field exclusion, output naming, and deterministic HTML generation. A real
recorded LLM battle is used for browser-size visual inspection at desktop and
mobile widths.
