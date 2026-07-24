# ShowdownMind

ShowdownMind is a research project for building and evaluating an LLM agent that
plays Pokémon Showdown without access to hidden opponent information.

The first milestone is deliberately small:

1. run a pinned Pokémon Showdown server locally;
2. connect with `poke-env`;
3. run built-in baseline players;
4. record reproducible smoke-test results.

LLM decision-making, belief tracking, damage tools, and planning are introduced
in later milestones so their effects can be measured separately.

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

## Tests

```bash
uv run pytest
```

The project design is documented in
[`docs/plans/2026-07-24-pokemon-showdown-agent-design.md`](docs/plans/2026-07-24-pokemon-showdown-agent-design.md).
