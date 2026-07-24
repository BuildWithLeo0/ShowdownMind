# Learning 01: See the Agent Loop

An Agent is not just an LLM call. It is a loop that connects a model to an
environment:

```text
observe → list safe actions → decide → validate → act → record → observe again
```

ShowdownMind keeps these jobs separate so you can change and test one idea at a
time.

## 1. Observe

`observation.py` converts the live `poke-env` battle object into a
`BattleSnapshot`. It includes the full information available to our player and
only information already revealed by the opponent.

The viewer's **可见状态** tab is a direct picture of this boundary. If hidden
opponent moves appear there, the Agent has an information leak.

## 2. List safe actions

`actions.py` enumerates the moves and switches that Showdown currently allows.
Each receives a stable `action_id`.

The viewer's **候选动作** tab displays this whitelist. The model chooses one ID
instead of inventing a Showdown command. This is a common Agent pattern: let the
program define what is possible, then let the model choose.

## 3. Decide and validate

`policy.py` compiles a model input and exposes one forced native tool:

```text
choose_battle_action(
  action_id,
  confidence,
  reason_codes,
  short_rationale
)
```

This is a real model tool call, but the model does not execute a Showdown
command. It asks the host program to use one legal `action_id`; the program then
validates the arguments and decides whether to honor that request. This
separation is the safety boundary between model reasoning and environment
control.

All four arguments are required. The short rationale is a public, concise
explanation for humans and experiments, not private chain-of-thought. The
policy permits one repair attempt and can fall back safely.

The viewer's **决策** tab shows the accepted choice and public short rationale.
The **执行记录** tab shows attempts, validation errors, fallback use, latency,
token usage, and the provider's tool-call ID. It does not claim to reveal
private chain-of-thought.

## 4. Act and record

`agent.py` translates the validated `action_id` back into a legal `poke-env`
order. `storage.py` appends the snapshot and result to JSONL before the loop
continues.

`viewer.py` later joins those records to the native Showdown replay using their
shared `battle_id`. The generated HTML is a learning tool: the animation shows
what happened in the environment while the inspector shows what crossed the
Agent boundary.

## 5. Synchronize environment time with Agent time

A single turn number is not always enough. If the Agent's Pokémon faints after
choosing a move, Showdown asks it to choose a replacement during the same turn.

`viewer.py` therefore gives each recorded decision a replay protocol step:

```text
|turn|2                    → choose the turn-2 move
...
|faint|p1a: Zekrom
|switch|p1a: Slowbro       → choose the turn-2 replacement
```

The browser reads the native player's current protocol step and activates the
latest decision whose anchor has been reached. It never replaces Showdown's
playback subscription or changes battle state.

This illustrates another general Agent lesson: the environment's clock and the
Agent's decision clock are often different. A useful trace needs a shared event
identifier, not just a timestamp or human-facing turn number.

## A useful exercise

Open a real viewer and find a decision with two attempts:

1. watch what happened around that turn in the replay;
2. inspect all legal actions;
3. read the first error under **执行记录**;
4. check which action was finally accepted;
5. decide whether a different observation or tool would have helped.

That exercise is the basic rhythm of Agent engineering: inspect a trace,
form a hypothesis, change one component, and compare the next experiment.
