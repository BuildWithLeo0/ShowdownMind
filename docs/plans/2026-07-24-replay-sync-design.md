# ShowdownMind Replay Synchronization

## Goal

The native Pokémon Showdown replay drives the Agent inspector automatically.
Play, pause, reset, previous turn, next turn, and go-to-turn operations update
the visible decision without requiring a second manual timeline click.

## Why turn number alone is insufficient

One Showdown turn can contain more than one Agent request. In the first live
GPT-5.6 Luna battle, turns 2, 10, and 12 each contain:

```text
choose a move → own Pokémon faints → choose a replacement
```

Both decisions share one `turn`, but happen at different positions in the
replay protocol. Synchronization therefore uses both the native player's
`Battle.turn` and `Battle.currentStep`.

## Chosen integration

The downloaded replay loads the official `replay-embed.js`, which exposes its
player as `Replays.battle`. The parent viewer reads four fields without
modifying the player:

- `turn`;
- `currentStep`;
- `paused`;
- `ended`.

This is intentionally a read-only bridge. Replacing `Battle.subscribe` would
take over its single subscription slot and break the replay's own controls.
Watching rendered DOM text would couple ShowdownMind to presentation markup.

The viewer polls the small state object while it is open. If the player is not
ready or its structure changes, the decision timeline remains fully usable in
manual mode.

## Protocol anchors

At build time, Python extracts the `battle-log-data` script from replay HTML.
Every first decision in a turn is anchored immediately after its `|turn|N`
protocol line. Additional decisions in that turn are anchored to the matching
player `|switch|...` or `|move|...` line.

The Agent side is discovered from the `|player|p1|ResearchPlayer...` or
`|player|p2|ResearchPlayer...` line. Move and species labels are normalized to
the same lowercase IDs used by decision records.

Each viewer decision receives a `replay_step`. During playback, the last
decision whose anchor is not greater than `currentStep` becomes active.

## Interaction behavior

Replay-follow mode is enabled by default. Its status distinguishes connecting,
playing, paused, ended, and unavailable states.

Manual previous, next, or timeline selection disables follow mode so the page
does not immediately snap back. The user can re-enable follow with one button,
which immediately catches the inspector up to the current replay step.

Changing inspector tabs does not disable follow mode.

## Validation

Unit tests verify protocol extraction, player-side detection, turn anchors,
forced-switch anchors, and missing-anchor fallback. Browser validation uses the
real 18-decision battle and checks:

- continuous play advances the inspector;
- native previous/next-turn controls move it backward and forward;
- the forced Slowbro switch becomes the second decision in turn 2;
- manual selection pauses follow and re-enabling catches up;
- replay loading failure leaves manual navigation intact.
