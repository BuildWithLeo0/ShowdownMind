# ShowdownMind Token-efficient Policy Input

## Goal

Reduce the repeated input cost of live-model battles without removing
player-visible battle facts or making experiments impossible to audit.

The first live `gpt-5.6-luna` battle used 39,307 input tokens across 23
decisions. The full `BattleSnapshot` is intentionally readable and complete,
but it repeats verbose field names, null values, labels, and metadata that the
model does not need to choose an action.

## Chosen approach

Add a versioned policy-input compiler between `BattleSnapshot` and
`SingleCallPolicy`.

- `full-v1` sends the existing snapshot unchanged and remains available as an
  experimental control.
- `pruned-v1` preserves familiar field names while removing empty values,
  execution metadata, nicknames, and redundant information-scope labels. It is
  the default because provider token accounting showed that shorter compact
  field names did not necessarily reduce billed input tokens.
- `compact-v1` keeps the same public facts but removes empty values, nicknames,
  redundant information-scope labels, and duplicated action descriptions.
- The full snapshot remains the authoritative audit record.
- The exact compiled model input, compiler version, character count, and hash
  are recorded with every decision.

This is deliberately not a delta protocol. Each turn remains self-contained,
so a retry, reconnect, or isolated log replay does not depend on earlier model
requests.

## Reduced representations

Both reduced inputs retain:

- turn and battle format;
- both active Pokémon;
- our complete team and the opponent's revealed team;
- HP fraction, fainted state, status, types, boosts, known item, known ability,
  revealed moves, and known Tera type;
- side conditions, weather, fields, and Tera/switch/trap resources;
- every legal action and all decision-relevant action details.

They remove:

- battle and request IDs, which are execution metadata;
- empty collections, null values, and false optional flags;
- Pokémon nickname and `information_scope`, which are not tactical facts;
- action labels that duplicate move or species IDs;
- nested action-detail wrappers.

The last two removals apply only to `compact-v1`; `pruned-v1` keeps the
human-readable action schema.

## Validation

Tests must prove that:

1. hidden opponent values still cannot enter either representation;
2. every non-empty tactical fact maps to each reduced representation;
3. reduced outputs are deterministic and hashable;
4. all formats drive the same legal-action validator;
5. deterministic and live-model paths continue to work;
6. historical real-battle snapshots show a material character reduction.

No win-rate claim will be made from a single battle. The new format is an
experimental variable, not an assumed improvement.

## Initial measurements

Recompiling the 23 decisions from the first complete live-model battle reduced
the serialized input characters as follows:

- `pruned-v1`: 31.16% fewer characters than `full-v1`;
- `compact-v1`: 41.03% fewer characters than `full-v1`.

Provider-reported tokens did not follow character count for the aggressively
compact schema. Repeated calls on the same small battle state reported:

| Format | Characters | Provider input tokens |
|---|---:|---:|
| `full-v1` | 729 | 275 |
| `pruned-v1` | 463 | 214 |
| `compact-v1` | 397 | 486 |

`pruned-v1` therefore becomes the default. This measurement is specific to the
configured `gpt-5.6-luna` endpoint; experiments must record both exact inputs
and provider usage rather than treating character count as billed tokens.
