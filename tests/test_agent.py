import hashlib
import json
from types import SimpleNamespace

import pytest
from poke_env.battle import Move
from poke_env.player.battle_order import SingleBattleOrder

from showdown_mind.agent import ResearchPlayer
from showdown_mind.models import ScriptedModelClient
from showdown_mind.observation import BattleSnapshotBuilder
from showdown_mind.policy import SingleCallPolicy


def fake_pokemon(species: str) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        name=species,
        revealed=True,
        active=True,
        fainted=False,
        current_hp_fraction=1.0,
        status=None,
        types=[],
        boosts={},
        item=None,
        ability=None,
        moves={},
        tera_type=None,
    )


def fake_battle() -> SimpleNamespace:
    own = fake_pokemon("Pikachu")
    opponent = fake_pokemon("Eevee")
    return SimpleNamespace(
        battle_tag="battle-idempotent",
        last_request={"rqid": 11},
        turn=2,
        format="gen9randombattle",
        team={"own": own},
        opponent_team={"opponent": opponent},
        active_pokemon=own,
        opponent_active_pokemon=opponent,
        side_conditions={},
        opponent_side_conditions={},
        weather={},
        fields={},
        can_tera=False,
        used_tera=False,
        opponent_used_tera=False,
        force_switch=False,
        trapped=False,
        valid_orders=[SingleBattleOrder(Move("tackle", gen=9))],
    )


def player_without_network(policy: SingleCallPolicy) -> ResearchPlayer:
    player = object.__new__(ResearchPlayer)
    player._policy = policy
    player._fallback_seed = "test-seed"
    player._snapshot_builder = BattleSnapshotBuilder()
    player._decision_sink = None
    player.decision_records = []
    player._request_cache = {}
    return player


@pytest.mark.asyncio
async def test_duplicate_request_reuses_the_first_decision() -> None:
    client = ScriptedModelClient(['{"action_id": "move:tackle"}'])
    player = player_without_network(SingleCallPolicy(client))
    battle = fake_battle()

    first = await player.choose_move(battle)
    second = await player.choose_move(battle)

    assert first is second
    assert len(client.requests) == 1
    assert len(player.decision_records) == 1
    assert player.decision_records[0].policy_input_format == "pruned-v1"
    assert player.decision_records[0].policy_input_characters > 0
    assert player.decision_records[0].policy_input["schema"] == "pruned-v1"
    policy_input = json.dumps(
        player.decision_records[0].policy_input,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert (
        player.decision_records[0].policy_input_hash
        == "sha256:" + hashlib.sha256(policy_input.encode()).hexdigest()
    )


@pytest.mark.asyncio
async def test_invalid_model_output_uses_logged_deterministic_fallback() -> None:
    client = ScriptedModelClient(["bad", "still bad"])
    player = player_without_network(SingleCallPolicy(client))

    order = await player.choose_move(fake_battle())

    assert isinstance(order, SingleBattleOrder)
    assert player.decision_records[0].fallback_used
    assert player.decision_records[0].attempts == 2
    assert player.decision_records[0].elapsed_seconds >= 0
    canonical_snapshot = json.dumps(
        player.decision_records[0].snapshot,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert (
        player.decision_records[0].snapshot_hash
        == "sha256:" + hashlib.sha256(canonical_snapshot.encode("utf-8")).hexdigest()
    )
