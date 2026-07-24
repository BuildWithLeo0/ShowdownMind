import json
from types import SimpleNamespace

import pytest

from showdown_mind.experiment_artifacts import (
    ArtifactPaths,
    ExperimentArtifactWriter,
    ExperimentSpec,
)


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        battle_format="gen9randombattle",
        opponent="random",
        requested_battles=1,
        prompt_format="pruned-v1",
        timeout_seconds=60,
    )


def test_artifact_paths_are_siblings_of_decision_log(tmp_path) -> None:
    paths = ArtifactPaths.from_decision_log(tmp_path / "run.jsonl")

    assert paths.manifest.name == "run.manifest.json"
    assert paths.summary.name == "run.summary.json"
    assert paths.failure.name == "run.failure.json"


def test_manifest_sanitizes_provider_url_and_excludes_credentials(tmp_path) -> None:
    writer = ExperimentArtifactWriter(tmp_path / "run.jsonl")
    model = SimpleNamespace(
        model_id="test-model",
        base_url="https://user:password@provider.example:8443/v1?api_key=secret",
    )

    writer.assert_new_run()
    writer.write_manifest(model, spec())

    encoded = writer.paths.manifest.read_text(encoding="utf-8")
    manifest = json.loads(encoded)
    assert manifest["model"]["base_url"] == "https://provider.example:8443/v1"
    assert "password" not in encoded
    assert "api_key" not in encoded
    assert "secret" not in encoded


def test_failure_record_redacts_common_secret_shapes(tmp_path) -> None:
    writer = ExperimentArtifactWriter(tmp_path / "run.jsonl")

    writer.write_failure(
        RuntimeError(
            "sk-abcdefghijk Bearer token-value "
            "api_key=query-secret https://user:password@example.com/v1"
        )
    )

    encoded = writer.paths.failure.read_text(encoding="utf-8")
    assert "abcdefghijk" not in encoded
    assert "token-value" not in encoded
    assert "query-secret" not in encoded
    assert "password" not in encoded
    assert encoded.count("[REDACTED]") == 4


def test_existing_artifact_prevents_mixed_runs(tmp_path) -> None:
    writer = ExperimentArtifactWriter(tmp_path / "run.jsonl")
    writer.paths.decisions.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        writer.assert_new_run()
