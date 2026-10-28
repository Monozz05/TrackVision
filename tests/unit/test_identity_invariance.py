"""The config hash and experiment ID depend only on canonical configuration and the ID inputs.

Timestamps, Git state, environment details and output locations are recorded in the manifest
but must never change an experiment's identity.
"""

import inspect
import json

import yaml

from trackvision import cli, config, repro

from conftest import BASE_PATH, REPO_ROOT, SMOKE_PATH

# Pinned so that any change to base.yaml, smoke.yaml or the canonical serialisation is noticed
# deliberately. If base.yaml legitimately changes (a new spec version), update this value in
# the same change and say why in the commit message.
GOLDEN_SMOKE_HASH = "32a95bb9586fb8480ce78577ea23448f81cc29a40bff0d2801b525e1aca7b058"


def _hash_and_id(exp_path, base_path=BASE_PATH):
    result = config.validate(exp_path, base_path=base_path, repo_root=REPO_ROOT)
    assert result.status == config.RUNNABLE, result.errors
    cfg = result.config
    return cfg.hash, repro.make_experiment_id(cfg.experiment["name"], cfg.hash, None)


def _reorder(value):
    """Recursively reverse mapping key order (lists keep their order: it is meaningful)."""
    if isinstance(value, dict):
        return {k: _reorder(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [_reorder(v) for v in value]
    return value


def test_formatting_key_order_and_comments_do_not_change_identity(write_yaml):
    reference = _hash_and_id(SMOKE_PATH)

    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    reformatted_base = "# a comment that is not in the original\n" + yaml.safe_dump(
        _reorder(base), sort_keys=False, default_flow_style=False, indent=4, width=200)
    smoke = yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))
    reformatted_smoke = yaml.safe_dump(_reorder(smoke), default_flow_style=True) + "\n# trailing comment\n"

    assert _hash_and_id(write_yaml(reformatted_smoke), write_yaml(reformatted_base)) == reference


def _run(monkeypatch, runs_dir, timestamp, dirty, env_tag):
    monkeypatch.setattr(cli, "_now", lambda: timestamp)
    monkeypatch.setattr(repro, "git_state", lambda root: {"sha": None, "dirty": dirty})
    monkeypatch.setattr(repro, "environment_info", lambda: {"tag": env_tag})
    assert cli.main(["run", "--config", str(SMOKE_PATH), "--runs-dir", str(runs_dir)]) == 0
    (run_dir,) = list(runs_dir.iterdir())
    return run_dir.name, json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_runtime_provenance_does_not_change_identity(tmp_path, monkeypatch):
    id_a, manifest_a = _run(monkeypatch, tmp_path / "a", "2026-01-01T00:00:00+00:00", False, "env-a")
    id_b, manifest_b = _run(monkeypatch, tmp_path / "elsewhere" / "b", "2026-01-01T01:00:00+00:00", True, "env-b")

    assert id_a == id_b
    assert manifest_a["config_hash"] == manifest_b["config_hash"]
    # The differing inputs were recorded, so the test really exercised them.
    assert manifest_a["timestamp_utc"] != manifest_b["timestamp_utc"]
    assert manifest_a["git"]["dirty"] is False and manifest_b["git"]["dirty"] is True
    assert manifest_a["environment"] != manifest_b["environment"]


def test_real_configuration_changes_do_change_identity(write_yaml):
    reference_hash, reference_id = _hash_and_id(SMOKE_PATH)
    smoke = yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))

    moved = yaml.safe_load(yaml.safe_dump(smoke))
    moved["experiment"]["selection"]["regime"] = "moving"
    assert _hash_and_id(write_yaml(moved))[0] != reference_hash

    renamed = yaml.safe_load(yaml.safe_dump(smoke))
    renamed["experiment"]["name"] = "smoke-renamed"
    new_hash, new_id = _hash_and_id(write_yaml(renamed))
    assert new_hash != reference_hash and new_id != reference_id

    assert repro.make_experiment_id("smoke", reference_hash, 0) != reference_id


def test_smoke_hash_matches_pinned_golden_value():
    smoke_hash, smoke_id = _hash_and_id(SMOKE_PATH)
    assert smoke_hash == GOLDEN_SMOKE_HASH, (
        f"smoke config hash is {smoke_hash}; if this change is deliberate, update GOLDEN_SMOKE_HASH"
    )
    assert smoke_id == f"smoke-{smoke_hash[:10]}-sna"


def test_identity_functions_cannot_reach_runtime_provenance():
    forbidden = ("time", "datetime", "git_state", "environment_info", "getcwd", "platform")
    for func in (config.config_hash, config.canonical_json, repro.make_experiment_id):
        source = inspect.getsource(func)
        body = source.split(":", 1)[1]  # ignore the signature line's annotations
        for word in forbidden:
            assert word not in body, f"{func.__name__} refers to {word!r}"
