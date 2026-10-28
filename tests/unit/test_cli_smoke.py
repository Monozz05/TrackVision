"""End-to-end checks of the command line, run as a real subprocess."""

import json
import subprocess
import sys

import yaml

from trackvision import cli

from conftest import REPO_ROOT, SMOKE_PATH


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "trackvision", *map(str, args)],
                          capture_output=True, text=True, cwd=REPO_ROOT, check=False)


def test_smoke_run_writes_the_three_artifacts(tmp_path):
    proc = _cli("run", "--config", SMOKE_PATH, "--runs-dir", tmp_path)
    assert proc.returncode == cli.EXIT_OK, proc.stdout + proc.stderr
    (run_dir,) = list(tmp_path.iterdir())
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config.resolved.yaml", "fixture_summary.json", "manifest.json"]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == run_dir.name
    assert manifest["kind"] == "smoke"
    assert manifest["scientific_validity"].startswith("none")
    assert manifest["research_choices_used"] == []
    assert manifest["seed"] is None
    assert manifest["environment"]["gpu"] == "not checked (Phase 0B)"

    summary = json.loads((run_dir / "fixture_summary.json").read_text(encoding="utf-8"))
    assert all(summary["checks"].values())

    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["experiment"]["name"] == "smoke"
    assert resolved["research_choices"]["seeds"]["value"] is None


def test_two_runs_into_separate_directories_share_id_and_hash(tmp_path):
    ids, hashes = [], []
    for sub in ("first", "second"):
        proc = _cli("run", "--config", SMOKE_PATH, "--runs-dir", tmp_path / sub)
        assert proc.returncode == cli.EXIT_OK, proc.stdout + proc.stderr
        (run_dir,) = list((tmp_path / sub).iterdir())
        ids.append(run_dir.name)
        hashes.append(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config_hash"])
    assert ids[0] == ids[1] and hashes[0] == hashes[1]


def test_existing_run_directory_is_not_overwritten(tmp_path):
    assert _cli("run", "--config", SMOKE_PATH, "--runs-dir", tmp_path).returncode == cli.EXIT_OK
    (run_dir,) = list(tmp_path.iterdir())
    marker = run_dir / "user-notes.txt"
    marker.write_text("keep me", encoding="utf-8")
    before = (run_dir / "manifest.json").read_bytes()

    proc = _cli("run", "--config", SMOKE_PATH, "--runs-dir", tmp_path)
    assert proc.returncode == cli.EXIT_EXISTS
    assert (run_dir / "manifest.json").read_bytes() == before

    assert _cli("run", "--config", SMOKE_PATH, "--runs-dir", tmp_path, "--force").returncode == cli.EXIT_OK
    assert marker.read_text(encoding="utf-8") == "keep me"  # --force rewrites only its own files


def test_invalid_experiment_exits_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment:\n  name: bad\n  kind: smoke\n  uses_research_choices: []\n"
                   "  selection: {regime: static, error_mode: F}\n", encoding="utf-8")
    proc = _cli("run", "--config", bad, "--runs-dir", tmp_path / "runs")
    assert proc.returncode == cli.EXIT_INVALID
    assert "INVALID" in proc.stdout
    assert not (tmp_path / "runs").exists()


def test_unresolved_research_experiment_exits_3_and_names_the_choice(tmp_path):
    exp = tmp_path / "research.yaml"
    exp.write_text("experiment:\n  name: clean-baseline\n  kind: research\n"
                   "  uses_research_choices: [seeds, inner_validation_rotation]\n"
                   "  selection: {regime: static}\n", encoding="utf-8")
    proc = _cli("run", "--config", exp, "--runs-dir", tmp_path / "runs")
    assert proc.returncode == cli.EXIT_UNRESOLVED
    assert "UNRESOLVED" in proc.stdout
    assert "seeds  (used by this experiment)" in proc.stdout
    assert not (tmp_path / "runs").exists()


def test_config_validate_reports_unresolved_choices_and_exits_0():
    proc = _cli("config", "validate")
    assert proc.returncode == cli.EXIT_OK, proc.stdout + proc.stderr
    assert "status: VALID" in proc.stdout
    for name in ("seeds", "inner_validation_rotation", "moving_dataset"):
        assert f"- {name}" in proc.stdout


def test_config_hash_command_matches_manifest(tmp_path):
    proc = _cli("config", "hash", "--config", SMOKE_PATH)
    assert proc.returncode == cli.EXIT_OK
    assert _cli("run", "--config", SMOKE_PATH, "--runs-dir", tmp_path).returncode == cli.EXIT_OK
    (run_dir,) = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert proc.stdout.strip() == manifest["config_hash"]
