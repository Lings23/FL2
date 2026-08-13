"""Resource gates, cache validation, and per-spec retry behavior."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

import main
from experiments import periodic_attack


def _round_frame(rounds: int) -> pd.DataFrame:
    return pd.DataFrame({
        "round": list(range(rounds + 1)),
        "planned_attack_active": [0] * (rounds + 1),
        "server_loss": np.linspace(2.0, 1.0, rounds + 1),
        "server_accuracy": np.linspace(0.1, 0.5, rounds + 1),
    })


def _args(rounds: int = 2, retries: int = 1) -> argparse.Namespace:
    return argparse.Namespace(
        rounds=rounds,
        smoke=False,
        max_spec_retries=retries,
        ray_min_available_memory_mb=0,
        ray_memory_wait_seconds=0.0,
    )


def _spec(name: str = "byzantine") -> dict:
    return {
        "attack": name,
        "attack_group": "untargeted",
        "period": "short_1_1",
        "on_rounds": 1,
        "off_rounds": 1,
        "malicious_fraction": 0.2,
        "seed": 42,
        "defense": "fedavg",
        "defense_type": "none",
        "custom_params": {},
        "attack_start_round": 1,
        "attack_end_round": 2,
        "partition": "iid",
        "dirichlet_alpha": 0.5,
        "participation_rate": 0.5,
        "boost_factor": 5.0,
        "benchmark_version": periodic_attack.BENCHMARK_VERSION,
    }


def test_memory_preflight_returns_immediately_when_capacity_is_sufficient(monkeypatch):
    monkeypatch.setattr(main, "estimate_available_memory_bytes", lambda: 2048 * 1024 * 1024)

    available = main.wait_for_available_memory(1024, timeout_seconds=0)

    assert available == 2048 * 1024 * 1024


def test_memory_preflight_waits_until_capacity_recovers(monkeypatch):
    readings = iter([100, 200])
    monkeypatch.setattr(
        main,
        "estimate_available_memory_bytes",
        lambda: next(readings) * 1024 * 1024,
    )

    available = main.wait_for_available_memory(150, timeout_seconds=1, poll_seconds=0.01)

    assert available == 200 * 1024 * 1024


def test_memory_preflight_timeout_reports_observed_and_required(monkeypatch):
    monkeypatch.setattr(main, "estimate_available_memory_bytes", lambda: 100 * 1024 * 1024)

    with pytest.raises(
        main.RayResourcePreflightError,
        match=r"available=100.0 MiB.*required=150 MiB",
    ):
        main.wait_for_available_memory(150, timeout_seconds=0)


@pytest.mark.parametrize("raises", [False, True])
def test_run_simulation_always_cleans_up_ray(monkeypatch, tmp_path, raises):
    cfg = main.load_config("config/config.yaml")
    cfg.project.log_dir = str(tmp_path)
    cfg.ray.min_available_memory_mb = 0
    cleanup_calls = []

    class Strategy:
        last_client_records = []

        @staticmethod
        def evaluate(*_args):
            return None

        @staticmethod
        def aggregate_fit(*_args):
            return None, {}

        @staticmethod
        def aggregate_evaluate(*_args):
            return None, {}

    class Server:
        strategy = Strategy()

    monkeypatch.setattr(main, "shutdown_ray_runtime", lambda: cleanup_calls.append(True))
    monkeypatch.setattr(
        main,
        "build_data_pipeline",
        lambda *_args, **_kwargs: ({}, object(), None),
    )
    monkeypatch.setattr(main, "build_model_factory", lambda _cfg: lambda: object())
    monkeypatch.setattr(main, "get_model", lambda **_kwargs: object())
    monkeypatch.setattr(main, "determine_malicious_ids", lambda _cfg: set())
    monkeypatch.setattr(main, "make_client_fn", lambda **_kwargs: lambda _cid: None)
    monkeypatch.setattr(
        main,
        "build_server",
        lambda *_args, **_kwargs: (Server(), object()),
    )

    def simulation(**_kwargs):
        if raises:
            raise RuntimeError("injected failure")

    monkeypatch.setattr(main.fl.simulation, "start_simulation", simulation)

    if raises:
        with pytest.raises(RuntimeError, match="injected failure"):
            main.run_simulation(cfg, experiment_name="cleanup-test")
    else:
        main.run_simulation(cfg, experiment_name="cleanup-test")

    assert len(cleanup_calls) == 2


def test_round_cache_requires_all_unique_rounds_and_finite_metrics(tmp_path):
    path = tmp_path / "rounds.csv"
    _round_frame(2).to_csv(path, index=False)
    assert periodic_attack.validate_round_cache(path, 2)[0] is True

    _round_frame(1).to_csv(path, index=False)
    valid, reason, _ = periodic_attack.validate_round_cache(path, 2)
    assert valid is False
    assert "missing rounds" in reason

    duplicate = pd.concat([_round_frame(2), _round_frame(2).iloc[[1]]], ignore_index=True)
    duplicate.to_csv(path, index=False)
    assert periodic_attack.validate_round_cache(path, 2)[1] == "duplicate round values"

    nonfinite = _round_frame(2)
    nonfinite.loc[1, "server_accuracy"] = np.nan
    nonfinite.to_csv(path, index=False)
    assert "finite" in periodic_attack.validate_round_cache(path, 2)[1]


def test_legacy_all_reverse_cache_recovers_exact_asr_from_confusion(tmp_path):
    path = tmp_path / "rounds.csv"
    frame = _round_frame(2)
    frame["planned_attack_active"] = [0, 1, 1]
    frame["server_confusion_matrix_json"] = [
        json.dumps([[5, 5], [4, 6]]),
        json.dumps([[1, 9], [8, 2]]),
        json.dumps([[2, 8], [6, 4]]),
    ]
    frame.to_csv(path, index=False)
    spec = _spec("label_flip_all_reverse")

    valid, reason, cached = periodic_attack.validate_round_cache(path, 2, spec)
    summary = periodic_attack.summarize_run(cached, spec)

    assert valid is True, reason
    assert summary["active_asr"] == pytest.approx((0.85 + 0.70) / 2)
    assert summary["peak_asr"] == pytest.approx(0.85)

    frame.loc[1, "server_confusion_matrix_json"] = "not-json"
    frame.to_csv(path, index=False)
    valid, reason, _ = periodic_attack.validate_round_cache(path, 2, spec)
    assert valid is False
    assert "recoverable" in reason


def test_spec_failure_is_retried_once_then_succeeds(tmp_path):
    spec = _spec()
    round_path = tmp_path / "rounds" / f"{periodic_attack.run_id(spec)}.csv"
    status_path = tmp_path / "status" / f"{periodic_attack.run_id(spec)}.json"
    attempts = []

    def launch(_spec, _payload, _output, path, _status, attempt, _offset):
        attempts.append(attempt)
        if attempt == 2:
            periodic_attack._atomic_write_csv(_round_frame(2), path)
            return 0, 202
        return 1, 101

    periodic_attack._run_spec_with_retries(
        spec, _args(), tmp_path, round_path, status_path, launch_fn=launch,
    )

    assert attempts == [1, 2]
    assert periodic_attack._read_status(status_path)["state"] == "completed"


def test_spec_stops_after_second_failure(tmp_path):
    spec = _spec()
    round_path = tmp_path / "rounds" / f"{periodic_attack.run_id(spec)}.csv"
    status_path = tmp_path / "status" / f"{periodic_attack.run_id(spec)}.json"
    attempts = []

    def launch(*call_args):
        attempts.append(call_args[-2])
        return 1, 303

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        periodic_attack._run_spec_with_retries(
            spec, _args(), tmp_path, round_path, status_path, launch_fn=launch,
        )

    assert attempts == [1, 2]
    assert periodic_attack._read_status(status_path)["state"] == "failed"


def test_matrix_does_not_launch_next_spec_after_exhausted_failure(monkeypatch, tmp_path):
    specs = [_spec("byzantine"), _spec("label_flip_all_reverse")]
    args = _args()
    args.rerun = True
    launched = []

    monkeypatch.setattr(periodic_attack, "_write_execution_environment", lambda *_args: None)
    monkeypatch.setattr(
        periodic_attack,
        "_write_current_summary",
        lambda *_args: pd.DataFrame(),
    )

    def fail(spec, *_args):
        launched.append(spec["attack"])
        raise RuntimeError("exhausted")

    monkeypatch.setattr(periodic_attack, "_run_spec_with_retries", fail)

    with pytest.raises(RuntimeError, match="exhausted"):
        periodic_attack._run_specs(specs, args, tmp_path, tmp_path / "rounds")

    assert launched == ["byzantine"]


def test_complete_cache_is_reused_without_launching_child(monkeypatch, tmp_path):
    spec = _spec()
    args = _args()
    args.rerun = False
    args.ray_object_store_memory_mb = 3072
    rounds_dir = tmp_path / "rounds"
    round_path = rounds_dir / f"{periodic_attack.run_id(spec)}.csv"
    periodic_attack._atomic_write_csv(_round_frame(2), round_path)

    def unexpected_launch(*_args, **_kwargs):
        raise AssertionError("cached specification must not launch a child")

    monkeypatch.setattr(periodic_attack, "_run_spec_with_retries", unexpected_launch)

    summary = periodic_attack._run_specs([spec], args, tmp_path, rounds_dir)

    assert len(summary) == 1
    status = periodic_attack._read_status(
        tmp_path / "status" / f"{periodic_attack.run_id(spec)}.json"
    )
    assert status["state"] == "completed_cached"


def test_atomic_temporary_csv_is_not_a_cache_hit(tmp_path):
    final_path = tmp_path / "run.csv"
    temporary = tmp_path / ".run.csv.123.tmp"
    _round_frame(2).to_csv(temporary, index=False)

    assert final_path.exists() is False
