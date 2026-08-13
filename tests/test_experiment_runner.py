"""Tests for the stable unified experiments entry."""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from experiments import run


def test_registered_profiles_cover_existing_execution_families():
    assert set(run.PROFILES) == {"periodic", "rtc-fedavg", "sweep"}


def test_profile_is_required_unless_listing_profiles():
    with pytest.raises(SystemExit):
        run.parse_args([])
    args = run.parse_args(["--list-profiles"])
    assert args.list_profiles is True
    assert args.profile is None


@pytest.mark.parametrize(
    "group, expected",
    [("targeted", "main"), ("untargeted", "untargeted"), ("all", "all")],
)
def test_periodic_auto_mode_follows_attack_group(group, expected):
    assert run._periodic_mode("auto", group) == expected
    assert run._periodic_mode("ablation", group) == "ablation"


def test_rtc_profile_translates_common_plural_parameters(monkeypatch):
    captured = {}

    def fake_runner(args: argparse.Namespace) -> pd.DataFrame:
        captured.update(vars(args))
        return pd.DataFrame()

    monkeypatch.setattr(run.rtc_fedavg_comparison, "run_comparison", fake_runner)
    args = run.parse_args([
        "--profile", "rtc-fedavg",
        "--attack-groups", "untargeted",
        "--defenses", "fedavg,clip_only,rtc_full",
        "--seeds", "7",
        "--malicious-fractions", "0.4",
    ])

    run.PROFILES["rtc-fedavg"].runner(args)

    assert captured["seed"] == 7
    assert captured["malicious_fraction"] == pytest.approx(0.4)
    assert captured["attacks"] == "label_flip_all_reverse,byzantine,gaussian_noise"
    assert captured["periods"] == "continuous_1_0"
    assert captured["defenses"] == "fedavg,clip_only,rtc_full"


def test_rtc_profile_rejects_multiple_seeds():
    args = run.parse_args([
        "--profile", "rtc-fedavg", "--seeds", "42,43",
    ])
    with pytest.raises(ValueError, match="exactly one seed"):
        run.PROFILES["rtc-fedavg"].runner(args)


def test_rtc_profile_preserves_skip_clean(monkeypatch):
    captured = {}

    def fake_runner(args: argparse.Namespace) -> pd.DataFrame:
        captured.update(vars(args))
        return pd.DataFrame()

    monkeypatch.setattr(run.rtc_fedavg_comparison, "run_comparison", fake_runner)
    args = run.parse_args([
        "--profile", "rtc-fedavg",
        "--seeds", "42",
        "--malicious-fractions", "0.2",
        "--skip-clean",
    ])

    run.PROFILES["rtc-fedavg"].runner(args)

    assert captured["skip_clean"] is True


def test_override_parser_preserves_scalar_types():
    parsed = run._parse_overrides([
        "flag=true", "count=3", "rate=0.25", "name=rtc", "missing=null",
    ])
    assert parsed == {
        "flag": True, "count": 3, "rate": 0.25, "name": "rtc", "missing": None,
    }


def test_matrix_runner_defaults_to_guarded_ray_resources():
    args = run.parse_args(["--profile", "rtc-fedavg"])

    assert args.batch_size == 48
    assert args.num_clients == 20
    assert args.participation_rate == pytest.approx(0.5)
    assert args.ray_client_num_cpus == pytest.approx(1.0)
    assert args.ray_client_num_gpus == pytest.approx(0.0)
    assert args.ray_object_store_memory_mb == 3072
    assert args.ray_min_available_memory_mb == 10240
    assert args.ray_memory_wait_seconds == pytest.approx(120.0)
    assert args.max_spec_retries == 1


def test_sweep_profile_applies_batch48_and_ten_client_concurrency(monkeypatch):
    captured = {}

    def fake_sweep(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(run.sweep, "run_sweep", fake_sweep)
    args = run.parse_args([
        "--profile", "sweep",
        "--attacks", "none",
        "--defenses", "none",
        "--rounds", "1",
        "--pairing-mode", "legacy",
    ])

    run.PROFILES["sweep"].runner(args)

    overrides = captured["extra_overrides"]
    assert overrides["client.batch_size"] == 48
    assert overrides["federation.num_clients"] == 20
    assert overrides["federation.clients_per_round"] == 10
    assert overrides["federation.min_fit_clients"] == 10
    assert overrides["federation.min_available_clients"] == 20
    assert overrides["ray.client_num_cpus"] == pytest.approx(1.0)
    assert overrides["ray.client_num_gpus"] == pytest.approx(0.0)


def test_matrix_spec_config_consumes_optimized_execution_defaults(tmp_path):
    args = run.parse_args(["--profile", "rtc-fedavg", "--rounds", "1"])
    spec = {
        "custom_params": {},
        "defense_type": "none",
        "seed": 42,
        "partition": "iid",
        "dirichlet_alpha": 0.5,
        "participation_rate": 0.5,
        "attack": "none",
        "malicious_fraction": 0.0,
        "attack_start_round": 1,
        "attack_end_round": -1,
        "on_rounds": 1,
        "off_rounds": 0,
    }

    cfg = run.periodic_attack._build_spec_config(spec, args, tmp_path)

    assert cfg.client.batch_size == 48
    assert cfg.federation.num_clients == 20
    assert cfg.federation.clients_per_round == 10
    assert cfg.federation.min_fit_clients == 10
    assert cfg.ray.client_num_cpus == pytest.approx(1.0)
    assert cfg.ray.client_num_gpus == pytest.approx(0.0)


def test_matrix_runner_accepts_explicit_ray_resource_overrides():
    args = run.parse_args([
        "--profile", "rtc-fedavg",
        "--ray-object-store-memory-mb", "4096",
        "--ray-min-available-memory-mb", "12288",
        "--ray-memory-wait-seconds", "30",
        "--max-spec-retries", "0",
    ])

    assert args.ray_object_store_memory_mb == 4096
    assert args.ray_min_available_memory_mb == 12288
    assert args.ray_memory_wait_seconds == pytest.approx(30.0)
    assert args.max_spec_retries == 0


def test_matrix_runner_accepts_rtc_v3_contract_arguments():
    args = run.parse_args([
        "--profile", "rtc-fedavg",
        "--defenses", "fedavg,rtc_v3",
        "--rtc-v3-manifest", "calibration.json",
        "--rtc-v3-phase", "5",
        "--rtc-v3-principal-map", "principals.json",
        "--rtc-v3-parameter-roles", "roles.json",
    ])

    assert args.defenses == "fedavg,rtc_v3"
    assert args.rtc_v3_manifest == "calibration.json"
    assert args.rtc_v3_phase == 5
    assert args.rtc_v3_principal_map == "principals.json"
    assert args.rtc_v3_parameter_roles == "roles.json"
