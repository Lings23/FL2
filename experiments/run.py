"""Unified command-line entry for repository experiments.

Experiment implementations stay in their focused modules; this file is the
stable public entry.  New reusable matrices should be registered as profiles
instead of adding another executable script.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import periodic_attack
from experiments import rtc_fedavg_comparison
from experiments import sweep
from experiments.rtc_v3 import byzantine


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    description: str
    runner: Callable[[argparse.Namespace], pd.DataFrame]


def _run_periodic(args: argparse.Namespace) -> pd.DataFrame:
    args.pairing_mode = args.pairing_mode or "strict"
    args.mode = _periodic_mode(args.mode, args.attack_groups)
    args.output = args.output or "logs/periodic_attack_v2"
    args.rounds = 60 if args.rounds is None else args.rounds
    if args.dry_run:
        matrix = periodic_attack.select_matrix(
            periodic_attack.build_matrix(args.mode, args.smoke), args
        )
        output = Path(args.output)
        if args.pairing_mode == "strict":
            matrix = periodic_attack.attach_trial_plans(matrix, args, output)
            clean = [] if args.skip_clean else periodic_attack.paired_fedavg_clean_specs(matrix)
        else:
            clean = [] if args.skip_clean else periodic_attack.clean_baselines(matrix, args.smoke)
        specs = periodic_attack._deduplicate(clean if args.clean_only else matrix + clean)
        periodic_attack.write_experiment_manifest(specs, output)
        _write_matrix(specs, output / "experiment_matrix.csv")
        logger.info("Dry run: periodic profile selected %d runs", len(specs))
        return pd.DataFrame()
    return periodic_attack.run_experiments(args)


def _run_rtc_fedavg(args: argparse.Namespace) -> pd.DataFrame:
    args.pairing_mode = args.pairing_mode or "strict"
    args.output = args.output or "logs/rtc_fedavg_60round"
    args.rounds = 60 if args.rounds is None else args.rounds
    seeds = _csv(args.seeds or "42")
    fractions = _csv(args.malicious_fractions or "0.2")
    if len(seeds) != 1 or len(fractions) != 1:
        raise ValueError("rtc-fedavg profile requires exactly one seed and one malicious fraction")
    args.seed = int(seeds[0])
    args.malicious_fraction = float(fractions[0])
    args.attacks = args.attacks or ",".join(_attacks_for_group(args.attack_groups))
    args.periods = args.periods or ",".join(rtc_fedavg_comparison.DEFAULT_PERIODS)
    args.defenses = args.defenses or ",".join(rtc_fedavg_comparison.DEFAULT_DEFENSES)
    args.smoke = False
    args.mode = "rtc_fedavg_comparison"
    # ``rtc-fedavg`` normally adds one paired FedAvg-clean counterfactual for
    # every attack-specific TrialPlan.  Keep the public ``--skip-clean`` flag
    # effective so a second defense-only pass (for example an old-manifest
    # baseline) can reuse the already evaluated shared counterfactual instead
    # of training a duplicate trajectory.
    args.skip_clean = bool(getattr(args, "skip_clean", False))
    args.clean_only = False
    return rtc_fedavg_comparison.run_comparison(args)


def _run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    args.pairing_mode = args.pairing_mode or "strict"
    attacks = _csv(args.attacks) if args.attacks else (
        ["none", "label_flip_targeted"] if args.quick else list(sweep.DEFAULT_ATTACKS)
    )
    defenses = _csv(args.defenses) if args.defenses else (
        ["none", "krum"] if args.quick else list(sweep.DEFAULT_DEFENSES)
    )
    seeds = [int(value) for value in _csv(args.seeds or "42")]
    rounds = 5 if args.quick else (50 if args.rounds is None else args.rounds)
    output = args.output or "logs/sweep"
    overrides = _parse_overrides(args.override)
    num_clients = int(args.num_clients)
    clients_per_round = max(
        1,
        min(num_clients, int(math.ceil(num_clients * float(args.participation_rate)))),
    )
    overrides.setdefault("client.batch_size", args.batch_size)
    overrides.setdefault("federation.num_clients", num_clients)
    overrides.setdefault("federation.clients_per_round", clients_per_round)
    overrides.setdefault("federation.min_fit_clients", clients_per_round)
    overrides.setdefault("federation.min_available_clients", num_clients)
    overrides.setdefault("ray.client_num_cpus", args.ray_client_num_cpus)
    overrides.setdefault("ray.client_num_gpus", args.ray_client_num_gpus)
    overrides.setdefault("ray.force_cpu", args.force_cpu)
    overrides.setdefault("ray.object_store_memory_mb", args.ray_object_store_memory_mb)
    overrides.setdefault("ray.min_available_memory_mb", args.ray_min_available_memory_mb)
    overrides.setdefault("ray.memory_wait_seconds", args.ray_memory_wait_seconds)
    overrides.setdefault("security.attack.source_label", args.label_flip_source_label)
    overrides.setdefault("security.attack.target_label", args.label_flip_target_label)
    overrides.setdefault(
        "security.attack.label_flip_poison_fraction", args.label_flip_poison_fraction
    )
    if args.pairing_mode == "strict":
        return _run_strict_sweep(
            args=args,
            attacks=attacks,
            defenses=defenses,
            seeds=seeds,
            rounds=rounds,
            output=Path(output),
        )
    if args.dry_run:
        rows = [
            {"attack": attack, "defense": defense, "seed": seed, "rounds": rounds}
            for attack in attacks for defense in defenses for seed in seeds
        ]
        _write_matrix(rows, Path(output) / "experiment_matrix.csv")
        logger.info("Dry run: sweep profile selected %d runs", len(rows))
        return pd.DataFrame(rows)
    return sweep.run_sweep(
        config_path=args.config,
        attacks=attacks,
        defenses=defenses,
        num_rounds=rounds,
        output_dir=output,
        extra_overrides=overrides,
        seeds=seeds,
    )


def _run_strict_sweep(
    *,
    args: argparse.Namespace,
    attacks: Sequence[str],
    defenses: Sequence[str],
    seeds: Sequence[int],
    rounds: int,
    output: Path,
) -> pd.DataFrame:
    """Run the generic grid through the same strict protocol as formal matrices."""
    attacked_names = [name for name in attacks if name != "none"]
    if not attacked_names:
        raise ValueError(
            "Strict sweep clean runs are attack-plan counterfactuals; select at least one attack"
        )
    periods = _csv(args.periods) if args.periods else ["continuous_1_0"]
    unknown_periods = set(periods).difference(periodic_attack.PERIODS)
    if unknown_periods:
        raise ValueError(f"Unsupported strict-sweep periods: {sorted(unknown_periods)}")
    fractions = [
        float(value) for value in _csv(args.malicious_fractions or "0.2")
    ]
    normalized_defenses = ["fedavg" if name == "none" else name for name in defenses]
    definitions = {**periodic_attack.DEFENSES, **periodic_attack.ABLATIONS}
    rows: list[dict[str, Any]] = []
    for attack in attacked_names:
        for period in periods:
            on_rounds, off_rounds = periodic_attack.PERIODS[period]
            for fraction in fractions:
                for seed in seeds:
                    for defense in normalized_defenses:
                        if defense in definitions:
                            defense_type, custom = definitions[defense]
                        else:
                            defense_type, custom = defense, {}
                        row: dict[str, Any] = {
                            "benchmark_version": periodic_attack.BENCHMARK_VERSION,
                            "attack_group": (
                                "targeted"
                                if attack in periodic_attack.TARGETED_ATTACKS
                                else "untargeted"
                            ),
                            "attack": attack,
                            "period": period,
                            "on_rounds": int(on_rounds),
                            "off_rounds": int(off_rounds),
                            "malicious_fraction": float(fraction),
                            "seed": int(seed),
                            "defense": defense,
                            "defense_type": defense_type,
                            "custom_params": dict(custom),
                            "attack_start_round": int(args.attack_start_round),
                            "attack_end_round": int(args.attack_end_round),
                            "partition": args.partition,
                            "dirichlet_alpha": float(args.dirichlet_alpha),
                            "participation_rate": float(args.participation_rate),
                            "boost_factor": float(args.boost_factor),
                        }
                        if attack in periodic_attack.LABEL_FLIP_ATTACKS:
                            row["label_flip_poison_fraction"] = float(
                                args.label_flip_poison_fraction
                            )
                            if attack == periodic_attack.LABEL_FLIP_TARGETED:
                                row["label_flip_source_label"] = int(
                                    args.label_flip_source_label
                                )
                                row["label_flip_target_label"] = int(
                                    args.label_flip_target_label
                                )
                        rows.append(row)
    args.rounds = int(rounds)
    args.smoke = False
    attacked = periodic_attack.attach_trial_plans(rows, args, output)
    clean = periodic_attack.paired_fedavg_clean_specs(attacked)
    specs = periodic_attack._deduplicate([*attacked, *clean])
    periodic_attack.write_experiment_manifest(specs, output)
    _write_matrix(specs, output / "experiment_matrix.csv")
    if args.dry_run:
        logger.info("Dry run: strict sweep selected %d runs", len(specs))
        return pd.DataFrame(specs)

    rounds_dir = output / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    summary = periodic_attack._run_specs(specs, args, output, rounds_dir)
    summary = periodic_attack.add_accuracy_drop_auc(
        periodic_attack.add_accuracy_drop(summary), rounds_dir
    )
    expected_ids = set(summary["run_id"].astype(str))
    gates = periodic_attack.validate_sampling_manifests(
        rounds_dir, output / "raw", expected_ids=expected_ids
    )
    periodic_attack.write_quality_gates(gates, output / "quality_gates.csv")
    failed = [gate for gate in gates if not gate["passed"]]
    if failed:
        raise RuntimeError(
            "Strict sweep pairing gates failed: "
            + "; ".join(str(gate["gate"]) for gate in failed)
        )
    summary.to_csv(output / "sweep_results.csv", index=False)
    periodic_attack.bootstrap_summary(summary).to_csv(
        output / "sweep_summary.csv", index=False
    )
    periodic_attack.statistical_tests(summary).to_csv(
        output / "statistical_tests.csv", index=False
    )
    return summary


def _run_rtc_byzantine(args: argparse.Namespace) -> pd.DataFrame:
    fractions = _csv(args.malicious_fractions or "0.2")
    if len(fractions) != 1:
        raise ValueError("rtc-byzantine requires exactly one malicious fraction")
    args.malicious_fraction = float(fractions[0])
    return byzantine.run(args)


def _run_rtc_byzantine_screen(args: argparse.Namespace) -> pd.DataFrame:
    fractions = _csv(args.malicious_fractions or "0.2")
    if len(fractions) != 1:
        raise ValueError("rtc-byzantine-screen requires exactly one malicious fraction")
    if len(_csv(args.seeds or "42")) != 1:
        raise ValueError("rtc-byzantine-screen requires exactly one seed")
    args.malicious_fraction = float(fractions[0])
    args.byzantine_screening = True
    args.defenses = "fedavg"
    args.output = args.output or "logs/rtc_v3_byzantine_screen"
    return byzantine.run(args)


PROFILES = {
    profile.name: profile for profile in (
        ExperimentProfile(
            "periodic",
            "General periodic benchmark, including the existing main/untargeted/ablation modes.",
            _run_periodic,
        ),
        ExperimentProfile(
            "rtc-fedavg",
            "Single-seed continuous-attack RTC comparison with objective-aware reports.",
            _run_rtc_fedavg,
        ),
        ExperimentProfile(
            "sweep",
            "Generic attack-by-defense grid sweep.",
            _run_sweep,
        ),
        ExperimentProfile(
            "rtc-byzantine",
            "Strictly paired generalized-Byzantine matrix for promoted RTC-V3.",
            _run_rtc_byzantine,
        ),
        ExperimentProfile(
            "rtc-byzantine-screen",
            "FedAvg-only weak/medium/strong Byzantine attack screening.",
            _run_rtc_byzantine_screen,
        ),
    )
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified FedSec experiment runner")
    parser.add_argument("--profile", choices=tuple(PROFILES), help="Registered experiment profile")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--attacks", default="", help="Comma-separated attack names")
    parser.add_argument(
        "--strength-levels",
        default="",
        help="Comma-separated weak/medium/strong filter for rtc-byzantine-screen",
    )
    parser.add_argument(
        "--attack-groups", choices=("targeted", "untargeted", "all"), default="all",
        help="Used when --attacks is omitted",
    )
    parser.add_argument("--defenses", default="", help="Comma-separated defense/profile names")
    parser.add_argument("--periods", default="", help="Comma-separated periodic schedules")
    parser.add_argument(
        "--rtc-v3-manifest",
        default=str(rtc_fedavg_comparison.RTC_V3_PROMOTED_MANIFEST),
        help="RTC-v3 manifest (defaults to the engineering-promoted manifest)",
    )
    parser.add_argument(
        "--rtc-v3-phase", type=int, default=6, choices=range(1, 7),
        help="RTC-v3 implementation phase (1-6)",
    )
    parser.add_argument(
        "--rtc-v3-principal-map", default="",
        help="Optional JSON client-to-principal mapping",
    )
    parser.add_argument(
        "--rtc-v3-parameter-roles", default="",
        help="Optional JSON parameter-index-to-role mapping",
    )
    parser.add_argument(
        "--byzantine-attack-freeze",
        default="",
        help="Frozen RTCByzantineAttackFreezeV1 JSON for formal rtc-byzantine runs",
    )
    parser.add_argument("--seeds", "--seed", dest="seeds", default="")
    parser.add_argument(
        "--malicious-fractions", "--malicious-fraction",
        dest="malicious_fractions", default="",
    )
    parser.add_argument(
        "--mode", choices=("auto", "main", "ablation", "untargeted", "mpaf", "all"), default="auto",
        help="Periodic profile matrix mode",
    )
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument(
        "--batch-size", type=int, default=48,
        help="Local client batch size (optimized default: 48)",
    )
    parser.add_argument("--partition", choices=("iid", "non_iid", "dirichlet"), default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--boost-factor", type=float, default=10.0)
    parser.add_argument("--label-flip-source-label", type=int, default=5)
    parser.add_argument("--label-flip-target-label", type=int, default=3)
    parser.add_argument("--label-flip-poison-fraction", type=float, default=1.0)
    parser.add_argument("--attack-start-round", type=int, default=11)
    parser.add_argument("--attack-end-round", type=int, default=-1)
    parser.add_argument("--max-client-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--ray-client-num-cpus", type=float, default=1.0)
    parser.add_argument("--ray-client-num-gpus", type=float, default=0.0)
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="Force the server and all Ray clients to use CPU; independent of the GPU resource quota",
    )
    parser.add_argument("--ray-object-store-memory-mb", type=int, default=3072)
    parser.add_argument("--ray-min-available-memory-mb", type=int, default=10240)
    parser.add_argument("--ray-memory-wait-seconds", type=float, default=120.0)
    parser.add_argument("--max-spec-retries", type=int, default=1)
    parser.add_argument(
        "--pairing-mode", choices=("strict", "legacy"), default=None,
        help="All experiment profiles default to strict; use legacy only for compatibility",
    )
    parser.add_argument(
        "--sampling-protocol",
        choices=("principal_uniform", "endpoint_uniform"),
        default="principal_uniform",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Periodic six-round smoke matrix")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Sweep five-round minimal grid")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.rounds is not None and args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_clients <= 0:
        parser.error("--num-clients must be positive")
    if not 0.0 < args.participation_rate <= 1.0:
        parser.error("--participation-rate must be in (0, 1]")
    if args.ray_client_num_cpus <= 0:
        parser.error("--ray-client-num-cpus must be positive")
    if args.ray_client_num_gpus < 0:
        parser.error("--ray-client-num-gpus must be non-negative")
    if args.ray_object_store_memory_mb < 0:
        parser.error("--ray-object-store-memory-mb must be non-negative")
    if args.ray_min_available_memory_mb < 0:
        parser.error("--ray-min-available-memory-mb must be non-negative")
    if args.ray_memory_wait_seconds < 0:
        parser.error("--ray-memory-wait-seconds must be non-negative")
    if args.max_spec_retries < 0:
        parser.error("--max-spec-retries must be non-negative")
    if not 0.0 <= args.label_flip_poison_fraction <= 1.0:
        parser.error("--label-flip-poison-fraction must be in [0, 1]")
    if not args.list_profiles and not args.profile:
        parser.error("--profile is required; use --list-profiles to inspect choices")
    return args


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    if args.list_profiles:
        for profile in PROFILES.values():
            print(f"{profile.name:12} {profile.description}")
        return pd.DataFrame()
    assert args.profile is not None
    logger.info("Experiment profile: %s", args.profile)
    return PROFILES[args.profile].runner(args)


def _periodic_mode(mode: str, attack_group: str) -> str:
    if mode != "auto":
        return mode
    return {"targeted": "main", "untargeted": "untargeted", "all": "all"}[attack_group]


def _attacks_for_group(group: str) -> tuple[str, ...]:
    if group == "targeted":
        return tuple(periodic_attack.TARGETED_ATTACKS)
    if group == "untargeted":
        return tuple(periodic_attack.UNTARGETED_ATTACKS)
    return tuple((*periodic_attack.TARGETED_ATTACKS, *periodic_attack.UNTARGETED_ATTACKS))


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        key, separator, raw = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid override: {item!r}; expected KEY=VALUE")
        result[key.strip()] = _parse_scalar(raw.strip())
    return result


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _write_matrix(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        serializable.append({
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
    pd.DataFrame(serializable).to_csv(path, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
