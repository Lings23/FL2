"""Strictly paired RTC-V2/V3 defense-overhead benchmark.

This tool is diagnostic only.  It never writes promotion metadata or an
attestation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.periodic_attack import (
    _run_specs,
    run_id,
    validate_sampling_manifests,
    write_experiment_manifest,
)
from experiments.rtc_fedavg_comparison import build_rtc_v3_custom_params
from experiments.trial_plan import attach_trial_plans


def build_performance_specs(
    *,
    seeds: Sequence[int],
    baseline_manifest: str | Path,
    candidate_manifest: str | Path,
    num_clients: int,
    participation_rate: float,
    partition: str,
    dirichlet_alpha: float,
) -> list[dict[str, Any]]:
    definitions = {
        "rtc_v2_baseline": build_rtc_v3_custom_params(
            manifest_path=baseline_manifest,
            implementation_phase=6,
            num_clients=num_clients,
        ),
        "rtc_v3_semantic": build_rtc_v3_custom_params(
            manifest_path=candidate_manifest,
            implementation_phase=6,
            num_clients=num_clients,
        ),
    }
    return [
        {
            "benchmark_version": 3,
            "attack_group": "performance_clean",
            "attack": "none",
            "period": "performance_clean",
            "on_rounds": 1,
            "off_rounds": 0,
            "malicious_fraction": 0.0,
            "seed": int(seed),
            "defense": defense,
            "defense_type": "rtc_v3_candidate",
            "custom_params": dict(custom),
            "attack_start_round": 1,
            "attack_end_round": -1,
            "partition": str(partition),
            "dirichlet_alpha": float(dirichlet_alpha),
            "participation_rate": float(participation_rate),
            "boost_factor": 10.0,
            "calibration_role": "paired_performance_only",
        }
        for seed in seeds
        for defense, custom in definitions.items()
    ]


def audit_performance_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "round",
        "fit_trial_plan_hash",
        "fit_total_defense_seconds",
        "fit_solver_seconds",
    }
    if not required.issubset(baseline) or not required.issubset(candidate):
        raise ValueError("paired performance frames are missing timing/plan columns")
    candidate_required = {
        "fit_semantic_total_seconds",
        "fit_rtc_v3_cone_update_applied_count",
    }
    if not candidate_required.issubset(candidate):
        raise ValueError("semantic candidate frame is missing frozen/performance metrics")
    keys = ["fit_trial_plan_hash", "round"]
    left = baseline.loc[pd.to_numeric(baseline["round"], errors="coerce") > 0].copy()
    right = candidate.loc[pd.to_numeric(candidate["round"], errors="coerce") > 0].copy()
    paired = left[keys + ["fit_total_defense_seconds", "fit_solver_seconds"]].merge(
        right[
            keys
            + [
                "fit_total_defense_seconds",
                "fit_semantic_total_seconds",
                "fit_solver_seconds",
                "fit_rtc_v3_cone_update_applied_count",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("performance runs are not exactly paired by plan hash and round")
    base = paired["fit_total_defense_seconds_baseline"].to_numpy(dtype=np.float64)
    cand = paired["fit_total_defense_seconds_candidate"].to_numpy(dtype=np.float64)
    solver_increment = np.maximum(
        0.0,
        paired["fit_solver_seconds_candidate"].to_numpy(dtype=np.float64)
        - paired["fit_solver_seconds_baseline"].to_numpy(dtype=np.float64),
    )
    paired["semantic_solver_increment_seconds"] = solver_increment
    semantic = (
        paired["fit_semantic_total_seconds"].to_numpy(dtype=np.float64)
        + solver_increment
    )
    paired["semantic_total_with_solver_increment_seconds"] = semantic
    if (
        not np.all(np.isfinite(base))
        or not np.all(np.isfinite(cand))
        or not np.all(np.isfinite(semantic))
        or np.any(base <= 0.0)
        or np.any(cand <= 0.0)
        or np.any(semantic < 0.0)
    ):
        raise ValueError("performance timings must be finite and non-negative")
    semantic_mean = float(np.mean(semantic))
    semantic_p95 = float(np.quantile(semantic, 0.95))
    baseline_mean = float(np.mean(base))
    candidate_mean = float(np.mean(cand))
    baseline_p95 = float(np.quantile(base, 0.95))
    candidate_p95 = float(np.quantile(cand, 0.95))
    mean_increase = candidate_mean / baseline_mean - 1.0
    p95_increase = candidate_p95 / baseline_p95 - 1.0
    frozen = bool(
        np.all(
            paired["fit_rtc_v3_cone_update_applied_count"].to_numpy(
                dtype=np.float64
            )
            == 0.0
        )
    )
    gates = {
        "paired_rounds": int(len(paired)),
        "semantic_mean_seconds": semantic_mean,
        "semantic_p95_seconds": semantic_p95,
        "baseline_defense_mean_seconds": baseline_mean,
        "candidate_defense_mean_seconds": candidate_mean,
        "baseline_defense_p95_seconds": baseline_p95,
        "candidate_defense_p95_seconds": candidate_p95,
        "defense_mean_increase_fraction": mean_increase,
        "defense_p95_increase_fraction": p95_increase,
        "semantic_mean_gate": semantic_mean <= 0.15,
        "semantic_p95_gate": semantic_p95 <= 0.25,
        "defense_mean_increase_gate": mean_increase <= 0.05,
        "defense_p95_increase_gate": p95_increase <= 0.10,
        "online_cones_frozen_gate": frozen,
    }
    gates["passed"] = bool(
        gates["semantic_mean_gate"]
        and gates["semantic_p95_gate"]
        and gates["defense_mean_increase_gate"]
        and gates["defense_p95_increase_gate"]
        and gates["online_cones_frozen_gate"]
    )
    return gates, paired


def _load_defense_frames(
    specs: Sequence[Mapping[str, Any]], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {
        "rtc_v2_baseline": [],
        "rtc_v3_semantic": [],
    }
    for spec in specs:
        path = output / "rounds" / f"{run_id(dict(spec))}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing performance run: {path}")
        frame = pd.read_csv(path)
        frame["seed"] = int(spec["seed"])
        frames[str(spec["defense"])].append(frame)
    return (
        pd.concat(frames["rtc_v2_baseline"], ignore_index=True),
        pd.concat(frames["rtc_v3_semantic"], ignore_index=True),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--baseline-manifest",
        default="config/rtc_v3_manifest_formal_iid_cone_v2.json",
    )
    parser.add_argument(
        "--candidate-manifest",
        default="config/rtc_v3_manifest_formal_iid_semantic_candidate.json",
    )
    parser.add_argument("--output", default="logs/rtc_v3_semantic_performance")
    parser.add_argument("--seeds", default="43")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.5)
    parser.add_argument("--partition", default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-client-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--ray-client-num-cpus", type=float, default=1.0)
    parser.add_argument("--ray-client-num-gpus", type=float, default=0.25)
    parser.add_argument("--ray-object-store-memory-mb", type=int, default=3072)
    parser.add_argument("--ray-min-available-memory-mb", type=int, default=10240)
    parser.add_argument("--ray-memory-wait-seconds", type=float, default=120.0)
    parser.add_argument("--max-spec-retries", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    args.seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    args.pairing_mode = "strict"
    args.sampling_protocol = "principal_uniform"
    args.smoke = False
    args.dry_run = not args.run
    args.skip_completed = not args.rerun
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = build_performance_specs(
        seeds=args.seeds,
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
        num_clients=args.num_clients,
        participation_rate=args.participation_rate,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    specs = attach_trial_plans(specs, args, output)
    write_experiment_manifest(specs, output)
    if not args.run:
        print(f"Prepared {len(specs)} paired performance runs in {output}")
        return output
    runner_args = SimpleNamespace(**vars(args))
    rounds = output / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    summary = _run_specs(specs, runner_args, output, rounds)
    summary.to_csv(output / "performance_runs.csv", index=False)
    expected_ids = {run_id(spec) for spec in specs}
    strict = validate_sampling_manifests(
        rounds, output / "raw", expected_ids=expected_ids
    )
    pd.DataFrame(strict).to_csv(output / "execution_validation.csv", index=False)
    if not strict or not all(bool(item["passed"]) for item in strict):
        raise RuntimeError("paired performance strict gates failed")
    baseline, candidate = _load_defense_frames(specs, output)
    gates, paired = audit_performance_frames(baseline, candidate)
    paired.to_csv(output / "paired_round_timings.csv", index=False)
    (output / "performance_gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not gates["passed"]:
        raise RuntimeError("semantic performance gates failed")
    print(json.dumps(gates, indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    main()
