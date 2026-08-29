"""Rebuild the complete Byzantine screen from immutable raw run artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import periodic_attack
from experiments.rtc_v3.byzantine import (
    CANONICAL_ATTACKS,
    annotate_screening_outcomes,
    select_screened_strengths,
)


TARGETED = {"scaling_backdoor", "label_flip_targeted", "dba"}


def _attack_group(attack: str) -> str:
    return "targeted" if attack in TARGETED else "untargeted"


def _strength(attack: str, cfg: dict[str, Any]) -> str | None:
    values = cfg.get("security", {}).get("attack", {})
    if attack == "gaussian_noise":
        value = float(values.get("gaussian_noise_std", math.nan))
        grid = {0.01: "weak", 0.05: "medium", 0.1: "strong"}
    elif attack == "random_noise":
        value = float(values.get("random_noise_scale", math.nan))
        grid = {1.0: "weak", 5.0: "medium", 10.0: "strong"}
    elif attack == "sign_flip":
        value = float(values.get("sign_flip_scale", math.nan))
        grid = {1.0: "weak", 5.0: "medium", 10.0: "strong"}
    elif attack in {"lie", "alie"}:
        value = float(values.get("lie_z", math.nan))
        grid = {0.25: "weak", 0.5: "medium", 1.0: "strong"}
    elif attack in {"min_max", "min_sum"}:
        value = float(values.get("optimization_gamma_fraction", math.nan))
        grid = {0.25: "weak", 0.5: "medium", 1.0: "strong"}
    elif attack in {"scaling_backdoor", "dba"}:
        poison = float(values.get("poison_fraction", math.nan))
        if bool(values.get("aggregation_aware_scaling", False)):
            gain = float(values.get("replacement_gain", math.nan))
            pairs = {
                (0.35, 0.1): "weak",
                (0.65, 0.2): "medium",
                (1.0, 0.3): "strong",
            }
            if attack == "dba":
                pairs = {
                    (0.4, 0.1): "weak",
                    (0.7, 0.2): "medium",
                    (1.0, 0.3): "strong",
                }
            return pairs.get((gain, poison))
        boost = float(values.get("dba_boost_factor", values.get("model_replacement_boost_factor", math.nan)))
        pairs = {
            (2.0, 0.1): "weak",
            (5.0, 0.3): "medium",
            (10.0, 0.5): "strong",
        }
        if attack == "dba":
            pairs = {(2.0, 0.1): "weak", (4.0, 0.2): "medium", (5.0, 0.3): "strong"}
        return pairs.get((boost, poison))
    elif attack in {"label_flip_targeted", "label_flip_all_reverse"}:
        value = float(values.get("label_flip_poison_fraction", math.nan))
        grid = {0.25: "weak", 0.5: "medium", 1.0: "strong"}
    else:
        return None
    for expected, level in grid.items():
        if math.isfinite(value) and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9):
            return level
    return None


def _spec_from_config(config: dict[str, Any], attack: str, level: str) -> dict[str, Any]:
    federation = config.get("federation", {})
    dataset = config.get("dataset", {})
    attack_cfg = config.get("security", {}).get("attack", {})
    spec: dict[str, Any] = {
        "benchmark_version": 4,
        "attack_group": _attack_group(attack),
        "attack": attack,
        "strength_level": level,
        "attack_version": "raw-artifact-reconstructed",
        "strength_source": "raw_config",
        "period": "continuous_1_0",
        "on_rounds": 1,
        "off_rounds": 0,
        "malicious_fraction": float(attack_cfg.get("malicious_fraction", 0.2)),
        "seed": int(config.get("project", {}).get("seed", 42)),
        "defense": "fedavg",
        "defense_type": "none",
        "attack_start_round": int(attack_cfg.get("attack_start_round", 1)),
        "attack_end_round": int(attack_cfg.get("attack_end_round", -1)),
        "partition": str(dataset.get("partition", "iid")),
        "dirichlet_alpha": float(dataset.get("dirichlet_alpha", 0.5)),
        "participation_rate": float(
            federation.get("clients_per_round", 10)
        ) / max(1, int(federation.get("num_clients", 20))),
        "pairing_mode": str(federation.get("pairing_mode", "strict")),
        "sampling_protocol": "principal_uniform",
        "trial_plan_hash": str(federation.get("trial_plan_hash", "")),
        "custom_params": {},
    }
    for key, value in attack_cfg.items():
        if key not in {"enabled", "type"}:
            spec[key] = value
    return spec


def _candidate_rank(attack: str, level: str, config: dict[str, Any], has_rounds: bool, path: Path) -> tuple[int, int, int]:
    values = config.get("security", {}).get("attack", {})
    current_dba = attack == "dba" and bool(
        values.get("aggregation_aware_scaling", False)
    ) and (
        (float(values.get("replacement_gain", -1)), float(values.get("poison_fraction", -1)))
        in {(0.4, 0.1), (0.7, 0.2), (1.0, 0.3)}
    )
    return (
        int(has_rounds),
        int(current_dba) if attack == "dba" else 0,
        int(path.stat().st_mtime),
    )


def _base_row(spec: dict[str, Any], run_id: str, state: str, source_config: Path) -> dict[str, Any]:
    row = dict(spec)
    row.update({
        "run_id": run_id,
        "state": state,
        "source_config": str(source_config),
        "source_rounds": "",
        "rounds_complete": False,
        "collapsed_invalid": state in {"diverged", "failed"},
        "numerical_divergence": state in {"diverged", "failed"},
        "asr_valid": False,
        "nan_rounds": math.nan,
        "invalid_asr_rounds": math.nan,
    })
    return row


def rebuild(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = root.resolve()
    raw = root / "raw"
    rounds = root / "rounds"
    candidates: dict[tuple[str, str], list[tuple[Path, dict[str, Any], Path | None]]] = {}
    for config_path in raw.glob("*_config.json"):
        run_id = config_path.name.removesuffix("_config.json")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        attack = str(config.get("security", {}).get("attack", {}).get("type", ""))
        if attack not in CANONICAL_ATTACKS:
            continue
        level = _strength(attack, config)
        if level is None:
            continue
        round_path = rounds / f"{run_id}.csv"
        candidates.setdefault((attack, level), []).append(
            (config_path, config, round_path if round_path.exists() else None)
        )

    rows: list[dict[str, Any]] = []
    selected_sources: dict[str, str] = {}
    for attack in CANONICAL_ATTACKS:
        for level in ("weak", "medium", "strong"):
            options = candidates.get((attack, level), [])
            if not options:
                rows.append({"attack": attack, "strength_level": level, "state": "missing", "collapsed_invalid": True})
                continue
            options.sort(
                key=lambda item: _candidate_rank(
                    attack, level, item[1], item[2] is not None, item[0]
                ),
                reverse=True,
            )
            config_path, config, round_path = options[0]
            run_id = config_path.name.removesuffix("_config.json")
            spec = _spec_from_config(config, attack, level)
            status_path = root / "status" / f"{run_id}.json"
            status = "completed" if round_path is not None else "missing_rounds"
            if status_path.exists():
                try:
                    status = str(json.loads(status_path.read_text()).get("state", status))
                except (OSError, ValueError):
                    pass
            row = _base_row(spec, run_id, status, config_path)
            if round_path is not None:
                frame = pd.read_csv(round_path)
                summary = periodic_attack.summarize_run(frame, spec)
                summary["run_id"] = run_id
                summary["state"] = status
                summary["source_config"] = str(config_path)
                summary["source_rounds"] = str(round_path)
                summary["rounds_complete"] = set(pd.to_numeric(frame["round"], errors="coerce").dropna().astype(int)) >= set(range(26))
                row = summary
                row["collapsed_invalid"] = bool(
                    row.get("collapsed_invalid", False)
                    or status in {"diverged", "failed"}
                )
            rows.append(row)
            selected_sources[f"{attack}:{level}"] = run_id

    matrix = pd.DataFrame(rows)
    # Restore one clean counterfactual per TrialPlan so untargeted accuracy
    # drops can be calculated against the exact paired baseline.
    selected_plans = {
        str(value)
        for value in matrix.loc[
            matrix["attack"].isin(CANONICAL_ATTACKS), "trial_plan_hash"
        ].dropna()
    }
    clean_candidates: dict[str, list[tuple[Path, dict[str, Any], Path | None]]] = {}
    for config_path in raw.glob("none*_config.json"):
        run_id = config_path.name.removesuffix("_config.json")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        plan = str(config.get("federation", {}).get("trial_plan_hash", ""))
        if plan not in selected_plans:
            continue
        round_path = rounds / f"{run_id}.csv"
        clean_candidates.setdefault(plan, []).append(
            (config_path, config, round_path if round_path.exists() else None)
        )
    for plan in sorted(selected_plans):
        options = clean_candidates.get(plan, [])
        if not options:
            continue
        options.sort(
            key=lambda item: (int(item[2] is not None), int(item[0].stat().st_mtime)),
            reverse=True,
        )
        config_path, config, round_path = options[0]
        run_id = config_path.name.removesuffix("_config.json")
        spec = _spec_from_config(config, "none", "clean")
        spec["attack_group"] = "clean"
        status_path = root / "status" / f"{run_id}.json"
        status = "completed" if round_path is not None else "missing_rounds"
        if status_path.exists():
            try:
                status = str(json.loads(status_path.read_text()).get("state", status))
            except (OSError, ValueError):
                pass
        if round_path is not None:
            frame = pd.read_csv(round_path)
            clean_row = periodic_attack.summarize_run(frame, spec)
            clean_row.update({
                "run_id": run_id,
                "state": status,
                "source_config": str(config_path),
                "source_rounds": str(round_path),
                "rounds_complete": set(pd.to_numeric(frame["round"], errors="coerce").dropna().astype(int)) >= set(range(26)),
                "collapsed_invalid": False,
            })
        else:
            clean_row = _base_row(spec, run_id, status, config_path)
        rows.append(clean_row)

    matrix = pd.DataFrame(rows)
    matrix = periodic_attack.add_accuracy_drop(matrix)
    matrix["attack_order"] = matrix["attack"].map({name: index for index, name in enumerate(CANONICAL_ATTACKS)})
    matrix["attack_order"] = matrix["attack_order"].fillna(-1)
    matrix["strength_order"] = matrix["strength_level"].map({"weak": 0, "medium": 1, "strong": 2, "clean": 3})
    matrix["strength_order"] = matrix["strength_order"].fillna(4)
    matrix = matrix.sort_values(["attack_order", "strength_order"]).drop(columns=["attack_order", "strength_order"])
    recommendations, _, screening_gates = select_screened_strengths(matrix)
    matrix = annotate_screening_outcomes(matrix, recommendations)
    matrix.attrs["selected_sources"] = selected_sources
    matrix.attrs["screening_gates"] = screening_gates
    return matrix, recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="logs/rtc_v3_byzantine_screen")
    args = parser.parse_args()
    root = Path(args.root)
    matrix, recommendations = rebuild(root)
    matrix.to_csv(root / "byzantine_attack_runs_reconstructed.csv", index=False)
    recommendations.to_csv(root / "attack_strength_screening_reconstructed.csv", index=False)
    reconstructed_gates = pd.DataFrame(matrix.attrs.get("screening_gates", []))
    original_gates_path = root / "quality_gates.csv"
    if original_gates_path.is_file():
        original_gates = pd.read_csv(original_gates_path)
        original_gates = original_gates[
            ~original_gates["gate"].astype(str).str.startswith("attack_strength:")
        ]
        reconstructed_gates = pd.concat(
            [original_gates, reconstructed_gates], ignore_index=True, sort=False
        )
    reconstructed_gates.to_csv(
        root / "quality_gates_reconstructed.csv", index=False
    )
    manifest = {
        "source": "raw config sidecars and rounds CSVs",
        "canonical_attacks": list(CANONICAL_ATTACKS),
        "rows": int(len(matrix)),
        "complete_round_rows": int(matrix["rounds_complete"].fillna(False).astype(bool).sum()),
        "collapsed_or_invalid_rows": int(matrix["collapsed_invalid"].fillna(False).astype(bool).sum()),
        "selected_sources": matrix.attrs.get("selected_sources", {}),
    }
    (root / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(root / "byzantine_attack_runs_reconstructed.csv")
    print(root / "attack_strength_screening_reconstructed.csv")


if __name__ == "__main__":
    main()
