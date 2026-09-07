"""Fail-closed analyzer for the 182-cell formal comparison.

Nine attacks use two seeds.  The explicit clean diagnostic is one FedAvg run
per seed.  The two seeds are a deliberately descriptive engineering comparison;
this module does not emit p-values or confidence intervals because n=2 is not a
credible inferential sample size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Allow the analyzer to be launched directly by absolute path from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.spec import attack_source_hash
from experiments import periodic_attack


ATTACKS = (
    "none",
    "gaussian_noise",
    "random_noise",
    "sign_flip",
    "lie",
    "min_max",
    "min_sum",
    "scaling_backdoor",
    "label_flip_all_reverse",
    "dba",
)
DEFENSES = (
    "fedavg",
    "rtc_cumulative_q_cap_accepted_anchor",
    "krum",
    "multi_krum",
    "trimmed_mean",
    "median",
    "foolsgold",
    "rfa",
    "freqfed",
    "fltrust",
)
SEEDS = (42, 46)
CLEAN_SEEDS = SEEDS
CLEAN_DEFENSES = ("fedavg",)
RTC = "rtc_cumulative_q_cap_accepted_anchor"
TARGETED = {"scaling_backdoor", "dba"}
EXPECTED_CELLS = len(CLEAN_DEFENSES) * len(CLEAN_SEEDS) + (len(ATTACKS) - 1) * len(DEFENSES) * len(SEEDS)
CLIENTS_PER_ROUND = 10


def _seeds_for_attack(attack: str) -> tuple[int, ...]:
    return CLEAN_SEEDS if attack == "none" else SEEDS


def _defenses_for_attack(attack: str) -> tuple[str, ...]:
    return CLEAN_DEFENSES if attack == "none" else DEFENSES


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _mean(frame: pd.DataFrame, column: str, default: float = math.nan) -> float:
    if column not in frame:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else default


def _rate(
    frame: pd.DataFrame,
    mask: pd.Series,
    column: str,
    value: str | None = None,
) -> float:
    if column not in frame:
        return math.nan
    selected = frame.loc[mask]
    if selected.empty:
        return math.nan
    if value is None:
        return float(selected[column].map(_truth).mean())
    return float(selected[column].astype(str).str.strip().str.lower().eq(value).mean())


def _expected_custom_params(spec: dict[str, Any]) -> None:
    if spec["defense"] != RTC:
        return
    custom = dict(spec.get("custom_params", {}))
    required = {
        "semantic_intervention_risk_floor": 0.5,
        "cumulative_q_cap_power": 1,
        "anchor_recycle_fraction": 0.51,
        "anchor_recycle_weighting": "accepted",
    }
    if any(custom.get(key) != value for key, value in required.items()):
        raise ValueError(f"RTC B3R-F0.51 parameter drift: {spec['attack']}/seed{spec['seed']}")
    if {"norm_clip_mad_k", "residual_rank_cap_top_k"}.intersection(custom):
        raise ValueError(f"Rejected B4/B5 parameter leaked into {spec['attack']}/seed{spec['seed']}")


def _validate_spec(spec: dict[str, Any], attack: str) -> None:
    if str(spec.get("attack")) != attack:
        raise ValueError(f"attack mismatch in {attack} manifest")
    if str(spec.get("defense")) not in _defenses_for_attack(attack) or int(spec.get("seed", -1)) not in _seeds_for_attack(attack):
        raise ValueError(f"unexpected defense/seed in {attack} manifest")
    if not math.isclose(float(spec.get("malicious_fraction", -1)), 0.3):
        raise ValueError("malicious_fraction must be 0.3")
    if str(spec.get("partition")) != "iid" or not math.isclose(float(spec.get("participation_rate", -1)), 0.5):
        raise ValueError("data/participation contract drift")
    if int(spec.get("attack_start_round", -1)) != 11 or int(spec.get("attack_end_round", 0)) != -1:
        raise ValueError("attack window must be rounds 11-60")
    if int(spec.get("krum_num_malicious", -1)) != 3:
        raise ValueError("Byzantine budget must be f=3")
    if spec["defense"] == "multi_krum" and int(spec.get("krum_num_to_select", -1)) != 5:
        raise ValueError("Multi-Krum must select 5 clients")
    if attack == "none":
        if spec.get("attack_group") != "clean" or spec.get("attack_parameter_status") != "not_applicable":
            raise ValueError("explicit clean contract drift")
    elif spec.get("attack_parameter_status") != "frozen":
        raise ValueError(f"attack parameters are not frozen: {attack}")
    if attack == "lie" and not math.isclose(float(spec.get("lie_z", -1)), 0.5):
        raise ValueError("LIE must use z=0.5")
    if attack == "sign_flip" and not math.isclose(float(spec.get("sign_flip_scale", -1)), 1.0):
        raise ValueError("Sign-flip must use the numerically valid scale=1")
    _expected_custom_params(spec)


def load_protocol(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock_path = root / "manifest_lock.json"
    lock = _read_json(lock_path)
    if lock.get("schema_version") != "RTCFormalAllAttacksTwoSeedLockV1":
        raise ValueError(f"unsupported or missing protocol lock: {lock_path}")
    protocol = lock.get("protocol", {})
    if tuple(protocol.get("attacks", [])) != ATTACKS:
        raise ValueError("locked attack inventory drifted")
    if tuple(protocol.get("defenses", [])) != DEFENSES:
        raise ValueError("locked defense inventory drifted")
    if (tuple(protocol.get("seeds", [])) != SEEDS or
            tuple(protocol.get("clean_seeds", [])) != CLEAN_SEEDS or
            tuple(protocol.get("clean_defenses", [])) != CLEAN_DEFENSES or
            int(protocol.get("total_cells", -1)) != EXPECTED_CELLS):
        raise ValueError("locked seed/cell count drifted")
    if not math.isclose(float(protocol.get("lie_z", -1)), 0.5):
        raise ValueError("locked LIE z drifted")

    freeze_path = root.parents[1] / "config" / "rtc_v3_formal_all_attacks_two_seed.freeze.json"
    freeze = _read_json(freeze_path)
    if freeze.get("implementation_source_sha256") != attack_source_hash():
        raise ValueError("attack freeze is stale relative to the current attack implementation")
    if set(freeze.get("attacks", {})) != set(ATTACKS).difference({"none"}):
        raise ValueError("attack freeze does not cover the exact canonical attack inventory")

    specs: list[dict[str, Any]] = []
    entries = list(lock.get("manifests", []))
    if len(entries) != len(ATTACKS):
        raise ValueError("protocol lock does not contain 11 submanifests")
    for entry in entries:
        attack = str(entry.get("attack"))
        if attack not in ATTACKS:
            raise ValueError(f"unknown locked attack: {attack}")
        batch_root = Path(str(entry["output"])).resolve()
        manifest_path = batch_root / "experiment_manifest.json"
        if _sha256(manifest_path) != str(entry.get("manifest_sha256")):
            raise ValueError(f"manifest hash mismatch: {attack}")
        batch_specs = list(_read_json(manifest_path).get("specs", []))
        attack_seeds = _seeds_for_attack(attack)
        attack_defenses = _defenses_for_attack(attack)
        expected_specs = len(attack_defenses) * len(attack_seeds)
        if len(batch_specs) != expected_specs:
            raise ValueError(f"expected {expected_specs} specs for {attack}, got {len(batch_specs)}")
        for spec in batch_specs:
            _validate_spec(spec, attack)
            spec["_batch_root"] = str(batch_root)
        for seed in attack_seeds:
            paired = [spec for spec in batch_specs if int(spec["seed"]) == seed]
            if {str(spec["defense"]) for spec in paired} != set(attack_defenses):
                raise ValueError(f"defense coverage mismatch: {attack}/seed{seed}")
            if len({str(spec["trial_plan_hash"]) for spec in paired}) != 1:
                raise ValueError(f"trial-plan mismatch: {attack}/seed{seed}")
            if len({str(spec["attack_implementation_hash"]) for spec in paired}) != 1:
                raise ValueError(f"attack implementation mismatch: {attack}/seed{seed}")
        specs.extend(batch_specs)
    keys = {(str(spec["attack"]), str(spec["defense"]), int(spec["seed"])) for spec in specs}
    expected = {
        (attack, defense, seed)
        for attack in ATTACKS
        for defense in _defenses_for_attack(attack)
        for seed in _seeds_for_attack(attack)
    }
    if len(specs) != EXPECTED_CELLS or keys != expected:
        raise ValueError("formal manifest matrix is not the exact 182-cell attack/clean design")
    return lock, specs


def _validate_quality_gates(specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_root_value in sorted({str(spec["_batch_root"]) for spec in specs}):
        batch_root = Path(batch_root_value)
        path = batch_root / "quality_gates.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing quality gates: {path}")
        gates = pd.read_csv(path)
        if gates.empty or not {"gate", "passed"}.issubset(gates.columns):
            raise ValueError(f"malformed quality gates: {path}")
        failed = gates.loc[~gates["passed"].map(_truth)]
        if not failed.empty:
            raise ValueError(f"failed quality gates in {batch_root}: {failed['gate'].astype(str).tolist()}")
        for row in gates.to_dict("records"):
            rows.append({"batch_root": str(batch_root), **row})
    return rows


def _metric_window(rounds: pd.DataFrame, attack: str) -> pd.DataFrame:
    round_values = pd.to_numeric(rounds["round"], errors="raise").astype(int)
    if attack == "none":
        active = rounds.loc[round_values.between(1, 60)].copy()
    else:
        active = rounds.loc[pd.to_numeric(rounds["planned_attack_active"], errors="coerce") == 1].copy()
    expected = 60 if attack == "none" else 50
    if len(active) != expected:
        raise ValueError(f"wrong metric window for attack={attack}: {len(active)} != {expected}")
    return active


def _load_run(spec: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    batch_root = Path(spec["_batch_root"])
    run_id = periodic_attack.run_id(spec)
    status_path = batch_root / "status" / f"{run_id}.json"
    status = _read_json(status_path)
    if status.get("state") not in {"completed", "completed_cached"} or int(status.get("exit_code", -1)) != 0 or int(status.get("last_round", -1)) != 60:
        raise ValueError(f"incomplete run: {run_id}")
    rounds_path = batch_root / "rounds" / f"{run_id}.csv"
    rounds = pd.read_csv(rounds_path)
    round_values = pd.to_numeric(rounds["round"], errors="raise").astype(int)
    if len(rounds) != 61 or set(round_values) != set(range(61)) or round_values.duplicated().any():
        raise ValueError(f"rounds must be unique 0-60: {run_id}")
    active = _metric_window(rounds, str(spec["attack"]))
    active_rounds = set(pd.to_numeric(active["round"], errors="raise").astype(int))

    clients_path = batch_root / "raw" / f"{run_id}_clients.csv"
    clients = pd.read_csv(clients_path)
    clients["round"] = pd.to_numeric(clients["round"], errors="raise").astype(int)
    clients = clients.loc[clients["round"].isin(active_rounds)].copy()
    if len(clients) != CLIENTS_PER_ROUND * len(active) or clients.duplicated(["round", "cid"]).any():
        raise ValueError(f"client diagnostics are incomplete or duplicated: {run_id}")
    benign = ~clients["is_malicious"].map(_truth)
    active_attackers = clients["attack_active"].map(_truth)
    nonattackers = ~active_attackers

    malicious_weight = _mean(active, "fit_malicious_aggregation_weight_share", 0.0)
    weight_sum = pd.to_numeric(active.get("fit_rtc_v3_weight_sum"), errors="coerce").dropna() if "fit_rtc_v3_weight_sum" in active else pd.Series(dtype=float)
    benign_weight = float(weight_sum.mean()) - malicious_weight if not weight_sum.empty else 1.0 - malicious_weight
    malicious_impact = _mean(active, "fit_malicious_impact_share", 0.0)
    server_asr = pd.to_numeric(active.get("server_asr"), errors="coerce").dropna() if "server_asr" in active else pd.Series(dtype=float)
    targeted = str(spec["attack"]) in TARGETED
    result = {
        "attack": str(spec["attack"]),
        "defense": str(spec["defense"]),
        "seed": int(spec["seed"]),
        "run_id": run_id,
        "source_root": str(batch_root),
        "trial_plan_hash": str(spec["trial_plan_hash"]),
        "attack_implementation_hash": str(spec["attack_implementation_hash"]),
        "metric_rounds": len(active),
        "mean_accuracy": _mean(active, "server_accuracy"),
        "final_accuracy": float(rounds.loc[round_values == 60, "server_accuracy"].iloc[0]),
        "mean_asr": float(server_asr.mean()) if targeted and not server_asr.empty else math.nan,
        "peak_asr": float(server_asr.max()) if targeted and not server_asr.empty else math.nan,
        "malicious_weight_share": malicious_weight,
        "benign_weight_share": benign_weight,
        "malicious_impact_share": malicious_impact,
        "benign_impact_share": 1.0 - malicious_impact,
        "zero_update_mass": _mean(active, "fit_rtc_v3_zero_update_mass", 0.0),
        "effective_update_mass": _mean(active, "fit_rtc_v3_effective_update_mass", 1.0),
        "anchor_recycle_mass": _mean(active, "fit_rtc_v3_anchor_recycle_mass", 0.0),
        "benign_clipping_rate": _rate(clients, benign, "clipped"),
        "benign_watch_rate": _rate(clients, benign, "state", "watch"),
        "benign_restricted_rate": _rate(clients, benign, "state", "restricted"),
        "benign_quarantined_rate": _rate(clients, benign, "state", "quarantined"),
        "nonattacker_clipping_rate": _rate(clients, nonattackers, "clipped"),
        "active_attacker_clipping_rate": _rate(clients, active_attackers, "clipped"),
        "active_attacker_watch_rate": _rate(clients, active_attackers, "state", "watch"),
        "active_attacker_restricted_rate": _rate(clients, active_attackers, "state", "restricted"),
        "active_attacker_quarantined_rate": _rate(clients, active_attackers, "state", "quarantined"),
        "mean_aggregation_seconds": _mean(active, "fit_aggregation_time_seconds"),
        "mean_train_seconds": _mean(active, "fit_train_time"),
    }
    return result, clients


def _add_clean_reference(runs: pd.DataFrame) -> pd.DataFrame:
    clean = runs.loc[runs["attack"].eq("none"), ["seed", "mean_accuracy", "final_accuracy"]].rename(
        columns={"mean_accuracy": "same_seed_fedavg_clean_mean_accuracy", "final_accuracy": "same_seed_fedavg_clean_final_accuracy"}
    )
    merged = runs.merge(clean, on=["seed"], how="left", validate="many_to_one")
    merged["mean_accuracy_drop_vs_same_seed_fedavg_clean"] = merged["same_seed_fedavg_clean_mean_accuracy"] - merged["mean_accuracy"]
    merged["final_accuracy_drop_vs_same_seed_fedavg_clean"] = merged["same_seed_fedavg_clean_final_accuracy"] - merged["final_accuracy"]
    return merged


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_accuracy", "final_accuracy", "mean_asr", "peak_asr",
        "mean_accuracy_drop_vs_same_seed_fedavg_clean", "final_accuracy_drop_vs_same_seed_fedavg_clean",
        "malicious_weight_share", "benign_weight_share", "malicious_impact_share",
        "benign_impact_share", "zero_update_mass", "effective_update_mass",
        "anchor_recycle_mass", "benign_clipping_rate", "benign_watch_rate",
        "benign_restricted_rate", "benign_quarantined_rate",
        "active_attacker_clipping_rate", "active_attacker_watch_rate",
        "active_attacker_restricted_rate", "active_attacker_quarantined_rate",
        "mean_aggregation_seconds", "mean_train_seconds",
    ]
    rows: list[dict[str, Any]] = []
    for (attack, defense), group in runs.groupby(["attack", "defense"], sort=False):
        row: dict[str, Any] = {"attack": attack, "defense": defense, "seeds": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = float(values.mean()) if not values.empty else math.nan
            row[f"sd_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            row[f"min_{metric}"] = float(values.min()) if not values.empty else math.nan
            row[f"max_{metric}"] = float(values.max()) if not values.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _rtc_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        if attack == "none":
            continue
        for seed in _seeds_for_attack(attack):
            selected = runs.loc[runs["attack"].eq(attack) & runs["seed"].eq(seed)].set_index("defense")
            rtc = selected.loc[RTC]
            for comparator in DEFENSES:
                if comparator == RTC:
                    continue
                other = selected.loc[comparator]
                rows.append(
                    {
                        "attack": attack,
                        "seed": seed,
                        "comparator": comparator,
                        "mean_accuracy_delta_rtc_minus_comparator": float(rtc["mean_accuracy"] - other["mean_accuracy"]),
                        "final_accuracy_delta_rtc_minus_comparator": float(rtc["final_accuracy"] - other["final_accuracy"]),
                        "mean_asr_improvement_comparator_minus_rtc": (
                            float(other["mean_asr"] - rtc["mean_asr"])
                            if attack in TARGETED else math.nan
                        ),
                        "peak_asr_improvement_comparator_minus_rtc": (
                            float(other["peak_asr"] - rtc["peak_asr"])
                            if attack in TARGETED else math.nan
                        ),
                        "malicious_weight_delta_rtc_minus_comparator": float(rtc["malicious_weight_share"] - other["malicious_weight_share"]),
                        "malicious_impact_delta_rtc_minus_comparator": float(rtc["malicious_impact_share"] - other["malicious_impact_share"]),
                        "zero_mass_delta_rtc_minus_comparator": float(rtc["zero_update_mass"] - other["zero_update_mass"]),
                        "benign_clipping_delta_rtc_minus_comparator": float(rtc["benign_clipping_rate"] - other["benign_clipping_rate"]),
                        "descriptive_only": True,
                    }
                )
    return pd.DataFrame(rows)


def _rankings(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        selected = summary.loc[summary["attack"].eq(attack)].copy()
        if attack in TARGETED:
            selected = selected.sort_values(["mean_asr", "mean_accuracy"], ascending=[True, False])
            primary = "mean_asr_ascending_then_accuracy"
        else:
            selected = selected.sort_values("mean_accuracy", ascending=False)
            primary = "mean_accuracy_descending"
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            rows.append(
                {
                    "attack": attack,
                    "rank": rank,
                    "defense": row["defense"],
                    "ranking_rule": primary,
                    "mean_accuracy": row["mean_accuracy"],
                    "mean_asr": row["mean_asr"],
                    "accuracy_drop_vs_same_seed_fedavg_clean": row["mean_accuracy_drop_vs_same_seed_fedavg_clean"],
                }
            )
    return pd.DataFrame(rows)


def _different(left: object, right: object, numeric: bool) -> bool:
    if pd.isna(left) and pd.isna(right):
        return False
    if pd.isna(left) != pd.isna(right):
        return True
    if numeric:
        return not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
    return str(left) != str(right)


def _first_divergences(client_frames: dict[tuple[str, str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = (
        ("aggregation_weight", True),
        ("effective_weight", True),
        ("clipped_delta_norm", True),
        ("clipped", False),
        ("state", False),
    )
    for attack in ATTACKS:
        if attack == "none":
            continue
        for seed in _seeds_for_attack(attack):
            rtc = client_frames[(attack, RTC, seed)].copy()
            for comparator in DEFENSES:
                if comparator == RTC:
                    continue
                other = client_frames[(attack, comparator, seed)].copy()
                merged = rtc.merge(other, on=["round", "cid"], suffixes=("_rtc", "_comparator"), validate="one_to_one")
                candidates: list[dict[str, Any]] = []
                for _, row in merged.sort_values(["round", "cid"]).iterrows():
                    changed = []
                    for field, numeric in fields:
                        left = row.get(f"{field}_rtc", math.nan)
                        right = row.get(f"{field}_comparator", math.nan)
                        if _different(left, right, numeric):
                            changed.append(field)
                    if changed:
                        candidates.append(
                            {
                                "round": int(row["round"]),
                                "cid": int(row["cid"]),
                                "is_malicious": _truth(row.get("is_malicious_rtc", False)),
                                "attack_active": _truth(row.get("attack_active_rtc", False)),
                                "changed_fields": "|".join(changed),
                            }
                        )
                        break
                first = candidates[0] if candidates else {}
                rows.append(
                    {
                        "attack": attack,
                        "seed": seed,
                        "comparator": comparator,
                        "first_metric_round": first.get("round", math.nan),
                        "first_metric_cid": first.get("cid", math.nan),
                        "first_metric_is_malicious": first.get("is_malicious", math.nan),
                        "first_metric_attack_active": first.get("attack_active", math.nan),
                        "changed_fields": first.get("changed_fields", ""),
                    }
                )
    return pd.DataFrame(rows)


def analyze(root: Path, output: Path, *, preflight_only: bool = False) -> None:
    root = root.resolve()
    output = output.resolve()
    lock, specs = load_protocol(root)
    if preflight_only:
        print(json.dumps({"preflight": "passed", "cells": len(specs), "attacks": len(ATTACKS) - 1, "defenses": len(DEFENSES), "attack_seeds": list(SEEDS), "clean_runs": len(CLEAN_SEEDS)}, indent=2))
        return

    merged_gates = _validate_quality_gates(specs)
    run_rows: list[dict[str, Any]] = []
    client_frames: dict[tuple[str, str, int], pd.DataFrame] = {}
    for spec in specs:
        result, clients = _load_run(spec)
        run_rows.append(result)
        client_frames[(result["attack"], result["defense"], result["seed"])] = clients
    runs = _add_clean_reference(pd.DataFrame(run_rows))
    if len(runs) != EXPECTED_CELLS or runs[["attack", "defense", "seed"]].duplicated().any():
        raise ValueError("analyzed run matrix is incomplete or duplicated")
    if runs[["mean_accuracy", "final_accuracy"]].isna().any().any():
        raise ValueError("primary accuracy metrics contain missing values")
    targeted = runs["attack"].isin(TARGETED)
    if runs.loc[targeted, ["mean_asr", "peak_asr"]].isna().any().any():
        raise ValueError("targeted attack ASR metrics contain missing values")

    summary = _summarize(runs)
    comparisons = _rtc_comparisons(runs)
    rankings = _rankings(summary)
    divergences = _first_divergences(client_frames)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "summary_by_attack_defense.csv", index=False)
    comparisons.to_csv(output / "rtc_vs_defenses_by_seed.csv", index=False)
    rankings.to_csv(output / "rankings.csv", index=False)
    divergences.to_csv(output / "first_rtc_divergence.csv", index=False)
    pd.DataFrame(merged_gates).to_csv(output / "merged_quality_gates.csv", index=False)
    decision = {
        "schema_version": "RTCFormalAllAttacksTwoSeedAnalysisV1",
        "status": "complete",
        "scope": {
            "seeds": list(SEEDS),
            "clean_seeds": list(CLEAN_SEEDS),
            "clean_defenses": list(CLEAN_DEFENSES),
            "attacks": list(ATTACKS),
            "defenses": list(DEFENSES),
            "cells": EXPECTED_CELLS,
            "malicious_fraction": 0.3,
            "rounds": 60,
            "attack_window": [11, 60],
            "lie_z": 0.5,
            "sign_flip_scale": 1.0,
        },
        "checks": {
            "all_cells_complete": True,
            "all_quality_gates_pass": True,
            "all_defenses_paired_by_trial_plan": True,
            "all_attack_implementation_hashes_match": True,
            "rtc_parameters_exact_b3r_f051": True,
            "all_primary_metrics_present": True,
            "all_rtc_comparators_have_first_divergence": len(divergences) == (
                len(DEFENSES) - 1
            ) * (len(ATTACKS) - 1) * len(SEEDS),
        },
        "inference": {
            "scope": "descriptive_two_seed_paired",
            "paired_seed_count": 2,
            "clean_seed_count": 2,
            "p_values_emitted": False,
            "confidence_intervals_emitted": False,
            "interpretation_guard": "Two seeds support engineering comparison only; they do not prove cross-seed superiority or equivalence.",
        },
        "protocol_lock_created_at": lock.get("created_at"),
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="logs/rtc_v3_formal_all_attacks_two_seed_mf03")
    parser.add_argument("--output", default="logs/rtc_v3_formal_all_attacks_two_seed_mf03/formal_analysis")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(Path(args.root), Path(args.output), preflight_only=args.preflight_only)
