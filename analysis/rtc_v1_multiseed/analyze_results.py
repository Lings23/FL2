"""Build the final paired V1 RTC B3R-F0.51 versus Multi-Krum evidence tables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import periodic_attack  # noqa: E402


SEEDS = (42, 46, 47)
RTC = "rtc_cumulative_q_cap_accepted_anchor"
MULTI_KRUM = "multi_krum"
CONDITIONS = ("clean", "lie_z025", "lie_z05", "dba")
PAIR_METRICS = (
    "mean_accuracy",
    "final_accuracy",
    "mean_asr",
    "malicious_weight_share",
    "benign_weight_share",
    "malicious_impact_share",
    "benign_impact_share",
    "zero_update_mass",
    "effective_update_mass",
    "anchor_recycle_mass",
    "benign_clipping_rate",
    "benign_watch_rate",
    "benign_restricted_rate",
    "benign_quarantined_rate",
    "nonattacker_clipping_rate",
    "nonattacker_watch_rate",
    "nonattacker_restricted_rate",
    "nonattacker_quarantined_rate",
    "active_attacker_clipping_rate",
    "active_attacker_watch_rate",
    "active_attacker_restricted_rate",
    "active_attacker_quarantined_rate",
)
DEFAULT_OUTPUT = ROOT / "analysis" / "rtc_v1_multiseed"
DEFAULT_DIRS = {
    "clean": ROOT / "logs" / "rtc_v3_v1_clean_seeds42_46_47_mf03",
    "lie_z025": ROOT / "logs" / "rtc_v3_v1_lie_z025_seeds42_46_47_mf03",
    "seed42": ROOT / "logs" / "rtc_v3_v1_lie_z05_dba_seed42_multikrum_mf03",
    "remaining": ROOT / "logs" / "rtc_v3_v1_lie_z05_dba_seeds46_47_mf03",
    "lie_z05_rtc_seed42": ROOT
    / "logs"
    / "rtc_v3_residual_rank_cap_b4_lie_baseline_seed42_mf03",
    "dba_rtc_seed42": ROOT / "logs" / "rtc_v3_anchor_recycle_b3r_f051_seed42_mf03",
}


def _load_specs(directory: Path) -> list[dict[str, object]]:
    path = directory / "experiment_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(spec) for spec in payload["specs"]]


def _all_quality_gates_pass(directory: Path) -> None:
    path = directory / "quality_gates.csv"
    frame = pd.read_csv(path)
    if frame.empty or not frame["passed"].astype(str).str.lower().eq("true").all():
        raise RuntimeError(f"quality gates are missing or failed: {directory}")


def _find_spec(
    specs: list[dict[str, object]], *, attack: str, defense: str, seed: int, lie_z: float | None
) -> dict[str, object]:
    matches = [
        spec
        for spec in specs
        if str(spec["attack"]) == attack
        and str(spec["defense"]) == defense
        and int(spec["seed"]) == seed
        and (lie_z is None or math.isclose(float(spec["lie_z"]), lie_z))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one spec for attack={attack} defense={defense} seed={seed} "
            f"lie_z={lie_z}; got {len(matches)}"
        )
    return matches[0]


def _source_key(condition: str, defense: str, seed: int) -> str:
    if condition in {"clean", "lie_z025"}:
        return condition
    if seed != 42:
        return "remaining"
    if defense == MULTI_KRUM:
        return "seed42"
    return "lie_z05_rtc_seed42" if condition == "lie_z05" else "dba_rtc_seed42"


def _condition_contract(condition: str) -> tuple[str, float | None]:
    if condition == "clean":
        return "none", None
    if condition == "lie_z025":
        return "lie", 0.25
    if condition == "lie_z05":
        return "lie", 0.5
    return "dba", None


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _numeric_mean(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> float:
    if column not in frame:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else default


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _client_rate(frame: pd.DataFrame, mask: pd.Series, column: str, value: str | None = None) -> float:
    selected = frame.loc[mask]
    if selected.empty or column not in selected:
        return math.nan
    if value is None:
        hits = _bool_series(selected[column])
    else:
        hits = selected[column].astype(str).str.strip().str.lower().eq(value)
    return float(hits.mean())


def _client_metrics(frame: pd.DataFrame, active_rounds: set[int]) -> dict[str, float]:
    clients = frame.copy()
    clients["round"] = pd.to_numeric(clients["round"], errors="raise").astype(int)
    clients = clients[clients["round"].isin(active_rounds)].copy()
    expected_rows = 10 * len(active_rounds)
    if len(clients) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} client rows in metric window, got {len(clients)}"
        )
    if clients.duplicated(["round", "cid"]).any():
        raise RuntimeError("duplicate round/cid rows in client diagnostics")
    identities_benign = ~_bool_series(clients["is_malicious"])
    active_attackers = _bool_series(clients["attack_active"])
    nonattackers = ~active_attackers
    return {
        "benign_clipping_rate": _client_rate(clients, identities_benign, "clipped"),
        "benign_watch_rate": _client_rate(clients, identities_benign, "state", "watch"),
        "benign_restricted_rate": _client_rate(
            clients, identities_benign, "state", "restricted"
        ),
        "benign_quarantined_rate": _client_rate(
            clients, identities_benign, "state", "quarantined"
        ),
        "nonattacker_clipping_rate": _client_rate(clients, nonattackers, "clipped"),
        "nonattacker_watch_rate": _client_rate(clients, nonattackers, "state", "watch"),
        "nonattacker_restricted_rate": _client_rate(
            clients, nonattackers, "state", "restricted"
        ),
        "nonattacker_quarantined_rate": _client_rate(
            clients, nonattackers, "state", "quarantined"
        ),
        "active_attacker_clipping_rate": _client_rate(
            clients, active_attackers, "clipped"
        ),
        "active_attacker_watch_rate": _client_rate(
            clients, active_attackers, "state", "watch"
        ),
        "active_attacker_restricted_rate": _client_rate(
            clients, active_attackers, "state", "restricted"
        ),
        "active_attacker_quarantined_rate": _client_rate(
            clients, active_attackers, "state", "quarantined"
        ),
    }


def _validate_spec(spec: dict[str, object], condition: str, defense: str) -> None:
    if float(spec["malicious_fraction"]) != 0.3:
        raise RuntimeError(f"{condition}/{defense}: malicious_fraction drifted")
    if str(spec.get("partition")) != "iid" or float(spec["participation_rate"]) != 0.5:
        raise RuntimeError(f"{condition}/{defense}: data or participation contract drifted")
    if int(spec["krum_num_malicious"]) != 3:
        raise RuntimeError(f"{condition}/{defense}: Byzantine budget is not f=3")
    if condition == "clean":
        if str(spec["attack"]) != "none":
            raise RuntimeError("clean condition is not explicit attack=none")
    elif condition.startswith("lie"):
        expected = 0.25 if condition == "lie_z025" else 0.5
        if not math.isclose(float(spec["lie_z"]), expected):
            raise RuntimeError(f"{condition}: LIE z drifted")
    else:
        if not (
            math.isclose(float(spec["replacement_gain"]), 1.0)
            and math.isclose(float(spec["poison_fraction"]), 0.3)
            and bool(spec["dba_scale_update"])
        ):
            raise RuntimeError("DBA strong contract drifted")
    custom = dict(spec.get("custom_params", {}))
    if defense == RTC:
        required = {
            "semantic_intervention_risk_floor": 0.5,
            "cumulative_q_cap_power": 1,
            "anchor_recycle_fraction": 0.51,
            "anchor_recycle_weighting": "accepted",
        }
        if any(custom.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"{condition}: frozen RTC B3R-F0.51 parameters drifted")
        forbidden = {"norm_clip_mad_k", "residual_rank_cap_top_k"}
        if forbidden.intersection(custom):
            raise RuntimeError(f"{condition}: rejected B4/B5 parameter leaked into V1")
    elif not (
        str(spec["defense_type"]) == "krum"
        and int(spec["krum_num_to_select"]) == 5
    ):
        raise RuntimeError(f"{condition}: Multi-Krum contract drifted")


def _load_run(directory: Path, spec: dict[str, object], condition: str) -> dict[str, object]:
    run_id = periodic_attack.run_id(spec)
    status = json.loads((directory / "status" / f"{run_id}.json").read_text(encoding="utf-8"))
    if not (
        status.get("state") in {"completed", "completed_cached"}
        and int(status.get("exit_code", -1)) == 0
        and int(status.get("last_round", -1)) == 60
    ):
        raise RuntimeError(f"incomplete V1 run: {run_id}")
    frame = pd.read_csv(directory / "rounds" / f"{run_id}.csv")
    rounds = pd.to_numeric(frame["round"], errors="raise").astype(int)
    if len(frame) != 61 or set(rounds) != set(range(61)):
        raise RuntimeError(f"expected rounds 0-60 for {run_id}")
    if condition == "clean":
        active = frame[rounds.between(1, 60)]
    else:
        active = frame[pd.to_numeric(frame["planned_attack_active"], errors="coerce") == 1]
    if len(active) != (60 if condition == "clean" else 50):
        raise RuntimeError(f"wrong metric window for {run_id}: {len(active)}")
    active_rounds = set(pd.to_numeric(active["round"], errors="raise").astype(int))
    accuracy = pd.to_numeric(active["server_accuracy"], errors="raise")
    final_accuracy = float(
        pd.to_numeric(frame.loc[rounds == 60, "server_accuracy"], errors="raise").iloc[0]
    )
    asr = _numeric_series(active, "server_asr")
    malicious_weight = _numeric_mean(active, "fit_malicious_aggregation_weight_share")
    malicious_impact = _numeric_mean(active, "fit_malicious_impact_share")
    weight_sum = _numeric_series(active, "fit_rtc_v3_weight_sum").dropna()
    benign_weight = (
        float(weight_sum.mean()) - malicious_weight
        if not weight_sum.empty
        else 1.0 - malicious_weight
    )
    client_path = directory / "raw" / f"{run_id}_clients.csv"
    if not client_path.exists():
        raise RuntimeError(f"missing client diagnostics: {client_path}")
    result = {
        "run_id": run_id,
        "active_rounds": len(active),
        "mean_accuracy": float(accuracy.mean()),
        "final_accuracy": final_accuracy,
        "mean_asr": float(asr.mean()) if condition == "dba" and asr.notna().any() else math.nan,
        "peak_asr": float(asr.max()) if condition == "dba" and asr.notna().any() else math.nan,
        "malicious_weight_share": malicious_weight,
        "benign_weight_share": benign_weight,
        "malicious_impact_share": malicious_impact,
        "benign_impact_share": 1.0 - malicious_impact,
        "zero_update_mass": _numeric_mean(active, "fit_rtc_v3_zero_update_mass"),
        "effective_update_mass": _numeric_mean(
            active, "fit_rtc_v3_effective_update_mass", default=1.0
        ),
        "anchor_recycle_mass": _numeric_mean(active, "fit_rtc_v3_anchor_recycle_mass"),
    }
    result.update(_client_metrics(pd.read_csv(client_path), active_rounds))
    return result


def _first_client_divergence(
    rtc: pd.DataFrame,
    multi_krum: pd.DataFrame,
    *,
    condition: str,
    seed: int,
) -> dict[str, object]:
    keys = ["round", "cid"]
    left = rtc.copy()
    right = multi_krum.copy()
    for frame in (left, right):
        frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
        frame["cid"] = frame["cid"].astype(str)
        if frame.duplicated(keys).any():
            raise RuntimeError(f"duplicate client diagnostics for {condition}/seed{seed}")
    columns = keys + [
        "is_malicious",
        "attack_active",
        "aggregation_weight",
        "effective_weight",
        "clipped_delta_norm",
        "clipped",
        "quarantined",
        "state",
    ]
    merged = left[columns].merge(
        right[columns], on=keys, how="outer", suffixes=("_rtc", "_multi_krum"), indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"client row pairing failed for {condition}/seed{seed}")
    numeric_fields = ("aggregation_weight", "effective_weight", "clipped_delta_norm")
    changed = pd.Series(False, index=merged.index)
    for field in numeric_fields:
        rtc_values = pd.to_numeric(merged[f"{field}_rtc"], errors="coerce").fillna(0.0)
        mk_values = pd.to_numeric(
            merged[f"{field}_multi_krum"], errors="coerce"
        ).fillna(0.0)
        changed |= (rtc_values - mk_values).abs() > 1e-12
    for field in ("clipped", "quarantined", "state"):
        changed |= (
            merged[f"{field}_rtc"].fillna("").astype(str)
            != merged[f"{field}_multi_krum"].fillna("").astype(str)
        )
    merged = merged.assign(_changed=changed).sort_values(["round", "cid"])

    def first_for(round_min: int) -> dict[str, object]:
        candidates = merged[(merged["round"] >= round_min) & merged["_changed"]]
        if candidates.empty:
            raise RuntimeError(
                f"no client mechanism divergence for {condition}/seed{seed} from round {round_min}"
            )
        row = candidates.iloc[0]
        reasons: list[str] = []
        for field in numeric_fields:
            rtc_value = pd.to_numeric(row[f"{field}_rtc"], errors="coerce")
            mk_value = pd.to_numeric(row[f"{field}_multi_krum"], errors="coerce")
            rtc_value = 0.0 if pd.isna(rtc_value) else float(rtc_value)
            mk_value = 0.0 if pd.isna(mk_value) else float(mk_value)
            if abs(rtc_value - mk_value) > 1e-12:
                reasons.append(field)
        for field in ("clipped", "quarantined", "state"):
            if str(row[f"{field}_rtc"]) != str(row[f"{field}_multi_krum"]):
                reasons.append(field)
        return {
            "round": int(row["round"]),
            "cid": str(row["cid"]),
            "is_malicious": bool(
                _bool_series(pd.Series([row["is_malicious_rtc"]])).iloc[0]
            ),
            "attack_active": bool(
                _bool_series(pd.Series([row["attack_active_rtc"]])).iloc[0]
            ),
            "rtc_weight": float(row["aggregation_weight_rtc"]),
            "multi_krum_weight": float(row["aggregation_weight_multi_krum"]),
            "rtc_state": str(row["state_rtc"]),
            "multi_krum_state": str(row["state_multi_krum"]),
            "changed_fields": "|".join(reasons),
        }

    any_divergence = first_for(1)
    metric_window_divergence = first_for(1 if condition == "clean" else 11)
    return {
        "condition": condition,
        "seed": seed,
        **{f"first_any_{key}": value for key, value in any_divergence.items()},
        **{
            f"first_metric_window_{key}": value
            for key, value in metric_window_divergence.items()
        },
    }


def _paired_stats(values: pd.Series) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="raise").dropna().astype(float)
    n = len(clean)
    if n != len(SEEDS):
        raise RuntimeError(f"expected {len(SEEDS)} paired seeds, got {n}")
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1))
    se = sd / math.sqrt(n)
    margin = float(stats.t.ppf(0.975, n - 1) * se)
    test = stats.ttest_1samp(clean, popmean=0.0)
    return {
        "n": n,
        "degrees_of_freedom": n - 1,
        "confidence_level": 0.95,
        "inference_scope": "exploratory_three_seed_paired",
        "equivalence_test_performed": False,
        "mean_delta": mean,
        "sd_delta": sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "paired_t": float(test.statistic),
        "paired_p_two_sided": float(test.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for key, default in DEFAULT_DIRS.items():
        parser.add_argument(f"--{key.replace('_', '-')}-dir", type=Path, default=default)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    directories = {
        key: getattr(args, f"{key}_dir").resolve() for key in DEFAULT_DIRS
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in set(directories.values()):
        _all_quality_gates_pass(directory)
    manifests = {key: _load_specs(path) for key, path in directories.items()}

    rows: list[dict[str, object]] = []
    divergence_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        attack, lie_z = _condition_contract(condition)
        for seed in SEEDS:
            paired_specs: dict[str, dict[str, object]] = {}
            paired_sources: dict[str, tuple[Path, dict[str, object]]] = {}
            for defense in (RTC, MULTI_KRUM):
                key = _source_key(condition, defense, seed)
                spec = _find_spec(
                    manifests[key], attack=attack, defense=defense, seed=seed, lie_z=lie_z
                )
                _validate_spec(spec, condition, defense)
                metrics = _load_run(directories[key], spec, condition)
                rows.append(
                    {
                        "condition": condition,
                        "attack": attack,
                        "lie_z": lie_z,
                        "seed": seed,
                        "defense": defense,
                        "source_directory": str(directories[key]),
                        "trial_plan_hash": str(spec["trial_plan_hash"]),
                        "attack_implementation_hash": str(spec["attack_implementation_hash"]),
                        **metrics,
                    }
                )
                paired_specs[defense] = spec
                paired_sources[defense] = (directories[key], spec)
            rtc_spec, mk_spec = paired_specs[RTC], paired_specs[MULTI_KRUM]
            if rtc_spec["trial_plan_hash"] != mk_spec["trial_plan_hash"]:
                raise RuntimeError(f"trial plan mismatch: {condition}/seed{seed}")
            if rtc_spec["attack_implementation_hash"] != mk_spec["attack_implementation_hash"]:
                raise RuntimeError(f"attack implementation mismatch: {condition}/seed{seed}")
            rtc_directory, rtc_source_spec = paired_sources[RTC]
            mk_directory, mk_source_spec = paired_sources[MULTI_KRUM]
            rtc_run_id = periodic_attack.run_id(rtc_source_spec)
            mk_run_id = periodic_attack.run_id(mk_source_spec)
            divergence_rows.append(
                _first_client_divergence(
                    pd.read_csv(rtc_directory / "raw" / f"{rtc_run_id}_clients.csv"),
                    pd.read_csv(mk_directory / "raw" / f"{mk_run_id}_clients.csv"),
                    condition=condition,
                    seed=seed,
                )
            )

    runs = pd.DataFrame(rows).sort_values(["condition", "seed", "defense"])
    expected_cells = len(CONDITIONS) * len(SEEDS) * 2
    if len(runs) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} final V1 cells, got {len(runs)}")
    runs.to_csv(output / "runs.csv", index=False)

    wide = runs.pivot(index=["condition", "seed"], columns="defense")
    paired_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            row = wide.loc[(condition, seed)]
            paired_row: dict[str, object] = {"condition": condition, "seed": seed}
            for metric in PAIR_METRICS:
                rtc_value = row[(metric, RTC)]
                mk_value = row[(metric, MULTI_KRUM)]
                paired_row[f"rtc_{metric}"] = rtc_value
                paired_row[f"multi_krum_{metric}"] = mk_value
                paired_row[f"{metric}_delta_rtc_minus_multi_krum"] = rtc_value - mk_value
            paired_rows.append(paired_row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "paired_by_seed.csv", index=False)
    divergences = pd.DataFrame(divergence_rows).sort_values(["condition", "seed"])
    expected_pairs = len(CONDITIONS) * len(SEEDS)
    if len(divergences) != expected_pairs:
        raise RuntimeError(
            f"expected {expected_pairs} paired divergence rows, got {len(divergences)}"
        )
    divergences.to_csv(output / "first_mechanism_divergence.csv", index=False)

    summaries: list[dict[str, object]] = []
    for condition in CONDITIONS:
        selected = paired[paired["condition"] == condition]
        for metric in PAIR_METRICS:
            values = selected[f"{metric}_delta_rtc_minus_multi_krum"]
            if pd.to_numeric(values, errors="coerce").notna().all():
                result = _paired_stats(values)
                summaries.append({"condition": condition, "metric": metric, **result})
    statistical = pd.DataFrame(summaries)
    statistical.to_csv(output / "statistical_tests.csv", index=False)
    aggregations: dict[str, tuple[str, str]] = {
        "seeds": ("seed", "nunique"),
        "sd_accuracy": ("mean_accuracy", "std"),
        "peak_asr": ("peak_asr", "max"),
    }
    aggregations.update({metric: (metric, "mean") for metric in PAIR_METRICS})
    defense_summary = runs.groupby(["condition", "defense"], as_index=False).agg(
        **aggregations
    )
    defense_summary.to_csv(output / "summary_by_condition.csv", index=False)
    decision = {
        "stage": "v1_multiseed_final_validation",
        "validation_complete": True,
        "final_rtc_candidate": "B3R-F0.51",
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "cells": len(runs),
        "new_cells": 22,
        "reused_current_contract_cells": 2,
        "statistical_inference": {
            "paired_seed_count": len(SEEDS),
            "degrees_of_freedom": len(SEEDS) - 1,
            "scope": "exploratory_three_seed_paired",
            "equivalence_test_performed": False,
            "interpretation_guard": (
                "A non-significant paired t-test does not establish equivalence."
            ),
        },
        "checks": {
            "all_24_cells_complete": True,
            "all_quality_gates_pass": True,
            "all_pairs_share_trial_plan_hash": True,
            "all_pairs_share_attack_implementation_hash": True,
            "rtc_parameters_exact_b3r_f051": True,
            "multi_krum_f3_select5": True,
            "lie_strengths_exact_0_25_and_0_5": True,
            "all_required_weight_impact_mass_and_false_positive_metrics_present": True,
            "all_12_pairs_have_first_mechanism_divergence": True,
            "three_seed_statistical_limit_recorded": True,
        },
    }
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
