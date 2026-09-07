"""Independent, stdlib-only verification of the V1 aggregate outputs.

This intentionally does not import ``analyze_results.py``.  It recomputes every
reported run-level metric directly from the round and client CSV files and
checks the aggregate files for complete condition/seed/defense coverage.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "rtc_v1_multiseed"
RUNS = OUT / "runs.csv"
RTC = "rtc_cumulative_q_cap_accepted_anchor"
DEFENSES = {RTC, "multi_krum"}
CONDITIONS = {"clean", "lie_z025", "lie_z05", "dba"}
SEEDS = {42, 46, 47}
TOL = 1e-12


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "").strip()
        if raw:
            values.append(float(raw))
    return values


def mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def rate(rows: list[dict[str, str]], predicate, column: str, value: str | None = None) -> float:
    selected = [row for row in rows if predicate(row)]
    if not selected:
        return math.nan
    if value is None:
        return sum(truthy(row[column]) for row in selected) / len(selected)
    return sum(row[column].strip().lower() == value for row in selected) / len(selected)


def close(actual: float, expected_text: str, label: str) -> None:
    if not expected_text.strip() and math.isnan(actual):
        return
    expected = float(expected_text)
    if math.isnan(actual) and math.isnan(expected):
        return
    if not math.isclose(actual, expected, rel_tol=TOL, abs_tol=TOL):
        raise AssertionError(f"{label}: recomputed={actual!r}, reported={expected!r}")


def verify_run(run: dict[str, str]) -> None:
    source = Path(run["source_directory"])
    run_id = run["run_id"]
    rounds = read_rows(source / "rounds" / f"{run_id}.csv")
    if len(rounds) != 61 or {int(row["round"]) for row in rounds} != set(range(61)):
        raise AssertionError(f"{run_id}: rounds are not exactly 0..60")

    if run["condition"] == "clean":
        active = [row for row in rounds if 1 <= int(row["round"]) <= 60]
    else:
        active = [row for row in rounds if float(row["planned_attack_active"] or 0) == 1]
    expected_active = 60 if run["condition"] == "clean" else 50
    if len(active) != expected_active:
        raise AssertionError(f"{run_id}: active rows={len(active)}, expected={expected_active}")

    final = next(float(row["server_accuracy"]) for row in rounds if int(row["round"]) == 60)
    mal_weight = mean(numeric(active, "fit_malicious_aggregation_weight_share"))
    mal_impact = mean(numeric(active, "fit_malicious_impact_share"))
    rtc_weight_sums = numeric(active, "fit_rtc_v3_weight_sum")
    benign_weight = mean(rtc_weight_sums) - mal_weight if rtc_weight_sums else 1 - mal_weight

    metrics = {
        "mean_accuracy": mean(numeric(active, "server_accuracy")),
        "final_accuracy": final,
        "malicious_weight_share": mal_weight,
        "benign_weight_share": benign_weight,
        "malicious_impact_share": mal_impact,
        "benign_impact_share": 1 - mal_impact,
        "zero_update_mass": mean(numeric(active, "fit_rtc_v3_zero_update_mass")),
        "effective_update_mass": mean(numeric(active, "fit_rtc_v3_effective_update_mass"), 1.0),
        "anchor_recycle_mass": mean(numeric(active, "fit_rtc_v3_anchor_recycle_mass")),
    }
    if run["condition"] == "dba":
        asr = numeric(active, "server_asr")
        metrics["mean_asr"] = mean(asr)
        metrics["peak_asr"] = max(asr)

    active_rounds = {int(row["round"]) for row in active}
    clients = [
        row
        for row in read_rows(source / "raw" / f"{run_id}_clients.csv")
        if int(row["round"]) in active_rounds
    ]
    if len(clients) != 10 * expected_active:
        raise AssertionError(f"{run_id}: client rows={len(clients)}")
    if len({(row["round"], row["cid"]) for row in clients}) != len(clients):
        raise AssertionError(f"{run_id}: duplicate round/cid client rows")

    benign = lambda row: not truthy(row["is_malicious"])
    attacker = lambda row: truthy(row["attack_active"])
    nonattacker = lambda row: not truthy(row["attack_active"])
    for prefix, predicate in (
        ("benign", benign),
        ("nonattacker", nonattacker),
        ("active_attacker", attacker),
    ):
        metrics[f"{prefix}_clipping_rate"] = rate(clients, predicate, "clipped")
        for state in ("watch", "restricted", "quarantined"):
            metrics[f"{prefix}_{state}_rate"] = rate(clients, predicate, "state", state)

    for name, actual in metrics.items():
        close(actual, run[name], f"{run_id}/{name}")

    round_hashes = {row["trial_plan_hash"] for row in active}
    attack_hashes = {row["attack_implementation_hash"] for row in active}
    if round_hashes != {run["trial_plan_hash"]}:
        raise AssertionError(f"{run_id}: trial-plan hash mismatch")
    if attack_hashes != {run["attack_implementation_hash"]}:
        raise AssertionError(f"{run_id}: attack implementation hash mismatch")


def main() -> None:
    runs = read_rows(RUNS)
    expected_keys = {(condition, seed, defense) for condition in CONDITIONS for seed in SEEDS for defense in DEFENSES}
    actual_keys = {(row["condition"], int(row["seed"]), row["defense"]) for row in runs}
    if len(runs) != 24 or actual_keys != expected_keys:
        raise AssertionError("runs.csv does not contain the exact 24-cell V1 matrix")
    for run in runs:
        verify_run(run)

    paired = read_rows(OUT / "paired_by_seed.csv")
    divergences = read_rows(OUT / "first_mechanism_divergence.csv")
    if len(paired) != 12 or len(divergences) != 12:
        raise AssertionError("paired/divergence output is incomplete")
    if {(row["condition"], int(row["seed"])) for row in divergences} != {
        (condition, seed) for condition in CONDITIONS for seed in SEEDS
    }:
        raise AssertionError("first-divergence coverage is incomplete")
    print("independent verification passed: 24 runs, 12 paired comparisons, all raw metrics match")


if __name__ == "__main__":
    main()
