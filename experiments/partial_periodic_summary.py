"""Build a descriptive-only report from completed periodic-attack run files.

This helper never edits raw/round CSV files.  It reconstructs summaries only
from runs that have a complete per-round file and explicitly excludes invalid
runs (for example, any run containing NaN/Inf server metrics) from condition
means while retaining them in the run-level audit table.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from periodic_attack import (
    run_id,
    summarize_run,
    validate_sampling_manifests,
)


GROUP_KEYS = ["attack", "period", "malicious_fraction", "defense"]
MEAN_METRICS = [
    "final_accuracy",
    "active_accuracy",
    "accuracy_drop",
    "active_accuracy_drop",
    "malicious_aggregation_weight_share",
    "malicious_impact_share",
    "active_malicious_aggregation_weight_share",
    "active_malicious_impact_share",
    "peak_active_malicious_aggregation_weight_share",
    "peak_active_malicious_impact_share",
    "min_active_accuracy",
    "inactive_accuracy",
    "freqfed_fallback",
    "freqfed_selected_clients",
    "freqfed_rejected_clients",
    "freqfed_noise_clients",
    "freqfed_cluster_count",
    "freqfed_selected_ratio",
    "active_freqfed_selected_ratio",
    "active_freqfed_noise_clients",
    "aggregation_time_seconds",
]


def _finite(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return bool(len(values) and np.isfinite(values).all())


def _mean(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    if column not in frame:
        return math.nan
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def _max(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    if column not in frame:
        return math.nan
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna()
    return float(values.max()) if len(values) else math.nan


def _min(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    if column not in frame:
        return math.nan
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna()
    return float(values.min()) if len(values) else math.nan


def _std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if len(array) > 1 else math.nan


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_partial_report(output: Path) -> None:
    rounds_dir = output / "rounds"
    raw_dir = output / "raw"
    with open(output / "experiment_manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    specs = list(manifest["specs"])
    spec_by_id = {run_id(spec): spec for spec in specs}

    completed_paths = sorted(
        (path for path in rounds_dir.glob("*.csv") if path.stem in spec_by_id),
        key=lambda path: path.stat().st_mtime,
    )
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for path in completed_paths:
        identifier = path.stem
        spec = spec_by_id[identifier]
        frame = pd.read_csv(path)
        frames[identifier] = frame
        result = summarize_run(frame, spec)

        expected_rounds = set(range(61))
        actual_rounds = set(pd.to_numeric(frame.get("round"), errors="coerce").dropna().astype(int))
        complete = len(frame) == 61 and actual_rounds == expected_rounds
        finite_accuracy = "server_accuracy" in frame and _finite(frame["server_accuracy"])
        finite_loss = "server_loss" in frame and _finite(frame["server_loss"])
        fit = frame.dropna(subset=["fit_selected_partition_ids"]) if "fit_selected_partition_ids" in frame else pd.DataFrame()
        schedule_match = bool(
            len(fit)
            and "fit_planned_partition_ids" in fit
            and (
                fit["fit_selected_partition_ids"].astype(str)
                == fit["fit_planned_partition_ids"].astype(str)
            ).all()
        )
        fallback = float(result.get("freqfed_fallback", math.nan))
        fallback_valid = spec["defense"] != "freqfed" or (
            np.isfinite(fallback) and fallback <= 0.25
        )
        invalid_reasons = []
        if not complete:
            invalid_reasons.append("incomplete_rounds")
        if not finite_accuracy:
            invalid_reasons.append("nonfinite_accuracy")
        if not finite_loss:
            invalid_reasons.append("nonfinite_loss")
        if not schedule_match:
            invalid_reasons.append("sampling_plan_mismatch")
        if not fallback_valid:
            invalid_reasons.append("freqfed_fallback_above_25pct")

        active = pd.to_numeric(frame["planned_attack_active"], errors="coerce").fillna(0).eq(1)
        inactive = (
            pd.to_numeric(frame["round"], errors="coerce").ge(int(spec.get("attack_start_round", 11)))
            & ~active
        )
        result.update({
            "round_file": path.name,
            "round_rows": len(frame),
            "max_round": int(pd.to_numeric(frame["round"], errors="coerce").max()),
            "run_complete": complete,
            "finite_server_accuracy": finite_accuracy,
            "finite_server_loss": finite_loss,
            "sampling_plan_match": schedule_match,
            "fallback_valid": fallback_valid,
            "valid_for_analysis": not invalid_reasons,
            "invalid_reasons": ",".join(invalid_reasons),
            "descriptive_only": True,
            "active_malicious_aggregation_weight_share": _mean(
                frame, active, "fit_malicious_aggregation_weight_share"
            ),
            "inactive_malicious_aggregation_weight_share": _mean(
                frame, inactive, "fit_malicious_aggregation_weight_share"
            ),
            "active_malicious_impact_share": _mean(
                frame, active, "fit_malicious_impact_share"
            ),
            "inactive_malicious_impact_share": _mean(
                frame, inactive, "fit_malicious_impact_share"
            ),
            "peak_active_malicious_aggregation_weight_share": _max(
                frame, active, "fit_malicious_aggregation_weight_share"
            ),
            "peak_active_malicious_impact_share": _max(
                frame, active, "fit_malicious_impact_share"
            ),
            "min_active_accuracy": _min(frame, active, "server_accuracy"),
            "inactive_accuracy": _mean(frame, inactive, "server_accuracy"),
            "active_freqfed_selected_ratio": _mean(
                frame, active, "fit_freqfed_selected_ratio"
            ),
            "inactive_freqfed_selected_ratio": _mean(
                frame, inactive, "fit_freqfed_selected_ratio"
            ),
            "active_freqfed_noise_clients": _mean(
                frame, active, "fit_freqfed_noise_clients"
            ),
            "inactive_freqfed_noise_clients": _mean(
                frame, inactive, "fit_freqfed_noise_clients"
            ),
        })
        rows.append(result)

    runs = pd.DataFrame(rows)
    clean_fedavg_frames: dict[int, pd.DataFrame] = {}
    for identifier, frame in frames.items():
        spec = spec_by_id[identifier]
        if spec["attack"] == "none" and spec["defense"] == "fedavg":
            clean_fedavg_frames[int(spec["seed"])] = frame

    runs["accuracy_drop"] = np.nan
    runs["active_accuracy_drop"] = np.nan
    runs["paired_clean_available"] = False
    for index, row in runs.iterrows():
        clean_frame = clean_fedavg_frames.get(int(row["seed"]))
        if clean_frame is None:
            continue
        runs.at[index, "paired_clean_available"] = True
        clean_final = float(pd.to_numeric(clean_frame["server_accuracy"], errors="coerce").dropna().iloc[-1])
        runs.at[index, "accuracy_drop"] = clean_final - float(row["final_accuracy"])
        attack_frame = frames[str(row["run_id"])]
        if row["attack"] == "none":
            continue
        merged = attack_frame[["round", "planned_attack_active", "server_accuracy"]].merge(
            clean_frame[["round", "server_accuracy"]],
            on="round",
            suffixes=("_attack", "_clean"),
        )
        active = merged["planned_attack_active"].eq(1)
        drops = (
            pd.to_numeric(merged.loc[active, "server_accuracy_clean"], errors="coerce")
            - pd.to_numeric(merged.loc[active, "server_accuracy_attack"], errors="coerce")
        ).dropna()
        if len(drops):
            runs.at[index, "active_accuracy_drop"] = float(drops.mean())

    runs.to_csv(output / "partial_runs.csv", index=False)

    manifest_frame = pd.DataFrame(specs)
    coverage_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for group_key, expected in manifest_frame.groupby(GROUP_KEYS, dropna=False):
        key = dict(zip(GROUP_KEYS, group_key))
        mask = pd.Series(True, index=runs.index)
        for column, value in key.items():
            mask &= runs[column].eq(value)
        completed = runs[mask]
        valid = completed[completed["valid_for_analysis"]]
        expected_seeds = sorted(int(value) for value in expected["seed"].unique())
        completed_seeds = sorted(int(value) for value in completed["seed"].unique())
        valid_seeds = sorted(int(value) for value in valid["seed"].unique())
        missing_seeds = sorted(set(expected_seeds) - set(completed_seeds))
        invalid_ids = completed.loc[~completed["valid_for_analysis"], "run_id"].astype(str).tolist()
        condition_complete = len(valid) == len(expected)
        status = (
            "complete_descriptive" if condition_complete
            else "invalid_run_present" if invalid_ids
            else "partial" if len(completed)
            else "not_started"
        )
        coverage = {
            **key,
            "n_expected": len(expected),
            "n_completed": len(completed),
            "n_valid": len(valid),
            "coverage_fraction": len(completed) / len(expected),
            "valid_coverage_fraction": len(valid) / len(expected),
            "expected_seeds": ",".join(map(str, expected_seeds)),
            "completed_seeds": ",".join(map(str, completed_seeds)),
            "valid_seeds": ",".join(map(str, valid_seeds)),
            "missing_seeds": ",".join(map(str, missing_seeds)),
            "invalid_run_ids": ",".join(invalid_ids),
            "status": status,
            "descriptive_only": True,
            "inference_allowed": False,
        }
        coverage_rows.append(coverage)
        summary = dict(coverage)
        summary["paired_clean_count"] = int(valid["paired_clean_available"].sum()) if len(valid) else 0
        for metric in MEAN_METRICS:
            if metric not in valid:
                continue
            values = pd.to_numeric(valid[metric], errors="coerce").dropna()
            summary[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
            summary[f"{metric}_std"] = _std(values)
            if metric.startswith("peak_") and len(values):
                summary[f"{metric}_max"] = float(values.max())
            if metric.startswith("min_") and len(values):
                summary[f"{metric}_min"] = float(values.min())
        summary_rows.append(summary)

    coverage = pd.DataFrame(coverage_rows).sort_values(GROUP_KEYS)
    condition_summary = pd.DataFrame(summary_rows).sort_values(GROUP_KEYS)
    coverage.to_csv(output / "partial_coverage.csv", index=False)
    condition_summary.to_csv(output / "partial_condition_summary.csv", index=False)

    completed_ids = set(runs["run_id"].astype(str))
    sampling_gates = validate_sampling_manifests(
        rounds_dir, raw_dir, expected_ids=completed_ids
    )
    gates: list[dict[str, Any]] = [
        {
            "gate": gate["gate"], "passed": bool(gate["passed"]),
            "observed": _json(gate["observed"]), "required": gate["required"],
            "severity": "fatal", "interpretation": "实验公平性/可复现性检查",
        }
        for gate in sampling_gates
    ]
    gates.extend([
        {
            "gate": "completed_round_files", "passed": len(runs) == 19,
            "observed": str(len(runs)), "required": "19 after requested stop",
            "severity": "fatal", "interpretation": "请求停止点的完整 run 数",
        },
        {
            "gate": "all_completed_runs_have_61_round_rows",
            "passed": bool(runs["run_complete"].all()),
            "observed": str(int(runs["run_complete"].sum())), "required": str(len(runs)),
            "severity": "fatal", "interpretation": "每个完成 run 必须包含 round 0-60",
        },
        {
            "gate": "all_completed_runs_finite",
            "passed": bool((runs["finite_server_accuracy"] & runs["finite_server_loss"]).all()),
            "observed": str(int((runs["finite_server_accuracy"] & runs["finite_server_loss"]).sum())),
            "required": str(len(runs)), "severity": "fatal",
            "interpretation": "非有限 loss/accuracy 的 run 不进入条件均值",
        },
        {
            "gate": "freqfed_fallback_rate",
            "passed": bool((pd.to_numeric(runs.loc[runs["defense"].eq("freqfed"), "freqfed_fallback"], errors="coerce") <= 0.25).all()),
            "observed": str(float(pd.to_numeric(runs.loc[runs["defense"].eq("freqfed"), "freqfed_fallback"], errors="coerce").max())),
            "required": "<= 0.25 each completed run", "severity": "fatal",
            "interpretation": "禁止用高 fallback 结果冒充 FreqFed",
        },
    ])
    clean_fedavg = runs[(runs["attack"] == "none") & (runs["defense"] == "fedavg")]
    clean_freqfed = runs[(runs["attack"] == "none") & (runs["defense"] == "freqfed")]
    if len(clean_fedavg) and len(clean_freqfed):
        clean_drop = float(clean_fedavg.iloc[0]["final_accuracy"] - clean_freqfed.iloc[0]["final_accuracy"])
        selected_ratio = float(clean_freqfed.iloc[0]["freqfed_selected_ratio"])
        noise_ratio = float(clean_freqfed.iloc[0]["freqfed_noise_clients"] / 10.0)
        gates.extend([
            {
                "gate": "freqfed_clean_accuracy_drop", "passed": clean_drop <= 0.02,
                "observed": str(clean_drop), "required": "<= 0.02",
                "severity": "acceptance", "interpretation": "原实验 clean utility 成功标准",
            },
            {
                "gate": "freqfed_clean_selected_ratio", "passed": selected_ratio >= 0.60,
                "observed": str(selected_ratio), "required": ">= 0.60",
                "severity": "diagnostic", "interpretation": "避免良性客户端过度过滤",
            },
            {
                "gate": "freqfed_clean_noise_ratio", "passed": noise_ratio <= 0.40,
                "observed": str(noise_ratio), "required": "<= 0.40",
                "severity": "diagnostic", "interpretation": "clean HDBSCAN 噪声比例",
            },
        ])
    attack_prechecks = runs[(runs["attack"] != "none") & (runs["defense"] == "fedavg")]
    gates.extend([
        {
            "gate": "fedavg_attack_prechecks_present", "passed": len(attack_prechecks) >= 3,
            "observed": str(len(attack_prechecks)), "required": ">= 3 attack families",
            "severity": "fatal", "interpretation": "没有无防御攻击基线就不能判断防御有效性",
        },
        {
            "gate": "clean_baselines_all_seeds",
            "passed": set(clean_fedavg["seed"].astype(int)) == {42, 43, 44},
            "observed": ",".join(map(str, sorted(clean_fedavg["seed"].astype(int).unique()))),
            "required": "42,43,44", "severity": "fatal",
            "interpretation": "active accuracy drop 必须使用同种子 clean 轨迹",
        },
        {
            "gate": "formal_matrix_complete", "passed": len(runs) == len(specs),
            "observed": f"{len(runs)}/{len(specs)}", "required": f"{len(specs)}/{len(specs)}",
            "severity": "informational", "interpretation": "用户主动停止后的覆盖率",
        },
    ])
    gates_frame = pd.DataFrame(gates)
    gates_frame.to_csv(output / "partial_quality_gates.csv", index=False)

    overview = pd.DataFrame([
        {"section": "coverage", "metric": "expected_runs", "value": len(specs), "unit": "runs", "status": "reference"},
        {"section": "coverage", "metric": "completed_runs", "value": len(runs), "unit": "runs", "status": "partial"},
        {"section": "coverage", "metric": "valid_runs", "value": int(runs["valid_for_analysis"].sum()), "unit": "runs", "status": "warning"},
        {"section": "coverage", "metric": "invalid_runs", "value": int((~runs["valid_for_analysis"]).sum()), "unit": "runs", "status": "fail"},
        {"section": "coverage", "metric": "completion_fraction", "value": len(runs) / len(specs), "unit": "ratio", "status": "partial"},
        {"section": "statistics", "metric": "descriptive_only", "value": True, "unit": "boolean", "status": "warning"},
        {"section": "statistics", "metric": "inference_allowed", "value": False, "unit": "boolean", "status": "fail"},
    ])
    if len(clean_fedavg) and len(clean_freqfed):
        clean_drop = float(clean_fedavg.iloc[0]["final_accuracy"] - clean_freqfed.iloc[0]["final_accuracy"])
        overview = pd.concat([overview, pd.DataFrame([
            {"section": "clean", "metric": "fedavg_final_accuracy", "value": float(clean_fedavg.iloc[0]["final_accuracy"]), "unit": "ratio", "status": "reference"},
            {"section": "clean", "metric": "freqfed_final_accuracy", "value": float(clean_freqfed.iloc[0]["final_accuracy"]), "unit": "ratio", "status": "warning"},
            {"section": "clean", "metric": "freqfed_accuracy_drop", "value": clean_drop, "unit": "ratio", "status": "fail" if clean_drop > 0.02 else "pass"},
            {"section": "clean", "metric": "freqfed_selected_ratio", "value": float(clean_freqfed.iloc[0]["freqfed_selected_ratio"]), "unit": "ratio", "status": "fail"},
            {"section": "clean", "metric": "freqfed_noise_ratio", "value": float(clean_freqfed.iloc[0]["freqfed_noise_clients"] / 10.0), "unit": "ratio", "status": "fail"},
        ])], ignore_index=True)
    overview.to_csv(output / "partial_overview.csv", index=False)

    statistical_status = pd.DataFrame([{
        "descriptive_only": True,
        "inference_allowed": False,
        "p_values_emitted": False,
        "reason": (
            "partial matrix; no attacked FedAvg prechecks/comparators; clean baselines "
            "missing for seeds 43 and 44; fewer than 8 paired observations"
        ),
    }])
    statistical_status.to_csv(output / "partial_statistical_status.csv", index=False)

    workbook_tables = {
        "Overview": json.loads(overview.to_json(orient="split")),
        "Conditions": json.loads(condition_summary.to_json(orient="split")),
        "Coverage": json.loads(coverage.to_json(orient="split")),
        "Quality Gates": json.loads(gates_frame.to_json(orient="split")),
        "Run Audit": json.loads(runs.to_json(orient="split")),
        "Statistical Status": json.loads(statistical_status.to_json(orient="split")),
    }
    with open(output / "partial_workbook_data.json", "w", encoding="utf-8") as handle:
        json.dump(workbook_tables, handle, ensure_ascii=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_output": str(output.resolve()),
        "benchmark_version": manifest.get("benchmark_version"),
        "manifest_sha256": manifest.get("sha256"),
        "expected_runs": len(specs),
        "completed_runs": len(runs),
        "valid_runs": int(runs["valid_for_analysis"].sum()),
        "invalid_runs": int((~runs["valid_for_analysis"]).sum()),
        "descriptive_only": True,
        "inference_allowed": False,
        "stop_boundary": "after run 19 completed; run 20 not started beyond marker",
    }
    with open(output / "partial_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    label = condition_summary[
        condition_summary["attack"].isin(
            ["label_flip_targeted", "label_flip_all_reverse"]
        )
    ]
    byz = condition_summary[condition_summary["attack"].eq("byzantine")]
    invalid = runs[~runs["valid_for_analysis"]]
    report_lines = [
        "# FreqFed 非定向周期攻击阶段性报告",
        "",
        f"- 完成：{len(runs)}/{len(specs)} runs（{len(runs)/len(specs):.1%}）",
        f"- 有效：{int(runs['valid_for_analysis'].sum())} runs；无效：{len(invalid)} runs",
        "- 统计属性：`descriptive_only=true`，禁止显著性与防御优越性结论",
        "- 停止边界：第 19 个 run 完成后停止，第 20 个仅写入启动 marker",
        "",
        "## 核心结论",
        "",
        "1. Label flip 的 12 个 FreqFed 条件已覆盖，但缺少攻击版 FedAvg 和 seed 43/44 clean 基线，因此只能描述绝对准确率与聚类行为。",
        "2. Byzantine 仅覆盖 short 1/1；40% 条件含一个非有限 loss run，条件不完整。",
        "3. Gaussian noise、Byzantine long 3/3 尚未运行。",
        "4. FreqFed completed runs fallback 均不超过 25%，但 clean selected ratio 与 clean utility 不满足接受标准。",
        "",
        "## Label flip（有效 run 条件均值）",
        "",
        "| period | malicious | n valid | final accuracy | active accuracy | peak active impact (worst) | selected ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in label.iterrows():
        report_lines.append(
            f"| {row['period']} | {row['malicious_fraction']:.0%} | {int(row['n_valid'])} | "
            f"{row.get('final_accuracy_mean', math.nan):.2%} | "
            f"{row.get('active_accuracy_mean', math.nan):.2%} | "
            f"{row.get('peak_active_malicious_impact_share_max', math.nan):.2%} | "
            f"{row.get('freqfed_selected_ratio_mean', math.nan):.2%} |"
        )
    report_lines.extend([
        "",
        "## Byzantine（有效 run 条件均值）",
        "",
        "| period | malicious | completed/expected | n valid | final accuracy | active accuracy | peak active impact (worst) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in byz.iterrows():
        if not row["n_completed"]:
            continue
        report_lines.append(
            f"| {row['period']} | {row['malicious_fraction']:.0%} | "
            f"{int(row['n_completed'])}/{int(row['n_expected'])} | {int(row['n_valid'])} | "
            f"{row.get('final_accuracy_mean', math.nan):.2%} | "
            f"{row.get('active_accuracy_mean', math.nan):.2%} | "
            f"{row.get('peak_active_malicious_impact_share_max', math.nan):.2%} |"
        )
    report_lines.extend([
        "",
        "## 无效运行",
        "",
    ])
    if invalid.empty:
        report_lines.append("- 无")
    else:
        for _, row in invalid.iterrows():
            report_lines.append(f"- `{row['run_id']}`：{row['invalid_reasons']}")
    report_lines.extend([
        "",
        "## 必须修正后才能恢复正式矩阵",
        "",
        "- runner 必须主动生成各攻击的 FedAvg seed-42 预检，不能依赖 `--defenses` 中包含 FedAvg。",
        "- clean FedAvg/FreqFed 的 seed 42/43/44 应在攻击矩阵之前完成。",
        "- 增加 FreqFed clean accuracy、selected ratio、noise ratio 门槛。",
        "- 对 nonfinite loss fail-fast，避免数值异常 run 继续进入结果表。",
        "- 修正 FreqFed 过度过滤后，再决定是否恢复未完成配置。",
        "",
    ])
    (output / "PARTIAL_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_partial_report(args.output)


if __name__ == "__main__":
    main()
