"""Build the bounded Data Analytics artifact for the RTC-v3 V1 report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "rtc_v1_multiseed"
RTC = "rtc_cumulative_q_cap_accepted_anchor"
MK = "multi_krum"
GENERATED_AT = "2026-09-06T09:38:04+08:00"


def rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float | None:
    return float(value) if value.strip() else None


def pct(value: str) -> float | None:
    parsed = number(value)
    return None if parsed is None else 100 * parsed


def r4(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def main() -> None:
    summary = rows("summary_by_condition.csv")
    stats = rows("statistical_tests.csv")
    paired = rows("paired_by_seed.csv")
    divergence = rows("first_mechanism_divergence.csv")

    summary_by_key = {(row["condition"], row["defense"]): row for row in summary}
    stats_by_key = {(row["condition"], row["metric"]): row for row in stats}
    labels = {
        "clean": "Clean",
        "lie_z025": "LIE z=0.25",
        "lie_z05": "LIE z=0.5",
        "dba": "Strong DBA",
    }

    condition_summary: list[dict[str, object]] = []
    mechanism_summary: list[dict[str, object]] = []
    for condition in ("clean", "lie_z025", "lie_z05", "dba"):
        rtc = summary_by_key[(condition, RTC)]
        mk = summary_by_key[(condition, MK)]
        test = stats_by_key[(condition, "mean_accuracy")]
        condition_summary.append(
            {
                "condition": condition,
                "condition_label": labels[condition],
                "rtc_mean_acc_pct": r4(pct(rtc["mean_accuracy"])),
                "mk_mean_acc_pct": r4(pct(mk["mean_accuracy"])),
                "mean_acc_delta_pp": r4(100 * float(test["mean_delta"])),
                "ci95_low_pp": r4(100 * float(test["ci95_low"])),
                "ci95_high_pp": r4(100 * float(test["ci95_high"])),
                "paired_p": r4(number(test["paired_p_two_sided"])),
                "rtc_final_acc_pct": r4(pct(rtc["final_accuracy"])),
                "mk_final_acc_pct": r4(pct(mk["final_accuracy"])),
                "rtc_mean_asr_pct": r4(pct(rtc["mean_asr"])),
                "mk_mean_asr_pct": r4(pct(mk["mean_asr"])),
                "seeds": 3,
                "metric_window": "rounds 1–60" if condition == "clean" else "attack-active rounds 11–60",
            }
        )
        mechanism_summary.append(
            {
                "condition": condition,
                "condition_label": labels[condition],
                "rtc_malicious_weight_pct": r4(pct(rtc["malicious_weight_share"])),
                "mk_malicious_weight_pct": r4(pct(mk["malicious_weight_share"])),
                "malicious_weight_delta_pp": r4(100 * (float(rtc["malicious_weight_share"]) - float(mk["malicious_weight_share"]))),
                "rtc_malicious_impact_pct": r4(pct(rtc["malicious_impact_share"])),
                "mk_malicious_impact_pct": r4(pct(mk["malicious_impact_share"])),
                "malicious_impact_delta_pp": r4(100 * (float(rtc["malicious_impact_share"]) - float(mk["malicious_impact_share"]))),
                "rtc_zero_mass_pct": r4(pct(rtc["zero_update_mass"])),
                "rtc_anchor_recycle_pct": r4(pct(rtc["anchor_recycle_mass"])),
                "rtc_benign_clip_pct": r4(pct(rtc["benign_clipping_rate"])),
                "rtc_benign_watch_pct": r4(pct(rtc["benign_watch_rate"])),
                "rtc_attacker_clip_pct": r4(pct(rtc["active_attacker_clipping_rate"])),
                "rtc_attacker_quarantine_pct": r4(pct(rtc["active_attacker_quarantined_rate"])),
            }
        )

    paired_rows = [
        {
            "condition": row["condition"],
            "condition_label": labels[row["condition"]],
            "seed": int(row["seed"]),
            "mean_acc_delta_pp": r4(100 * float(row["mean_accuracy_delta_rtc_minus_multi_krum"])),
            "final_acc_delta_pp": r4(100 * float(row["final_accuracy_delta_rtc_minus_multi_krum"])),
            "mean_asr_delta_pp": r4(pct(row["mean_asr_delta_rtc_minus_multi_krum"])),
            "malicious_weight_delta_pp": r4(100 * float(row["malicious_weight_share_delta_rtc_minus_multi_krum"])),
            "malicious_impact_delta_pp": r4(100 * float(row["malicious_impact_share_delta_rtc_minus_multi_krum"])),
            "zero_mass_delta_pp": r4(100 * float(row["zero_update_mass_delta_rtc_minus_multi_krum"])),
            "benign_clip_delta_pp": r4(100 * float(row["benign_clipping_rate_delta_rtc_minus_multi_krum"])),
        }
        for row in paired
    ]

    divergence_rows = [
        {
            "condition": row["condition"],
            "condition_label": labels[row["condition"]],
            "seed": int(row["seed"]),
            "first_any_round": int(row["first_any_round"]),
            "first_any_cid": int(row["first_any_cid"]),
            "first_any_malicious": row["first_any_is_malicious"].lower() == "true",
            "first_metric_round": int(row["first_metric_window_round"]),
            "first_metric_cid": int(row["first_metric_window_cid"]),
            "first_metric_malicious": row["first_metric_window_is_malicious"].lower() == "true",
            "first_metric_attack_active": row["first_metric_window_attack_active"].lower() == "true",
            "changed_fields": row["first_metric_window_changed_fields"],
        }
        for row in divergence
    ]

    dba_rtc = summary_by_key[("dba", RTC)]
    dba_mk = summary_by_key[("dba", MK)]
    dba_test = stats_by_key[("dba", "mean_asr")]
    dba_asr = [
        {
            "defense": "RTC B3R-F0.51",
            "mean_asr_pct": r4(pct(dba_rtc["mean_asr"])),
            "peak_asr_pct": r4(pct(dba_rtc["peak_asr"])),
            "mean_acc_pct": r4(pct(dba_rtc["mean_accuracy"])),
            "seeds": 3,
            "metric_window": "attack-active rounds 11–60",
        },
        {
            "defense": "Multi-Krum",
            "mean_asr_pct": r4(pct(dba_mk["mean_asr"])),
            "peak_asr_pct": r4(pct(dba_mk["peak_asr"])),
            "mean_acc_pct": r4(pct(dba_mk["mean_accuracy"])),
            "seeds": 3,
            "metric_window": "attack-active rounds 11–60",
        },
    ]

    macro_rtc = sum(float(summary_by_key[(c, RTC)]["mean_accuracy"]) for c in labels) / 4
    macro_mk = sum(float(summary_by_key[(c, MK)]["mean_accuracy"]) for c in labels) / 4
    lie05_rtc = summary_by_key[("lie_z05", RTC)]
    lie05_mk = summary_by_key[("lie_z05", MK)]
    clean_rtc = summary_by_key[("clean", RTC)]
    clean_mk = summary_by_key[("clean", MK)]
    headline = [
        {
            "macro_rtc_acc": macro_rtc,
            "macro_mk_acc": macro_mk,
            "macro_acc_delta_pp": 100 * (macro_rtc - macro_mk),
            "lie05_rtc_acc": float(lie05_rtc["mean_accuracy"]),
            "lie05_mk_acc": float(lie05_mk["mean_accuracy"]),
            "lie05_acc_delta_pp": 100 * (float(lie05_rtc["mean_accuracy"]) - float(lie05_mk["mean_accuracy"])),
            "dba_rtc_asr": float(dba_rtc["mean_asr"]),
            "dba_mk_asr": float(dba_mk["mean_asr"]),
            "dba_asr_delta_pp": 100 * float(dba_test["mean_delta"]),
            "clean_rtc_final_acc": float(clean_rtc["final_accuracy"]),
            "clean_mk_final_acc": float(clean_mk["final_accuracy"]),
            "clean_final_acc_delta_pp": 100 * (float(clean_rtc["final_accuracy"]) - float(clean_mk["final_accuracy"])),
        }
    ]

    source_summary = {
        "id": "source-summary",
        "label": "V1 condition summary",
        "path": "analysis/rtc_v1_multiseed/summary_by_condition.csv",
        "query": {
            "sql": "SELECT * FROM read_csv_auto('analysis/rtc_v1_multiseed/summary_by_condition.csv', header=true);",
            "description": "Aggregates 24 completed paired runs by condition and defense.",
            "engine": "DuckDB",
            "language": "sql",
            "executed_at": GENERATED_AT,
            "tables_used": ["analysis/rtc_v1_multiseed/summary_by_condition.csv"],
            "filters": ["seeds 42, 46, 47", "clean rounds 1–60", "attack-active rounds 11–60"],
            "metric_definitions": [
                "mean ACC is the arithmetic mean of server_accuracy over the condition metric window",
                "mean ASR is the arithmetic mean of server_asr over DBA attack-active rounds",
                "macro ACC is the unweighted mean of the four condition-level mean ACC values",
            ],
        },
    }
    source_paired = {
        "id": "source-paired",
        "label": "V1 paired-seed comparisons and tests",
        "path": "analysis/rtc_v1_multiseed/paired_by_seed.csv",
        "query": {
            "sql": "SELECT * FROM read_csv_auto('analysis/rtc_v1_multiseed/paired_by_seed.csv', header=true);",
            "description": "Pairs RTC and Multi-Krum within identical condition, seed, trial plan, and attack implementation.",
            "engine": "DuckDB",
            "language": "sql",
            "executed_at": GENERATED_AT,
            "tables_used": ["analysis/rtc_v1_multiseed/paired_by_seed.csv"],
            "filters": ["exact trial-plan hash match", "exact attack-implementation hash match", "n=3 paired seeds"],
            "metric_definitions": [
                "delta is RTC minus Multi-Krum in percentage points",
                "95% CI and two-sided paired t-test use n=3 and df=2",
                "no equivalence test was performed",
            ],
        },
    }
    source_condition_comparison = {
        "id": "source-condition-comparison",
        "label": "V1 condition-level paired inference",
        "path": "analysis/rtc_v1_multiseed/statistical_tests.csv",
        "query": {
            "sql": "WITH s AS (SELECT * FROM read_csv_auto('analysis/rtc_v1_multiseed/summary_by_condition.csv', header=true)), t AS (SELECT * FROM read_csv_auto('analysis/rtc_v1_multiseed/statistical_tests.csv', header=true)) SELECT * FROM s JOIN t USING (condition) WHERE t.metric IN ('mean_accuracy','mean_asr');",
            "description": "Joins condition/defense summaries to the preregistered paired-seed inference rows used by the charts and condition table.",
            "engine": "DuckDB",
            "language": "sql",
            "executed_at": GENERATED_AT,
            "tables_used": [
                "analysis/rtc_v1_multiseed/summary_by_condition.csv",
                "analysis/rtc_v1_multiseed/statistical_tests.csv",
            ],
            "filters": ["metrics mean_accuracy and DBA mean_asr", "n=3 paired seeds", "df=2"],
            "metric_definitions": [
                "delta is RTC minus Multi-Krum in percentage points",
                "95% CI and two-sided paired t-test use n=3 and df=2",
                "no equivalence test was performed",
            ],
        },
    }
    source_runs = {
        "id": "source-runs",
        "label": "V1 run-level audit metrics",
        "path": "analysis/rtc_v1_multiseed/runs.csv",
        "query": {
            "sql": "SELECT * FROM read_csv_auto('analysis/rtc_v1_multiseed/runs.csv', header=true);",
            "description": "Run-level metrics recomputed from raw rounds and client diagnostics.",
            "engine": "DuckDB",
            "language": "sql",
            "executed_at": GENERATED_AT,
            "tables_used": ["analysis/rtc_v1_multiseed/runs.csv"],
            "filters": ["24/24 runs complete", "52/52 quality gates pass", "rounds exactly 0–60"],
            "metric_definitions": [
                "benign rates use non-malicious client identities",
                "nonattacker rates use clients whose attack_active flag is false",
                "zero-update and anchor-recycle mass are round means over the metric window",
            ],
        },
    }
    source_divergence = {
        "id": "source-divergence",
        "label": "First client-level mechanism divergence",
        "path": "analysis/rtc_v1_multiseed/first_mechanism_divergence.csv",
        "query": {
            "sql": "SELECT * FROM read_csv_auto('analysis/rtc_v1_multiseed/first_mechanism_divergence.csv', header=true);",
            "description": "First RTC/Multi-Krum client diagnostic mismatch overall and within the metric window.",
            "engine": "DuckDB",
            "language": "sql",
            "executed_at": GENERATED_AT,
            "tables_used": ["analysis/rtc_v1_multiseed/first_mechanism_divergence.csv"],
            "filters": ["same condition/seed/round/cid", "all 12 pairs covered"],
        },
    }

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "RTC-v3 V1 三 Seed 最终验证",
        "description": "冻结 RTC B3R-F0.51 与 Multi-Krum 在 clean、LIE 和 strong DBA 上的配对验证。",
        "generatedAt": GENERATED_AT,
        "sources": [source_summary, source_condition_comparison, source_paired, source_runs, source_divergence],
        "cards": [
            {
                "id": "card-macro-acc",
                "dataset": "headline",
                "sourceId": "source-summary",
                "description": "四个条件等权宏平均的 round-window ACC。",
                "metrics": [
                    {"label": "RTC 宏平均 ACC", "field": "macro_rtc_acc", "format": "percent"},
                    {"label": "Multi-Krum", "field": "macro_mk_acc", "format": "percent"},
                    {"label": "差值 pp", "field": "macro_acc_delta_pp", "format": "number", "signed": True},
                ],
            },
            {
                "id": "card-lie05-acc",
                "dataset": "headline",
                "sourceId": "source-summary",
                "description": "LIE z=0.5 攻击活跃轮 11–60 的平均 ACC。",
                "metrics": [
                    {"label": "RTC LIE z=.5 ACC", "field": "lie05_rtc_acc", "format": "percent"},
                    {"label": "Multi-Krum", "field": "lie05_mk_acc", "format": "percent"},
                    {"label": "差值 pp", "field": "lie05_acc_delta_pp", "format": "number", "signed": True},
                ],
            },
            {
                "id": "card-dba-asr",
                "dataset": "headline",
                "sourceId": "source-condition-comparison",
                "description": "Strong DBA 攻击活跃轮的平均攻击成功率；越低越好。",
                "metrics": [
                    {"label": "RTC DBA 平均 ASR", "field": "dba_rtc_asr", "format": "percent"},
                    {"label": "Multi-Krum", "field": "dba_mk_asr", "format": "percent"},
                    {"label": "退化 pp", "field": "dba_asr_delta_pp", "format": "number", "signed": True},
                ],
            },
            {
                "id": "card-clean-final",
                "dataset": "headline",
                "sourceId": "source-summary",
                "description": "显式 clean 条件第 60 轮 ACC。",
                "metrics": [
                    {"label": "RTC clean 最终 ACC", "field": "clean_rtc_final_acc", "format": "percent"},
                    {"label": "Multi-Krum", "field": "clean_mk_final_acc", "format": "percent"},
                    {"label": "差值 pp", "field": "clean_final_acc_delta_pp", "format": "number", "signed": True},
                ],
            },
        ],
        "charts": [
            {
                "id": "chart-mean-acc-delta",
                "title": "各条件平均 ACC 差值（RTC − Multi-Krum）",
                "subtitle": "三 seed 点估计均为正，但所有 95% CI 均跨越 0。",
                "type": "bar",
                "intent": "comparison",
                "dataset": "condition_summary",
                "sourceId": "source-condition-comparison",
                "encodings": {
                    "x": {"field": "condition_label", "type": "nominal", "label": "条件"},
                    "y": {"field": "mean_acc_delta_pp", "type": "quantitative", "label": "平均 ACC 差值", "unit": "pp"},
                    "tooltip": [
                        {"field": "rtc_mean_acc_pct", "type": "quantitative", "label": "RTC ACC", "unit": "%"},
                        {"field": "mk_mean_acc_pct", "type": "quantitative", "label": "Multi-Krum ACC", "unit": "%"},
                        {"field": "ci95_low_pp", "type": "quantitative", "label": "95% CI 下界", "unit": "pp"},
                        {"field": "ci95_high_pp", "type": "quantitative", "label": "95% CI 上界", "unit": "pp"},
                        {"field": "paired_p", "type": "quantitative", "label": "paired p"},
                    ],
                },
                "valueFormat": "number",
                "unit": "pp",
                "palette": {"kind": "categorical", "name": "research-blue"},
                "labels": {"values": "all"},
                "referenceLines": [{"axis": "y", "value": 0, "label": "相同", "color": "neutral", "lineStyle": "solid"}],
                "settings": {"orientation": "vertical", "groupMode": "single", "sort": "none", "showValues": True},
                "surface": {"surface": "card", "showControls": False, "viewMode": "visualization"},
            },
            {
                "id": "chart-dba-asr",
                "title": "Strong DBA 攻击活跃轮平均 ASR",
                "subtitle": "RTC 比 Multi-Krum 高 0.601 pp；三 seed 配对 95% CI 为 0.019–1.183 pp。",
                "type": "bar",
                "intent": "comparison",
                "dataset": "dba_asr",
                "sourceId": "source-condition-comparison",
                "encodings": {
                    "x": {"field": "defense", "type": "nominal", "label": "防御"},
                    "y": {"field": "mean_asr_pct", "type": "quantitative", "label": "平均 ASR", "unit": "%"},
                    "tooltip": [
                        {"field": "peak_asr_pct", "type": "quantitative", "label": "跨 seed 峰值 ASR", "unit": "%"},
                        {"field": "mean_acc_pct", "type": "quantitative", "label": "平均 ACC", "unit": "%"},
                        {"field": "seeds", "type": "quantitative", "label": "seed 数"},
                    ],
                },
                "valueFormat": "number",
                "unit": "%",
                "palette": {"kind": "categorical", "name": "research-purple"},
                "labels": {"values": "all"},
                "settings": {"orientation": "vertical", "groupMode": "single", "sort": "none", "showValues": True},
                "surface": {"surface": "card", "showControls": False, "viewMode": "visualization"},
            },
        ],
        "tables": [
            {
                "id": "table-condition-summary",
                "title": "条件级效用与安全结果",
                "subtitle": "ACC/ASR 为百分比；差值为 RTC − Multi-Krum，单位 pp。",
                "dataset": "condition_summary",
                "sourceId": "source-condition-comparison",
                "defaultSort": {"field": "mean_acc_delta_pp", "direction": "desc"},
                "density": "dense",
                "columns": [
                    {"field": "condition_label", "label": "条件", "type": "text"},
                    {"field": "rtc_mean_acc_pct", "label": "RTC 平均 ACC (%)", "format": "number"},
                    {"field": "mk_mean_acc_pct", "label": "MK 平均 ACC (%)", "format": "number"},
                    {"field": "mean_acc_delta_pp", "label": "平均 ACC 差 (pp)", "format": "number", "movement": True},
                    {"field": "ci95_low_pp", "label": "95% CI 下界", "format": "number"},
                    {"field": "ci95_high_pp", "label": "95% CI 上界", "format": "number"},
                    {"field": "rtc_final_acc_pct", "label": "RTC 最终 ACC (%)", "format": "number"},
                    {"field": "mk_final_acc_pct", "label": "MK 最终 ACC (%)", "format": "number"},
                    {"field": "rtc_mean_asr_pct", "label": "RTC 平均 ASR (%)", "format": "number"},
                    {"field": "mk_mean_asr_pct", "label": "MK 平均 ASR (%)", "format": "number"},
                ],
            },
            {
                "id": "table-mechanism",
                "title": "RTC 机制指标与 Multi-Krum 差异",
                "subtitle": "说明问题不只是良性误伤：LIE 下 RTC 同时保留了更多恶意权重与 impact。",
                "dataset": "mechanism_summary",
                "sourceId": "source-runs",
                "defaultSort": {"field": "malicious_impact_delta_pp", "direction": "desc"},
                "density": "dense",
                "columns": [
                    {"field": "condition_label", "label": "条件", "type": "text"},
                    {"field": "malicious_weight_delta_pp", "label": "恶意权重差 (pp)", "format": "number", "movement": True},
                    {"field": "malicious_impact_delta_pp", "label": "恶意 impact 差 (pp)", "format": "number", "movement": True},
                    {"field": "rtc_zero_mass_pct", "label": "RTC zero mass (%)", "format": "number"},
                    {"field": "rtc_anchor_recycle_pct", "label": "RTC anchor mass (%)", "format": "number"},
                    {"field": "rtc_benign_clip_pct", "label": "RTC 良性 clipping (%)", "format": "number"},
                    {"field": "rtc_benign_watch_pct", "label": "RTC 良性 watch (%)", "format": "number"},
                    {"field": "rtc_attacker_clip_pct", "label": "RTC 攻击者 clipping (%)", "format": "number"},
                    {"field": "rtc_attacker_quarantine_pct", "label": "RTC 攻击者 quarantine (%)", "format": "number"},
                ],
            },
            {
                "id": "table-paired-seeds",
                "title": "逐 seed 配对差值",
                "subtitle": "所有差值均为 RTC − Multi-Krum；正 ACC 为 RTC 更高，正 ASR 为 RTC 更差。",
                "dataset": "paired_seeds",
                "sourceId": "source-paired",
                "defaultSort": {"field": "mean_acc_delta_pp", "direction": "desc"},
                "density": "dense",
                "columns": [
                    {"field": "condition_label", "label": "条件", "type": "text"},
                    {"field": "seed", "label": "Seed", "format": "number"},
                    {"field": "mean_acc_delta_pp", "label": "平均 ACC 差 (pp)", "format": "number", "movement": True},
                    {"field": "final_acc_delta_pp", "label": "最终 ACC 差 (pp)", "format": "number", "movement": True},
                    {"field": "mean_asr_delta_pp", "label": "平均 ASR 差 (pp)", "format": "number", "movement": True},
                    {"field": "malicious_weight_delta_pp", "label": "恶意权重差 (pp)", "format": "number", "movement": True},
                    {"field": "malicious_impact_delta_pp", "label": "恶意 impact 差 (pp)", "format": "number", "movement": True},
                    {"field": "zero_mass_delta_pp", "label": "zero mass 差 (pp)", "format": "number", "movement": True},
                    {"field": "benign_clip_delta_pp", "label": "良性 clipping 差 (pp)", "format": "number", "movement": True},
                ],
            },
            {
                "id": "table-divergence",
                "title": "首次客户端级机制分叉",
                "subtitle": "所有攻击条件的指标窗口首次分叉均在 round 11。",
                "dataset": "divergence",
                "sourceId": "source-divergence",
                "defaultSort": {"field": "first_metric_round", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "condition_label", "label": "条件", "type": "text"},
                    {"field": "seed", "label": "Seed", "format": "number"},
                    {"field": "first_any_round", "label": "首次任意分叉轮", "format": "number"},
                    {"field": "first_metric_round", "label": "指标窗口首次轮", "format": "number"},
                    {"field": "first_metric_cid", "label": "CID", "format": "number"},
                    {"field": "first_metric_malicious", "label": "恶意身份", "type": "text"},
                    {"field": "first_metric_attack_active", "label": "攻击活跃", "type": "text"},
                    {"field": "changed_fields", "label": "首处分叉字段", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# RTC-v3 V1 三 Seed 最终验证", "layout": "full"},
            {
                "id": "executive-summary",
                "type": "markdown",
                "body": "## 技术摘要\n\n冻结候选 **B3R-F0.51** 已把 RTC 从“平均 ACC 一直略低”推进到四个条件点估计均不低于 Multi-Krum：四条件宏平均 ACC 高 **0.605 pp**。但严格的‘效用提升且安全不退化’目标仍未完全满足：strong DBA 平均 ASR 为 **2.396% vs 1.795%**，RTC 高 **0.601 pp**。因此它是当前最优 RTC 研究基线，却不是 Multi-Krum 的无条件替代。",
                "layout": "full",
            },
            {"id": "headline-metrics", "type": "metric-strip", "cardIds": ["card-macro-acc", "card-lie05-acc", "card-dba-asr", "card-clean-final"], "layout": "full"},
            {
                "id": "key-findings",
                "type": "markdown",
                "body": "## 关键发现\n\n1. **ACC 差距已基本消除。** RTC 的条件级平均 ACC 差分别为 clean **+0.720 pp**、LIE z=.25 **+1.255 pp**、LIE z=.5 **+0.343 pp**、DBA **+0.100 pp**。\n2. **LIE z=.5 仍不稳定。** 三个 seed 的平均 ACC 差为 **+1.106/-0.673/+0.597 pp**，且 RTC 第 60 轮 ACC 平均低 **0.747 pp**。\n3. **DBA 暴露安全代价。** RTC 平均 ASR 高 **0.601 pp**，探索性配对 95% CI **[0.019, 1.183] pp**、p=**0.047**；这一结果不能被小幅 ACC 增益覆盖。",
                "layout": "full",
            },
            {"id": "mean-acc-chart", "type": "chart", "chartId": "chart-mean-acc-delta", "layout": "full"},
            {"id": "condition-table", "type": "table", "tableId": "table-condition-summary", "layout": "full"},
            {
                "id": "mechanism-diagnosis",
                "type": "markdown",
                "body": "## 机制诊断\n\n“RTC 错误损失良性质量”只解释了一部分。RTC 在 clean/LIE z=.5 的良性 clipping 分别为 **6.01%/5.07%**，并在 LIE z=.5 保留 **1.87% zero-update mass**，说明仍有良性效用税；但 clean ACC 反而高于 Multi-Krum，证明它已不是当前主导瓶颈。更关键的是，LIE z=.5 下 RTC 比 Multi-Krum 多保留 **9.31 pp 恶意权重**和 **11.08 pp 恶意 impact**，表明 RTC 对恶意更新的抑制仍不够。DBA 下 RTC 的直接恶意权重仅 **0.24%**、impact 仅 **0.42%**，ASR 却更高，说明权重/impact 代理没有完整刻画后门方向以及 anchor recycle 的几何效应。",
                "layout": "full",
            },
            {"id": "mechanism-table", "type": "table", "tableId": "table-mechanism", "layout": "full"},
            {"id": "dba-asr-chart", "type": "chart", "chartId": "chart-dba-asr", "layout": "full"},
            {"id": "paired-table", "type": "table", "tableId": "table-paired-seeds", "layout": "full"},
            {
                "id": "first-divergence",
                "type": "markdown",
                "body": "## 首次机制分叉\n\nRTC 与 Multi-Krum 在所有 12 个 condition/seed 配对中从首轮就因聚合选择不同而分叉；对三个攻击条件，指标窗口内首次客户端级分叉全部出现在攻击开始的 **round 11**。因此后续 ACC/ASR 差异是聚合机制从攻击起点持续累积的结果，不是末轮偶发噪声。",
                "layout": "full",
            },
            {"id": "divergence-table", "type": "table", "tableId": "table-divergence", "layout": "full"},
            {
                "id": "scope-method",
                "type": "markdown",
                "body": "## 范围、数据与方法\n\nV1 包含 seeds **42/46/47**、四个条件（显式 clean、LIE z=.25、LIE z=.5、strong DBA）和两种防御，共 **24 个单元**。RTC 固定 floor=.5、linear cumulative-q cap、accepted-anchor recycle fraction=.51、MAD k=2.5；Multi-Krum 固定 f=3、select=5；恶意比例=.3、IID、参与率=.5。clean 使用 rounds 1–60，攻击使用 planned-attack-active rounds 11–60。所有单元 exit code=0、rounds 完整 0–60、**52/52** 质量门通过；12 对均共享 trial-plan 与 attack-implementation hash。主分析器之外，纯标准库复核脚本从原始 rounds/client CSV 重算了全部指标。",
                "layout": "full",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## 局限性与不确定性\n\n三 seed 配对检验只有 **2 个自由度**，置信区间较宽，只适合工程方向判断；没有做等价性检验，‘不显著’不能解释为‘等价’。例如 LIE z=.5 平均 ACC 差的 95% CI 为 **[-1.933, 2.620] pp**。此外，clean 中‘良性身份’会排除名义恶意客户端，而‘非攻击者’包含所有未激活攻击的客户端；两种口径必须分开解释。若用于论文或强泛化声明，应补到至少五 seed，并预注册主要终点。",
                "layout": "full",
            },
            {
                "id": "recommendations",
                "type": "markdown",
                "body": "## 结论与下一步\n\n- 冻结 **B3R-F0.51** 作为当前最优 RTC 候选；它已达到缩小/逆转 ACC 差距的工程目标。\n- 不应宣称 RTC 全面优于 Multi-Krum，也不应在 DBA 安全优先场景直接替代 Multi-Krum。\n- 若开启下一项独立研究，优先针对 **LIE 恶意权重/impact 过高**与 **DBA 的方向敏感 anchor 安全门**，不要继续在同一三 seed 上做无预注册参数搜索。\n- 三 seed 足以停止当前微调并做工程决策；只有论文、等价性或稳健泛化声明才有必要补五 seed。",
                "layout": "full",
            },
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "condition_summary": condition_summary,
                "mechanism_summary": mechanism_summary,
                "paired_seeds": paired_rows,
                "dba_asr": dba_asr,
                "divergence": divergence_rows,
            },
        },
        "sources": [source_summary, source_condition_comparison, source_paired, source_runs, source_divergence],
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()
