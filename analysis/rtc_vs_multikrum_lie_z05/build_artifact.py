"""Build the canonical Data Analytics report artifact for the RTC diagnosis."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


OUTPUT = Path(__file__).resolve().parent
TITLE = "RTC 为什么经常不是最佳：跨攻击机制诊断与 LIE 深挖"


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None})
    return clean.to_dict(orient="records")


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary = json.loads((OUTPUT / "analysis_summary.json").read_text(encoding="utf-8"))
    round_comparison = pd.read_csv(OUTPUT / "round_comparison.csv")
    windows = pd.read_csv(OUTPUT / "window_summary.csv")
    signals = pd.read_csv(OUTPUT / "rtc_signal_separation.csv")
    states = pd.read_csv(OUTPUT / "rtc_state_summary.csv")
    weights = pd.read_csv(OUTPUT / "weight_mechanism.csv")
    scenarios = pd.read_csv(OUTPUT / "weight_scenario_summary.csv")
    defense_accuracy = pd.read_csv(OUTPUT / "defense_accuracy_summary.csv")
    cross_attacks = pd.read_csv(OUTPUT / "cross_attack_summary.csv")
    cross_attack_labels = {
        "dba": "DBA",
        "gaussian_noise": "Gaussian noise",
        "lie": "LIE",
        "min_max": "Min-Max",
        "min_sum": "Min-Sum",
        "scaling_backdoor": "Scaling backdoor",
    }
    cross_attacks["attack_label"] = cross_attacks["attack"].map(cross_attack_labels)

    accuracy_rows: list[dict] = []
    for _, row in round_comparison.iterrows():
        accuracy_rows.extend(
            [
                {
                    "round": int(row["round"]),
                    "series": "RTC",
                    "defense": "rtc_full",
                    "accuracy": float(row["rtc_accuracy"]),
                    "attack_active": int(row["attack_active"]),
                    "phase": str(row["phase"]),
                    "attacker_weight": None if pd.isna(row["rtc_attacker_weight"]) else float(row["rtc_attacker_weight"]),
                    "zero_update_mass": None if pd.isna(row["rtc_zero_update_mass"]) else float(row["rtc_zero_update_mass"]),
                    "selected_attackers": None if pd.isna(row["selected_attackers"]) else float(row["selected_attackers"]),
                },
                {
                    "round": int(row["round"]),
                    "series": "Multi-Krum",
                    "defense": "multi_krum",
                    "accuracy": float(row["multi_krum_accuracy"]),
                    "attack_active": int(row["attack_active"]),
                    "phase": str(row["phase"]),
                    "attacker_weight": None if pd.isna(row["multi_krum_attacker_weight"]) else float(row["multi_krum_attacker_weight"]),
                    "zero_update_mass": 0.0,
                    "selected_attackers": None if pd.isna(row["selected_attackers"]) else float(row["selected_attackers"]),
                },
            ]
        )

    signal_labels = {
        "semantic total risk": "语义总风险",
        "semantic max z": "语义 max-z",
        "raw update norm": "原始更新范数",
        "residual norm": "残差范数",
        "one minus cone similarity": "锥相似度反向",
        "one minus cumulative q": "累计证据 1-q",
    }
    selected_signals = signals[signals["signal"].isin(signal_labels)].copy()
    selected_signals["signal_label"] = selected_signals["signal"].map(signal_labels)
    selected_signals["auc_vs_random_pp"] = (
        selected_signals["malicious_detection_auc"] - 0.5
    ) * 100.0

    state_map = states.set_index("group")
    headline = [
        {
            "active_gap": summary["accuracy"]["rtc_minus_multi_krum_active_pp"] / 100.0,
            "rtc_active": summary["accuracy"]["rtc_active_mean"],
            "multi_krum_active": summary["accuracy"]["multi_krum_active_mean"],
            "rtc_attacker_weight": summary["mechanism"]["rtc_attacker_weight_share"],
            "multi_krum_attacker_weight": summary["mechanism"]["multi_krum_attacker_weight_share"],
            "attacker_weight_reduction": summary["mechanism"]["multi_krum_weight_reduction_vs_rtc"],
            "rtc_zero_update_mass": summary["mechanism"]["rtc_mean_zero_update_mass"],
            "benign_cap_rate": float(state_map.loc["benign", "cap_rate"]),
            "malicious_cap_rate": float(state_map.loc["malicious", "cap_rate"]),
        }
    ]

    tuning_plan = [
        {
            "priority": 2,
            "change": "仅对 LIE profile 关闭有害语义惩罚",
            "setting": "LIE: custom_params.semantic_ablation = observe",
            "evidence": "恶意语义风险均值 0；良性 0.0348。恶意 semantic max-z AUC=0.000。",
            "expected_effect": "去掉针对良性的 q/soft/exposure 压权；目标先回收约 0.5 pp 的平均 ACC 税。",
            "risk_gate": "clean RTC 不低于当前 RTC；恶意权重不高于 21.4%。",
        },
        {
            "priority": 3,
            "change": "累计 q 直接进入客户端 cap",
            "setting": "cap_i = nominal_i × min(semantic_q_i, cumulative_q_i, direction_q_i)",
            "evidence": "累计证据 AUC=0.705，但当前只缩残差预算，恶意客户端 cap 率仍为 0。",
            "expected_effect": "日志重放的权重上限估计把恶意质量从 21.4% 降到 19.1%；q² 版本降到 17.6%。",
            "risk_gate": "良性 cap <5%；active ACC 至少比当前 RTC +0.3 pp。",
        },
        {
            "priority": 1,
            "change": "零质量回填到稳健 anchor",
            "setting": "aggregate += (1 - Σw_i) × coordinate_median_anchor",
            "evidence": "RTC 平均 4.87% 零更新质量，50% 主动轮非零；会减慢学习。",
            "expected_effect": "恢复更新幅度，同时不把质量返还给单个高风险客户端。",
            "risk_gate": "clean/attack 都验证无数值放大；anchor 范数与 loss 无异常。",
        },
        {
            "priority": 4,
            "change": "增加轻量残差 rank-cap",
            "setting": "每轮 residual norm 前 2 名 cap×0.5，剩余 10% 质量给 anchor",
            "evidence": "残差范数 AUC=0.835；离线权重路径估计恶意质量降至 15.3%。",
            "expected_effect": "接近 Multi-Krum 的 12.3%，但只软压 2/10 客户端。",
            "risk_gate": "IID clean ACC 税 <0.3 pp；Non-IID 单独重标定，不直接复用。",
        },
        {
            "priority": 5,
            "change": "小步收紧 norm clipping",
            "setting": "把硬编码 2.5×MAD 参数化，试 2.25 与 2.0",
            "evidence": "恶意裁剪率 29.0%，良性 4.6%，原始范数 AUC=0.790。",
            "expected_effect": "降低 LIE 单次更新影响；不能替代权重筛选。",
            "risk_gate": "良性裁剪 <8%；clean ACC 税 <0.3 pp。",
        },
    ]

    experiment_grid = [
        {"order": 1, "cell": "B0", "configuration": "当前 RTC", "purpose": "基线复现"},
        {"order": 2, "cell": "B1", "configuration": "semantic_ablation=observe", "purpose": "量化语义误伤税"},
        {"order": 3, "cell": "B2", "configuration": "B1 + direct cumulative cap", "purpose": "验证信号到权重连接"},
        {"order": 4, "cell": "B3", "configuration": "B2 + anchor mass recycle", "purpose": "恢复有效步长"},
        {"order": 5, "cell": "B4", "configuration": "B3 + top-2 residual cap×0.5", "purpose": "接近 Multi-Krum 攻击者质量"},
        {"order": 6, "cell": "B5", "configuration": "B3 + clip 2.25×MAD", "purpose": "检验轻量裁剪增益"},
    ]

    report_frames = {
        "headline": pd.DataFrame(headline),
        "accuracy_timeline": pd.DataFrame(accuracy_rows),
        "window_summary": windows,
        "signal_auc": selected_signals,
        "rtc_states": states,
        "weight_mechanism": weights,
        "weight_scenarios": scenarios,
        "defense_accuracy": defense_accuracy,
        "tuning_plan": pd.DataFrame(tuning_plan),
        "experiment_grid": pd.DataFrame(experiment_grid),
        "cross_attack_summary": cross_attacks,
    }
    report_queries = {
        "headline": "SELECT * FROM headline",
        "accuracy_timeline": "SELECT * FROM accuracy_timeline ORDER BY round, series",
        "window_summary": "SELECT * FROM window_summary ORDER BY start_round",
        "signal_auc": "SELECT * FROM signal_auc ORDER BY malicious_detection_auc",
        "rtc_states": "SELECT * FROM rtc_states ORDER BY group_name",
        "weight_mechanism": "SELECT * FROM weight_mechanism ORDER BY defense",
        "weight_scenarios": "SELECT * FROM weight_scenarios ORDER BY attacker_client_mass DESC",
        "defense_accuracy": "SELECT * FROM defense_accuracy ORDER BY active_accuracy DESC",
        "tuning_plan": "SELECT * FROM tuning_plan ORDER BY priority",
        "experiment_grid": "SELECT * FROM experiment_grid ORDER BY \"order\"",
        "cross_attack_summary": "SELECT * FROM cross_attack_summary ORDER BY rtc_balanced_gap_pp",
    }
    database_path = OUTPUT / "report_snapshot.sqlite"
    with sqlite3.connect(database_path) as connection:
        for table_name, frame in report_frames.items():
            sqlite_frame = frame.rename(columns={"group": "group_name"}) if table_name == "rtc_states" else frame
            sqlite_frame.to_sql(table_name, connection, if_exists="replace", index=False)
        report_datasets = {
            dataset: records(pd.read_sql_query(query, connection))
            for dataset, query in report_queries.items()
        }

    sources = [
        {
            "id": "src_rounds",
            "label": "严格配对 LIE z=0.5 逐轮分析表",
            "path": "analysis/rtc_vs_multikrum_lie_z05/report_snapshot.sqlite",
            "query": {
                "sql": report_queries["accuracy_timeline"],
                "description": "从经诊断脚本核验的逐轮分析表读取 RTC 与 Multi-Krum 轨迹；上游为两种防御的 61 行 round CSV。",
                "engine": "sqlite",
                "language": "sql",
                "filters": ["seed=42", "lie_z=0.5", "malicious_fraction=0.2", "partition=iid", "active rounds=11-60"],
                "tables_used": [
                    "logs/rtc_v3_lie_z05_seed42_mf03/rounds/*rtc_full*.csv",
                    "logs/rtc_v3_lie_z05_seed42_mf03/rounds/*multi_krum*.csv",
                    "logs/rtc_v3_lie_z05_seed42_mf03/rounds/*fedavg*.csv",
                ],
                "metric_definitions": [
                    "active ACC = round 11-60 server_accuracy arithmetic mean",
                    "attacker weight = active_attacker_weight_share arithmetic mean",
                    "gap = RTC ACC - Multi-Krum ACC in percentage points",
                ],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_clients",
            "label": "RTC 客户端信号 AUC 分析表",
            "path": "analysis/rtc_vs_multikrum_lie_z05/report_snapshot.sqlite",
            "query": {
                "sql": report_queries["signal_auc"],
                "description": "读取按真实恶意标签计算的客户端信号 AUC；上游为 RTC 与 Multi-Krum 各 600 行 client CSV。",
                "engine": "sqlite",
                "language": "sql",
                "filters": ["rounds=11-60", "10 selected clients per round", "true is_malicious label"],
                "tables_used": [
                    "logs/rtc_v3_lie_z05_seed42_mf03/raw/*rtc_full*_clients.csv",
                    "logs/rtc_v3_lie_z05_seed42_mf03/raw/*multi_krum*_clients.csv",
                ],
                "metric_definitions": [
                    "detection AUC = pairwise rank probability that a malicious observation has a higher suspiciousness score",
                    "cap rate = capped=true observations / group observations",
                    "positive weight = aggregation_weight > 1e-12",
                ],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_analysis",
            "label": "离线权重路径分析表",
            "path": "analysis/rtc_vs_multikrum_lie_z05/report_snapshot.sqlite",
            "query": {
                "sql": report_queries["weight_scenarios"],
                "description": "读取诊断脚本产生的离线权重路径估计；这些估计不模拟模型参数或反事实 ACC。",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": [
                    "analysis/rtc_vs_multikrum_lie_z05/round_comparison.csv",
                    "analysis/rtc_vs_multikrum_lie_z05/rtc_signal_separation.csv",
                    "analysis/rtc_vs_multikrum_lie_z05/weight_scenario_summary.csv",
                ],
                "metric_definitions": [
                    "block bootstrap CI uses circular five-round blocks and describes within-run temporal uncertainty only",
                    "weight scenarios do not claim counterfactual ACC",
                ],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_config",
            "label": "RTC V3 标定清单与实现",
            "path": "config/rtc_v3_manifest_formal_iid_semantic.json",
            "query": {
                "description": "核对 semantic temporal、cumulative、direction、QP cap 与 clipping 的实际连接方式。",
                "tables_used": [
                    "config/rtc_v3_manifest_formal_iid_semantic.json",
                    "defenses/rtc/v3.py",
                    "defenses/rtc/solver.py",
                    "defenses/rtc/semantic_temporal.py",
                ],
                "metric_definitions": [
                    "current QP client cap uses nominal × semantic_q",
                    "cumulative_q currently scales principal residual budgets rather than the direct client cap",
                    "clip norm uses median + 2.5 × 1.4826 × MAD, bounded by phase limits",
                ],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_headline",
            "label": "报告头部指标分析表",
            "path": "analysis/rtc_vs_multikrum_lie_z05/report_snapshot.sqlite",
            "query": {
                "sql": report_queries["headline"],
                "description": "读取由严格配对逐轮和客户端分析计算的报告头部指标。",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": ["headline"],
                "filters": ["active rounds=11-60", "seed=42", "LIE z=0.5"],
                "metric_definitions": [
                    "active_gap = RTC active ACC - Multi-Krum active ACC",
                    "attacker_weight_reduction = 1 - Multi-Krum attacker weight / RTC attacker weight",
                    "cap rate = capped client observations / group observations",
                ],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_windows",
            "label": "分阶段 ACC 与权重分析表",
            "path": "analysis/rtc_vs_multikrum_lie_z05/report_snapshot.sqlite",
            "query": {
                "sql": report_queries["window_summary"],
                "description": "读取按预热期和十轮主动窗口聚合的 ACC 与攻击者权重。",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": ["window_summary"],
                "filters": ["rounds=1-60", "fixed windows"],
                "metric_definitions": ["gap pp = 100 × (RTC ACC - Multi-Krum ACC)"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_tuning",
            "label": "RTC 微调优先级表",
            "path": "analysis/rtc_vs_multikrum_lie_z05/report_snapshot.sqlite",
            "query": {
                "sql": report_queries["tuning_plan"],
                "description": "读取由信号分离、误伤率、权重路径和实现核对共同支持的微调清单。",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": ["tuning_plan"],
                "filters": ["priority=1-5"],
                "metric_definitions": ["priority is ordered by expected utility, implementation size, and validation risk"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_experiments",
            "label": "最小消融实验矩阵",
            "path": "analysis/rtc_vs_multikrum_lie_z05/report_snapshot.sqlite",
            "query": {
                "sql": report_queries["experiment_grid"],
                "description": "读取逐项新增改动的 B0-B5 验证矩阵。",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": ["experiment_grid"],
                "filters": ["B0-B5 in execution order"],
                "metric_definitions": ["each cell adds one change for attribution"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_cross_attack",
            "label": "Seed-42 强攻击跨防御与客户端机制分析表",
            "path": "analysis/rtc_vs_multikrum_lie_z05/report_snapshot.sqlite",
            "query": {
                "sql": report_queries["cross_attack_summary"],
                "description": "读取更正后的 seed-42 强攻击目录：六种完整攻击的五防御比较、内部 clean FedAvg 反事实，以及 RTC/Multi-Krum 客户端权重。",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": [
                    "cross_attack_summary",
                    "logs/rtc_v3_byzantine_strong_seed42_mf03/periodic_attack_runs.csv",
                    "logs/rtc_v3_byzantine_strong_seed42_mf03/raw/*_clients.csv",
                    "logs/rtc_v3_byzantine_strong_seed42_mf03/rounds/*.csv",
                ],
                "filters": ["seed=42", "formal malicious_fraction=0.3", "active rounds=11-60", "complete five-defense attacks only"],
                "metric_definitions": [
                    "balanced best = highest active ACC; for targeted attacks only defenses with active ASR <= 10% are eligible",
                    "benign mass loss = nominal benign client share - actual benign aggregation mass",
                    "zero-update mass = 1 - sum of client aggregation weights",
                    "attacker retention = actual attacker mass / nominal selected-attacker share",
                ],
                "executed_at": generated_at,
            },
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Seed-42 cross-attack diagnosis of RTC utility and robustness, with a detailed LIE mechanism analysis and ACC-focused tuning plan.",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "card_gap",
                "dataset": "headline",
                "sourceId": "src_headline",
                "description": "第 11–60 轮 RTC 相对 Multi-Krum 的平均 ACC 差。",
                "metrics": [
                    {"label": "RTC − Multi-Krum", "field": "active_gap", "format": "percent", "signed": True},
                    {"label": "RTC", "field": "rtc_active", "format": "percent"},
                    {"label": "Multi-Krum", "field": "multi_krum_active", "format": "percent"},
                ],
            },
            {
                "id": "card_weight",
                "dataset": "headline",
                "sourceId": "src_headline",
                "description": "主动期恶意客户端获得的绝对聚合质量。",
                "metrics": [
                    {"label": "RTC 恶意权重", "field": "rtc_attacker_weight", "format": "percent"},
                    {"label": "Multi-Krum", "field": "multi_krum_attacker_weight", "format": "percent"},
                    {"label": "相对减少", "field": "attacker_weight_reduction", "format": "percent"},
                ],
            },
            {
                "id": "card_utility_tax",
                "dataset": "headline",
                "sourceId": "src_headline",
                "description": "RTC 未分配给客户端的质量与错误压低良性权重的比例。",
                "metrics": [
                    {"label": "RTC 零更新质量", "field": "rtc_zero_update_mass", "format": "percent"},
                    {"label": "良性 cap", "field": "benign_cap_rate", "format": "percent"},
                    {"label": "恶意 cap", "field": "malicious_cap_rate", "format": "percent"},
                ],
            },
        ],
        "charts": [
            {
                "id": "chart_accuracy",
                "title": "RTC 与 Multi-Krum 的服务器准确率",
                "subtitle": "60 轮严格配对；第 11 轮开始连续 LIE，后半程差距与攻击者权重分叉同步扩大。",
                "intent": "trend",
                "question": "两种防御的 ACC 差距何时出现？",
                "rationale": "逐轮轨迹能区分早期 Multi-Krum 震荡与后期稳定收益。",
                "comparisonContext": {"baseline": "same TrialPlan", "grain": "federated round", "unit": "accuracy rate", "denominator": "CIFAR-10 test set"},
                "type": "line",
                "dataset": "accuracy_timeline",
                "sourceId": "src_rounds",
                "encodings": {
                    "x": {"field": "round", "type": "quantitative", "label": "Round"},
                    "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Server accuracy"},
                    "color": {"field": "series", "type": "nominal", "label": "Defense"},
                    "tooltip": [
                        {"field": "phase", "type": "nominal", "label": "RTC phase"},
                        {"field": "attacker_weight", "type": "quantitative", "format": "percent", "label": "Attacker weight"},
                        {"field": "zero_update_mass", "type": "quantitative", "format": "percent", "label": "Zero-update mass"},
                        {"field": "selected_attackers", "type": "quantitative", "label": "Selected attackers"},
                    ],
                },
                "valueFormat": "percent",
                "palette": {"kind": "categorical"},
                "referenceLines": [{"axis": "x", "value": 11, "label": "Attack starts", "color": "neutral", "lineStyle": "dashed"}],
                "settings": {"showPoints": "hover"},
                "surface": {"surface": "card", "viewMode": "visualization"},
            },
            {
                "id": "chart_signal_auc",
                "title": "RTC 客户端信号的恶意识别 AUC",
                "subtitle": "0.5 为随机；语义信号低于随机，残差与累计信号高于随机。",
                "intent": "comparison",
                "question": "哪些现有 RTC 信号对 LIE 真正有分离力？",
                "rationale": "AUC 统一不同量纲，可直接比较分离方向。",
                "comparisonContext": {"baseline": "AUC=0.5", "grain": "selected client-round", "unit": "rank AUC", "denominator": "107 malicious and 393 benign observations"},
                "type": "bar",
                "dataset": "signal_auc",
                "sourceId": "src_clients",
                "encodings": {
                    "x": {"field": "signal_label", "type": "nominal", "label": "Signal"},
                    "y": {"field": "malicious_detection_auc", "type": "quantitative", "format": "percent", "label": "Detection AUC"},
                    "tooltip": [
                        {"field": "malicious_mean", "type": "quantitative", "label": "Malicious mean"},
                        {"field": "benign_mean", "type": "quantitative", "label": "Benign mean"},
                        {"field": "auc_vs_random_pp", "type": "quantitative", "label": "vs random, pp"},
                    ],
                },
                "valueFormat": "percent",
                "palette": {"kind": "single"},
                "referenceLines": [{"axis": "y", "value": 0.5, "label": "Random", "color": "neutral", "lineStyle": "dashed"}],
                "surface": {"surface": "card", "viewMode": "visualization"},
            },
            {
                "id": "chart_cross_attack_gap",
                "title": "RTC active ACC 与安全合格最佳防御的差距",
                "subtitle": "Seed 42；DBA/Scaling 只和 active ASR≤10% 的防御比较，负值表示 RTC 较低。",
                "intent": "comparison",
                "question": "RTC 的非第一名表现是普遍小差距，还是由少数攻击主导？",
                "rationale": "按攻击展示与安全合格最佳防御的差距，可区分排名噪声与实质失效。",
                "comparisonContext": {"baseline": "best security-qualified defense", "grain": "attack", "unit": "percentage points", "denominator": "CIFAR-10 test accuracy"},
                "type": "bar",
                "dataset": "cross_attack_summary",
                "sourceId": "src_cross_attack",
                "encodings": {
                    "x": {"field": "attack_label", "type": "nominal", "label": "Attack"},
                    "y": {"field": "rtc_balanced_gap_pp", "type": "quantitative", "format": "number", "label": "RTC gap, pp"},
                    "tooltip": [
                        {"field": "rtc_active_accuracy", "type": "quantitative", "format": "percent", "label": "RTC active ACC"},
                        {"field": "best_balanced_defense", "type": "nominal", "label": "Balanced best"},
                        {"field": "attacker_mass_retention", "type": "quantitative", "format": "percent", "label": "Attacker retention"},
                        {"field": "benign_mass_loss_vs_nominal", "type": "quantitative", "format": "percent", "label": "Benign mass loss"},
                        {"field": "zero_update_mass", "type": "quantitative", "format": "percent", "label": "Zero mass"},
                    ],
                },
                "valueFormat": "number",
                "palette": {"kind": "single"},
                "referenceLines": [{"axis": "y", "value": 0, "label": "Best", "color": "neutral", "lineStyle": "solid"}],
                "surface": {"surface": "card", "viewMode": "visualization"},
            },
        ],
        "tables": [
            {
                "id": "table_cross_attack",
                "title": "跨攻击 RTC 性能与权重机制",
                "subtitle": "Seed 42、恶意比例 30%、第 11–60 轮；targeted 攻击的 best 需 active ASR≤10%。",
                "dataset": "cross_attack_summary",
                "sourceId": "src_cross_attack",
                "density": "spacious",
                "defaultSort": {"field": "rtc_balanced_gap_pp", "direction": "asc"},
                "columns": [
                    {"field": "attack_label", "label": "Attack", "type": "text"},
                    {"field": "rtc_active_accuracy", "label": "RTC ACC", "format": "percent"},
                    {"field": "best_balanced_defense", "label": "Balanced best", "type": "text"},
                    {"field": "rtc_balanced_gap_pp", "label": "Gap, pp", "format": "number", "movement": True},
                    {"field": "attacker_mass_retention", "label": "Attacker retained", "format": "percent"},
                    {"field": "benign_mass_loss_vs_nominal", "label": "Benign mass loss", "format": "percent"},
                    {"field": "zero_update_mass", "label": "Zero mass", "format": "percent"},
                    {"field": "diagnosis", "label": "Mechanism", "type": "text"},
                ],
            },
            {
                "id": "table_windows",
                "title": "分阶段 ACC 与攻击者权重",
                "subtitle": "gap 为 RTC − Multi-Krum；正值表示 RTC 更高。",
                "dataset": "window_summary",
                "sourceId": "src_windows",
                "density": "spacious",
                "defaultSort": {"field": "start_round", "direction": "asc"},
                "columns": [
                    {"field": "window", "label": "Window", "type": "text"},
                    {"field": "start_round", "label": "Start", "format": "number"},
                    {"field": "rtc_accuracy", "label": "RTC ACC", "format": "percent"},
                    {"field": "multi_krum_accuracy", "label": "Multi-Krum ACC", "format": "percent"},
                    {"field": "rtc_minus_multi_krum_pp", "label": "Gap, pp", "format": "number", "movement": True},
                    {"field": "rtc_attacker_weight", "label": "RTC attacker", "format": "percent"},
                    {"field": "multi_krum_attacker_weight", "label": "MK attacker", "format": "percent"},
                ],
            },
            {
                "id": "table_weight_scenarios",
                "title": "权重路径离线估计",
                "subtitle": "只估计恶意/良性/anchor 质量，不是反事实 ACC。",
                "dataset": "weight_scenarios",
                "sourceId": "src_analysis",
                "density": "spacious",
                "defaultSort": {"field": "attacker_client_mass", "direction": "desc"},
                "columns": [
                    {"field": "scenario", "label": "Scenario", "type": "text"},
                    {"field": "attacker_client_mass", "label": "Attacker mass", "format": "percent"},
                    {"field": "benign_client_mass", "label": "Benign mass", "format": "percent"},
                    {"field": "robust_anchor_or_zero_mass", "label": "Anchor/zero mass", "format": "percent"},
                    {"field": "attacker_mass_reduction_vs_current", "label": "Reduction", "format": "percent", "movement": True},
                ],
            },
            {
                "id": "table_tuning",
                "title": "ACC 微调优先级",
                "subtitle": "先恢复良性学习质量，再把有效信号接入权重 cap。",
                "dataset": "tuning_plan",
                "sourceId": "src_tuning",
                "density": "spacious",
                "defaultSort": {"field": "priority", "direction": "asc"},
                "columns": [
                    {"field": "priority", "label": "P", "format": "number"},
                    {"field": "change", "label": "Change", "type": "text"},
                    {"field": "setting", "label": "Setting", "type": "text"},
                    {"field": "expected_effect", "label": "Expected mechanism", "type": "text"},
                    {"field": "risk_gate", "label": "Gate", "type": "text"},
                ],
            },
            {
                "id": "table_experiments",
                "title": "最小实验矩阵",
                "subtitle": "每次只新增一项，避免无法归因。",
                "dataset": "experiment_grid",
                "sourceId": "src_experiments",
                "density": "spacious",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "#", "format": "number"},
                    {"field": "cell", "label": "Cell", "type": "text"},
                    {"field": "configuration", "label": "Configuration", "type": "text"},
                    {"field": "purpose", "label": "Purpose", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {TITLE}", "layout": "full"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "layout": "full",
                "body": "## 技术摘要\n\n**不能把 RTC 在所有攻击下非第一名统一归因为‘错误损失良性质量’。** 在更正后的 seed-42、恶意比例 30% 强攻击五防御对照中，RTC 对 Min-Max 的 active ACC 是最高，对 Min-Sum 只落后 0.19 pp。按 targeted 攻击先要求 active ASR≤10% 后再比 ACC，六种完整攻击中有五种 RTC 距最佳不超过 0.7 pp，唯一实质性异常是 LIE（−16.01 pp）。\n\n跨攻击存在三类机制：LIE 同时保留 99.9% 恶意质量并损失 7.72 pp 良性质量；Gaussian noise 没有良性质量损失，但仍保留 93.4% 恶意质量；DBA/Scaling 则正确压掉约 99.7% 恶意质量，却留下 29.95%–33.45% 零更新质量，形成安全—效用税。Min-Max/Min-Sum 相对各自 clean FedAvg 的攻击降幅仅 3.80/2.75 pp，且 RTC 没有良性 state/cap 误伤。\n\n因此微调应分成两个公共方向：把被拒绝质量回填到稳健 anchor，解决所有‘正确拦截但步长缩水’场景；再按攻击家族把 residual/cumulative 信号接入直接权重 cap，解决 LIE/Gaussian 的‘检测信息未转成权重’。`semantic_ablation=observe` 主要是 LIE 专项，不应跨攻击默认关闭。",
            },
            {
                "id": "cross_attack_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_cross_attack",
                "body": "## 良性质量损失只在 LIE 中是主因，其他攻击由不同机制主导\n\n**LIE 是双重失效。** RTC 恶意质量保留率为 99.9%，良性质量相对名义份额少 7.72 pp，良性非 normal 率 13.7%；这与 −16.01 pp 的实质差距一致。**Gaussian noise 是漏压恶意而非误伤良性：**良性质量损失为 0，但恶意质量保留率仍有 93.4%，尽管恶意 clip 率为 100%。**DBA 与 Scaling 是有效防御后的步长税：**恶意质量都降到约 0.1%，ASR 分别为 2.25% 和 1.71%，但 33.45% 和 29.95% 的质量没有被重新分配；它们相对安全合格最佳防御只低 0.46/0.61 pp。\n\nMin-Max/Min-Sum 则提供反例：良性质量损失、零质量和良性 state 误伤均为 0；RTC 分别为最佳和只低 0.19 pp。更正目录内部的 clean FedAvg 反事实显示，这两种攻击使 FedAvg active ACC 分别下降 3.80/2.75 pp，明显弱于 LIE 的 25.90 pp，因此其防御排名证据应低权重解释。",
            },
            {"id": "cross_attack_chart_block", "type": "chart", "chartId": "chart_cross_attack_gap", "layout": "full"},
            {"id": "cross_attack_table_block", "type": "table", "tableId": "table_cross_attack", "layout": "full"},
            {
                "id": "lie_detail_intro",
                "type": "markdown",
                "layout": "full",
                "body": "## LIE 是需要单独修复的异常路径\n\n下面保留 LIE z=0.5、实际恶意比例 20% 的独立严格配对深挖。它与跨攻击强攻击集的 LIE（恶意比例 30%、强档）不是同一次运行，但两者共同指向：语义信号对 LIE 方向错误，累计/残差信号没有充分进入直接客户端权重。",
            },
            {"id": "kpis", "type": "metric-strip", "cardIds": ["card_gap", "card_weight", "card_utility_tax"], "layout": "full"},
            {
                "id": "accuracy_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_rounds",
                "body": "## RTC 并非全程更差，分叉从第 21 轮后形成\n\n攻击刚开始的第 11–20 轮，RTC 反而领先 5.50 pp，因为 Multi-Krum 当时仍给攻击者 28.3% 的聚合质量并出现明显震荡。第 21 轮后，Multi-Krum 的攻击者质量降到 0%–16.7% 的多数轮，而 RTC 始终接近当轮名义恶意占比。第 31–50 轮 RTC 平均落后约 3.1 pp。\n\n因此“RTC 一直略差”需要修正为：**长期平均略差，但早期领先、后期在攻击者权重分叉后落后。** 末轮 −4.78 pp 是路径依赖单点，主读数仍应是完整主动期 −0.60 pp。",
            },
            {"id": "accuracy_chart_block", "type": "chart", "chartId": "chart_accuracy", "layout": "full"},
            {"id": "windows_table_block", "type": "table", "tableId": "table_windows", "layout": "full"},
            {
                "id": "mechanism_finding",
                "type": "markdown",
                "layout": "full",
                "body": "## 在 LIE 中，Multi-Krum 赢在降攻击者质量，RTC 同时损失良性质量\n\nMulti-Krum 的客户端选择将恶意权重从名义 21.4% 降到 12.3%，相对减少 42.4%。RTC 的恶意权重仍是 21.4%，与 FedAvg 相同；与此同时其 ACC 比 FedAvg 低 0.56 pp，而 Multi-Krum 只比 FedAvg 高 0.03 pp。这说明本条件下的总体差距几乎全部可解释为：Multi-Krum 的筛选收益抵消了选择损失，而 RTC 没压攻击者、却承担了自身误伤和次概率质量损失。\n\n逐轮恶意权重差与 ACC 差的 Spearman 相关为 −0.862，但两条模型轨迹相互依赖，不能当作独立样本的因果估计。",
            },
            {"id": "weight_scenario_block", "type": "table", "tableId": "table_weight_scenarios", "layout": "full"},
            {
                "id": "signal_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_clients",
                "body": "## 语义信号方向反了，残差与累计信号才值得接入 cap\n\n恶意客户端的语义总风险均值为 0，良性为 0.0348；恶意 semantic max-z 均值 0.305，良性 2.947。语义 max-z 的恶意识别 AUC 为 0.000，即在这 500 个主动期观测中方向完全反向。降低语义 threshold、kappa 或 hard-exposure floor 会更强地惩罚良性客户端。\n\n相反，原始更新范数 AUC=0.790、残差范数 AUC=0.835、累计证据 AUC=0.705。现有 cumulative q 的确更常压到恶意客户端，但实现只用它缩小 principal residual budget；QP 的直接客户端 cap 仍主要是 `nominal × semantic_q`，所以恶意 cap 率最终仍为 0。",
            },
            {"id": "signal_chart_block", "type": "chart", "chartId": "chart_signal_auc", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_cross_attack",
                "body": "## 范围、数据与指标口径\n\n- 跨攻击正式数据仅使用 `rtc_v3_byzantine_strong_seed42_mf03`：CIFAR-10 / ResNet-18、IID、seed 42、20 个客户端、每轮 10 个、本地 5 epoch、60 轮。\n- 恶意比例为 30%，攻击从第 11 轮持续到实验结束；六个攻击均有 RTC/FedAvg/Krum/Multi-Krum/Median 五种完整有效结果，并各有一条目录内部 clean FedAvg 配对反事实。\n- Sign-flip 只有 RTC 一条运行，不进入防御排名。\n- LIE 专项深挖仍来自 `rtc_v3_lie_z05_seed42_mf03`，目录名虽含 `mf03`，实际执行恶意比例为 20%、z=0.5；它只用于解释 LIE 的信号到权重路径。\n- active ACC 是第 11–60 轮 server accuracy 算术平均；targeted 攻击的 balanced best 先要求 active ASR≤10%，再在合格防御中取最高 ACC。",
            },
            {
                "id": "cross_methodology",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_cross_attack",
                "body": "## 跨攻击方法：先按安全目标筛选，再分解权重质量\n\n诊断确认更正目录的 37 个 `run_id` 唯一、全部有效且每个逐轮文件完整覆盖第 0–60 轮，再只纳入具有五种完整防御的六类攻击。untargeted 攻击直接比较 active ACC；DBA/Scaling 先过滤 active ASR>10% 的防御，再比较合格防御 ACC。客户端层按第 11–60 轮对齐每种 RTC/Multi-Krum 运行的 500 个 selected-client 观测，分解名义恶意份额、实际恶意/良性质量、零质量、state/cap/clip 率与恶意识别 AUC。",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_analysis",
                "body": "## LIE 专项方法：复现差距并核对信号到权重路径\n\n诊断脚本检查 6 个状态文件均完成到第 60 轮、所有 round 表严格为 0–60、RTC/Multi-Krum client 表均为 600 行且 `(round,cid)` 唯一。随后按轮对齐 ACC 与攻击者权重，按真实恶意标签计算信号 AUC、clip/cap/state 率，并从实现核对哪些 q 实际进入 QP 上限。离线权重方案只重放 cap 和质量分配，不模拟模型参数，因此不把估计的攻击者质量变化写成 ACC 增益。",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": "## 限制、稳健性与反例\n\n- **两套正式分析都只有 seed 42。** 轮次是同一训练轨迹而非独立重复，不能把跨攻击的六个点当成独立统计重复。\n- **缺少 clean RTC/Multi-Krum 反事实。** 每个 TrialPlan 只有 clean FedAvg；零质量造成的 ACC 税需要 anchor-recycle 消融才能做因果确认。\n- **Min-Max/Min-Sum 的正式攻击效应较弱。** 相对目录内部 clean FedAvg 仅下降 3.80/2.75 pp，因此它们只能作为‘没有普遍误伤’的反例，不能证明强鲁棒性。\n- **Sign-flip 只有 RTC 一条运行。** 可观察其权重机制，但不能参与‘是否最佳’排名。\n- **Multi-Krum 理论前提并非每轮成立。** 实际抽中的恶意数量可超过配置 `f`，当前优势是经验结果。\n- **IID 限定。** residual rank-cap 在 Non-IID 可能误伤自然群组，必须重新标定。",
            },
            {
                "id": "tuning",
                "type": "markdown",
                "layout": "full",
                "body": "## 微调顺序：公共步长修复与 LIE 专项要分开验证\n\n跨攻击公共项应优先验证 anchor mass recycle，因为 DBA/Scaling 的 30%–33% 零质量主要来自正确剔除恶意客户端，不应简单归一化回客户端。LIE 专项再按 `semantic_ablation=observe` → direct cumulative cap → residual rank-cap 逐项消融，分别回答语义模块是否误伤、有效累计/残差信号能否转成直接权重。不要把 `semantic_ablation=observe` 当作所有攻击的全局默认；DBA/Scaling 的语义信号 AUC=1，关闭后可能损害后门防御。",
            },
            {"id": "tuning_table_block", "type": "table", "tableId": "table_tuning", "layout": "full"},
            {
                "id": "experiments",
                "type": "markdown",
                "layout": "full",
                "body": "## 推荐验证矩阵与晋级门槛\n\n使用 held-out seeds 42/46/47/48/51，至少覆盖 clean、LIE z=0.25/0.5/1.0；所有 cell 共享 TrialPlan。主门槛：z=0.5 active ACC 相对当前 RTC 至少 +0.3 pp，且不低于 Multi-Krum 0.2 pp；恶意权重 ≤15%；clean ACC 税 <0.3 pp；良性 cap <5%；anchor 回填后 zero mass≈0。报告均值、seed-level 配对差和 95% CI，不再用 50 个路径依赖轮次替代重复实验。",
            },
            {"id": "experiments_table_block", "type": "table", "tableId": "table_experiments", "layout": "full"},
            {
                "id": "further_questions",
                "type": "markdown",
                "layout": "full",
                "body": "## 进一步问题\n\n1. `semantic_ablation=observe` 在 clean RTC 上能回收多少 ACC，是否会恶化 label-flip 等已验证攻击？\n2. cumulative q 直接进 cap 后，最佳形式是 `q`、`q²`，还是 phase-specific 映射？\n3. anchor mass recycle 对优化步长和收敛速度的收益是否稳定，是否需要范数上限？\n4. residual rank-cap 在 Dirichlet α=0.5/0.3 下的良性误伤有多大？\n5. `mf03` 命名与实际 0.2 的不一致应先修正，避免下一轮实验错误归档。",
            },
        ],
    }

    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": report_datasets,
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }
    (OUTPUT / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifact": str(OUTPUT / "artifact.json"), "datasets": {k: len(v) for k, v in snapshot["datasets"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
