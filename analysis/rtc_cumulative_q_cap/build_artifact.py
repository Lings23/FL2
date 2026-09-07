"""Build the bounded technical-report artifact for the completed RTC-v3 B2 screen."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "rtc_cumulative_q_cap"
BASELINE = ROOT / "logs" / "rtc_v3_semantic_restricted_only_seed42_mf03"
CANDIDATE = ROOT / "logs" / "rtc_v3_cumulative_q_cap_seed42_mf03"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def one_file(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, got {len(matches)}")
    return matches[0]


def number(value: str | None) -> float | None:
    return None if value in {None, ""} else float(value)


def write_sqlite(tables: dict[str, list[dict[str, object]]]) -> None:
    database = OUT / "report_snapshot.sqlite"
    with sqlite3.connect(database) as connection:
        for table, rows in tables.items():
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            if not rows:
                continue
            fields = list(rows[0])
            types: dict[str, str] = {}
            for field in fields:
                values = [row[field] for row in rows if row[field] is not None]
                if values and all(isinstance(value, bool) for value in values):
                    types[field] = "INTEGER"
                elif values and all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in values
                ):
                    types[field] = "REAL"
                else:
                    types[field] = "TEXT"
            columns = ", ".join(f'"{field}" {types[field]}' for field in fields)
            connection.execute(f'CREATE TABLE "{table}" ({columns})')
            placeholders = ", ".join("?" for _ in fields)
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [tuple(row[field] for field in fields) for row in rows],
            )


def source(
    source_id: str,
    label: str,
    table: str,
    description: str,
    definitions: list[str],
    generated_at: str,
) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": "analysis/rtc_cumulative_q_cap/report_snapshot.sqlite",
        "query": {
            "description": description,
            "engine": "sqlite",
            "language": "sql",
            "filters": [
                "seed=42",
                "malicious_fraction=0.3",
                "partition=iid",
                "attack rounds=11-60",
                "LIE z=0.5 or strong DBA",
            ],
            "tables_used": [table],
            "metric_definitions": definitions,
            "executed_at": generated_at,
            "sql": f'SELECT * FROM "{table}"',
        },
    }


def main() -> None:
    comparison = [
        {
            key: (
                value
                if key in {"attack", "mode"}
                else number(value)
            )
            for key, value in row.items()
        }
        for row in read_csv(OUT / "comparison.csv")
    ]
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    divergence = [
        {
            key: (
                value
                if key == "attack"
                else str(value).lower() == "true"
                if key == "is_malicious"
                else int(value)
                if key in {"round", "cid"}
                else number(value)
            )
            for key, value in row.items()
        }
        for row in read_csv(OUT / "first_weight_divergence.csv")
    ]

    timeline: list[dict[str, object]] = []
    for label, directory, pattern in (
        ("B1R floor=0.5", BASELINE, "lie*rtc_semantic_restricted_only*.csv"),
        ("B2 linear q cap", CANDIDATE, "lie*rtc_cumulative_q_cap*.csv"),
    ):
        for row in read_csv(one_file(directory / "rounds", pattern)):
            timeline.append(
                {
                    "round": int(row["round"]),
                    "mode": label,
                    "accuracy": float(row["server_accuracy"]),
                    "malicious_weight_share": float(
                        row.get("fit_malicious_aggregation_weight_share") or 0.0
                    ),
                    "malicious_impact_share": float(
                        row.get("fit_malicious_impact_share") or 0.0
                    ),
                    "zero_update_mass": float(
                        row.get("fit_rtc_v3_zero_update_mass") or 0.0
                    ),
                    "client_q_cap_active_count": float(
                        row.get("fit_rtc_v3_client_q_cap_active_count") or 0.0
                    ),
                    "attack_active": int(row["round"]) >= 11,
                }
            )

    client_audit: list[dict[str, object]] = []
    for attack in ("lie", "dba"):
        rows = [
            row
            for row in read_csv(
                one_file(CANDIDATE / "raw", f"{attack}*rtc_cumulative_q_cap*_clients.csv")
            )
            if int(row["round"]) >= 11
        ]
        for malicious in (False, True):
            group = [
                row
                for row in rows
                if (str(row["is_malicious"]).lower() == "true") == malicious
            ]
            active = [
                row for row in group if float(row["rtc_v3_client_q_cap"]) < 1.0 - 1e-12
            ]
            violations = [
                float(row["aggregation_weight"])
                - float(row["nominal_mass"]) * float(row["rtc_v3_client_q_cap"])
                for row in group
                if float(row["aggregation_weight"])
                - float(row["nominal_mass"]) * float(row["rtc_v3_client_q_cap"])
                > 1e-10
            ]
            client_audit.append(
                {
                    "attack": attack,
                    "client_group": "malicious" if malicious else "benign",
                    "client_rounds": len(group),
                    "cap_active_client_rounds": len(active),
                    "cap_active_rate": len(active) / len(group) if group else 0.0,
                    "mean_client_q_cap": sum(
                        float(row["rtc_v3_client_q_cap"]) for row in group
                    )
                    / len(group),
                    "weight_bound_violations": len(violations),
                    "max_weight_bound_excess": max(violations, default=0.0),
                }
            )

    indexed = {(row["attack"], row["mode"]): row for row in comparison}
    lie_base = indexed[("lie", "rtc_semantic_restricted_only")]
    lie_cap = indexed[("lie", "rtc_cumulative_q_cap")]
    dba_base = indexed[("dba", "rtc_semantic_restricted_only")]
    dba_cap = indexed[("dba", "rtc_cumulative_q_cap")]
    headline = [
        {
            "lie_active_acc_gain_pp": 100.0
            * (lie_cap["active_accuracy"] - lie_base["active_accuracy"]),
            "lie_final_acc_gain_pp": 100.0
            * (lie_cap["final_accuracy"] - lie_base["final_accuracy"]),
            "lie_malicious_weight_reduction_pp": 100.0
            * (lie_base["malicious_weight_share"] - lie_cap["malicious_weight_share"]),
            "lie_malicious_impact_reduction_pp": 100.0
            * (lie_base["malicious_impact_share"] - lie_cap["malicious_impact_share"]),
            "lie_zero_mass_increase_pp": 100.0
            * (lie_cap["zero_update_mass"] - lie_base["zero_update_mass"]),
            "dba_active_asr": dba_cap["active_asr"],
            "dba_active_acc_delta_pp": 100.0
            * (dba_cap["active_accuracy"] - dba_base["active_accuracy"]),
            "quality_gate_count": 11,
            "accepted": bool(decision["accepted_for_next_stage"]),
        }
    ]
    lie_mechanism = []
    for metric, label in (
        ("active_accuracy", "Active ACC"),
        ("malicious_weight_share", "Malicious weight"),
        ("malicious_impact_share", "Malicious impact"),
        ("zero_update_mass", "Zero-update mass"),
    ):
        lie_mechanism.extend(
            [
                {
                    "metric": label,
                    "mode": "B1R floor=0.5",
                    "rate": lie_base[metric],
                    "attack": "LIE z=0.5",
                    "seed": 42,
                    "active_rounds": 50,
                },
                {
                    "metric": label,
                    "mode": "B2 linear q cap",
                    "rate": lie_cap[metric],
                    "attack": "LIE z=0.5",
                    "seed": 42,
                    "active_rounds": 50,
                },
            ]
        )
    decision_rows = [
        {"check": key, "passed": bool(value)}
        for key, value in decision["checks"].items()
    ]
    tables = {
        "headline": headline,
        "timeline": timeline,
        "comparison": comparison,
        "lie_mechanism": lie_mechanism,
        "client_audit": client_audit,
        "divergence": divergence,
        "decision_checks": decision_rows,
    }
    write_sqlite(tables)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sources = [
        source(
            "src_headline",
            "B2 判定摘要",
            "headline",
            "复算 B2 相对严格配对 B1R floor=0.5 的预设门槛。",
            [
                "active metric = arithmetic mean over rounds 11-60",
                "delta pp = 100 × (B2 - B1R)",
                "malicious-weight reduction pp = 100 × (B1R - B2)",
            ],
            generated_at,
        ),
        source(
            "src_timeline",
            "LIE z=0.5 逐轮配对日志",
            "timeline",
            "读取 B1R 与 B2 的 61 轮 ACC、恶意质量和 cap 活跃数。",
            [
                "accuracy = CIFAR-10 server test accuracy",
                "client cap active count = selected clients with client_q_cap < 1",
            ],
            generated_at,
        ),
        source(
            "src_comparison",
            "LIE/DBA 主动期比较",
            "comparison",
            "严格配对四个运行的效用、安全、误伤和缺失质量指标。",
            [
                "all active metrics are arithmetic means over rounds 11-60",
                "zero-update mass = 1 - effective update mass",
            ],
            generated_at,
        ),
        source(
            "src_clients",
            "B2 客户端 cap 审计",
            "client_audit",
            "逐行核对 cap 命中对象与 aggregation_weight ≤ nominal_mass × client_q_cap。",
            [
                "cap active means client_q_cap < 1 - 1e-12",
                "weight violation tolerance = 1e-10",
            ],
            generated_at,
        ),
        {
            "id": "src_code",
            "label": "RTC-v3 cumulative-q cap 实现",
            "path": "defenses/rtc/v3.py",
            "query": {
                "description": "核对 B2 的唯一机制变化、参数别名和运行时导出。",
                "engine": "local files",
                "language": "python",
                "filters": ["cumulative_q_cap_power=1", "semantic_intervention_risk_floor=0.5"],
                "tables_used": [
                    "defenses/rtc/v3.py",
                    "strategies/fed_strategy.py",
                    "experiments/rtc_v3/byzantine.py",
                ],
                "metric_definitions": [
                    "upper_bound_i = nominal_i × min(semantic_q_i, cumulative_q_i, direction_q_i)",
                    "B2 does not renormalize remaining mass",
                ],
                "executed_at": generated_at,
            },
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "RTC-v3 B2：线性 cumulative-q cap 通过并降低 LIE 恶意质量",
            "description": "seed42 严格配对 LIE z=0.5 与 strong DBA 的 B2 阶段验收。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "card_mal_weight",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "超过预设至少 3 pp 的 LIE 恶意权重下降门槛。",
                    "metrics": [
                        {"label": "LIE 恶意权重下降", "field": "lie_malicious_weight_reduction_pp", "format": "number", "unit": "pp"},
                        {"label": "LIE active ACC 变化", "field": "lie_active_acc_gain_pp", "format": "number", "unit": "pp", "signed": True},
                    ],
                },
                {
                    "id": "card_impact",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "恶意影响同步下降，未以更高 impact 换取权重下降。",
                    "metrics": [
                        {"label": "LIE 恶意 impact 下降", "field": "lie_malicious_impact_reduction_pp", "format": "number", "unit": "pp"},
                        {"label": "LIE final ACC 变化", "field": "lie_final_acc_gain_pp", "format": "number", "unit": "pp", "signed": True},
                    ],
                },
                {
                    "id": "card_dba",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "DBA 轨迹与 B1R 完全一致，安全护栏保留。",
                    "metrics": [
                        {"label": "DBA active ASR", "field": "dba_active_asr", "format": "percent"},
                        {"label": "DBA active ACC 变化", "field": "dba_active_acc_delta_pp", "format": "number", "unit": "pp", "signed": True},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "chart_lie_mechanism",
                    "title": "LIE 主动期效用与聚合质量",
                    "subtitle": "seed42、50 个攻击轮；B2 只减少恶意质量，良性权重保持 71.4%。",
                    "intent": "comparison",
                    "question": "线性 cumulative-q cap 改变了哪些 LIE 主动期指标？",
                    "rationale": "四个同尺度 rate 用分组柱图显示 B1R 与 B2 的离散比较。",
                    "comparisonContext": {"baseline": "B1R floor=0.5", "grain": "active-period mean", "unit": "rate", "denominator": "50 attack rounds"},
                    "type": "bar",
                    "dataset": "lie_mechanism",
                    "sourceId": "src_comparison",
                    "encodings": {
                        "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                        "y": {"field": "rate", "type": "quantitative", "format": "percent", "label": "Rate"},
                        "color": {"field": "mode", "type": "nominal", "label": "Mode"},
                        "tooltip": [
                            {"field": "attack", "type": "nominal", "label": "Attack"},
                            {"field": "active_rounds", "type": "quantitative", "label": "Active rounds"},
                        ],
                    },
                    "valueFormat": "percent",
                    "palette": {"kind": "categorical"},
                    "settings": {"orientation": "vertical", "grouping": "grouped", "showValues": True},
                    "surface": {"surface": "card", "viewMode": "visualization"},
                },
                {
                    "id": "chart_accuracy",
                    "title": "LIE z=0.5 逐轮服务器准确率",
                    "subtitle": "严格配对 61 轮；第 11–60 轮为攻击主动期，B2 的平均 ACC 未回退。",
                    "intent": "trend",
                    "question": "B2 的 ACC 改善是否只来自第 60 轮单点？",
                    "rationale": "61 轮双轨线图用于检查路径，而非把轮次当独立重复。",
                    "comparisonContext": {"baseline": "B1R floor=0.5", "grain": "federated round", "unit": "accuracy rate", "denominator": "CIFAR-10 test set"},
                    "type": "line",
                    "dataset": "timeline",
                    "sourceId": "src_timeline",
                    "encodings": {
                        "x": {"field": "round", "type": "quantitative", "label": "Round"},
                        "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Server accuracy"},
                        "color": {"field": "mode", "type": "nominal", "label": "Mode"},
                        "tooltip": [
                            {"field": "malicious_weight_share", "type": "quantitative", "format": "percent", "label": "Malicious weight"},
                            {"field": "zero_update_mass", "type": "quantitative", "format": "percent", "label": "Zero mass"},
                            {"field": "client_q_cap_active_count", "type": "quantitative", "label": "Cap-active clients"},
                        ],
                    },
                    "valueFormat": "percent",
                    "palette": {"kind": "categorical"},
                    "referenceLines": [{"axis": "x", "value": 11, "label": "Attack starts", "color": "neutral", "lineStyle": "dashed"}],
                    "settings": {"showPoints": "hover"},
                    "surface": {"surface": "card", "viewMode": "visualization"},
                },
            ],
            "tables": [
                {
                    "id": "table_comparison",
                    "title": "B2 效用、安全与误伤指标",
                    "subtitle": "active 指标均为第 11–60 轮算术平均；ASR 仅适用于 DBA。",
                    "dataset": "comparison",
                    "sourceId": "src_comparison",
                    "density": "spacious",
                    "defaultSort": {"field": "attack", "direction": "asc"},
                    "columns": [
                        {"field": "attack", "label": "Attack", "type": "text"},
                        {"field": "mode", "label": "Mode", "type": "text"},
                        {"field": "active_accuracy", "label": "Active ACC", "format": "percent"},
                        {"field": "final_accuracy", "label": "Final ACC", "format": "percent"},
                        {"field": "benign_weight_share", "label": "Benign weight", "format": "percent"},
                        {"field": "malicious_weight_share", "label": "Malicious weight", "format": "percent"},
                        {"field": "malicious_impact_share", "label": "Malicious impact", "format": "percent"},
                        {"field": "zero_update_mass", "label": "Zero mass", "format": "percent"},
                        {"field": "benign_watch_rate", "label": "Benign watch", "format": "percent"},
                        {"field": "benign_restricted_rate", "label": "Benign restricted", "format": "percent"},
                        {"field": "benign_quarantined_rate", "label": "Benign quarantined", "format": "percent"},
                        {"field": "active_asr", "label": "Active ASR", "format": "percent"},
                    ],
                },
                {
                    "id": "table_client_audit",
                    "title": "客户端 cap 命中与权重上限审计",
                    "subtitle": "第 11–60 轮共 500 个客户端行/攻击；容差 1e-10。",
                    "dataset": "client_audit",
                    "sourceId": "src_clients",
                    "density": "spacious",
                    "defaultSort": {"field": "attack", "direction": "asc"},
                    "columns": [
                        {"field": "attack", "label": "Attack", "type": "text"},
                        {"field": "client_group", "label": "Client group", "type": "text"},
                        {"field": "client_rounds", "label": "Client-rounds", "format": "number"},
                        {"field": "cap_active_client_rounds", "label": "Cap-active", "format": "number"},
                        {"field": "cap_active_rate", "label": "Cap-active rate", "format": "percent"},
                        {"field": "mean_client_q_cap", "label": "Mean q cap", "format": "percent"},
                        {"field": "weight_bound_violations", "label": "Violations", "format": "number"},
                        {"field": "max_weight_bound_excess", "label": "Max excess", "format": "number"},
                    ],
                },
                {
                    "id": "table_divergence",
                    "title": "B2 首次客户端权重分叉",
                    "subtitle": "同一 TrialPlan 下 B1R 与 B2 的首个非零权重差。",
                    "dataset": "divergence",
                    "sourceId": "src_clients",
                    "density": "spacious",
                    "defaultSort": {"field": "round", "direction": "asc"},
                    "columns": [
                        {"field": "attack", "label": "Attack", "type": "text"},
                        {"field": "round", "label": "Round", "format": "number"},
                        {"field": "cid", "label": "Client", "format": "number"},
                        {"field": "is_malicious", "label": "Malicious", "type": "text"},
                        {"field": "baseline_weight", "label": "B1R weight", "format": "percent"},
                        {"field": "candidate_weight", "label": "B2 weight", "format": "percent"},
                        {"field": "candidate_minus_full", "label": "Delta", "format": "percent", "movement": True},
                        {"field": "candidate_cumulative_q", "label": "Cumulative q", "format": "percent"},
                        {"field": "candidate_client_q_cap", "label": "Client q cap", "format": "percent"},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# RTC-v3 B2：线性 cumulative-q cap 通过并降低 LIE 恶意质量", "layout": "full"},
                {"id": "summary", "type": "markdown", "sourceId": "src_headline", "layout": "full", "body": "## 技术摘要\n\n**B2 通过全部预设门槛，可以冻结线性 cumulative-q cap 并进入 B3。** 相对 B1R，LIE 恶意权重下降 3.2687 pp、恶意 impact 下降 2.5195 pp，active ACC 反而提高 0.7092 pp、final ACC 提高 2.31 pp。DBA 的 ACC、ASR、恶意权重与 impact 完全不变。该结果支持“直接限制累计异常客户端质量”而非继续压低良性语义质量。"},
                {"id": "kpis", "type": "metric-strip", "cardIds": ["card_mal_weight", "card_impact", "card_dba"], "layout": "full"},
                {"id": "mechanism_finding", "type": "markdown", "sourceId": "src_comparison", "layout": "full", "body": "## LIE 恶意质量下降超过门槛，良性质量没有损失\n\nB2 将 LIE 恶意权重从 28.5775% 降至 25.3088%，超过至少 3 pp 的预设门槛；良性权重保持 71.4%，良性 watch/restricted/quarantined 比例也不变。代价是 zero-update mass 从 0.0225% 增至 3.2912%，但 active ACC 从 77.5966% 升至 78.3058%，因此当前没有观察到有效步长损失。"},
                {"id": "mechanism_chart", "type": "chart", "chartId": "chart_lie_mechanism", "layout": "full"},
                {"id": "path_finding", "type": "markdown", "sourceId": "src_timeline", "layout": "full", "body": "## ACC 改善不是只靠末轮单点\n\n门槛使用第 11–60 轮的 50 轮平均，而不是第 60 轮单点。B2 active ACC 提高 0.7092 pp，final ACC 同时提高 2.31 pp；逐轮路径用于检查模型路径差异，但这些轮次不被当作独立 seed。"},
                {"id": "accuracy_chart", "type": "chart", "chartId": "chart_accuracy", "layout": "full"},
                {"id": "security_finding", "type": "markdown", "sourceId": "src_comparison", "layout": "full", "body": "## DBA 安全轨迹完全保持\n\nDBA active ACC 仍为 83.9258%，active ASR 仍为 2.3753%，恶意权重仍为 0.2540%，所有逐轮汇总指标与 B1R 一致。DBA 恶意客户端的 cap 虽为 active，但原有 quarantined 权重已低于新上限，所以新机制没有改变解。"},
                {"id": "comparison_table", "type": "table", "tableId": "table_comparison", "layout": "full"},
                {"id": "audit_finding", "type": "markdown", "sourceId": "src_clients", "layout": "full", "body": "## 上限确实生效，且没有命中良性客户端\n\n逐行审计显示，两种攻击各 500 个活跃期客户端行均满足 aggregation weight ≤ nominal×client q cap，没有越界。LIE 有 61 个 cap-active 客户端行，全部为恶意；良性命中为 0。首次权重分叉在第 34 轮恶意 cid7：cumulative q=0.991141，权重从 10% 降至 9.9114%。"},
                {"id": "client_table", "type": "table", "tableId": "table_client_audit", "layout": "full"},
                {"id": "divergence_table", "type": "table", "tableId": "table_divergence", "layout": "full"},
                {"id": "scope", "type": "markdown", "sourceId": "src_headline", "layout": "full", "body": "## 范围、数据与指标定义\n\nCIFAR-10 IID，seed42，20 客户端、每轮采样10个，恶意比例30%。LIE 固定 z=0.5；DBA 使用冻结 strong 参数；攻击期均为第11–60轮。active 指标是50个攻击轮算术平均，final ACC 是第60轮。两个候选均 completed、exit code 0、last round 60、各61行，11/11质量门通过，trial-plan hash 与 B1R 完全相同。"},
                {"id": "method", "type": "markdown", "sourceId": "src_code", "layout": "full", "body": "## 方法与代码复核\n\nB2 冻结 semantic intervention floor=0.5，只增加 cumulative_q_cap_power=1。客户端上限为 nominal×min(semantic q, cumulative q, direction q)，剩余质量不重归一化。验收先检查完成性、质量门与严格配对，再独立复算逐轮指标、逐行核对上限并定位首次分叉。"},
                {"id": "limitations", "type": "markdown", "layout": "full", "body": "## 限制、稳健性与判定边界\n\n- 只有 seed42；不能解释为多 seed 显著性结论。\n- 当前覆盖 LIE z=0.5 与 strong DBA，尚不能外推到其他攻击或 Non-IID。\n- LIE 恶意权重仍为 25.31%，离 Multi-Krum 约 12.3% 的历史参考仍有差距。\n- B2 将 3.29% 质量留为空更新；B3 的 anchor recycle 必须证明回填不会抬高恶意 impact 或 ASR。"},
                {"id": "next", "type": "markdown", "layout": "full", "body": "## 下一步：B3 仅回填被拒绝质量到 coordinate-median anchor\n\n冻结 floor=0.5 与 linear q cap，只增加 anchor recycle。最小 seed42 筛选应严格配对并优先覆盖 DBA 与 scaling；验收要求 zero-update mass 至少减半或降至不高于2%，ACC 提升，且 ASR 与恶意 impact 不恶化。训练前需先完成候选别名、测试、dry-run、分析器和唯一人工命令。"},
                {"id": "questions", "type": "markdown", "layout": "full", "body": "## 进一步问题\n\n1. B3 是否能回填 B2 在 LIE 下新增的 3.29% 缺失质量，而不把恶意方向重新注入？\n2. 对 DBA 已有约31.55% zero mass，coordinate-median anchor 是否保持低 ASR？\n3. scaling 的严格配对基线能否复用，还是需要同 TrialPlan 补跑 B2 基线？"},
            ],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": tables,
            "accessIssues": [],
        },
        "sources": sources,
    }
    (OUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
