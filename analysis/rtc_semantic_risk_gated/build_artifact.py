"""Build the bounded technical-report artifact for the completed B1R screen."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "rtc_semantic_restricted_only"
CURRENT = ROOT / "logs" / "rtc_v3_semantic_restricted_only_seed42_mf03"
LIE_BASE = ROOT / "logs" / "rtc_v3_semantic_observe_lie_z05_seed42_mf03"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def one_file(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, got {len(matches)}")
    return matches[0]


def write_sqlite(tables: dict[str, list[dict[str, object]]]) -> None:
    database = OUT / "report_snapshot.sqlite"
    with sqlite3.connect(database) as connection:
        for table, rows in tables.items():
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            if not rows:
                continue
            fields = list(rows[0])
            types = {}
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
        "path": "analysis/rtc_semantic_restricted_only/report_snapshot.sqlite",
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
                float(value)
                if key not in {"attack", "mode"} and value not in {"", None}
                else value or None
            )
            for key, value in row.items()
        }
        for row in read_csv(OUT / "comparison.csv")
    ]
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    divergence = [
        {
            key: (
                None
                if value in {"", None}
                else int(value)
                if key in {"round", "cid"}
                else str(value).lower() == "true"
                if key == "is_malicious"
                else float(value)
                if key not in {"attack"}
                else value
            )
            for key, value in row.items()
        }
        for row in read_csv(OUT / "first_weight_divergence.csv")
    ]

    timeline: list[dict[str, object]] = []
    for label, path in (
        ("RTC full", one_file(LIE_BASE / "rounds", "lie*rtc_full*.csv")),
        (
            "Restricted-only 0.5",
            one_file(CURRENT / "rounds", "lie*rtc_semantic_restricted_only*.csv"),
        ),
    ):
        for row in read_csv(path):
            timeline.append(
                {
                    "round": int(row["round"]),
                    "mode": label,
                    "accuracy": float(row["server_accuracy"]),
                    "weight_sum": float(row.get("fit_rtc_v3_weight_sum") or 1.0),
                    "zero_update_mass": float(
                        row.get("fit_rtc_v3_zero_update_mass") or 0.0
                    ),
                    "malicious_weight_share": float(
                        row.get("fit_malicious_aggregation_weight_share") or 0.0
                    ),
                    "malicious_impact_share": float(
                        row.get("fit_malicious_impact_share") or 0.0
                    ),
                    "attack_active": int(row["round"]) >= 11,
                }
            )

    clients = read_csv(
        one_file(CURRENT / "raw", "lie*rtc_semantic_restricted_only*_clients.csv")
    )
    bins = [
        ("≤0.1 normal", -1.0, 0.1),
        ("0.1–0.5 watch", 0.1, 0.5),
        ("0.5–0.8 restricted", 0.5, 0.8),
        (">0.8 quarantined", 0.8, 1.000001),
    ]
    risk_buckets: list[dict[str, object]] = []
    for label, lower, upper in bins:
        selected = [
            row
            for row in clients
            if int(row["round"]) >= 11
            and str(row["is_malicious"]).lower() == "false"
            and lower < float(row["semantic_risk"]) <= upper
        ]
        missing = sum(
            max(0.0, float(row["nominal_mass"]) - float(row["aggregation_weight"]))
            for row in selected
        )
        risk_buckets.append(
            {
                "risk_bucket": label,
                "client_rounds": len(selected),
                "mean_risk": (
                    sum(float(row["semantic_risk"]) for row in selected)
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "missing_mass_per_round": missing / 50.0,
                "share_of_benign_missing_mass": 0.0,
            }
        )
    total_missing = sum(row["missing_mass_per_round"] for row in risk_buckets)
    for row in risk_buckets:
        row["share_of_benign_missing_mass"] = (
            row["missing_mass_per_round"] / total_missing if total_missing else 0.0
        )

    indexed = {(row["attack"], row["mode"]): row for row in comparison}
    lie_full = indexed[("lie", "rtc_full")]
    lie_gate = indexed[("lie", "rtc_semantic_restricted_only")]
    dba_gate = indexed[("dba", "rtc_semantic_restricted_only")]
    headline = [
        {
            "lie_active_gain_pp": 100.0
            * (lie_gate["active_accuracy"] - lie_full["active_accuracy"]),
            "lie_active_gain_target_pp": 0.3,
            "lie_final_gain_pp": 100.0
            * (lie_gate["final_accuracy"] - lie_full["final_accuracy"]),
            "recovered_mass_pp": 100.0
            * (lie_gate["weight_sum"] - lie_full["weight_sum"]),
            "dba_active_asr": dba_gate["active_asr"],
            "dba_asr_limit": 0.10,
            "dba_malicious_weight": dba_gate["malicious_weight_share"],
            "quality_gate_count": 11,
            "accepted": True,
        }
    ]
    decision_rows = [
        {"check": key, "passed": value}
        for key, value in decision["checks"].items()
    ]
    tables = {
        "headline": headline,
        "timeline": timeline,
        "comparison": comparison,
        "divergence": divergence,
        "risk_buckets": risk_buckets,
        "decision_checks": decision_rows,
    }
    write_sqlite(tables)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sources = [
        source(
            "src_headline",
            "B1R 判定摘要",
            "headline",
            "复算 B1R 相对严格配对 RTC full 的门槛指标。",
            [
                "active ACC = rounds 11-60 server_accuracy arithmetic mean",
                "gain pp = 100 × (candidate - rtc_full)",
                "DBA active ASR = rounds 11-60 full-trigger ASR arithmetic mean",
            ],
            generated_at,
        ),
        source(
            "src_timeline",
            "LIE z=0.5 逐轮配对日志",
            "timeline",
            "读取 RTC full 与 restricted-only 0.5 候选的 61 轮 ACC 和聚合质量。",
            [
                "weight sum = sum of client aggregation weights",
                "zero-update mass = 1 - effective update mass",
            ],
            generated_at,
        ),
        source(
            "src_comparison",
            "LIE/DBA 主动期比较",
            "comparison",
            "四个严格配对运行的主动期效用、安全和误伤指标。",
            [
                "all active metrics are arithmetic means over rounds 11-60",
                "benign weight = RTC weight sum - malicious weight share",
            ],
            generated_at,
        ),
        source(
            "src_clients",
            "客户端风险桶与首次分叉",
            "risk_buckets",
            "按 semantic state 阈值汇总 LIE 候选的良性客户端轮次和丢失质量。",
            [
                "missing mass = max(0, nominal mass - aggregation weight)",
                "per-round missing mass divides summed client-round mass by 50 attack rounds",
            ],
            generated_at,
        ),
        {
            "id": "src_code",
            "label": "RTC-v3 风险门控实现",
            "path": "defenses/rtc/v3.py",
            "query": {
                "description": "核对 semantic intervention floor 对 QP risk、q cap 和 hard exposure 的实现。",
                "engine": "local files",
                "language": "python",
                "filters": ["semantic_intervention_risk_floor"],
                "tables_used": [
                    "defenses/rtc/v3.py",
                    "defenses/rtc/semantic.py",
                    "experiments/rtc_v3/byzantine.py",
                ],
                "metric_definitions": [
                    "B1R accepted candidate uses floor=0.5",
                    "B2 proposed candidate adds cumulative_q_cap_power=1",
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
            "title": "RTC-v3 B1R restricted-only：效用门槛通过且安全保留",
            "description": "seed42 严格配对 LIE z=0.5 与 DBA 的语义风险门控诊断。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "card_lie_gain",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "LIE 主动期 ACC 增益超过预设 0.3 pp 门槛。",
                    "metrics": [
                        {"label": "LIE active ACC 增益", "field": "lie_active_gain_pp", "format": "number", "unit": "pp", "signed": True},
                        {"label": "通过门槛", "field": "lie_active_gain_target_pp", "format": "number", "unit": "pp"},
                    ],
                },
                {
                    "id": "card_mass",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "floor=0.5 基本消除 LIE 的缺失更新质量。",
                    "metrics": [
                        {"label": "恢复更新质量", "field": "recovered_mass_pp", "format": "number", "unit": "pp", "signed": True},
                        {"label": "末轮 ACC 增益", "field": "lie_final_gain_pp", "format": "number", "unit": "pp", "signed": True},
                    ],
                },
                {
                    "id": "card_dba",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "DBA 安全门显著低于 10% ASR 上限。",
                    "metrics": [
                        {"label": "DBA active ASR", "field": "dba_active_asr", "format": "percent"},
                        {"label": "ASR 上限", "field": "dba_asr_limit", "format": "percent"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "chart_accuracy",
                    "title": "LIE z=0.5 逐轮服务器准确率",
                    "subtitle": "严格配对 61 轮；第 11–60 轮为攻击主动期，候选达到预设平均收益门槛。",
                    "intent": "trend",
                    "question": "restricted-only floor=0.5 是否产生足够稳定的 ACC 改善？",
                    "rationale": "61 轮双轨线图可避免由末轮单点决定阶段通过。",
                    "comparisonContext": {"baseline": "RTC full", "grain": "federated round", "unit": "accuracy rate", "denominator": "CIFAR-10 test set"},
                    "type": "line",
                    "dataset": "timeline",
                    "sourceId": "src_timeline",
                    "encodings": {
                        "x": {"field": "round", "type": "quantitative", "label": "Round"},
                        "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Server accuracy"},
                        "color": {"field": "mode", "type": "nominal", "label": "Mode"},
                        "tooltip": [
                            {"field": "weight_sum", "type": "quantitative", "format": "percent", "label": "Weight sum"},
                            {"field": "zero_update_mass", "type": "quantitative", "format": "percent", "label": "Zero mass"},
                            {"field": "malicious_weight_share", "type": "quantitative", "format": "percent", "label": "Malicious weight"},
                        ],
                    },
                    "valueFormat": "percent",
                    "palette": {"kind": "categorical"},
                    "referenceLines": [{"axis": "x", "value": 11, "label": "Attack starts", "color": "neutral", "lineStyle": "dashed"}],
                    "settings": {"showPoints": "hover"},
                    "surface": {"surface": "card", "viewMode": "visualization"},
                }
            ],
            "tables": [
                {
                    "id": "table_comparison",
                    "title": "B1R 效用、安全与误伤指标",
                    "subtitle": "active 指标为第 11–60 轮算术平均；DBA ASR 仅适用于 DBA 行。",
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
                        {"field": "active_asr", "label": "Active ASR", "format": "percent"},
                    ],
                },
                {
                    "id": "table_risk_buckets",
                    "title": "LIE 候选的良性语义风险桶",
                    "subtitle": "50 个攻击轮；丢失质量按客户端 nominal−aggregation weight 汇总。",
                    "dataset": "risk_buckets",
                    "sourceId": "src_clients",
                    "density": "spacious",
                    "defaultSort": {"field": "missing_mass_per_round", "direction": "desc"},
                    "columns": [
                        {"field": "risk_bucket", "label": "Semantic state", "type": "text"},
                        {"field": "client_rounds", "label": "Client-rounds", "format": "number"},
                        {"field": "mean_risk", "label": "Mean risk", "format": "percent"},
                        {"field": "missing_mass_per_round", "label": "Missing/round", "format": "percent"},
                        {"field": "share_of_benign_missing_mass", "label": "Share of missing", "format": "percent"},
                    ],
                },
                {
                    "id": "table_divergence",
                    "title": "B1R 首次客户端权重分叉",
                    "subtitle": "同一 TrialPlan、客户端和分叉前模型；分别列出 LIE 与 DBA 的首次变化。",
                    "dataset": "divergence",
                    "sourceId": "src_clients",
                    "density": "spacious",
                    "defaultSort": {"field": "round", "direction": "asc"},
                    "columns": [
                        {"field": "attack", "label": "Attack", "type": "text"},
                        {"field": "round", "label": "Round", "format": "number"},
                        {"field": "cid", "label": "Client", "format": "number"},
                        {"field": "is_malicious", "label": "Malicious", "type": "text"},
                        {"field": "full_semantic_risk", "label": "Risk", "format": "percent"},
                        {"field": "full_weight", "label": "RTC full", "format": "percent"},
                        {"field": "candidate_weight", "label": "Candidate", "format": "percent"},
                        {"field": "candidate_minus_full", "label": "Delta", "format": "percent", "movement": True},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# RTC-v3 B1R restricted-only：效用门槛通过且安全保留", "layout": "full"},
                {"id": "summary", "type": "markdown", "sourceId": "src_headline", "layout": "full", "body": "## 技术摘要\n\n**B1R floor=0.5 通过全部预设门槛，可以冻结并进入 B2。** LIE active ACC 提高 0.4792 pp、final ACC 提高 2.15 pp，zero-update mass 从 4.5245% 降到 0.0225%；DBA active ACC 仅下降 0.0008 pp，active ASR 2.3753%、恶意权重 0.2540%，均显著低于安全上限。下一步只增加线性 cumulative-q 客户端 cap。"},
                {"id": "kpis", "type": "metric-strip", "cardIds": ["card_lie_gain", "card_mass", "card_dba"], "layout": "full"},
                {"id": "accuracy_finding", "type": "markdown", "sourceId": "src_timeline", "layout": "full", "body": "## LIE 主动期收益达到预设门槛\n\n候选 active ACC 为 77.5966%，RTC full 为 77.1174%，提高 0.4792 pp；final ACC 从 81.92% 提高到 84.07%。zero-update mass 从 4.5245% 降到 0.0225%，说明释放良性 watch 状态的质量损失是主要效用来源。"},
                {"id": "accuracy_chart", "type": "chart", "chartId": "chart_accuracy", "layout": "full"},
                {"id": "security_finding", "type": "markdown", "sourceId": "src_comparison", "layout": "full", "body": "## DBA 安全门保持，restricted-only 没有重现 observe 的失守\n\n候选 DBA active ASR 为 2.3753%，仅比 RTC full 高 0.1218 pp且远低于 10% 上限；恶意权重为 0.2540%，低于 5% 上限。active ACC 仅下降 0.0008 pp。DBA 恶意客户端仍全部处于 quarantined，安全分离保持。"},
                {"id": "comparison_table", "type": "table", "tableId": "table_comparison", "layout": "full"},
                {"id": "driver_finding", "type": "markdown", "sourceId": "src_clients", "layout": "full", "body": "## floor=0.5 回收的是良性 watch 质量，而非放松恶意语义压制\n\nLIE 良性权重从 66.9200% 恢复到 71.4000%，恶意权重基本不变（28.5556%→28.5775%），恶意 impact 反而从 31.4735% 降到 30.2719%。这支持“RTC 错误损失良性质量”的机制判断，但作用点应精确为 watch 状态。"},
                {"id": "risk_table", "type": "table", "tableId": "table_risk_buckets", "layout": "full"},
                {"id": "divergence_finding", "type": "markdown", "sourceId": "src_clients", "layout": "full", "body": "## 首次分叉符合预期，DBA 的变化发生在高风险恶意客户端\n\nLIE 首次分叉仍是第26轮良性 cid18，risk 5.9031%，权重从9.4097%恢复到10%。DBA 在攻击开始的第11轮即分叉：两名恶意客户端 risk 分别为93.35%和91.71%，候选权重略高但仍受 quarantined 约束；后续平均安全指标仍通过。"},
                {"id": "divergence_table", "type": "table", "tableId": "table_divergence", "layout": "full"},
                {"id": "scope", "type": "markdown", "sourceId": "src_headline", "layout": "full", "body": "## 范围、数据与指标定义\n\nCIFAR-10 IID，seed42，20 客户端、每轮采样10个，恶意比例30%。LIE 固定 z=0.5；DBA 使用冻结 strong 参数；攻击期均为第11–60轮。active ACC/ASR 是50个攻击轮算术平均，final ACC 是第60轮。两个候选均完成61行（round0–60）、exit code 0，11/11 质量门通过，且 trial-plan hash 与复用的 RTC full 基线完全相同。"},
                {"id": "method", "type": "markdown", "sourceId": "src_code", "layout": "full", "body": "## 方法与代码复核\n\n先验证两个候选均 completed、exit code 0、last round 60、61行 rounds、11/11质量门和 trial-plan hash，再从逐轮CSV独立复算 active 指标，并按 round/cid 一对一连接客户端记录。floor=0.5 同时控制 soft penalty、temporal q cap 和 hard exposure；其他 RTC 与攻击参数保持冻结。"},
                {"id": "limitations", "type": "markdown", "layout": "full", "body": "## 限制、稳健性与判定边界\n\n- 只有 seed42；50个轮次不是独立实验重复，不能给出显著性结论。\n- B1R 只证明 restricted-only 在 LIE z=0.5 与 strong DBA 上通过，尚未覆盖其他攻击或 Non-IID。\n- LIE 恶意权重仍为28.58%，效用恢复并未解决 inlier 恶意质量过高。\n- B2 可能以增加 zero-update mass 为代价降低恶意权重，必须独立检查 ACC。"},
                {"id": "next", "type": "markdown", "layout": "full", "body": "## 下一步：B2 线性 cumulative-q 客户端 cap\n\n冻结 floor=0.5，只增加 `cumulative_q_cap_power=1`，令客户端上限为 nominal×min(semantic q, cumulative q, direction q)。仍只运行 LIE z=0.5 与DBA两个候选单元并复用严格配对的B1R基线。通过条件：LIE恶意权重至少下降3 pp且active ACC降幅≤0.2 pp；DBA ASR≤10%、恶意权重≤5%、active ACC降幅≤0.2 pp。"},
                {"id": "questions", "type": "markdown", "layout": "full", "body": "## 进一步问题\n\n1. 线性 cap 能否实现离线重放预测的约3.54 pp LIE恶意权重下降？\n2. 新增缺失质量是否会使 active ACC 下降超过0.2 pp？\n3. 若线性 q 不足，是否有证据支持单独试 q²，而不是提前进入 anchor recycle？"},
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
