from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "rtc_v3_semantic_observe_lie_z05_seed42_mf03"
OUT_DIR = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def defense_name(path: Path) -> str:
    return "Semantic observe" if "semantic_observe" in path.name else "RTC full"


def source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    tables: list[str],
    metric_definitions: list[str],
    executed_at: str,
    sql: str | None = None,
) -> dict[str, object]:
    query: dict[str, object] = {
        "description": description,
        "engine": "sqlite" if sql else "local files",
        "language": "sql" if sql else "python",
        "filters": [
            "seed=42",
            "LIE z=0.5",
            "malicious_fraction=0.3",
            "partition=iid",
            "attack rounds=11-60",
        ],
        "tables_used": tables,
        "metric_definitions": metric_definitions,
        "executed_at": executed_at,
    }
    if sql:
        query["sql"] = sql
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": query,
    }


def write_sqlite(tables: dict[str, list[dict[str, object]]]) -> None:
    database = OUT_DIR / "report_snapshot.sqlite"
    with sqlite3.connect(database) as connection:
        for table, rows in tables.items():
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            if not rows:
                continue
            fields = list(rows[0])
            column_types: dict[str, str] = {}
            for field in fields:
                values = [row[field] for row in rows if row[field] is not None]
                if values and all(isinstance(value, bool) for value in values):
                    column_types[field] = "INTEGER"
                elif values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                    column_types[field] = "INTEGER"
                elif values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                    column_types[field] = "REAL"
                else:
                    column_types[field] = "TEXT"
            columns = ", ".join(
                f'"{field}" {column_types[field]}' for field in fields
            )
            connection.execute(f'CREATE TABLE "{table}" ({columns})')
            placeholders = ", ".join("?" for _ in fields)
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [tuple(row[field] for field in fields) for row in rows],
            )


def main() -> None:
    round_files = sorted((LOG_DIR / "rounds").glob("*.csv"))
    client_files = sorted((LOG_DIR / "raw").glob("*_clients.csv"))
    if len(round_files) != 2 or len(client_files) != 2:
        raise RuntimeError("Expected exactly two paired round and client files")

    round_rows: dict[str, list[dict[str, str]]] = {
        defense_name(path): read_csv(path) for path in round_files
    }
    client_rows: dict[str, list[dict[str, str]]] = {
        defense_name(path): read_csv(path) for path in client_files
    }
    runs = {row["defense"]: row for row in read_csv(LOG_DIR / "byzantine_attack_runs.csv")}
    full_run = runs["rtc_full"]
    observe_run = runs["rtc_semantic_observe"]

    timeline: list[dict[str, object]] = []
    for defense, rows in round_rows.items():
        for row in rows:
            if int(row["round"]) == 0:
                continue
            timeline.append(
                {
                    "round": int(row["round"]),
                    "defense": defense,
                    "accuracy": number(row, "server_accuracy"),
                    "loss": number(row, "server_loss"),
                    "weight_sum": number(row, "fit_rtc_v3_weight_sum"),
                    "zero_update_mass": number(row, "fit_rtc_v3_zero_update_mass"),
                    "malicious_weight_share": number(
                        row, "fit_malicious_aggregation_weight_share"
                    ),
                    "malicious_impact_share": number(
                        row, "fit_malicious_impact_share"
                    ),
                    "semantic_risk_mean": number(
                        row, "fit_rtc_v3_semantic_risk_mean"
                    ),
                    "attack_active": int(row["round"]) >= 11,
                }
            )

    comparison: list[dict[str, object]] = []
    for defense, run in (("RTC full", full_run), ("Semantic observe", observe_run)):
        active = [row for row in round_rows[defense] if int(row["round"]) >= 11]
        comparison.append(
            {
                "defense": defense,
                "final_accuracy": number(run, "final_accuracy"),
                "active_accuracy": number(run, "active_accuracy"),
                "mean_weight_sum": sum(number(row, "fit_rtc_v3_weight_sum") for row in active) / len(active),
                "mean_zero_update_mass": sum(number(row, "fit_rtc_v3_zero_update_mass") for row in active) / len(active),
                "mean_malicious_weight_share": sum(number(row, "fit_malicious_aggregation_weight_share") for row in active) / len(active),
                "mean_malicious_impact_share": sum(number(row, "fit_malicious_impact_share") for row in active) / len(active),
                "benign_restricted_rate": number(run, "benign_restricted_rate"),
                "valid": run["valid"],
            }
        )

    divergence: list[dict[str, object]] = []
    full_clients = {
        row["cid"]: row
        for row in client_rows["RTC full"]
        if int(row["round"]) == 26 and row["split"] == "client"
    }
    observe_clients = {
        row["cid"]: row
        for row in client_rows["Semantic observe"]
        if int(row["round"]) == 26 and row["split"] == "client"
    }
    if full_clients.keys() != observe_clients.keys():
        raise RuntimeError("Round-26 client identities do not match")
    for cid in sorted(full_clients, key=int):
        base = full_clients[cid]
        candidate = observe_clients[cid]
        divergence.append(
            {
                "cid": cid,
                "is_malicious": base["is_malicious"],
                "semantic_risk": number(base, "semantic_risk"),
                "nominal_mass": number(base, "nominal_mass"),
                "rtc_full_weight": number(base, "aggregation_weight"),
                "observe_weight": number(candidate, "aggregation_weight"),
                "weight_delta": number(candidate, "aggregation_weight")
                - number(base, "aggregation_weight"),
                "rtc_full_capped": base["capped"],
            }
        )

    headline = [
        {
            "final_gain_pp": 100.0
            * (number(observe_run, "final_accuracy") - number(full_run, "final_accuracy")),
            "active_gain_pp": 100.0
            * (number(observe_run, "active_accuracy") - number(full_run, "active_accuracy")),
            "recovered_mass_pp": 100.0
            * (comparison[1]["mean_weight_sum"] - comparison[0]["mean_weight_sum"]),
            "malicious_weight_delta_pp": 100.0
            * (
                comparison[1]["mean_malicious_weight_share"]
                - comparison[0]["mean_malicious_weight_share"]
            ),
            "malicious_impact_delta_pp": 100.0
            * (
                comparison[1]["mean_malicious_impact_share"]
                - comparison[0]["mean_malicious_impact_share"]
            ),
        }
    ]

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sources = [
        source(
            "src_rounds",
            "LIE z=0.5 严格配对逐轮日志",
            "analysis/rtc_semantic_observe_lie_z05/report_snapshot.sqlite",
            "读取两种 RTC 语义模式的 61 轮服务器指标与聚合诊断。",
            ["timeline"],
            [
                "active ACC = round 11-60 server_accuracy arithmetic mean",
                "zero-update mass = 1 - sum of client aggregation weights",
                "observe minus full deltas use the same paired round",
            ],
            generated_at,
            "SELECT * FROM timeline ORDER BY round, defense",
        ),
        source(
            "src_clients",
            "第 26 轮客户端级首次分叉",
            "analysis/rtc_semantic_observe_lie_z05/report_snapshot.sqlite",
            "按同一 cid 对齐首次权重分叉轮，核对恶意身份、语义风险与实际权重。",
            ["divergence"],
            [
                "weight delta = semantic-observe aggregation weight - rtc-full aggregation weight",
                "malicious identity comes from the frozen paired trial plan",
            ],
            generated_at,
            "SELECT * FROM divergence ORDER BY weight_delta DESC, cid",
        ),
        source(
            "src_summary",
            "运行汇总与质量门",
            "analysis/rtc_semantic_observe_lie_z05/report_snapshot.sqlite",
            "读取完成状态、最终/主动期指标、配对哈希和机制摘要。",
            ["headline"],
            [
                "final gain = 100 × (observe final ACC - full final ACC), percentage points",
                "active gain = 100 × (observe active ACC - full active ACC), percentage points",
            ],
            generated_at,
            "SELECT * FROM headline",
        ),
        source(
            "src_comparison",
            "主动期模式比较表",
            "analysis/rtc_semantic_observe_lie_z05/report_snapshot.sqlite",
            "读取两种语义模式的最终、主动期、质量和及恶意贡献指标；上游为运行汇总和逐轮 CSV。",
            ["comparison"],
            [
                "active metrics are arithmetic means over rounds 11-60",
                "benign restricted rate is the active-period client observation rate",
            ],
            generated_at,
            "SELECT * FROM comparison ORDER BY active_accuracy DESC",
        ),
        source(
            "src_code",
            "RTC V3 语义消融实现",
            "defenses/rtc/v3.py",
            "核对 observe 仍计算并记录语义证据，但不把 semantic q、risk 或 exposure budget 加入 QP。",
            ["defenses/rtc/v3.py", "experiments/rtc_v3/byzantine.py"],
            [
                "observe: solver cap remains nominal and semantic risk penalty is zero",
                "exposure: semantic q, risk penalty and semantic exposure budgets are active",
            ],
            generated_at,
        ),
    ]

    manifest: dict[str, object] = {
        "version": 1,
        "surface": "report",
        "title": "RTC 语义观察消融：LIE z=0.5 的良性质量误伤证据",
        "description": "严格配对 seed-42 实验对 RTC full 与 semantic observe 的效用和机制进行诊断。",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "card_final_gain",
                "dataset": "headline",
                "sourceId": "src_summary",
                "description": "Semantic observe 相对 RTC full 的末轮 ACC 差。",
                "metrics": [
                    {"label": "末轮 ACC 增益", "field": "final_gain_pp", "format": "number", "unit": "pp", "signed": True},
                    {"label": "主动期增益", "field": "active_gain_pp", "format": "number", "unit": "pp", "signed": True},
                ],
            },
            {
                "id": "card_mass",
                "dataset": "headline",
                "sourceId": "src_summary",
                "description": "移除语义干预后恢复的平均客户端更新质量。",
                "metrics": [
                    {"label": "恢复更新质量", "field": "recovered_mass_pp", "format": "number", "unit": "pp", "signed": True},
                    {"label": "恶意权重变化", "field": "malicious_weight_delta_pp", "format": "number", "unit": "pp", "signed": True},
                ],
            },
            {
                "id": "card_impact",
                "dataset": "headline",
                "sourceId": "src_summary",
                "description": "恶意更新对实际聚合向量范数的贡献份额变化。",
                "metrics": [
                    {"label": "恶意影响变化", "field": "malicious_impact_delta_pp", "format": "number", "unit": "pp", "signed": True}
                ],
            },
        ],
        "charts": [
            {
                "id": "chart_accuracy",
                "title": "两种语义模式的服务器准确率",
                "subtitle": "严格配对 61 轮；第 11 轮开始连续 LIE z=0.5，第 26 轮首次出现权重与 ACC 分叉。",
                "intent": "trend",
                "question": "关闭语义干预后 ACC 在何时、以何种幅度发生变化？",
                "rationale": "逐轮双轨线图可以区分末轮单点与持续性效应。",
                "comparisonContext": {
                    "baseline": "same TrialPlan and seed",
                    "grain": "federated round",
                    "unit": "accuracy rate",
                    "denominator": "CIFAR-10 test set",
                },
                "type": "line",
                "dataset": "timeline",
                "sourceId": "src_rounds",
                "encodings": {
                    "x": {"field": "round", "type": "quantitative", "label": "Round"},
                    "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Server accuracy"},
                    "color": {"field": "defense", "type": "nominal", "label": "Semantic mode"},
                    "tooltip": [
                        {"field": "weight_sum", "type": "quantitative", "format": "percent", "label": "Client weight sum"},
                        {"field": "zero_update_mass", "type": "quantitative", "format": "percent", "label": "Zero-update mass"},
                        {"field": "malicious_weight_share", "type": "quantitative", "format": "percent", "label": "Malicious weight"},
                        {"field": "malicious_impact_share", "type": "quantitative", "format": "percent", "label": "Malicious impact"},
                    ],
                },
                "valueFormat": "percent",
                "palette": {"kind": "categorical"},
                "referenceLines": [
                    {"axis": "x", "value": 11, "label": "Attack starts", "color": "neutral", "lineStyle": "dashed"},
                    {"axis": "x", "value": 26, "label": "First divergence", "color": "neutral", "lineStyle": "dotted"},
                ],
                "settings": {"showPoints": "hover"},
                "surface": {"surface": "card", "viewMode": "visualization"},
            },
            {
                "id": "chart_weight_sum",
                "title": "两种语义模式的客户端更新质量和",
                "subtitle": "第 26 轮后 RTC full 在 28 个主动轮产生质量缺口，observe 基本保持总质量为 1。",
                "intent": "trend",
                "question": "ACC 分叉是否与语义约束造成的更新质量损失同步？",
                "rationale": "逐轮质量和直接显示 QP 约束留下的零更新质量。",
                "comparisonContext": {
                    "baseline": "same TrialPlan and seed",
                    "grain": "federated round",
                    "unit": "aggregation mass",
                    "denominator": "nominal selected-client mass",
                },
                "type": "line",
                "dataset": "timeline",
                "sourceId": "src_rounds",
                "encodings": {
                    "x": {"field": "round", "type": "quantitative", "label": "Round"},
                    "y": {"field": "weight_sum", "type": "quantitative", "format": "percent", "label": "Client weight sum"},
                    "color": {"field": "defense", "type": "nominal", "label": "Semantic mode"},
                    "tooltip": [
                        {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Server accuracy"},
                        {"field": "zero_update_mass", "type": "quantitative", "format": "percent", "label": "Zero-update mass"},
                        {"field": "semantic_risk_mean", "type": "quantitative", "format": "percent", "label": "Mean semantic risk"},
                    ],
                },
                "valueFormat": "percent",
                "palette": {"kind": "categorical"},
                "referenceLines": [
                    {"axis": "x", "value": 26, "label": "First divergence", "color": "neutral", "lineStyle": "dotted"}
                ],
                "settings": {"showPoints": "hover"},
                "surface": {"surface": "card", "viewMode": "visualization"},
            },
        ],
        "tables": [
            {
                "id": "table_comparison",
                "title": "主动期效用与安全指标",
                "subtitle": "第 11–60 轮算术平均；末轮 ACC 单独列示。",
                "dataset": "comparison",
                "sourceId": "src_comparison",
                "density": "spacious",
                "defaultSort": {"field": "active_accuracy", "direction": "desc"},
                "columns": [
                    {"field": "defense", "label": "Mode", "type": "text"},
                    {"field": "final_accuracy", "label": "Final ACC", "format": "percent"},
                    {"field": "active_accuracy", "label": "Active ACC", "format": "percent"},
                    {"field": "mean_weight_sum", "label": "Weight sum", "format": "percent"},
                    {"field": "mean_zero_update_mass", "label": "Zero mass", "format": "percent"},
                    {"field": "mean_malicious_weight_share", "label": "Malicious weight", "format": "percent"},
                    {"field": "mean_malicious_impact_share", "label": "Malicious impact", "format": "percent"},
                    {"field": "benign_restricted_rate", "label": "Benign restricted", "format": "percent"},
                ],
            },
            {
                "id": "table_divergence",
                "title": "第 26 轮客户端级首次分叉",
                "subtitle": "同一模型、客户端选择和本地随机流；按 observe−full 权重差降序。",
                "dataset": "divergence",
                "sourceId": "src_clients",
                "density": "spacious",
                "defaultSort": {"field": "weight_delta", "direction": "desc"},
                "columns": [
                    {"field": "cid", "label": "Client", "type": "text"},
                    {"field": "is_malicious", "label": "Malicious", "type": "text"},
                    {"field": "semantic_risk", "label": "Semantic risk", "format": "percent"},
                    {"field": "nominal_mass", "label": "Nominal", "format": "percent"},
                    {"field": "rtc_full_weight", "label": "RTC full", "format": "percent"},
                    {"field": "observe_weight", "label": "Observe", "format": "percent"},
                    {"field": "weight_delta", "label": "Delta", "format": "percent", "movement": True},
                    {"field": "rtc_full_capped", "label": "Full capped", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# RTC 语义观察消融：LIE z=0.5 的良性质量误伤证据", "layout": "full"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_summary",
                "body": "## 技术摘要\n\n**本次严格配对实验直接支持‘RTC 的语义干预错误损失良性质量’。** 将语义模块从 exposure 改为 observe 后，末轮 ACC 提高 2.15 pp，主动期平均 ACC 提高 0.479 pp；同时平均恢复 4.502 pp 客户端更新质量。恶意聚合权重只增加 0.022 pp，恶意影响份额反而下降 1.202 pp。\n\n最强机制证据来自第 26 轮首次分叉：两组此前轨迹完全相同，RTC full 只压低良性 cid 18 的权重（10.00%→9.41%），当轮 5 个恶意客户端的语义风险全为 0、权重完全未减。observe 因此是值得保留的 LIE 候选，但单 seed 不能证明跨攻击安全，尚不能提升为默认。",
            },
            {"id": "kpis", "type": "metric-strip", "cardIds": ["card_final_gain", "card_mass", "card_impact"], "layout": "full"},
            {
                "id": "accuracy_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_rounds",
                "body": "## 收益从第 26 轮语义约束首次生效后出现\n\n第 1–25 轮两组 ACC 与聚合质量完全一致；第 26 轮 RTC full 首次留下 0.59 pp 零更新质量，随后模型轨迹分叉。第 26–60 轮 observe 平均 ACC 高 0.685 pp，最后 10 轮平均高 0.943 pp，说明 2.15 pp 的末轮收益不是唯一正向点，但仍受单轨迹路径依赖影响。",
            },
            {"id": "accuracy_chart_block", "type": "chart", "chartId": "chart_accuracy", "layout": "full"},
            {
                "id": "mass_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_rounds",
                "body": "## ACC 改善与恢复客户端更新质量同步\n\n攻击期 RTC full 的平均客户端权重和为 95.48%，observe 为 99.98%；差值 4.502 pp。RTC full 在 28 个主动轮的权重和低于 99.9%，observe 只有 3 个。observe 几乎没有增加恶意权重，却降低了恶意影响份额，因此该收益更符合‘保留良性有效更新并稀释恶意影响’，而不是放松后让恶意更新受益。",
            },
            {"id": "weight_chart_block", "type": "chart", "chartId": "chart_weight_sum", "layout": "full"},
            {"id": "comparison_table_block", "type": "table", "tableId": "table_comparison", "layout": "full"},
            {
                "id": "causal_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_clients",
                "body": "## 首次分叉明确指向良性误伤，而非恶意筛除\n\n第 26 轮两组共享相同 TrialPlan、客户端身份、本地随机流和分叉前模型。唯一受 RTC full 语义干预减权的 cid 18 是良性客户端，语义风险为 5.90%；5 个恶意客户端语义风险均为 0。这个同状态、单轮对照比后续平均相关更接近机制证据：LIE 的协同内点更新没有被当前语义方向捕获，反而由自然良性类别头变化触发约束。",
            },
            {"id": "divergence_table_block", "type": "table", "tableId": "table_divergence", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_summary",
                "body": "## 范围、数据与指标口径\n\n- CIFAR-10 / IID，seed 42，20 个客户端，每轮抽取 10 个，正式恶意比例 30%。\n- LIE 强度固定为 z=0.5，攻击从第 11 轮持续到第 60 轮；两组均完成 60 轮且所有配对质量门通过。\n- `active ACC` 是第 11–60 轮服务器准确率的算术平均；`final ACC` 是第 60 轮服务器准确率。\n- `客户端更新质量和` 是 RTC QP 输出的客户端权重总和；未分配部分记为 zero-update mass。",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_code",
                "body": "## 方法：先验证配对，再寻找首次可归因分叉\n\n质量门确认两组 selected partitions、TrialPlan 序列、恶意身份、初始模型与客户端随机流一致。随后按轮对齐 ACC、权重和、恶意权重与恶意影响；再定位第一个非零权重差，并在客户端表按 cid 核对语义风险和权重。代码复核显示 observe 仍计算并提交语义观测，但不会把 semantic q、risk penalty 或 semantic exposure budgets 加入 QP，因此两组的设计差异正是‘观测但不干预’。",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": "## 限制、稳健性与反例\n\n- **只有一个 seed。** 50 个主动轮不是 50 个独立重复，无法给出跨 seed 显著性结论。\n- **observe 不是安全升级结论。** LIE z=0.5 下恶意权重基本不变，但 DBA、Gaussian noise 或 Sign-flip 可能依赖语义干预。\n- **第 26 轮之后是路径依赖。** 首次分叉可以归因于语义模式，后续风险与客户端更新随模型轨迹共同变化。\n- **仅限 IID 与恶意比例 30%。** Non-IID 下自然语义异质性可能使误伤更严重，也可能改变攻击可分性。",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "layout": "full",
                "body": "## 下一步：先做四单元跨攻击安全门\n\n保持 observe 为候选、不修改默认。下一实验仅运行 DBA 与 Gaussian noise，每种攻击严格配对 `rtc_full` 和 `rtc_semantic_observe`，共 4 个单元。DBA 重点看 active ASR、恶意权重与 ACC；Gaussian noise 重点看恶意权重、恶意影响与 ACC。若任一攻击的安全指标明显恶化，则 observe 只能保留为 LIE 条件分支；只有两类安全门都通过，才值得进入更多 seeds。",
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "layout": "full",
                "body": "## 进一步问题\n\n1. DBA 的低 ASR 是否依赖 semantic exposure，还是主要来自 residual/cone budgets？\n2. Gaussian noise 中语义观测到的恶意异常是否真正进入 QP 并降低恶意质量？\n3. 若 observe 只适合 LIE，是否应按 attack-family proxy 动态切换，而不是全局默认？\n4. 通过安全门后，seed 46/47 是否仍保持主动期 ACC 正增益？",
            },
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "headline": headline,
                "timeline": timeline,
                "comparison": comparison,
                "divergence": divergence,
            },
        },
        "sources": sources,
        "package_info": {
            "analysis": "rtc_semantic_observe_lie_z05",
            "chart_map": [
                {
                    "section": "accuracy_finding",
                    "question": "When did ACC diverge?",
                    "family": "Trend",
                    "type": "line",
                    "fields": ["round", "accuracy", "defense"],
                    "claim": "Accuracy diverges after semantic constraints first reduce client mass.",
                    "palette_policy": "hard two-root cap",
                },
                {
                    "section": "mass_finding",
                    "question": "Did client mass recovery accompany the gain?",
                    "family": "Trend",
                    "type": "line",
                    "fields": ["round", "weight_sum", "defense"],
                    "claim": "Observe restores client update mass from round 26 onward.",
                    "palette_policy": "hard two-root cap",
                },
            ],
            "audience": "technical",
            "required_structure_mapping": {
                "technical_summary": "technical_summary",
                "key_findings": ["accuracy_finding", "mass_finding", "causal_finding"],
                "scope": "scope",
                "methodology": "methodology",
                "limitations": "limitations",
                "recommended_next_steps": "next_steps",
                "further_questions": "further_questions",
            },
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_sqlite(
        {
            "headline": headline,
            "timeline": timeline,
            "comparison": comparison,
            "divergence": divergence,
        }
    )
    (OUT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUT_DIR / "artifact.json")


if __name__ == "__main__":
    main()
