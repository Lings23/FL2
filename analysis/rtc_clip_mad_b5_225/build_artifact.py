"""Build the bounded technical-report artifact for the completed B5 stage."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = frame.to_dict(orient="records")
    return [
        {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in row.items()}
        for row in rows
    ]


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    comparison = pd.read_csv(OUT / "comparison.csv")
    clipping = pd.read_csv(OUT / "clipping_summary.csv")
    deltas = pd.read_csv(OUT / "deltas.csv")
    divergence = pd.read_csv(OUT / "first_clipping_divergence.csv")
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    chart_rows: list[dict[str, object]] = []
    labels = {
        "active_accuracy": "Active ACC",
        "malicious_weight_share": "Malicious weight",
        "malicious_impact_share": "Malicious impact",
        "preattack_clean_clipping_rate": "Clean clipping",
        "active_benign_clipping_rate": "Active benign clipping",
        "active_malicious_clipping_rate": "Active malicious clipping",
    }
    for row in _records(deltas):
        for field, label in labels.items():
            chart_rows.append(
                {
                    "attack": str(row["attack"]).upper(),
                    "metric": label,
                    "delta_pp": 100.0 * float(row[f"{field}_candidate_minus_baseline"]),
                    "seed": 42,
                    "active_rounds": 50,
                    "baseline": "B3R-F0.51 / MAD k=2.5",
                    "candidate": "B5 / MAD k=2.25",
                }
            )
    check_rows = [
        {"check": key, "passed": bool(value)}
        for key, value in decision["checks"].items()
    ]
    database = OUT / "report_snapshot.sqlite"
    if database.exists():
        database.unlink()
    with sqlite3.connect(database) as connection:
        comparison.to_sql("comparison", connection, index=False)
        clipping.to_sql("clipping", connection, index=False)
        deltas.to_sql("deltas", connection, index=False)
        divergence.to_sql("first_divergence", connection, index=False)
        pd.DataFrame(chart_rows).to_sql("metric_deltas", connection, index=False)
        pd.DataFrame(check_rows).to_sql("checks", connection, index=False)
    sources = [
        {
            "id": "src_b5_comparison",
            "label": "B5 paired comparison",
            "path": "analysis/rtc_clip_mad_b5_225/report_snapshot.sqlite",
            "query": {
                "description": "Reads paired B3R-F0.51 versus MAD k=2.25 utility, safety and mass metrics.",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": ["comparison"],
                "filters": [
                    "seed=42",
                    "malicious_fraction=0.3",
                    "LIE z=0.5 or strong DBA",
                    "attack-active rounds=11-60",
                    "preattack clean rounds=1-10",
                ],
                "metric_definitions": [
                    "active ACC/ASR = arithmetic mean over rounds 11-60",
                    "clipping rate = clipped client rows divided by eligible client rows",
                    "delta pp = 100 × (MAD k=2.25 - frozen MAD k=2.5)",
                ],
                "executed_at": generated,
                "sql": "SELECT * FROM comparison",
            },
        },
        {
            "id": "src_b5_deltas",
            "label": "B5 metric deltas",
            "path": "analysis/rtc_clip_mad_b5_225/report_snapshot.sqlite",
            "query": {
                "description": "Reads percentage-point changes from the frozen baseline to MAD k=2.25.",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": ["metric_deltas"],
                "filters": ["seed=42", "candidate minus baseline"],
                "metric_definitions": ["delta_pp = 100 × (MAD k=2.25 - MAD k=2.5)"],
                "executed_at": generated,
                "sql": "SELECT * FROM metric_deltas",
            },
        },
        {
            "id": "src_b5_clipping",
            "label": "B5 clipping decomposition",
            "path": "analysis/rtc_clip_mad_b5_225/report_snapshot.sqlite",
            "query": {
                "description": "Reads client clipping rates by clean/active window and benign/malicious role.",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": ["clipping"],
                "filters": ["rounds 1-10 or 11-60", "seed=42"],
                "metric_definitions": ["clipping rate = clipped rows divided by eligible client rows"],
                "executed_at": generated,
                "sql": "SELECT * FROM clipping",
            },
        },
        {
            "id": "src_b5_divergence",
            "label": "B5 first clipping divergence",
            "path": "analysis/rtc_clip_mad_b5_225/report_snapshot.sqlite",
            "query": {
                "description": "Reads the first paired client rows where clipping status or clipped norm diverges.",
                "engine": "sqlite",
                "language": "sql",
                "tables_used": ["first_divergence"],
                "filters": ["earliest changed round per attack"],
                "metric_definitions": ["clipped norm delta = candidate clipped norm - baseline clipped norm"],
                "executed_at": generated,
                "sql": "SELECT * FROM first_divergence",
            },
        },
        {
            "id": "src_b5_code",
            "label": "B5 analysis and clipping implementation",
            "path": "analysis/rtc_clip_mad_b5_225/analyze_results.py",
            "query": {
                "description": "Reviews runtime parameter checks, round windows, strict pairing and the clipping mechanism.",
                "engine": "local files",
                "language": "python",
                "tables_used": [
                    "analysis/rtc_clip_mad_b5_225/analyze_results.py",
                    "defenses/rtc/v3.py",
                    "experiments/rtc_v3/byzantine.py",
                ],
                "metric_definitions": [
                    "round 0 is evaluation-only and excluded from fit-runtime validation",
                    "MAD threshold k is 2.5 in the baseline and 2.25 in B5",
                ],
                "executed_at": generated,
            },
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "RTC-v3 B5：更强裁剪未换来更高 ACC",
        "description": "seed42、mf=0.3 下 LIE z=0.5 与 strong DBA 的 B5 阶段技术判定。",
        "generatedAt": generated,
        "sources": sources,
        "charts": [
            {
                "id": "chart_b5_deltas",
                "title": "B5 相对冻结基线的指标变化",
                "subtitle": "seed42；单位为百分点，正值表示 k=2.25 更高。",
                "dataset": "metric_deltas",
                "sourceId": "src_b5_deltas",
                "intent": "comparison",
                "question": "Which utility and filtering metrics changed under stronger clipping?",
                "rationale": "A grouped bar exposes the direction and size of the two attack-specific changes on one common percentage-point scale.",
                "type": "bar",
                "encodings": {
                    "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                    "y": {"field": "delta_pp", "type": "quantitative", "label": "Candidate - baseline", "format": "number"},
                    "color": {"field": "attack", "type": "nominal", "label": "Attack"},
                    "tooltip": [
                        {"field": "seed", "type": "quantitative", "label": "Seed"},
                        {"field": "active_rounds", "type": "quantitative", "label": "Active rounds"},
                    ],
                },
                "valueFormat": "number",
                "palette": {"kind": "categorical"},
                "settings": {"orientation": "vertical", "grouping": "grouped", "showValues": True},
                "surface": {"surface": "card", "viewMode": "visualization"},
            }
        ],
        "tables": [
            {
                "id": "table_b5_comparison",
                "title": "冻结基线与 B5 候选的精确结果",
                "subtitle": "Active 指标覆盖第11–60轮；每行是一种攻击与参数方案。",
                "dataset": "comparison",
                "sourceId": "src_b5_comparison",
                "density": "spacious",
                "defaultSort": {"field": "attack", "direction": "asc"},
                "columns": [
                    {"field": "attack", "label": "Attack", "type": "text"},
                    {"field": "mode", "label": "Mode", "type": "text"},
                    {"field": "active_accuracy", "label": "Active ACC", "format": "percent"},
                    {"field": "final_accuracy", "label": "Final ACC", "format": "percent"},
                    {"field": "active_asr", "label": "Active ASR", "format": "percent"},
                    {"field": "malicious_weight_share", "label": "Malicious weight", "format": "percent"},
                    {"field": "malicious_impact_share", "label": "Malicious impact", "format": "percent"},
                    {"field": "zero_update_mass", "label": "Zero mass", "format": "percent"},
                ],
            },
            {
                "id": "table_b5_clipping",
                "title": "客户端裁剪率分解",
                "subtitle": "攻击前1–10轮与活跃期11–60轮，按良性/恶意客户端分开。",
                "dataset": "clipping",
                "sourceId": "src_b5_clipping",
                "density": "spacious",
                "defaultSort": {"field": "attack", "direction": "asc"},
                "columns": [
                    {"field": "attack", "label": "Attack", "type": "text"},
                    {"field": "mode", "label": "Mode", "type": "text"},
                    {"field": "preattack_clean_clipping_rate", "label": "Clean clipping", "format": "percent"},
                    {"field": "active_benign_clipping_rate", "label": "Active benign", "format": "percent"},
                    {"field": "active_malicious_clipping_rate", "label": "Active malicious", "format": "percent"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "layout": "full", "body": "# RTC-v3 B5：更强裁剪未换来更高 ACC"},
            {
                "id": "summary",
                "type": "markdown",
                "sourceId": "src_b5_comparison",
                "layout": "full",
                "body": "## 技术摘要\n\n**拒绝 MAD k=2.25，冻结方案仍为 B3R-F0.51（k=2.5）。** LIE active ACC下降0.039 pp，DBA下降0.244 pp，未满足两攻击ACC均不低于基线以及LIE严格改善的门槛。LIE恶意权重与impact分别下降0.298和0.330 pp，但DBA两者反而小幅增加；DBA攻击前clean clipping仍恰好为8%，也未满足严格小于8%的门。",
            },
            {
                "id": "finding",
                "type": "markdown",
                "sourceId": "src_b5_deltas",
                "layout": "full",
                "body": "## 安全收益局限在 LIE，效用损失在 DBA 更明显\n\nLIE中更强裁剪把active malicious clipping提高4.196 pp，同时降低恶意权重和impact，但没有提高ACC；DBA中恶意客户端本来就100%被裁剪，继续压低阈值没有新增恶意识别收益，active benign clipping却增加1.173 pp，ACC下降0.244 pp。下图的正负号均为候选减基线。",
            },
            {"id": "delta_chart", "type": "chart", "chartId": "chart_b5_deltas", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "sourceId": "src_b5_comparison",
                "layout": "full",
                "body": "## 比较口径与指标定义\n\n两组均为CIFAR-10 IID、seed42、恶意比例0.3、60轮、round11起攻击、每轮10/20客户端。基线是已接受的B3R-F0.51且MAD k=2.5，候选唯一变化为k=2.25。Active ACC/ASR取第11–60轮50个值的算术平均；clean clipping取同一trial plan的第1–10轮100个客户端行。",
            },
            {"id": "comparison_table", "type": "table", "tableId": "table_b5_comparison", "layout": "full"},
            {
                "id": "method",
                "type": "markdown",
                "sourceId": "src_b5_code",
                "layout": "full",
                "body": "## 严格配对与运行时验证\n\nLIE与DBA的trial-plan hash、attack implementation hash、恶意身份、采样序列和初始模型在防御对内一致；两个候选均完成60轮且全部quality gates通过。分析器只在fit轮1–60验证运行时k=2.25，明确排除没有fit字段的评估行round0。452项仓库测试通过。",
            },
            {
                "id": "robustness",
                "type": "markdown",
                "sourceId": "src_b5_divergence",
                "layout": "full",
                "body": "## 裁剪率反常不代表阈值变宽\n\nLIE候选的全程良性裁剪率低于基线，是round1之后模型轨迹已改变的内生结果。配对客户端日志显示，k=2.25在round1就比k=2.5多裁剪一个恶意客户端，并更强地缩短已被裁剪的更新；因此不能用后续较低的平均良性裁剪率推断更激进参数反而更温和。DBA在攻击前仍有8% clean clipping，直接触及失败门。",
            },
            {"id": "clipping_table", "type": "table", "tableId": "table_b5_clipping", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": "## 证据边界\n\n本阶段是单seed、两攻击的参数筛选，不是最终统计结论；均值差异可用于执行预注册门，但不能替代多seed置信区间。由于2.25已经沿着更强裁剪方向损害效用，而2.0只会继续同方向收紧，当前证据不支持再运行2.0。",
            },
            {
                "id": "next",
                "type": "markdown",
                "layout": "full",
                "body": "## 下一步：冻结 B3R-F0.51 并进入 V1\n\n不接受B5，也不追加k=2.0。V1将用seeds 42/46/47/48/51，在clean、LIE z=0.25、LIE z=0.5和strong DBA上严格配对B3R-F0.51与Multi-Krum，并输出配对置信区间与最终对照报告。",
            },
            {
                "id": "questions",
                "type": "markdown",
                "layout": "full",
                "body": "## 待 V1 回答的问题\n\n当前唯一会改变最终结论的不确定性，是seed间方差是否会放大或反转RTC相对Multi-Krum的ACC与DBA ASR差异；V1的五seed配对统计专门回答这一点。",
            },
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated,
            "datasets": {
                "comparison": _records(comparison),
                "clipping": _records(clipping),
                "metric_deltas": chart_rows,
                "checks": check_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    (OUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
