"""Build the bounded technical report for the completed RTC-v3 B3 screen."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "rtc_anchor_recycle_b3"
EXPERIMENT = ROOT / "logs" / "rtc_v3_anchor_recycle_b3_seed42_mf03"
BASELINE = "rtc_cumulative_q_cap"
CANDIDATE = "rtc_cumulative_q_cap_anchor"
ATTACK_LABELS = {"dba": "DBA", "scaling_backdoor": "Scaling backdoor"}
MODE_LABELS = {BASELINE: "B2 baseline", CANDIDATE: "B3 full recycle"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    return None if value in {None, ""} else float(value)


def one_round_file(attack: str, defense: str) -> Path:
    matches = [
        path
        for path in (EXPERIMENT / "rounds").glob("*.csv")
        if path.name.startswith(f"{attack}__") and f"__{defense}__" in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one rounds file for {attack}/{defense}, got {len(matches)}"
        )
    return matches[0]


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


def sqlite_source(
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
        "path": "analysis/rtc_anchor_recycle_b3/report_snapshot.sqlite",
        "query": {
            "description": description,
            "engine": "sqlite",
            "language": "sql",
            "filters": [
                "seed=42",
                "malicious_fraction=0.3",
                "partition=iid",
                "attack rounds=11-60",
                "strong DBA or strong scaling backdoor",
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
            key: value if key in {"attack", "mode"} else number(value)
            for key, value in row.items()
        }
        for row in read_csv(OUT / "comparison.csv")
    ]
    indexed = {(str(row["attack"]), str(row["mode"])): row for row in comparison}
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    events = [
        {
            key: (
                value
                if key in {"attack", "event"}
                else int(value)
                if key == "round"
                else number(value)
            )
            for key, value in row.items()
        }
        for row in read_csv(OUT / "asr_trajectory_events.csv")
    ]
    mass_audit = [
        {
            key: (
                value
                if key in {"attack", "mode"}
                else int(value)
                if key == "active_rounds"
                else number(value)
            )
            for key, value in row.items()
        }
        for row in read_csv(OUT / "mass_balance_audit.csv")
    ]

    timeline: list[dict[str, object]] = []
    for attack in ATTACK_LABELS:
        for defense in (BASELINE, CANDIDATE):
            for row in read_csv(one_round_file(attack, defense)):
                if int(row["round"]) < 11:
                    continue
                timeline.append(
                    {
                        "attack": ATTACK_LABELS[attack],
                        "attack_key": attack,
                        "mode": MODE_LABELS[defense],
                        "defense": defense,
                        "series": f"{ATTACK_LABELS[attack]} · {MODE_LABELS[defense]}",
                        "round": int(row["round"]),
                        "server_asr": float(row["server_asr"]),
                        "server_accuracy": float(row["server_accuracy"]),
                        "malicious_weight_share": float(
                            row.get("fit_malicious_aggregation_weight_share") or 0.0
                        ),
                        "malicious_impact_share": float(
                            row.get("fit_malicious_impact_share") or 0.0
                        ),
                        "benign_watch_rate": float(
                            row.get("fit_benign_watch_rate") or 0.0
                        ),
                        "benign_restricted_rate": float(
                            row.get("fit_benign_restricted_rate") or 0.0
                        ),
                        "weight_sum": float(row["fit_rtc_v3_weight_sum"]),
                        "zero_update_mass": float(
                            row["fit_rtc_v3_zero_update_mass"]
                        ),
                        "anchor_recycle_mass": float(
                            row["fit_rtc_v3_anchor_recycle_mass"]
                        ),
                    }
                )
    timeline_dba = [row for row in timeline if row["attack_key"] == "dba"]
    timeline_scaling = [
        row for row in timeline if row["attack_key"] == "scaling_backdoor"
    ]

    dba_base = indexed[("dba", BASELINE)]
    dba_candidate = indexed[("dba", CANDIDATE)]
    scaling_base = indexed[("scaling_backdoor", BASELINE)]
    scaling_candidate = indexed[("scaling_backdoor", CANDIDATE)]
    headline = [
        {
            "dba_asr_delta_pp": 100.0
            * (dba_candidate["active_asr"] - dba_base["active_asr"]),
            "dba_acc_delta_pp": 100.0
            * (dba_candidate["active_accuracy"] - dba_base["active_accuracy"]),
            "scaling_asr_delta_pp": 100.0
            * (scaling_candidate["active_asr"] - scaling_base["active_asr"]),
            "scaling_acc_delta_pp": 100.0
            * (
                scaling_candidate["active_accuracy"]
                - scaling_base["active_accuracy"]
            ),
            "dba_anchor_mass": dba_candidate["anchor_recycle_mass"],
            "scaling_anchor_mass": scaling_candidate["anchor_recycle_mass"],
            "dba_candidate_zero_mass": dba_candidate["zero_update_mass"],
            "scaling_candidate_zero_mass": scaling_candidate["zero_update_mass"],
            "quality_gate_count": 13,
            "accepted": bool(decision["accepted_for_next_stage"]),
        }
    ]
    proxy_gap: list[dict[str, object]] = []
    for attack, row in (
        ("DBA", dba_candidate),
        ("Scaling backdoor", scaling_candidate),
    ):
        proxy_gap.extend(
            [
                {
                    "attack": attack,
                    "metric": "Active ASR",
                    "rate": row["active_asr"],
                    "active_rounds": 50,
                    "seed": 42,
                },
                {
                    "attack": attack,
                    "metric": "Attributed malicious weight",
                    "rate": row["malicious_weight_share"],
                    "active_rounds": 50,
                    "seed": 42,
                },
            ]
        )
    mechanism = [
        {
            "stage": "Anchor construction",
            "timing": "Before solver",
            "weight_basis": "Nominal mass of every positive-mass client",
            "attacker_can_contribute": "Yes",
            "client_attribution": "Not applicable yet",
        },
        {
            "stage": "Client-weight solver",
            "timing": "After nominal anchor exists",
            "weight_basis": "Semantic, cumulative, direction and exposure caps",
            "attacker_can_contribute": "Usually nearly quarantined",
            "client_attribution": "Recorded",
        },
        {
            "stage": "Missing-mass recycle",
            "timing": "After solver",
            "weight_basis": "Full missing mass × pre-solver nominal anchor",
            "attacker_can_contribute": "Hidden through anchor",
            "client_attribution": "No client/principal attribution",
        },
    ]

    tables = {
        "headline": headline,
        "comparison": comparison,
        "timeline": timeline,
        "timeline_dba": timeline_dba,
        "timeline_scaling": timeline_scaling,
        "proxy_gap": proxy_gap,
        "asr_events": events,
        "mass_audit": mass_audit,
        "mechanism": mechanism,
    }
    write_sqlite(tables)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sources = [
        sqlite_source(
            "src_headline",
            "B3 stage decision",
            "headline",
            "Recomputes the pre-registered B3 decision deltas from paired active-round metrics.",
            [
                "active metric = arithmetic mean over rounds 11-60",
                "delta pp = 100 × (B3 - B2)",
                "B3 accepted only when every predefined check is true",
            ],
            generated_at,
        ),
        sqlite_source(
            "src_comparison",
            "B2/B3 paired comparison",
            "comparison",
            "Four strictly paired runs summarized for utility, security, weights, impact, mass and false positives.",
            [
                "active ACC/ASR = arithmetic mean over 50 attack-active rounds",
                "benign weight = total RTC client weight - attributed malicious client weight",
                "malicious impact excludes server-owned recycle anchor",
                "zero-update mass = 1 - effective update mass",
            ],
            generated_at,
        ),
        sqlite_source(
            "src_timeline",
            "B2/B3 round trajectories",
            "timeline",
            "Reads the paired round logs and retains ACC, ASR, client weights, impact, false-positive rates and mass accounting.",
            [
                "server ASR = targeted attack success rate on the configured trigger evaluation",
                "round grain = one completed federated round",
                "displayed population = attack-active rounds 11-60",
            ],
            generated_at,
        ),
        sqlite_source(
            "src_events",
            "B3 ASR divergence events",
            "asr_events",
            "Locates first ASR increase beyond the 0.5 pp gate, first 10% ASR crossing and candidate peak.",
            [
                "material ASR divergence = candidate - baseline > 0.005",
                "10% crossing = candidate ASR >= 0.10",
                "peak = maximum candidate ASR over rounds 11-60",
            ],
            generated_at,
        ),
        sqlite_source(
            "src_mass",
            "B3 mass-balance audit",
            "mass_audit",
            "Independently checks both RTC update-mass identities for every active round.",
            [
                "effective mass = client weight sum + recycled anchor mass",
                "zero mass = 1 - effective mass",
            ],
            generated_at,
        ),
        sqlite_source(
            "src_mechanism",
            "B3 anchor recycle code-path audit",
            "mechanism",
            "Materializes the reviewed implementation order and attribution boundary from defenses/rtc/v3.py.",
            [
                "anchor construction occurs before the client-weight solver",
                "recycle uses the pre-solver nominal-weight coordinate median",
                "recycled exposure has no client/principal residual or semantic attribution",
            ],
            generated_at,
        ),
        {
            "id": "src_code",
            "label": "RTC-v3 anchor recycle implementation",
            "path": "defenses/rtc/v3.py",
            "query": {
                "description": "Code review of anchor construction, solver ordering, recycle exposure and attribution.",
                "engine": "local files",
                "language": "python",
                "filters": [
                    "anchor_recycle_fraction=1.0",
                    "semantic_intervention_risk_floor=0.5",
                    "cumulative_q_cap_power=1",
                ],
                "tables_used": [
                    "defenses/rtc/v3.py",
                    "experiments/rtc_v3/byzantine.py",
                    "analysis/rtc_anchor_recycle_b3/analyze_results.py",
                ],
                "metric_definitions": [
                    "nominal anchor = weighted coordinate median of clipped positive-mass updates using nominal weights",
                    "recycle target = anchor_recycle_fraction × (1 - final client weight sum)",
                    "recycled anchor is server-owned and is absent from client/principal residual or semantic attribution",
                ],
                "executed_at": generated_at,
            },
        },
        {
            "id": "src_manifest",
            "label": "B3 experiment provenance",
            "path": "logs/rtc_v3_anchor_recycle_b3_seed42_mf03/experiment_manifest.json",
            "query": {
                "description": "Validates four completed specs, exact trial plans, attack contracts, statuses and quality gates.",
                "engine": "local files",
                "language": "json/csv",
                "filters": [
                    "seed=42",
                    "malicious_fraction=0.3",
                    "rounds=60",
                    "attack_start_round=11",
                ],
                "tables_used": [
                    "experiment_manifest.json",
                    "status/*.json",
                    "rounds/*.csv",
                    "quality_gates.csv",
                ],
                "metric_definitions": [
                    "completion = state completed, exit code 0, last round 60 and 61 round rows",
                    "strict pairing = matching trial-plan hash and attack contract within each attack",
                ],
                "executed_at": generated_at,
            },
        },
    ]

    chart_common = {
        "intent": "trend",
        "question": "When and how strongly did full anchor recycle change ASR?",
        "rationale": "Fifty paired active rounds reveal onset, escalation and peak rather than relying on the final round.",
        "type": "line",
        "encodings": {
            "x": {"field": "round", "type": "quantitative", "label": "Round"},
            "y": {
                "field": "server_asr",
                "type": "quantitative",
                "format": "percent",
                "label": "Attack success rate",
            },
            "color": {"field": "mode", "type": "nominal", "label": "Mode"},
            "tooltip": [
                {
                    "field": "server_accuracy",
                    "type": "quantitative",
                    "format": "percent",
                    "label": "Server ACC",
                },
                {
                    "field": "malicious_weight_share",
                    "type": "quantitative",
                    "format": "percent",
                    "label": "Attributed malicious weight",
                },
                {
                    "field": "anchor_recycle_mass",
                    "type": "quantitative",
                    "format": "percent",
                    "label": "Anchor recycle mass",
                },
            ],
        },
        "valueFormat": "percent",
        "palette": {"kind": "categorical"},
        "settings": {"showPoints": "hover"},
        "surface": {"surface": "card", "viewMode": "visualization"},
    }

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "RTC-v3 B3：完整 anchor 回填提升 ACC 但破坏后门防御",
            "description": "seed42 严格配对 strong DBA 与 scaling backdoor 的 B3 阶段拒绝报告。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "card_dba_asr",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "远超预设不高于 +0.5 pp 的安全上限。",
                    "metrics": [
                        {
                            "label": "DBA active ASR 变化",
                            "field": "dba_asr_delta_pp",
                            "format": "number",
                            "unit": "pp",
                            "signed": True,
                        },
                        {
                            "label": "Active ACC 变化",
                            "field": "dba_acc_delta_pp",
                            "format": "number",
                            "unit": "pp",
                            "signed": True,
                        },
                    ],
                },
                {
                    "id": "card_scaling_asr",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "第二种后门攻击独立复现同方向安全退化。",
                    "metrics": [
                        {
                            "label": "Scaling active ASR 变化",
                            "field": "scaling_asr_delta_pp",
                            "format": "number",
                            "unit": "pp",
                            "signed": True,
                        },
                        {
                            "label": "Active ACC 变化",
                            "field": "scaling_acc_delta_pp",
                            "format": "number",
                            "unit": "pp",
                            "signed": True,
                        },
                    ],
                },
                {
                    "id": "card_anchor_mass",
                    "dataset": "headline",
                    "sourceId": "src_headline",
                    "description": "完整回填消除了 zero mass，但回填量足以主导模型轨迹。",
                    "metrics": [
                        {
                            "label": "DBA 平均 anchor mass",
                            "field": "dba_anchor_mass",
                            "format": "percent",
                        },
                        {
                            "label": "Scaling 平均 anchor mass",
                            "field": "scaling_anchor_mass",
                            "format": "percent",
                        },
                    ],
                },
            ],
            "charts": [
                {
                    "id": "chart_dba_asr",
                    "title": "DBA 逐轮攻击成功率",
                    "subtitle": "第11–60轮；B3在第12轮越过+0.5 pp门，第58轮达到89.78%。",
                    "dataset": "timeline_dba",
                    "sourceId": "src_timeline",
                    "comparisonContext": {
                        "baseline": "B2 linear q cap",
                        "grain": "federated round",
                        "unit": "ASR rate",
                        "denominator": "triggered evaluation samples",
                    },
                    **chart_common,
                },
                {
                    "id": "chart_scaling_asr",
                    "title": "Scaling backdoor 逐轮攻击成功率",
                    "subtitle": "第11–60轮；B3在第15轮越过+0.5 pp门，第53轮达到67.84%。",
                    "dataset": "timeline_scaling",
                    "sourceId": "src_timeline",
                    "comparisonContext": {
                        "baseline": "B2 linear q cap",
                        "grain": "federated round",
                        "unit": "ASR rate",
                        "denominator": "triggered evaluation samples",
                    },
                    **chart_common,
                },
                {
                    "id": "chart_proxy_gap",
                    "title": "B3 主动期 ASR 与客户端恶意权重",
                    "subtitle": "同为比例；极低的客户端恶意权重没有反映 server-owned anchor 中的攻击方向。",
                    "intent": "comparison",
                    "question": "Why did client-level safety proxies fail to signal B3's backdoor exposure?",
                    "rationale": "A grouped bar compares the outcome metric with the attributed client proxy for both attacks.",
                    "comparisonContext": {
                        "baseline": "B3 candidate only",
                        "grain": "attack-level active-period mean",
                        "unit": "rate",
                        "denominator": "50 active rounds",
                    },
                    "type": "bar",
                    "dataset": "proxy_gap",
                    "sourceId": "src_comparison",
                    "encodings": {
                        "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                        "y": {
                            "field": "rate",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Rate",
                        },
                        "color": {"field": "attack", "type": "nominal", "label": "Attack"},
                        "tooltip": [
                            {
                                "field": "active_rounds",
                                "type": "quantitative",
                                "label": "Active rounds",
                            },
                            {"field": "seed", "type": "quantitative", "label": "Seed"},
                        ],
                    },
                    "valueFormat": "percent",
                    "palette": {"kind": "categorical"},
                    "settings": {
                        "orientation": "vertical",
                        "grouping": "grouped",
                        "showValues": True,
                    },
                    "surface": {"surface": "card", "viewMode": "visualization"},
                },
            ],
            "tables": [
                {
                    "id": "table_comparison",
                    "title": "B2 与 B3 效用、安全和质量指标",
                    "subtitle": "四个严格配对运行；active指标为第11–60轮算术平均。",
                    "dataset": "comparison",
                    "sourceId": "src_comparison",
                    "density": "spacious",
                    "defaultSort": {"field": "attack", "direction": "asc"},
                    "columns": [
                        {"field": "attack", "label": "Attack", "type": "text"},
                        {"field": "mode", "label": "Mode", "type": "text"},
                        {"field": "active_accuracy", "label": "Active ACC", "format": "percent"},
                        {"field": "final_accuracy", "label": "Final ACC", "format": "percent"},
                        {"field": "active_asr", "label": "Active ASR", "format": "percent"},
                        {"field": "peak_asr", "label": "Peak ASR", "format": "percent"},
                        {"field": "malicious_weight_share", "label": "Malicious weight", "format": "percent"},
                        {"field": "benign_weight_share", "label": "Benign weight", "format": "percent"},
                        {"field": "malicious_impact_share", "label": "Malicious impact", "format": "percent"},
                        {"field": "zero_update_mass", "label": "Zero mass", "format": "percent"},
                        {"field": "anchor_recycle_mass", "label": "Anchor mass", "format": "percent"},
                        {"field": "benign_watch_rate", "label": "Benign watch", "format": "percent"},
                        {"field": "benign_restricted_rate", "label": "Benign restricted", "format": "percent"},
                        {"field": "benign_quarantined_rate", "label": "Benign quarantined", "format": "percent"},
                    ],
                },
                {
                    "id": "table_events",
                    "title": "ASR 分叉与峰值事件",
                    "subtitle": "按攻击与事件定位首次越门、首次达到10%和候选峰值。",
                    "dataset": "asr_events",
                    "sourceId": "src_events",
                    "density": "spacious",
                    "defaultSort": {"field": "round", "direction": "asc"},
                    "columns": [
                        {"field": "attack", "label": "Attack", "type": "text"},
                        {"field": "event", "label": "Event", "type": "text"},
                        {"field": "round", "label": "Round", "format": "number"},
                        {"field": "baseline_asr", "label": "B2 ASR", "format": "percent"},
                        {"field": "candidate_asr", "label": "B3 ASR", "format": "percent"},
                        {"field": "candidate_minus_baseline", "label": "ASR delta", "format": "percent", "movement": True},
                        {"field": "anchor_recycle_mass", "label": "Anchor mass", "format": "percent"},
                    ],
                },
                {
                    "id": "table_mechanism",
                    "title": "Anchor recycle 代码路径",
                    "subtitle": "构造、求解与回填的执行顺序及归因边界。",
                    "dataset": "mechanism",
                    "sourceId": "src_mechanism",
                    "density": "spacious",
                    "defaultSort": {"field": "stage", "direction": "asc"},
                    "columns": [
                        {"field": "stage", "label": "Stage", "type": "text"},
                        {"field": "timing", "label": "Timing", "type": "text"},
                        {"field": "weight_basis", "label": "Weight basis", "type": "text"},
                        {"field": "attacker_can_contribute", "label": "Attacker contribution", "type": "text"},
                        {"field": "client_attribution", "label": "Client attribution", "type": "text"},
                    ],
                },
                {
                    "id": "table_mass_audit",
                    "title": "主动期质量守恒审计",
                    "subtitle": "两条质量恒等式在每个主动轮的最大绝对误差。",
                    "dataset": "mass_audit",
                    "sourceId": "src_mass",
                    "density": "spacious",
                    "defaultSort": {"field": "attack", "direction": "asc"},
                    "columns": [
                        {"field": "attack", "label": "Attack", "type": "text"},
                        {"field": "mode", "label": "Mode", "type": "text"},
                        {"field": "active_rounds", "label": "Active rounds", "format": "number"},
                        {"field": "max_weight_plus_anchor_minus_effective_abs", "label": "|weight+anchor-effective|max", "format": "number"},
                        {"field": "max_one_minus_effective_minus_zero_abs", "label": "|1-effective-zero|max", "format": "number"},
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "layout": "full",
                    "body": "# RTC-v3 B3：完整 anchor 回填提升 ACC 但破坏后门防御",
                },
                {
                    "id": "summary",
                    "type": "markdown",
                    "sourceId": "src_comparison",
                    "layout": "full",
                    "body": "## 技术摘要\n\n**B3 必须拒绝，不能进入 B4。** 完整 coordinate-median anchor 回填把 DBA active ASR 从2.38%推高到52.17%，把 scaling backdoor active ASR从1.73%推高到27.25%；相对B2分别恶化49.792和25.524 pp。虽然active ACC分别提高0.514和0.173 pp、zero-update mass均降至0，但这不满足‘安全不退化’的阶段目标。客户端恶意权重仍仅约0.25%和0.22%，说明该代理没有计入server-owned anchor携带的隐藏攻击方向。",
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["card_dba_asr", "card_scaling_asr", "card_anchor_mass"], "layout": "full"},
                {
                    "id": "finding_asr",
                    "type": "markdown",
                    "sourceId": "src_events",
                    "layout": "full",
                    "body": "## ASR 在回填启动后持续累积，而非单轮噪声\n\n两种攻击的anchor在第11轮首次激活。DBA在第12轮首次超过+0.5 pp安全门、第19轮超过10%，第58轮峰值89.78%；scaling在第15轮首次越门、第27轮超过10%，第53轮峰值67.84%。50轮轨迹显示这是持续模型漂移，不是final round或均值被单个异常点扭曲。",
                },
                {"id": "dba_chart", "type": "chart", "chartId": "chart_dba_asr", "layout": "full"},
                {"id": "scaling_chart", "type": "chart", "chartId": "chart_scaling_asr", "layout": "full"},
                {"id": "events_table", "type": "table", "tableId": "table_events", "layout": "full"},
                {
                    "id": "finding_proxy",
                    "type": "markdown",
                    "sourceId": "src_comparison",
                    "layout": "full",
                    "body": "## 客户端恶意权重低，但 server-owned anchor 绕过了归因\n\nB3下DBA与scaling的客户端恶意权重仅0.248%和0.224%，恶意impact代理仅0.432%和0.413%，却对应52.17%和27.25%的active ASR。这个巨大缺口不是质量计算错误：每轮`client weight + anchor mass = effective mass`，且`1 - effective mass = zero mass`。问题在于现有恶意权重与impact只归因客户端项，不包含回填anchor。",
                },
                {"id": "proxy_chart", "type": "chart", "chartId": "chart_proxy_gap", "layout": "full"},
                {"id": "comparison_table", "type": "table", "tableId": "table_comparison", "layout": "full"},
                {
                    "id": "finding_code",
                    "type": "markdown",
                    "sourceId": "src_code",
                    "layout": "full",
                    "body": "## 失败机制位于 anchor 构造顺序与归因边界\n\n代码先用所有正nominal质量客户端的clipped update构造coordinate median，之后solver才根据语义、累计证据、方向和暴露预算把攻击客户端近乎quarantine。B3随后把平均约32%的缺失质量乘到这个solver前anchor上，并仅登记server anchor/total exposure。由此，被客户端cap拒绝的后门方向可经anchor重新进入聚合，而客户端与principal指标仍显示低风险。",
                },
                {"id": "mechanism_table", "type": "table", "tableId": "table_mechanism", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "src_manifest",
                    "layout": "full",
                    "body": "## 范围、数据与指标定义\n\nCIFAR-10 IID，seed42，20客户端、每轮采样10个，恶意比例30%，攻击期第11–60轮。DBA与scaling均使用冻结strong参数、replacement gain=1、poison fraction=0.3。active指标为50个攻击轮算术平均，final ACC为第60轮。四个运行均completed、exit code 0、last round 60、各61行，13/13质量门通过；每个攻击内B2/B3的trial-plan hash、attack contract、初始模型、采样序列和恶意身份一致。",
                },
                {
                    "id": "method",
                    "type": "markdown",
                    "sourceId": "src_comparison",
                    "layout": "full",
                    "body": "## 实验设计与验收方法\n\nB3相对已冻结B2只增加`anchor_recycle_fraction=1.0`；floor=0.5与linear cumulative-q cap保持不变。预设门要求每种攻击的zero mass至少减半或不高于2%、active ACC不降低、active ASR/恶意impact/恶意权重增幅各不超过0.5 pp，并要求anchor实际生效。分析先验证完成性和严格配对，再从round/client日志复算全部指标、定位首次机制与ASR分叉，并独立核对质量恒等式。",
                },
                {"id": "mass_table", "type": "table", "tableId": "table_mass_audit", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## 限制、稳健性与判定边界\n\n- 当前只有seed42，不能把效应量解释为多seed置信区间或显著性结论。\n- 但两种独立后门攻击、严格配对、同方向大幅ASR退化以及明确代码路径共同支持B3机制拒绝；无需用更多seed挽救当前full-recycle候选。\n- 恶意impact与恶意权重在当前实现中是客户端归因代理，不能用于单独证明server-owned anchor安全。\n- 本阶段未覆盖LIE，因为B3预注册安全筛选优先针对DBA与scaling；B3R通过安全门后仍需补回效用检查。",
                },
                {
                    "id": "next",
                    "type": "markdown",
                    "sourceId": "src_code",
                    "layout": "full",
                    "body": "## 下一步：B3R 只改变 recycle anchor 的权重来源\n\n保留B2的floor=0.5、linear q cap和full recycle fraction，仅把recycle vector改为solver后按最终accepted client weights计算的coordinate median，并用该向量重新计算server anchor/total exposure。若accepted质量为0则不回退到nominal anchor、直接不回填。这样针对已验证失败源，同时不混入fraction、q²、residual rank-cap或clipping变化。先准备代码、测试、两单元candidate-only dry-run和分析器；达到训练边界后由人工执行。",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## 进一步问题\n\n1. Accepted-weight anchor能否在DBA与scaling下同时维持ASR增幅不超过0.5 pp？\n2. 它能保留多少B3的ACC收益，以及是否仍能把zero mass至少减半？\n3. 若安全但回填不足，后续应只调recycle fraction，还是将B3判定为不可用并保留B2进入B4？",
                },
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
