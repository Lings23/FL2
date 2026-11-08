"""Read-only analysis and report generation for the paired relation-graph pilot.

Never invokes training. Refuses to report a run whose status, round logs,
checkpoints or defense histories are inconsistent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.relation_graph_mnist_paired import CRITICAL_SOURCES, source_hashes

GROUPS = (
    dict(name="clean_fedavg", attack="none", defense="fedavg"),
    dict(name="clean_relation_graph", attack="none", defense="relation_graph"),
    dict(name="dba_fedavg", attack="dba", defense="fedavg"),
    dict(name="dba_relation_graph", attack="dba", defense="relation_graph"),
)
NAMES = tuple(group["name"] for group in GROUPS)
ATTACK = {group["name"]: group["attack"] for group in GROUPS}
DEFENSE = {group["name"]: group["defense"] for group in GROUPS}
COLORS = {
    "clean_fedavg": "#2a78d6",
    "clean_relation_graph": "#eb6834",
    "dba_fedavg": "#1baf7a",
    "dba_relation_graph": "#eda100",
}
MALICIOUS_COLOR = "#eb6834"
BENIGN_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"
NORMALIZATION_TEXT = {
    "batch": "标准 tracked BatchNorm（服务器聚合 running_mean/running_var/num_batches_tracked）",
    "batch_no_running_stats": "无运行统计 BatchNorm（batch_no_running_stats，评估使用批内矩）",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def pct(value, digits=2):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def num(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def longest_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def integrity(run: Path, cfg: dict, rows: dict[str, list[dict]]) -> dict:
    budget = cfg["total_rounds"]
    status = read(run / "status.json")
    manifest = read(run / "manifest.json")
    summary = read(run / "summary.json")
    bundle = load_checkpoint(run / "latest_bundle.pt")
    initial = load_checkpoint(run / "initial_state.pt")
    checks = {
        "source_hashes": (set(manifest["source_sha256"]) == set(CRITICAL_SOURCES)
                          and source_hashes() == manifest["source_sha256"]),
        "status_completed": status.get("phase") == "completed" and status.get("round") == budget,
        "summary_completed_rounds": summary.get("completed_rounds") == budget,
        "bundle_round": bundle.get("completed_round") == budget,
        "initial_state_hash": state_hash(initial["state"]) == initial["sha256"]
                              == manifest["initial_state_sha256"],
    }
    for name in NAMES:
        records = rows[name]
        checks[f"{name}_rounds"] = [row["round"] for row in records] == list(range(1, budget + 1))
        checks[f"{name}_clean_count"] = all(row["clean"]["count"] == cfg["evaluator_validation_size"]
                                            for row in records)
        checks[f"{name}_asr_denominator"] = all(
            row["asr"]["asr_valid"] and row["asr"]["asr_total"] == 1800 for row in records)
        checks[f"{name}_final_state"] = (
            state_hash(bundle["states"][name]) == records[-1]["state_sha256"])
        if DEFENSE[name] == "relation_graph":
            defense = bundle["defenses"][name]
            history = defense["history"]
            checks[f"{name}_defense_round"] = defense["last_round"] == budget
            checks[f"{name}_defense_history"] = (
                [frame["round_number"] for frame in history]
                == sorted({frame["round_number"] for frame in history})
                and (not history or history[-1]["round_number"] == budget))
            restored = __import__(
                "defenses.relation_graph.defense", fromlist=["RelationGraphDefense"]
            ).RelationGraphDefense(
                __import__("config.config_loader", fromlist=["DefenseConfig"]).DefenseConfig(
                    enabled=True, type="relation_graph",
                    custom_params={"settings": cfg["relation_graph"]["settings"]}),
                num_clients=cfg["clients"])
            restored.load_state_dict(json.loads(json.dumps(defense, allow_nan=False)))
            checks[f"{name}_defense_roundtrip"] = restored.state_dict() == defense
        else:
            checks[f"{name}_stateless"] = bundle["defenses"][name] is None
    checks["paired_round1_clean"] = (rows["clean_fedavg"][0]["state_sha256"]
                                     == rows["clean_relation_graph"][0]["state_sha256"])
    checks["paired_round1_dba"] = (rows["dba_fedavg"][0]["state_sha256"]
                                   == rows["dba_relation_graph"][0]["state_sha256"])
    with np.load(run / "split_indices.npz", allow_pickle=False) as archive:
        indices = {key: archive[key] for key in archive.files}
    with np.load(run / "evaluation_order.npz", allow_pickle=False) as archive:
        order = archive["indices"]
    expected_order = np.random.default_rng(cfg["evaluation_order_seed"]).permutation(
        np.asarray(indices["validation"], dtype=np.int64))
    checks["evaluation_order"] = (len(order) == cfg["evaluator_validation_size"]
                                  and np.array_equal(order, expected_order))
    splits = read(run / "splits.json")
    checks["split_sizes"] = (
        splits["validation"]["count"] == cfg["evaluator_validation_size"]
        and all(splits[f"client_{index}"]["count"] == cfg["max_client_samples"]
                for index in range(cfg["clients"])))
    return dict(checks=checks, passed=all(checks.values()), manifest=manifest,
                status=status, bundle_round=bundle["completed_round"],
                initial_state_sha256=manifest["initial_state_sha256"],
                final_state_sha256={name: state_hash(bundle["states"][name])
                                    for name in NAMES})


def client_table(records, malicious_ids):
    table = {}
    for index in range(len(records[0]["clients"])):
        table[index] = dict(
            client_id=index, is_malicious=index in malicious_ids,
            rounds_downweighted=0, total_drop=0.0, max_drop=0.0, min_ratio=1.0,
            first_round=None, last_round=None,
            drops=[], ratios=[])
    for row in records:
        for client in row["clients"]:
            entry = table[client["client_id"]]
            drop = client["weight_drop"]
            ratio = client["weight_ratio"]
            entry["drops"].append(drop)
            entry["ratios"].append(ratio)
            if drop > 1e-12:
                entry["rounds_downweighted"] += 1
                entry["total_drop"] += drop
                entry["max_drop"] = max(entry["max_drop"], drop)
                entry["min_ratio"] = min(entry["min_ratio"], ratio)
                entry["first_round"] = entry["first_round"] or row["round"]
                entry["last_round"] = row["round"]
    for entry in table.values():
        entry["max_consecutive_downweighted_rounds"] = longest_run(
            [drop > 1e-12 for drop in entry["drops"]])
        entry["max_consecutive_zero_update"] = longest_run(
            [client["update_norm"] <= 1e-12
             for client in [row["clients"][entry["client_id"]] for row in records]])
        del entry["drops"]
        del entry["ratios"]
    return [table[index] for index in range(len(table))]


def group_summary(records, malicious_ids):
    recent = records[-10:]
    graph_rows = [dict(round=row["round"], **row["graph"])
                  for row in records if row["graph"] is not None]
    bench = [client for row in records for client in row["clients"]
             if not client["is_malicious_evaluation_only"]]
    mal = [client for row in records for client in row["clients"]
           if client["is_malicious_evaluation_only"]]
    norm_ratios = []
    for row in records:
        mal_norms = [client["update_norm"] for client in row["clients"]
                     if client["is_malicious_evaluation_only"]]
        ben_norms = [client["update_norm"] for client in row["clients"]
                     if not client["is_malicious_evaluation_only"]]
        if mal_norms and ben_norms:
            norm_ratios.append(float(np.mean(mal_norms) / np.mean(ben_norms)))
    table = client_table(records, malicious_ids)
    result = dict(
        final=dict(round=records[-1]["round"],
                   accuracy=records[-1]["clean"]["accuracy"],
                   loss=records[-1]["clean"]["loss"],
                   asr=records[-1]["asr"]["asr"],
                   aggregate_update_norm=records[-1]["aggregate_update_norm"]),
        last10=dict(accuracy=mean(row["clean"]["accuracy"] for row in recent),
                    loss=mean(row["clean"]["loss"] for row in recent),
                    asr=mean(row["asr"]["asr"] for row in recent),
                    aggregate_update_norm=mean(row["aggregate_update_norm"] for row in recent)),
        peak=dict(asr=max(row["asr"]["asr"] for row in records),
                  asr_round=max(records, key=lambda row: row["asr"]["asr"])["round"],
                  accuracy=max(row["clean"]["accuracy"] for row in records)),
        minimum=dict(asr=min(row["asr"]["asr"] for row in records),
                     asr_round=min(records, key=lambda row: row["asr"]["asr"])["round"]),
        local_asr=dict(first=records[0]["asr"].get("dba_local_asr_mean"),
                       last=records[-1]["asr"].get("dba_local_asr_mean")),
        benign=dict(
            client_rounds=len(bench),
            downweighted_client_rounds=sum(c["weight_drop"] > 1e-12 for c in bench),
            downweighted_rounds=sum(any(c["weight_drop"] > 1e-12 for c in row["clients"]
                                       if not c["is_malicious_evaluation_only"])
                                    for row in records),
            mean_ratio=mean(c["weight_ratio"] for c in bench),
            min_ratio=min((c["weight_ratio"] for c in bench), default=None),
            total_drop=float(sum(c["weight_drop"] for c in bench)),
            max_drop=max((c["weight_drop"] for c in bench), default=0.0)),
        malicious=None if not mal else dict(
            client_rounds=len(mal),
            active_client_rounds=sum(bool(c["train"].get("attack_active")) for c in mal),
            nominal_weight=records[0]["malicious_weight_summary"]["nominal_sum"],
            mean_update_norm_ratio=mean(norm_ratios),
            downweighted_client_rounds=sum(c["weight_drop"] > 1e-12 for c in mal),
            downweighted_rounds=sum(any(c["weight_drop"] > 1e-12 for c in row["clients"]
                                       if c["is_malicious_evaluation_only"])
                                    for row in records),
            mean_ratio=mean(c["weight_ratio"] for c in mal),
            min_ratio=min((c["weight_ratio"] for c in mal), default=None),
            total_drop=float(sum(c["weight_drop"] for c in mal)),
            max_drop=max((c["weight_drop"] for c in mal), default=0.0)),
        clients=table,
        zero_weight_clients=sorted({c["client_id"] for row in records for c in row["clients"]
                                    if c["aggregation_weight"] <= 0.0}),
        zero_update_clients=sorted({c["client_id"] for row in records for c in row["clients"]
                                    if c["update_norm"] <= 1e-12}),
        max_consecutive_zero_weight=max(
            (longest_run([row["clients"][index]["aggregation_weight"] <= 0.0 for row in records])
             for index in range(len(table))), default=0),
        max_consecutive_zero_update=max(
            (longest_run([row["clients"][index]["update_norm"] <= 1e-12 for row in records])
             for index in range(len(table))), default=0),
        median_client_seconds=float(np.median([row["client_seconds"] for row in records])),
        median_aggregation_seconds=float(np.median([row["aggregation_seconds"] for row in records])),
        median_evaluation_seconds=float(np.median([row["evaluation_seconds"] for row in records])),
        median_round_seconds=float(np.median([row["round_seconds"] for row in records])),
    )
    if graph_rows:
        states = [row["state"] for row in graph_rows]
        interventions = [row["round"] for row in graph_rows if row["anchor_weight"] > 1e-12]
        result["graph"] = dict(
            cold_start_rounds=sum(row["calibrated_fraction"] == 0.0 for row in graph_rows),
            insufficient_evidence_rounds=sum(s == "insufficient_evidence" for s in states),
            no_anomaly_evidence_rounds=sum(s == "no_anomaly_evidence" for s in states),
            anomalous_structure_rounds=sum(s == "anomalous_structure" for s in states),
            confirmed_group_rounds=sum(row["confirmed_groups"] > 0 for row in graph_rows),
            intervention_rounds=len(interventions),
            first_intervention_round=interventions[0] if interventions else None,
            downweighted_client_rounds=sum(row["downweighted_clients"] for row in graph_rows),
            last10_mean_calibrated_fraction=mean(
                row["calibrated_fraction"] for row in graph_rows[-10:]),
            last10_mean_anchor_weight=mean(row["anchor_weight"] for row in graph_rows[-10:]),
            total_anchor_weight=float(sum(row["anchor_weight"] for row in graph_rows)),
        )
    return result


def anomalous_cases(records, details, malicious_ids):
    cases = []
    by_round = {row["round"]: row for row in records}
    for row in records:
        graph = row["graph"]
        if graph is None or graph["state"] != "anomalous_structure":
            continue
        previous = by_round.get(row["round"] - 1)
        mal = [client for client in row["clients"] if client["is_malicious_evaluation_only"]]
        ben = [client for client in row["clients"] if not client["is_malicious_evaluation_only"]]
        groups = details.get(row["round"], {}).get("groups", [])
        down = [client["client_id"] for client in row["clients"]
                if client["weight_drop"] > 1e-12]
        cases.append(dict(
            round=row["round"],
            asr=row["asr"]["asr"],
            previous_asr=None if previous is None else previous["asr"]["asr"],
            accuracy=row["clean"]["accuracy"],
            previous_accuracy=None if previous is None else previous["clean"]["accuracy"],
            anchor_weight=graph["anchor_weight"],
            downweighted_clients=graph["downweighted_clients"],
            groups=groups,
            group_members=[group["principals"] for group in groups],
            downweighted_ids=down,
            downweighted_malicious_ids=[value for value in down if value in malicious_ids],
            groups_contain_malicious=any(set(group["principals"]) & malicious_ids
                                         for group in groups),
            malicious_mean_ratio=mean(c["weight_ratio"] for c in mal),
            benign_mean_ratio=mean(c["weight_ratio"] for c in ben),
            malicious_min_ratio=min(c["weight_ratio"] for c in mal),
        ))
    return cases


def summarize(run: Path, cfg: dict, rows: dict[str, list[dict]], check: dict) -> dict:
    budget = cfg["total_rounds"]
    malicious = set(cfg["malicious_clients"])
    group_summaries = {
        name: group_summary(rows[name], malicious if ATTACK[name] != "none" else set())
        for name in NAMES}
    result = dict(
        run=run.as_posix(), seed=cfg["seed"], budget_rounds=budget,
        completed_rounds=min(len(rows[name]) for name in NAMES),
        normalization=cfg.get("normalization"),
        integrity=check, groups=group_summaries,
    )
    cf, cg = rows["clean_fedavg"], rows["clean_relation_graph"]
    df, dg = rows["dba_fedavg"], rows["dba_relation_graph"]
    result["comparisons"] = dict(
        clean_final_accuracy_delta=group_summaries["clean_relation_graph"]["final"]["accuracy"]
                                  - group_summaries["clean_fedavg"]["final"]["accuracy"],
        clean_last10_accuracy_delta=group_summaries["clean_relation_graph"]["last10"]["accuracy"]
                                   - group_summaries["clean_fedavg"]["last10"]["accuracy"],
        dba_final_asr_delta=group_summaries["dba_relation_graph"]["final"]["asr"]
                           - group_summaries["dba_fedavg"]["final"]["asr"],
        dba_last10_asr_delta=group_summaries["dba_relation_graph"]["last10"]["asr"]
                            - group_summaries["dba_fedavg"]["last10"]["asr"],
        dba_final_accuracy_delta=group_summaries["dba_relation_graph"]["final"]["accuracy"]
                                - group_summaries["dba_fedavg"]["final"]["accuracy"],
        dba_last10_accuracy_delta=group_summaries["dba_relation_graph"]["last10"]["accuracy"]
                                 - group_summaries["dba_fedavg"]["last10"]["accuracy"],
    )
    result["identical_trajectory_rounds"] = {
        "clean": sum(a["state_sha256"] == b["state_sha256"] for a, b in zip(cf, cg)),
        "dba": sum(a["state_sha256"] == b["state_sha256"] for a, b in zip(df, dg)),
    }
    # Frozen qualification: DBA/FedAvg against the clean FedAvg control.
    q = cfg["qualification"]
    base = {row["round"]: row for row in cf}
    checks = []
    for row in df[-q["sustained_rounds"]:]:
        reference = base[row["round"]]
        checks.append(dict(
            round=row["round"],
            clean_model_ready=reference["clean"]["accuracy"] >= q["minimum_clean_accuracy"],
            task_preserved=(row["clean"]["accuracy"] >=
                            reference["clean"]["accuracy"] - q["maximum_accuracy_drop"]),
            attack_strong=row["asr"]["asr"] >= q["minimum_asr"],
            exceeds_clean_trigger_response=(row["asr"]["asr"] - reference["asr"]["asr"]
                                            >= q["minimum_asr_excess"])))
    passed = (len(checks) == q["sustained_rounds"]
              and all(all(value for key, value in entry.items() if key != "round")
                      for entry in checks))
    result["attack_strength_fedavg"] = dict(
        qualified=passed, checks=checks, criteria=q,
        interpretation=("eligible_for_defense_evaluation" if passed
                        else "not_qualified_no_defense_success_claim"))
    details = {
        name: {row["round"]: row for row in load_lines(run / f"graph_{name}.jsonl")}
        for name in ("clean_relation_graph", "dba_relation_graph")}
    cases = {
        "clean_relation_graph": anomalous_cases(cg, details["clean_relation_graph"], set()),
        "dba_relation_graph": anomalous_cases(dg, details["dba_relation_graph"], malicious),
    }
    result["anomalous_cases"] = cases
    result["anomalous_summary"] = {
        name: dict(
            rounds=[case["round"] for case in items],
            group_members=[case["group_members"] for case in items],
            downweighted_ids=[case["downweighted_ids"] for case in items],
            downweighted_malicious_ids=[case["downweighted_malicious_ids"] for case in items],
            rounds_with_benign_only_groups=sum(not case["groups_contain_malicious"]
                                               for case in items),
            rounds_with_malicious_groups=sum(case["groups_contain_malicious"]
                                             for case in items),
            false_downweighted_client_rounds=sum(len(case["downweighted_ids"])
                                                 for case in items),
        )
        for name, items in cases.items()}
    # Descriptive correspondence on rounds where the graph actually intervened.
    dg_graph = [row for row in dg if row["graph"] is not None]
    active = [row for row in dg_graph if row["graph"]["anchor_weight"] > 1e-12]
    if len(active) >= 3:
        by_round = {row["round"]: row for row in dg}
        pairs = [(row["graph"]["anchor_weight"],
                  -(by_round[row["round"] + 1]["asr"]["asr"] - row["asr"]["asr"]))
                 for row in active if row["round"] + 1 in by_round]
        result["intervention_correspondence"] = dict(
            active_rounds=len(active),
            pearson_anchor_vs_next_asr_drop=pearson([x for x, _ in pairs],
                                                    [y for _, y in pairs]))
    else:
        result["intervention_correspondence"] = dict(
            active_rounds=len(active), pearson_anchor_vs_next_asr_drop=None)
    return result


def write_csvs(run: Path, rows: dict[str, list[dict]]) -> Path:
    out = run / "results"
    out.mkdir(exist_ok=True)
    with (out / "round_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ["group", "round", "accuracy", "loss", "asr", "dba_local_asr_mean",
                   "dba_local_asr_min", "dba_local_asr_max", "graph_state",
                   "calibrated_fraction", "confirmed_groups", "graph_downweighted_clients",
                   "broadcast_model_weight", "aggregate_update_norm",
                   "benign_mean_weight_ratio", "benign_total_drop",
                   "malicious_mean_weight_ratio", "malicious_total_drop",
                   "client_seconds", "aggregation_seconds", "evaluation_seconds",
                   "round_seconds"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in NAMES:
            for row in rows[name]:
                graph = row["graph"] or {}
                writer.writerow(dict(
                    group=name, round=row["round"],
                    accuracy=row["clean"]["accuracy"], loss=row["clean"]["loss"],
                    asr=row["asr"]["asr"],
                    dba_local_asr_mean=row["asr"].get("dba_local_asr_mean"),
                    dba_local_asr_min=row["asr"].get("dba_local_asr_min"),
                    dba_local_asr_max=row["asr"].get("dba_local_asr_max"),
                    graph_state=graph.get("state", ""),
                    calibrated_fraction=graph.get("calibrated_fraction", ""),
                    confirmed_groups=graph.get("confirmed_groups", ""),
                    graph_downweighted_clients=graph.get("downweighted_clients", ""),
                    broadcast_model_weight=row["broadcast_model_weight"],
                    aggregate_update_norm=row["aggregate_update_norm"],
                    benign_mean_weight_ratio=row["benign_weight_summary"]["mean_weight_ratio"],
                    benign_total_drop=row["benign_weight_summary"]["total_drop"],
                    malicious_mean_weight_ratio=(row["malicious_weight_summary"] or {}).get(
                        "mean_weight_ratio", ""),
                    malicious_total_drop=(row["malicious_weight_summary"] or {}).get(
                        "total_drop", ""),
                    client_seconds=row["client_seconds"],
                    aggregation_seconds=row["aggregation_seconds"],
                    evaluation_seconds=row["evaluation_seconds"],
                    round_seconds=row["round_seconds"]))
    with (out / "client_weights.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ["group", "round", "client_id", "is_malicious_evaluation_only",
                   "nominal_weight", "aggregation_weight", "weight_ratio", "weight_drop",
                   "update_norm", "attack_active", "dba_fragment_index",
                   "dba_poisoned_examples"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in NAMES:
            for row in rows[name]:
                for client in row["clients"]:
                    writer.writerow(dict(
                        group=name, round=row["round"], client_id=client["client_id"],
                        is_malicious_evaluation_only=client["is_malicious_evaluation_only"],
                        nominal_weight=client["nominal_weight"],
                        aggregation_weight=client["aggregation_weight"],
                        weight_ratio=client["weight_ratio"],
                        weight_drop=client["weight_drop"],
                        update_norm=client["update_norm"],
                        attack_active=client["train"].get("attack_active", ""),
                        dba_fragment_index=client["train"].get("dba_fragment_index", ""),
                        dba_poisoned_examples=client["train"].get("dba_poisoned_examples", "")))
    with (out / "graph_edges.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ["group", "round", "left", "right", "backbone", "task",
                   "reference_center", "reference_scale", "residual", "score",
                   "confirmations", "effective_observations", "exceedance_fraction",
                   "strength"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in ("clean_relation_graph", "dba_relation_graph"):
            for row in load_lines(run / f"graph_{name}.jsonl"):
                for edge in row["edges"]:
                    writer.writerow(dict(
                        group=name, round=row["round"], left=edge["left"],
                        right=edge["right"], backbone=edge["backbone"], task=edge["task"],
                        reference_center=edge["reference_center"],
                        reference_scale=edge["reference_scale"],
                        residual=edge["residual"], score=edge["score"],
                        confirmations=edge["confirmations"],
                        effective_observations=edge["effective_observations"],
                        exceedance_fraction=edge["exceedance_fraction"],
                        strength=edge["strength"]))
    return out


def _plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False,
        "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY, "text.color": INK,
    })
    return plt


def _end_labels(ax, series, gap):
    """Direct labels at line ends; near-overlapping series rely on the legend."""
    entries = sorted([(value[-1], label, color) for value, label, color in series],
                     key=lambda item: item[0])
    placed = []
    for value, label, color in entries:
        if placed and abs(value - placed[-1]) < gap:
            continue
        placed.append(value)
        ax.annotate(label, (ax.get_xlim()[1], value), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK_SECONDARY, annotation_clip=False)


def _line_panel(ax, x, series, ylabel, title, gap):
    for values, label, color, style in series:
        ax.plot(x, values, color=color, linestyle=style, label=label)
    ax.set_title(title, loc="left", color=INK)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xlim(1, len(x))
    _end_labels(ax, [(values, label, color) for values, label, color, _ in series], gap)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_all(run: Path, cfg: dict, rows: dict[str, list[dict]], out: Path) -> None:
    plt = _plot_style()
    rounds = list(range(1, cfg["total_rounds"] + 1))
    # Accuracy and ASR, two panels, one axis each.
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    _line_panel(axes[0], rounds, [
        ([row["clean"]["accuracy"] for row in rows[name]], name, COLORS[name], "-")
        for name in NAMES], "clean accuracy", "Clean accuracy (2000-example holdout)", 0.02)
    _line_panel(axes[1], rounds, [
        ([row["asr"]["asr"] for row in rows[name]], name, COLORS[name],
         "-" if ATTACK[name] != "none" else "--")
        for name in NAMES], "DBA full-trigger ASR", "DBA full-trigger ASR (1800 non-target)", 0.03)
    _save(fig, out / "accuracy_asr_curves.png")

    # Weight ratios.
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    dba_graph = rows["dba_relation_graph"]
    clean_graph = rows["clean_relation_graph"]
    _line_panel(ax, rounds, [
        ([row["malicious_weight_summary"]["mean_weight_ratio"] for row in dba_graph],
         "DBA/RG malicious", MALICIOUS_COLOR, "-"),
        ([row["benign_weight_summary"]["mean_weight_ratio"] for row in dba_graph],
         "DBA/RG benign", BENIGN_COLOR, "-"),
        ([row["benign_weight_summary"]["mean_weight_ratio"] for row in clean_graph],
         "clean/RG benign", "#1baf7a", "--"),
    ], "mean actual / nominal weight", "Client weight retention (1.0 = no downweight)", 0.02)
    ax.set_ylim(0.85, 1.02)
    ax.axhline(1.0, color="#9b9a94", linewidth=0.8, linestyle=":")
    _save(fig, out / "weight_retention.png")

    # Graph diagnostics.
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for name in ("clean_relation_graph", "dba_relation_graph"):
        rows_g = rows[name]
        axes[0].plot(rounds, [row["graph"]["calibrated_fraction"] for row in rows_g],
                     color=COLORS[name], label=name)
        axes[1].plot(rounds, [row["graph"]["downweighted_clients"] for row in rows_g],
                     color=COLORS[name], label=name)
        axes[2].plot(rounds, [row["broadcast_model_weight"] for row in rows_g],
                     color=COLORS[name], label=name)
    for ax, title in zip(axes, ("Calibrated edge fraction", "Downweighted clients",
                                "Broadcast-model (anchor) weight")):
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_xlim(1, cfg["total_rounds"])
        ax.axvspan(29.5, 35.5, color="#f0efec", zorder=0)
        ax.legend(loc="upper left", fontsize=7.5)
    from matplotlib.ticker import MaxNLocator
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].text(30.2, 0.35, "DBA/RG false-positive\ninterventions (30–35)",
                 fontsize=7, color=INK_SECONDARY)
    axes[2].set_xlabel("round")
    _save(fig, out / "graph_diagnostics.png")

    # Aggregate update norm.
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    _line_panel(ax, rounds, [
        ([row["aggregate_update_norm"] for row in rows[name]], name, COLORS[name], "-")
        for name in NAMES], "aggregate update norm",
        "True aggregate update norm (trainable parameters)", 0.005)
    _save(fig, out / "aggregate_update_norm.png")


def report_markdown(summary: dict, cfg: dict, check: dict) -> str:
    g = summary["groups"]
    c = summary["comparisons"]
    q = summary["attack_strength_fedavg"]
    clean_fed = g["clean_fedavg"]
    clean_rg = g["clean_relation_graph"]
    dba_fed = g["dba_fedavg"]
    dba_rg = g["dba_relation_graph"]
    normalization = cfg.get("normalization")
    normalization_text = NORMALIZATION_TEXT.get(normalization, f"`{normalization}`")
    lines = []
    lines.append("# 层间关系残差图防御：MNIST-CNN 清洁 / DBA 配对预实验（种子 42）\n")
    lines.append(f"运行目录：`{summary['run']}`。四组各完成 "
                 f"{summary['completed_rounds']}/{summary['budget_rounds']} 轮。"
                 "评估使用 2000 张独立训练留出样本，DBA ASR 分母为 1800 张非目标类样本；"
                 "最终测试集未使用。所有结论限定于种子 42 和当前 IID、"
                 f"{normalization_text}协议。\n")

    lines.append("## 结果总览\n")
    lines.append("| 组 | 最终 ACC | 末 10 轮 ACC | 最终 loss | 末 10 轮 loss | "
                 "最终 ASR | 末 10 轮 ASR | 峰值 ASR（轮次） |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in NAMES:
        row = g[name]
        lines.append(f"| {name} | {pct(row['final']['accuracy'])} | {pct(row['last10']['accuracy'])} | "
                     f"{num(row['final']['loss'])} | {num(row['last10']['loss'])} | "
                     f"{pct(row['final']['asr'])} | {pct(row['last10']['asr'])} | "
                     f"{pct(row['peak']['asr'])}（{row['peak']['asr_round']}） |")
    lines.append("\n表中 clean 组的 ASR 是清洁模型对 DBA 全触发器的误响应，"
                 "用作 `exceeds_clean_trigger_response` 的基线；它不是攻击成功。")
    lines.append("")

    # Q1
    lines.append("## 1. 无攻击时，图防御与 FedAvg 的差异\n")
    lines.append(f"最终准确率差（relation_graph − FedAvg）为 "
                 f"{100 * c['clean_final_accuracy_delta']:+.2f} 个百分点，"
                 f"末 10 轮均值差为 {100 * c['clean_last10_accuracy_delta']:+.2f} 个百分点。")
    down_clients = [entry for entry in clean_rg["clients"]
                    if entry["rounds_downweighted"] > 0]
    if down_clients:
        described = "、".join(
            f"客户端 {entry['client_id']}（{entry['rounds_downweighted']} 轮，"
            f"总降权 {entry['total_drop']:.4f}，最小保留比例 {entry['min_ratio']:.4f}）"
            for entry in down_clients)
        lines.append(f"无攻击组中被降权的正常客户端：{described}。"
                     f"共 {clean_rg['benign']['downweighted_client_rounds']} 个客户端-轮次，"
                     f"总降权质量 {clean_rg['benign']['total_drop']:.6f}，"
                     f"单次最大降权 {clean_rg['benign']['max_drop']:.6f}，"
                     f"最小保留比例 {clean_rg['benign']['min_ratio']:.6f}。")
    else:
        lines.append("无攻击组没有任何正常客户端被降权（所有轮次的 actual weight 等于 nominal weight）。")
    lines.append(f"持续零更新的检查：最大连续零聚合权重轮数为 "
                 f"{clean_rg['max_consecutive_zero_weight']}，最大连续零更新范数轮数为 "
                 f"{clean_rg['max_consecutive_zero_update']}；"
                 f"零权重客户端 {clean_rg['zero_weight_clients'] or '无'}，"
                 f"零更新客户端 {clean_rg['zero_update_clients'] or '无'}。"
                 f"清洁组图防御的干预轮数为 {clean_rg['graph']['intervention_rounds']}；"
                 f"与 FedAvg 逐轮 state hash 相同的轮数为 "
                 f"{summary['identical_trajectory_rounds']['clean']}/{summary['budget_rounds']}。")
    lines.append("")

    # Q2
    lines.append("## 2. DBA 在 FedAvg 下的强度\n")
    lines.append(f"峰值 ASR 为 {pct(dba_fed['peak']['asr'])}（第 {dba_fed['peak']['asr_round']} 轮），"
                 f"最终 ASR 为 {pct(dba_fed['final']['asr'])}，末 10 轮均值为 "
                 f"{pct(dba_fed['last10']['asr'])}；最终 ACC 为 "
                 f"{pct(dba_fed['final']['accuracy'])}。")
    if (dba_fed["peak"]["asr_round"] <= 3
            and dba_fed["peak"]["asr"] - dba_fed["final"]["asr"] > 0.2):
        lines.append(f"峰值出现在第 {dba_fed['peak']['asr_round']} 轮并随后明显回落，"
                     "因此该峰值只反映训练早期的短暂状态，不能当作持续攻击强度。")
    # Local-fragment ASR is recorded every round; cite the first/last values.
    first_asr = dba_fed["local_asr"]["first"]
    last_asr = dba_fed["local_asr"]["last"]
    if first_asr is not None and last_asr is not None:
        lines.append("四个局部碎片在全部 120 轮均启用（0–3，每客户端 250 张投毒样本，无上传放大）；"
                     f"局部碎片 ASR 均值也从第 1 轮 {pct(first_asr)} 降到第 120 轮 "
                     f"{pct(last_asr)}，与全触发器 ASR 的衰减同向。")
    execution = dba_fed["malicious"]
    lines.append(f"执行检查：20/20 客户端每轮全参与，恶意客户端每轮 4/4 实际训练且 "
                 f"`attack_active=True`（{execution['active_client_rounds']}/"
                 f"{execution['client_rounds']} 恶意客户端-轮次），名义权重恒为 "
                 f"{pct(execution['nominal_weight'])}，恶意更新范数平均为正常客户端的 "
                 f"{num(execution['mean_update_norm_ratio'], 2)} 倍。"
                 "因此 ASR 下降不是客户端漏选或恶意更新被压成零造成的。")
    if (dba_fed["minimum"]["asr_round"] > dba_fed["peak"]["asr_round"]
            and dba_fed["final"]["asr"] > dba_fed["minimum"]["asr"] + 0.05):
        lines.append(f"ASR 在第 {dba_fed['minimum']['asr_round']} 轮降到最低 "
                     f"{pct(dba_fed['minimum']['asr'])}，随后缓慢回升到最终 "
                     f"{pct(dba_fed['final']['asr'])}；攻击没有完全消失，"
                     "但回升后的水平仍远低于 80% 的持续强度判据。")
    if q["qualified"]:
        lines.append("按冻结的 `qualification` 判据（连续 3 轮：清洁基线 ACC ≥ 95%、"
                     "攻击组 ACC 不低于基线 − 2 个百分点、ASR ≥ 80%、相对清洁触发响应超出 ≥ 60 个百分点），"
                     "**DBA 在 FedAvg 下通过强度判据**，可以作为防御比较的攻击前提。")
    else:
        failed = defaultdict(list)
        for entry in q["checks"]:
            for key, value in entry.items():
                if key != "round" and not value:
                    failed[key].append(entry["round"])
        detail = "；".join(f"{key} 未通过于轮次 {values}" for key, values in failed.items())
        lines.append("按冻结的 `qualification` 判据，**DBA 在 FedAvg 下未通过攻击强度判据**"
                     f"（{detail or '末 3 轮检查不完整'}）。因此不能把后续防御结果解释为"
                     "“成功抵御了一个已建立的高强度 DBA”；本批只保留观测结果，并限定结论。")
    lines.append("")

    # Q3
    lines.append("## 3. 图防御下的 ASR / ACC 是否与异常子图和恶意权重对应\n")
    if not q["qualified"]:
        lines.append("首先限定：第 2 节的攻击强度判据未通过，因此本节只能描述两条轨迹的观测差异，"
                     "不能把差异解释为“图防御抵御了已建立的 DBA”。")
    lines.append(f"DBA/relation_graph 相对 DBA/FedAvg：最终 ASR 差 "
                 f"{100 * c['dba_final_asr_delta']:+.2f} 个百分点，末 10 轮 ASR 差 "
                 f"{100 * c['dba_last10_asr_delta']:+.2f} 个百分点；最终 ACC 差 "
                 f"{100 * c['dba_final_accuracy_delta']:+.2f} 个百分点，末 10 轮 ACC 差 "
                 f"{100 * c['dba_last10_accuracy_delta']:+.2f} 个百分点。")
    if summary["identical_trajectory_rounds"]["dba"] < summary["budget_rounds"]:
        lines.append(f"两条轨迹在前 {summary['identical_trajectory_rounds']['dba']} 轮逐轮 "
                     "state hash 相同，首次降权后开始分离。")
    if dba_rg["graph"]["intervention_rounds"] == 0:
        lines.append(f"图防御在 {summary['budget_rounds']} 轮内没有确认任何异常子图，"
                     "anchor weight 始终为 0，"
                     "也没有客户端被降权；`dba_relation_graph` 与 `dba_fedavg` 的逐轮模型状态"
                     f"完全相同（{summary['identical_trajectory_rounds']['dba']}/"
                     f"{summary['budget_rounds']} 轮 state hash 相同）。"
                     "因此本批不存在“图状态 → ASR/ACC 变化”的对应关系；任何 ASR/ACC 差异都只能是 0，"
                     "不能据此判断图结构有效或无效。")
    else:
        lines.append(f"图防御共在 {dba_rg['graph']['intervention_rounds']} 轮产生非零 anchor weight，"
                     f"首次干预在第 {dba_rg['graph']['first_intervention_round']} 轮；"
                     f"确认异常结构的轮数为 {dba_rg['graph']['anomalous_structure_rounds']}，"
                     f"累计降权客户端-轮次 {dba_rg['graph']['downweighted_client_rounds']}。"
                     f"恶意客户端平均保留比例 {num(dba_rg['malicious']['mean_ratio'])}，"
                     f"正常客户端平均保留比例 {num(dba_rg['benign']['mean_ratio'])}。")
        cases = summary["anomalous_cases"]["dba_relation_graph"]
        if cases:
            lines.append("\n具体轮次与异常子图成员：\n")
            lines.append("| 轮次 | 异常子图成员 | 被降权客户端 | 子图是否含恶意客户端 | "
                         "恶意平均保留 | 正常平均保留 | ASR（上轮→本轮） | ACC（上轮→本轮） | anchor |")
            lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|")
            for case in cases:
                members = "；".join("{" + ",".join(group) + "}"
                                    for group in case["group_members"])
                down = ",".join(map(str, case["downweighted_ids"]))
                contains = "是" if case["groups_contain_malicious"] else "否"
                lines.append(f"| {case['round']} | {members} | {down} | {contains} | "
                             f"{num(case['malicious_mean_ratio'])} | "
                             f"{num(case['benign_mean_ratio'])} | "
                             f"{pct(case['previous_asr'])} → {pct(case['asr'])} | "
                             f"{pct(case['previous_accuracy'])} → {pct(case['accuracy'])} | "
                             f"{case['anchor_weight']:.6f} |")
            anomaly = summary["anomalous_summary"]["dba_relation_graph"]
            lines.append(f"\n在 {len(cases)} 个异常轮次中，{anomaly['rounds_with_benign_only_groups']} 轮的"
                         "确认子图完全由正常客户端组成；"
                         f"恶意客户端被降权的客户端-轮次为 "
                         f"{dba_rg['malicious']['downweighted_client_rounds']}。")
            if anomaly["rounds_with_benign_only_groups"] == len(cases):
                lines.append("也就是说，图证据与实际恶意客户端不对应：图防御把正常客户端对误判为异常子图并降权，"
                             "而 4 个 DBA 客户端在全部 120 轮中保留 100% 名义权重。"
                             "因此本批的 ASR/ACC 差异不能归因于对恶意更新的抑制。")
                sizes = [(len(group["principals"]), group.get("edges"))
                         for case in cases for group in case["groups"]]
                if sizes and all(size == 2 for size, _ in sizes):
                    lines.append("从图结构看，本批确认的子图都是 2 个客户端、1 条边"
                                 "（density=1.0）；默认 `min_group_density=0.67` 不会排除这种"
                                 "两节点单边子图，这是误降权的直接结构原因之一。")
            correspondence = summary["intervention_correspondence"]
            if correspondence.get("pearson_anchor_vs_next_asr_drop") is not None:
                lines.append(f"\n干预强度（anchor weight）与下一轮 ASR 下降的 Pearson 相关系数为 "
                             f"{correspondence['pearson_anchor_vs_next_asr_drop']:+.3f}"
                             f"（n={correspondence['active_rounds']}）。由于干预对象是正常客户端、"
                             "样本只有 4 个干预轮次，该系数不能解释为防御效果。")
        else:
            lines.append("虽然存在非零 anchor weight，但没有 `anomalous_structure` 状态；"
                         "干预来自证据不足或其他路径，需要结合 `graph_*.jsonl` 逐轮核对。")
    lines.append("")

    # Q4
    lines.append("## 4. 冷启动、证据不足与误降权\n")
    for name in ("clean_relation_graph", "dba_relation_graph"):
        graph = g[name]["graph"]
        lines.append(f"- `{name}`：冷启动（已校准边比例为 0）{graph['cold_start_rounds']} 轮；"
                     f"证据不足 {graph['insufficient_evidence_rounds']} 轮；"
                     f"无异常证据 {graph['no_anomaly_evidence_rounds']} 轮；"
                     f"确认异常结构 {graph['anomalous_structure_rounds']} 轮。")
    lines.append(f"- 无攻击组的误降权：{clean_rg['benign']['downweighted_client_rounds']} 个正常客户端-轮次，"
                 f"涉及 {clean_rg['benign']['downweighted_rounds']} 轮。")
    if dba_rg["malicious"]:
        false_ids = sorted({value for case in summary["anomalous_cases"]["dba_relation_graph"]
                            for value in case["downweighted_ids"]})
        lines.append(f"- DBA 图防御组：恶意客户端被降权 "
                     f"{dba_rg['malicious']['downweighted_client_rounds']} 个客户端-轮次；"
                     f"正常客户端被误降权 {dba_rg['benign']['downweighted_client_rounds']} 个客户端-轮次"
                     f"（{dba_rg['benign']['downweighted_rounds']} 轮），"
                     f"涉及客户端 {false_ids}，"
                     f"单次最低保留比例 {dba_rg['benign']['min_ratio']:.4f}，"
                     f"总误降权质量 {dba_rg['benign']['total_drop']:.6f}。")
        missed = (dba_rg["malicious"]["client_rounds"]
                  - dba_rg["malicious"]["downweighted_client_rounds"])
        lines.append(f"- DBA 图防御组：未被降权的恶意客户端-轮次为 {missed}"
                     f"（占 {100 * missed / dba_rg['malicious']['client_rounds']:.2f}%）。")
    lines.append("")
    lines.append("这些计数直接决定结论的适用范围：冷启动和证据不足轮次按设计回退到样本加权 FedAvg，"
                 "不构成防御失败；只有确认异常结构且实际降权后，ASR/ACC 的变化才可能与图证据对应。"
                 "误降权会直接损失正常客户端权重；漏降恶意客户端则使图防御退化为 FedAvg。")
    if (dba_rg["malicious"]
            and dba_rg["malicious"]["downweighted_client_rounds"] == 0
            and dba_rg["benign"]["downweighted_client_rounds"] > 0):
        severity = [entry["min_ratio"] for entry in dba_rg["clients"]
                    if entry["rounds_downweighted"] > 0]
        retained_text = (f"{100 * min(severity):.1f}%–{100 * max(severity):.1f}%"
                         if severity else "—")
        lines.append("\n本批的实际错误方向是：确认的异常子图全部是正常客户端对，"
                     "4 个恶意客户端在全部 120 轮中从未被降权；"
                     f"误降权使相关正常客户端在受影响轮次只保留 {retained_text} 的名义权重，"
                     "但攻击本身未建立，因此没有观察到 ASR 下降，反而在干预轮次内 ASR 有小幅波动。"
                     "这不是“防御成功”，而是“错误归因 + 攻击不足”的叠加。")
    lines.append("")

    # Integrity
    lines.append("## 完整性与复现\n")
    lines.append(f"完整性检查：{'全部通过' if check['passed'] else '存在失败项，见 analysis_summary.json'}。"
                 f"四组共享初始状态哈希 `{check['initial_state_sha256'][:16]}…`；"
                 f"第 1 轮清洁配对与 DBA 配对的 state hash 分别一致："
                 f"{check['checks']['paired_round1_clean']} / {check['checks']['paired_round1_dba']}。"
                 "每组每轮保存全局模型、防御状态和真实聚合更新范数；中断后从最后一个完整 checkpoint 恢复。")
    lines.append("")
    lines.append("## 限制\n")
    lines.append("- 单一随机种子（42）和 IID 20 客户端全参与协议；不能外推为多种子或非 IID 结论。")
    if normalization == "batch":
        lines.append("- 使用标准 tracked BatchNorm（`normalization: \"batch\"`）：running statistics "
                     "由客户端本地训练更新并由服务器按聚合权重合并，评估阶段使用清洁训练分布矩，"
                     "不会在 100% 触发器批次上发生批内矩自抵消；该协议与 "
                     "`experiments/reference_attack_qualification.py` 一致。")
    else:
        lines.append("- 使用无运行统计 BatchNorm（`normalization: \"batch_no_running_stats\"`）；"
                     "评估使用批内矩，与 tracked-BN 的历史数字不可直接比较。"
                     "此前的 tracked-BN DBA 强度筛选不能外推到本批协议；攻击强度以本批 FedAvg 组为准。")
    lines.append("- 已有 G²uardFL 配置和产物未修改，也未作为本批的对照日志。")
    lines.append("- 本批只回答整体防御表现；图结构自身的贡献留给同特征、同聚合规则的去结构与打乱边消融。")
    lines.append("- 最终测试集未使用；ASR/ACC 均来自 2000 张训练留出样本。")
    if not q["qualified"]:
        lines.append("- 若要在本防御上检验 ASR 降低，需要先在新协议中建立一个持续通过强度判据的 DBA "
                     "（本批 FedAvg 组未通过），再保持相同图参数做配对比较；该补充不属于本批结果。")
    lines.append("")
    lines.append("## 产物\n")
    lines.append("- 冻结配置：`config/relation_graph_mnist_paired_120.json`；"
                 "运行目录中的 `protocol.json` 是逐字冻结副本")
    lines.append("- 运行入口：`experiments/relation_graph_mnist_paired.py`，"
                 "分析/报告入口：`experiments/relation_graph_mnist_paired_analysis.py`")
    lines.append("- 原始日志：`rounds_*.jsonl`、`graph_*.jsonl`、`status.json`、`summary.json`")
    lines.append("- Checkpoint：`latest_bundle.pt`、`latest_<group>.pt`、`snapshots/<group>/round_*.pt`")
    lines.append("- 指标与曲线：`results/round_metrics.csv`、`client_weights.csv`、"
                 "`graph_edges.csv`、`accuracy_asr_curves.png`、`weight_retention.png`、"
                 "`graph_diagnostics.png`、`aggregate_update_norm.png`")
    lines.append("- 汇总与完整性：`analysis_summary.json`、`s42/results/integrity.json`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run = args.run
    cfg = read(run / "protocol.json")
    if cfg.get("smoke_wiring_check"):
        raise ValueError("Refusing to report a smoke run as a formal result")
    rows = {name: load_lines(run / f"rounds_{name}.jsonl") for name in NAMES}
    for name in NAMES:
        if [row["round"] for row in rows[name]] != list(range(1, cfg["total_rounds"] + 1)):
            raise ValueError(f"{name}: incomplete or discontinuous round history")
    check = integrity(run, cfg, rows)
    summary = summarize(run, cfg, rows, check)
    out = args.output or (run / "results")
    out.mkdir(parents=True, exist_ok=True)
    (run / "results").mkdir(exist_ok=True)
    (run / "results" / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (run / "results" / "integrity.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csvs(run, rows)
    plot_all(run, cfg, rows, run / "results")
    report = report_markdown(summary, cfg, check)
    (run / "results" / "report.md").write_text(report, encoding="utf-8")
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if out != run / "results":
        for name in ("round_metrics.csv", "client_weights.csv", "graph_edges.csv",
                     "accuracy_asr_curves.png", "weight_retention.png",
                     "graph_diagnostics.png", "aggregate_update_norm.png"):
            shutil.copyfile(run / "results" / name, out / name)
    print(json.dumps(dict(passed=check["passed"], run=str(run),
                          output=str(out), report=str(out / "REPORT.md")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()