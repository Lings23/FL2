"""Read-only analysis and report generation for the multi-attack paired pilot.

Never invokes training. Refuses a run whose status, round logs, checkpoints or
defense histories are inconsistent. Produces per-attack four-curve reports for
BadNet, Neurotoxin and PGD under FedAvg and the relation-graph group-zero
defense.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.relation_graph_mnist_paired import (
    ATTACK_TYPES, CRITICAL_SOURCES, group_name, group_names, source_hashes,
    state_hash,
)

COLORS = {
    "clean_fedavg": "#2a78d6",
    "clean_relation_graph": "#eb6834",
    "badnet_fedavg": "#1baf7a",
    "badnet_relation_graph": "#eda100",
    "neurotoxin_fedavg": "#7b5aa6",
    "neurotoxin_relation_graph": "#d64f8c",
    "pgd_fedavg": "#4f8f2f",
    "pgd_relation_graph": "#8a6d3b",
}
ATTACK_LABEL = {"badnet": "BadNet", "neurotoxin": "Neurotoxin", "pgd": "PGD"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def pct(value, digits=2):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def num(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def longest_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def integrity(run: Path, cfg: dict, rows: dict[str, list[dict]]) -> dict:
    budget = cfg["total_rounds"]
    names = group_names(cfg)
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
    for name in names:
        records = rows[name]
        checks[f"{name}_rounds"] = [row["round"] for row in records] == list(range(1, budget + 1))
        checks[f"{name}_clean_count"] = all(
            row["clean"]["count"] == cfg["evaluator_validation_size"] for row in records)
        checks[f"{name}_asr_denominator"] = all(
            row["asr"]["asr_valid"] and row["asr"]["asr_total"] == 1800 for row in records)
        checks[f"{name}_final_state"] = (
            state_hash(bundle["states"][name]) == records[-1]["state_sha256"])
        if cfg["groups"][names.index(name)]["defense"] == "relation_graph":
            defense = bundle["defenses"][name]
            history = defense["history"]
            checks[f"{name}_defense_round"] = defense["last_round"] == budget
            checks[f"{name}_defense_history"] = (
                [frame["round_number"] for frame in history]
                == sorted({frame["round_number"] for frame in history})
                and (not history or history[-1]["round_number"] == budget))
            from defenses.relation_graph.defense import RelationGraphDefense
            from config.config_loader import DefenseConfig
            restored = RelationGraphDefense(
                DefenseConfig(enabled=True, type="relation_graph",
                              custom_params={"settings": cfg["relation_graph"]["settings"]}),
                num_clients=cfg["clients"])
            restored.load_state_dict(json.loads(json.dumps(defense, allow_nan=False)))
            checks[f"{name}_defense_roundtrip"] = restored.state_dict() == defense
        else:
            checks[f"{name}_stateless"] = bundle["defenses"][name] is None
    clean_fed = group_name(cfg, "none", "fedavg")
    clean_rg = group_name(cfg, "none", "relation_graph")
    checks["paired_round1_clean"] = (rows[clean_fed][0]["state_sha256"]
                                     == rows[clean_rg][0]["state_sha256"])
    for attack in ATTACK_TYPES:
        fed = group_name(cfg, attack, "fedavg")
        rg = group_name(cfg, attack, "relation_graph")
        checks[f"paired_round1_{attack}"] = (
            rows[fed][0]["state_sha256"] == rows[rg][0]["state_sha256"])
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
                                    for name in names})


def group_summary(records, malicious_ids):
    malicious_ids = set(malicious_ids)
    recent = records[-10:]
    benign = [client for row in records for client in row["clients"]
              if not client["is_malicious_evaluation_only"]]
    malicious = [client for row in records for client in row["clients"]
                 if client["is_malicious_evaluation_only"]]
    graph_rows = [dict(round=row["round"], **row["graph"])
                  for row in records if row["graph"] is not None]
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
        windows={f"{lo}-{hi}": dict(
            asr=mean(row["asr"]["asr"] for row in records[lo - 1:hi]),
            accuracy=mean(row["clean"]["accuracy"] for row in records[lo - 1:hi]))
            for lo, hi in ((1, 30), (31, 60), (61, 90), (91, 120))},
        benign=dict(
            client_rounds=len(benign),
            downweighted_client_rounds=sum(c["weight_drop"] > 1e-12 for c in benign),
            downweighted_rounds=sum(any(c["weight_drop"] > 1e-12 for c in row["clients"]
                                       if not c["is_malicious_evaluation_only"])
                                    for row in records),
            mean_ratio=mean(c["weight_ratio"] for c in benign),
            min_ratio=min((c["weight_ratio"] for c in benign), default=None),
            total_drop=float(sum(c["weight_drop"] for c in benign))),
        malicious=None if not malicious else dict(
            client_rounds=len(malicious),
            active_client_rounds=sum(bool(c["train"].get("attack_active")) for c in malicious),
            downweighted_client_rounds=sum(c["weight_drop"] > 1e-12 for c in malicious),
            downweighted_rounds=sum(any(c["weight_drop"] > 1e-12 for c in row["clients"]
                                       if c["is_malicious_evaluation_only"])
                                    for row in records),
            zero_weight_client_rounds=sum(c["aggregation_weight"] <= 0.0 for c in malicious),
            mean_ratio=mean(c["weight_ratio"] for c in malicious),
            min_ratio=min((c["weight_ratio"] for c in malicious), default=None),
            total_drop=float(sum(c["weight_drop"] for c in malicious))),
        zero_weight_clients=sorted({
            c["client_id"] for row in records for c in row["clients"]
            if c["aggregation_weight"] <= 0.0}),
        max_consecutive_zero_weight=max(
            (longest_run([row["clients"][index]["aggregation_weight"] <= 0.0
                          for row in records])
             for index in range(len(records[0]["clients"]))), default=0),
        median_client_seconds=float(np.median([row["client_seconds"] for row in records])),
        median_aggregation_seconds=float(np.median([row["aggregation_seconds"] for row in records])),
        median_evaluation_seconds=float(np.median([row["evaluation_seconds"] for row in records])),
        median_round_seconds=float(np.median([row["round_seconds"] for row in records])),
    )
    if graph_rows:
        states = [row["state"] for row in graph_rows]
        result["graph"] = dict(
            cold_start_rounds=sum(row["calibrated_fraction"] == 0.0 for row in graph_rows),
            insufficient_evidence_rounds=sum(s == "insufficient_evidence" for s in states),
            no_anomaly_evidence_rounds=sum(s == "no_anomaly_evidence" for s in states),
            anomalous_structure_rounds=sum(s == "anomalous_structure" for s in states),
            confirmed_group_rounds=sum(row["confirmed_groups"] > 0 for row in graph_rows),
            intervention_rounds=sum(row["anchor_weight"] > 1e-12 for row in graph_rows),
            first_intervention_round=next(
                (row["round"] for row in graph_rows if row["anchor_weight"] > 1e-12), None),
            downweighted_client_rounds=sum(row["downweighted_clients"] for row in graph_rows),
            last10_mean_calibrated_fraction=mean(
                row["calibrated_fraction"] for row in graph_rows[-10:]),
            last10_mean_anchor_weight=mean(row["anchor_weight"] for row in graph_rows[-10:]),
        )
    return result


def qualification(cfg, baseline, attacked):
    q = cfg["qualification"]
    base = {row["round"]: row for row in baseline}
    checks = []
    for row in attacked[-q["sustained_rounds"]:]:
        reference = base.get(row["round"])
        if reference is None:
            continue
        checks.append(dict(
            round=row["round"],
            clean_model_ready=reference["clean"]["accuracy"] >= q["minimum_clean_accuracy"],
            task_preserved=(row["clean"]["accuracy"] >=
                            reference["clean"]["accuracy"] - q["maximum_accuracy_drop"]),
            attack_strong=row["asr"]["asr"] >= q["minimum_asr"],
            exceeds_clean_trigger_response=(row["asr"]["asr"] - reference["asr"]["asr"]
                                            >= q["minimum_asr_excess"]),
        ))
    passed = (len(checks) == q["sustained_rounds"] and
              all(all(value for key, value in check.items() if key != "round")
                  for check in checks))
    return dict(qualified=passed, checks=checks, criteria=q)


def summarize(run: Path, cfg: dict, rows: dict[str, list[dict]], check: dict) -> dict:
    names = group_names(cfg)
    malicious = set(cfg["malicious_clients"])
    attack_by_name = {group["name"]: group["attack"] for group in cfg["groups"]}
    summaries = {name: group_summary(rows[name],
                                     malicious if attack_by_name[name] != "none" else set())
                 for name in names}
    clean_fed = group_name(cfg, "none", "fedavg")
    clean_rg = group_name(cfg, "none", "relation_graph")
    attacks = {}
    for attack in ATTACK_TYPES:
        fed_name = group_name(cfg, attack, "fedavg")
        rg_name = group_name(cfg, attack, "relation_graph")
        fed, rg = summaries[fed_name], summaries[rg_name]
        attacks[attack] = dict(
            fedavg=fed_name, relation_graph=rg_name,
            final_asr_delta=rg["final"]["asr"] - fed["final"]["asr"],
            last10_asr_delta=rg["last10"]["asr"] - fed["last10"]["asr"],
            final_accuracy_delta=(rg["final"]["accuracy"] - fed["final"]["accuracy"]),
            last10_accuracy_delta=(rg["last10"]["accuracy"] - fed["last10"]["accuracy"]),
            attack_strength=qualification(cfg, rows[clean_fed], rows[fed_name]),
            defended_attack_strength=qualification(cfg, rows[clean_rg], rows[rg_name]),
        )
        mal_tasks, ben_tasks, mal_backbones, ben_backbones = [], [], [], []
        max_malicious_score = None
        for detail in load_lines(run / f"graph_{rg_name}.jsonl"):
            for edge in detail["edges"]:
                malicious_edge = (edge["left"] in {"0", "1", "2", "3"}
                                  and edge["right"] in {"0", "1", "2", "3"})
                benign_edge = (edge["left"] not in {"0", "1", "2", "3"}
                               and edge["right"] not in {"0", "1", "2", "3"})
                if malicious_edge:
                    mal_tasks.append(edge["task"])
                    mal_backbones.append(edge["backbone"])
                    if edge["score"] is not None:
                        max_malicious_score = (
                            edge["score"] if max_malicious_score is None
                            else max(max_malicious_score, edge["score"]))
                elif benign_edge:
                    ben_tasks.append(edge["task"])
                    ben_backbones.append(edge["backbone"])
        attacks[attack]["detection_diagnostics"] = dict(
            malicious_task=float(np.median(mal_tasks)) if mal_tasks else None,
            benign_task=float(np.median(ben_tasks)) if ben_tasks else None,
            malicious_backbone=float(np.median(mal_backbones)) if mal_backbones else None,
            benign_backbone=float(np.median(ben_backbones)) if ben_backbones else None,
            max_malicious_score=max_malicious_score,
        )
        if attack == "pgd":
            mal_metrics = [client["train"] for row in rows[fed_name]
                           for client in row["clients"]
                           if client["is_malicious_evaluation_only"]]
            radii = [m.get("pgd_radius") for m in mal_metrics]
            attacks[attack]["pgd_execution"] = dict(
                first_radius=next((r for r in radii if r is not None and r > 0), None),
                last_radius=next((r for r in reversed(radii)
                                  if r is not None and r > 0), None),
                max_projection_count=max(
                    (m.get("pgd_projection_count") or 0 for m in mal_metrics), default=0),
                round1_reference_available=bool(
                    mal_metrics[0].get("pgd_reference_available")) if mal_metrics else None,
            )
    result = dict(
        run=run.as_posix(), seed=cfg["seed"], budget_rounds=cfg["total_rounds"],
        completed_rounds=min(len(rows[name]) for name in names),
        normalization=cfg.get("normalization"),
        defense_variant=cfg.get("defense_variant"),
        groups=summaries, attacks=attacks, integrity=check,
        clean_delta=dict(
            final_accuracy=summaries[clean_rg]["final"]["accuracy"]
                          - summaries[clean_fed]["final"]["accuracy"],
            last10_accuracy=summaries[clean_rg]["last10"]["accuracy"]
                           - summaries[clean_fed]["last10"]["accuracy"],
            identical_rounds=sum(a["state_sha256"] == b["state_sha256"]
                                 for a, b in zip(rows[clean_fed], rows[clean_rg])),
        ),
        series={name: [dict(round=row["round"],
                            accuracy=row["clean"]["accuracy"],
                            asr=row["asr"]["asr"]) for row in rows[name]]
                for name in names},
    )
    return result


def write_csvs(run: Path, cfg: dict, rows: dict[str, list[dict]]) -> Path:
    out = run / "results"
    out.mkdir(exist_ok=True)
    attack_by_name = {group["name"]: group["attack"] for group in cfg["groups"]}
    defense_by_name = {group["name"]: group["defense"] for group in cfg["groups"]}
    with (out / "round_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ["group", "attack", "defense", "round", "accuracy", "loss", "asr",
                   "graph_state", "calibrated_fraction", "confirmed_groups",
                   "graph_downweighted_clients", "broadcast_model_weight",
                   "aggregate_update_norm", "benign_mean_weight_ratio",
                   "benign_total_drop", "malicious_mean_weight_ratio",
                   "malicious_total_drop", "client_seconds", "aggregation_seconds",
                   "evaluation_seconds", "round_seconds"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in group_names(cfg):
            for row in rows[name]:
                graph = row["graph"] or {}
                writer.writerow(dict(
                    group=name, attack=attack_by_name[name], defense=defense_by_name[name],
                    round=row["round"], accuracy=row["clean"]["accuracy"],
                    loss=row["clean"]["loss"], asr=row["asr"]["asr"],
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
                   "update_norm", "attack_active"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in group_names(cfg):
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
                        attack_active=client["train"].get("attack_active", "")))
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


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _line_panel(ax, x, series, ylabel, title):
    for values, label, color, style in series:
        ax.plot(x, values, color=color, linestyle=style, label=label)
    ax.set_title(title, loc="left", color=INK)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xlim(1, len(x))
    ax.legend(loc="upper left", fontsize=7, ncol=2)


def plot_all(run: Path, cfg: dict, rows: dict[str, list[dict]], out: Path) -> None:
    plt = _plot_style()
    names = group_names(cfg)
    rounds = list(range(1, cfg["total_rounds"] + 1))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    _line_panel(axes[0], rounds, [
        ([row["clean"]["accuracy"] for row in rows[name]], name, COLORS[name], "-")
        for name in names], "clean accuracy", "Clean accuracy (2000-example holdout)")
    _line_panel(axes[1], rounds, [
        ([row["asr"]["asr"] for row in rows[name]], name, COLORS[name],
         "-" if "relation_graph" in name else "--")
        for name in names], "targeted ASR", "Reference-trigger ASR (1800 non-target)")
    _save(fig, out / "accuracy_asr_curves.png")
    for attack in ATTACK_TYPES:
        subset = [group_name(cfg, "none", "fedavg"),
                  group_name(cfg, "none", "relation_graph"),
                  group_name(cfg, attack, "fedavg"),
                  group_name(cfg, attack, "relation_graph")]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        _line_panel(axes[0], rounds, [
            ([row["clean"]["accuracy"] for row in rows[name]], name, COLORS[name], "-")
            for name in subset], "clean accuracy", f"{ATTACK_LABEL[attack]}: clean accuracy")
        _line_panel(axes[1], rounds, [
            ([row["asr"]["asr"] for row in rows[name]], name, COLORS[name],
             "-" if "relation_graph" in name else "--")
            for name in subset], "targeted ASR", f"{ATTACK_LABEL[attack]}: targeted ASR")
        _save(fig, out / f"accuracy_asr_{attack}.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for attack in ATTACK_TYPES:
        name = group_name(cfg, attack, "relation_graph")
        axes[0].plot(rounds, [row["malicious_weight_summary"]["mean_weight_ratio"]
                              for row in rows[name]], color=COLORS[name],
                     label=f"{ATTACK_LABEL[attack]} malicious")
        axes[1].plot(rounds, [row["benign_weight_summary"]["mean_weight_ratio"]
                              for row in rows[name]], color=COLORS[name],
                     label=f"{ATTACK_LABEL[attack]} benign")
        axes[2].plot(rounds, [row["graph"]["confirmed_groups"] for row in rows[name]],
                     color=COLORS[name], label=ATTACK_LABEL[attack])
    for ax, title in zip(axes, ("Malicious mean weight ratio", "Benign mean weight ratio",
                                "Confirmed group size")):
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_xlim(1, cfg["total_rounds"])
        ax.legend(loc="upper right", fontsize=7)
    axes[2].set_xlabel("round")
    _save(fig, out / "weight_and_graph_diagnostics.png")


def report_markdown(summary: dict, cfg: dict, check: dict) -> str:
    g = summary["groups"]
    names = group_names(cfg)
    attacks = summary["attacks"]
    clean_fed = group_name(cfg, "none", "fedavg")
    clean_rg = group_name(cfg, "none", "relation_graph")
    lines = []
    lines.append("# 关系图闭环防御：MNIST 上 BadNet / Neurotoxin / PGD 配对实验（种子 42）\n")
    lines.append(f"运行目录：`{summary['run']}`。八组各完成 "
                 f"{summary['completed_rounds']}/{summary['budget_rounds']} 轮。"
                 "评估使用 2000 张独立训练留出样本，目标 ASR 分母为 1800 张非目标类样本；"
                 "最终测试集未使用。协议为 IID 20 客户端、标准 tracked BatchNorm、"
                 "`group_zero_weight` 闭环 relation-graph 防御。\n")
    lines.append("攻击共享恶意客户端 0–3、目标类 7、50% 投毒与相同初始模型；"
                 "BadNet、Neurotoxin、PGD 均为 AlignIns 参考后门触发器。"
                 "Neurotoxin 使用本地 clean-gradient GradMask（keep ratio 0.25，逐层）；"
                 "PGD 使用 parameter-scope 投影，半径 = 1.0 × 上一轮客户端参数更新范数中位数。"
                 "清洁组的 ASR 是清洁模型对同一参考触发器的误响应。\n")

    lines.append("## 结果总览\n")
    lines.append("| 组 | 最终 ACC | 末 10 轮 ACC | 最终 ASR | 末 10 轮 ASR | 峰值 ASR（轮次） |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in names:
        row = g[name]
        lines.append(f"| {name} | {pct(row['final']['accuracy'])} | {pct(row['last10']['accuracy'])} "
                     f"| {pct(row['final']['asr'])} | {pct(row['last10']['asr'])} "
                     f"| {pct(row['peak']['asr'])}（{row['peak']['asr_round']}） |")
    lines.append("")

    lines.append("## 结论摘要\n")
    for attack in ATTACK_TYPES:
        label = ATTACK_LABEL[attack]
        fed_name = group_name(cfg, attack, "fedavg")
        rg_name = group_name(cfg, attack, "relation_graph")
        fed, rg = g[fed_name], g[rg_name]
        entry = attacks[attack]
        qualified = entry["attack_strength"]["qualified"]
        rg_graph = rg.get("graph") or {}
        confirmed = rg_graph.get("confirmed_group_rounds", 0)
        malicious = rg["malicious"]
        zero = malicious["zero_weight_client_rounds"] if malicious else 0
        client_rounds = malicious["client_rounds"] if malicious else 0
        if not qualified:
            verdict = "攻击基线未达冻结强度门槛，防御差异仅作描述"
        elif confirmed == 0:
            verdict = "图防御未确认任何异常团伙，ASR 未下降"
        elif entry["last10_asr_delta"] <= -0.5:
            verdict = "图防御显著压制后门 ASR"
        else:
            verdict = "图防御仅有限压制"
        lines.append(
            f"- **{label}**：FedAvg 最终 ASR {pct(fed['final']['asr'])}"
            f"（{'通过' if qualified else '未通过'}攻击强度判据）；"
            f"relation_graph 最终 ASR {pct(rg['final']['asr'])}"
            f"（{100 * entry['final_asr_delta']:+.2f} pp）；"
            f"团伙确认 {confirmed}/{summary['budget_rounds']} 轮，"
            f"恶意零权重 {zero}/{client_rounds} 个客户端-轮次，"
            f"正常客户端误降权 {rg['benign']['downweighted_client_rounds']} 个客户端-轮次。"
            f"判定：{verdict}。")
    lines.append("")

    lines.append("## 清洁对照\n")
    delta = summary["clean_delta"]
    lines.append(f"`clean_relation_graph` 相对 `clean_fedavg`：最终 ACC 差 "
                 f"{100 * delta['final_accuracy']:+.2f} 个百分点，末 10 轮 ACC 差 "
                 f"{100 * delta['last10_accuracy']:+.2f} 个百分点；"
                 f"逐轮 state hash 相同的轮数为 {delta['identical_rounds']}/"
                 f"{summary['budget_rounds']}。"
                 f"清洁组正常客户端误降权 {g[clean_rg]['benign']['downweighted_client_rounds']}/"
                 f"{g[clean_rg]['benign']['client_rounds']} 个客户端-轮次。\n")

    key_rounds = sorted({1, summary["budget_rounds"]}
                        | set(range(10, summary["budget_rounds"] + 1, 10)))
    for attack in ATTACK_TYPES:
        label = ATTACK_LABEL[attack]
        fed = group_name(cfg, attack, "fedavg")
        rg = group_name(cfg, attack, "relation_graph")
        entry = attacks[attack]
        lines.append(f"## {label}\n")
        lines.append("| 指标 | FedAvg | relation_graph | 差 / 说明 |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| 最终 ASR | {pct(g[fed]['final']['asr'])} | {pct(g[rg]['final']['asr'])} "
                     f"| {100 * entry['final_asr_delta']:+.2f} pp |")
        lines.append(f"| 末 10 轮 ASR | {pct(g[fed]['last10']['asr'])} | "
                     f"{pct(g[rg]['last10']['asr'])} | {100 * entry['last10_asr_delta']:+.2f} pp |")
        lines.append(f"| 峰值 ASR | {pct(g[fed]['peak']['asr'])}（第 {g[fed]['peak']['asr_round']} 轮） "
                     f"| {pct(g[rg]['peak']['asr'])}（第 {g[rg]['peak']['asr_round']} 轮） | — |")
        lines.append(f"| 最终 ACC | {pct(g[fed]['final']['accuracy'])} | "
                     f"{pct(g[rg]['final']['accuracy'])} | "
                     f"{100 * entry['final_accuracy_delta']:+.2f} pp |")
        lines.append(f"| 末 10 轮 ACC | {pct(g[fed]['last10']['accuracy'])} | "
                     f"{pct(g[rg]['last10']['accuracy'])} | "
                     f"{100 * entry['last10_accuracy_delta']:+.2f} pp |")
        for window in ("1-30", "31-60", "61-90", "91-120"):
            lines.append(f"| {window} 轮平均 ASR | {pct(g[fed]['windows'][window]['asr'])} | "
                         f"{pct(g[rg]['windows'][window]['asr'])} | "
                         f"{100 * (g[rg]['windows'][window]['asr'] - g[fed]['windows'][window]['asr']):+.2f} pp |")
        malicious = g[rg]["malicious"]
        if malicious:
            lines.append(f"| 恶意客户端降权率 | — | "
                         f"{malicious['downweighted_client_rounds']}/{malicious['client_rounds']} "
                         f"= {pct(malicious['downweighted_client_rounds'] / malicious['client_rounds'])} | — |")
        lines.append(f"| 正常客户端误降权率 | — | "
                     f"{g[rg]['benign']['downweighted_client_rounds']}/"
                     f"{g[rg]['benign']['client_rounds']} = "
                     f"{pct(g[rg]['benign']['downweighted_client_rounds'] / g[rg]['benign']['client_rounds'])} | — |")
        if g[rg].get("graph"):
            confirmed = g[rg]["graph"]["confirmed_group_rounds"]
            lines.append(f"| 团伙确认轮次占比 | — | {confirmed}/{summary['budget_rounds']} = "
                         f"{pct(confirmed / summary['budget_rounds'])} | — |")
        lines.append("")
        lines.append("关键轮次轨迹（完整逐轮 ACC/ASR 见 `results/round_metrics.csv` 与曲线图）：\n")
        subset = [clean_fed, clean_rg, fed, rg]
        lines.append("| 轮次 | " + " | ".join(f"{name} ACC | {name} ASR" for name in subset) + " |")
        lines.append("|---:|" + "---:|" * (2 * len(subset)))
        series = {name: {item["round"]: item for item in summary["series"][name]}
                  for name in subset}
        for number in key_rounds:
            cells = []
            for name in subset:
                item = series[name][number]
                cells.extend((pct(item["accuracy"]), pct(item["asr"])))
            lines.append(f"| {number} | " + " | ".join(cells) + " |")
        lines.append("")
        q = entry["attack_strength"]
        if q["qualified"]:
            lines.append(f"{label} 在 FedAvg 下通过冻结的攻击强度判据"
                         "（连续 3 轮：清洁基线 ACC ≥ 95%、攻击组 ACC 不低于基线 − 2 个百分点、"
                         "ASR ≥ 80%、相对清洁触发响应超出 ≥ 60 个百分点）。\n")
        else:
            failed = defaultdict(list)
            for check_row in q["checks"]:
                for key, value in check_row.items():
                    if key != "round" and not value:
                        failed[key].append(check_row["round"])
            detail = "；".join(f"{key} 未通过于轮次 {values}"
                              for key, values in failed.items())
            lines.append(f"{label} 在 FedAvg 下未通过攻击强度判据"
                         f"（{detail or '末 3 轮检查不完整'}）。\n")
        diag = entry.get("detection_diagnostics")
        if diag:
            lines.append(f"检测诊断：恶意客户端对的 task 相似度中位数 "
                         f"{num(diag['malicious_task'], 3)}，正常客户端对 "
                         f"{num(diag['benign_task'], 3)}；backbone 相似度中位数 "
                         f"{num(diag['malicious_backbone'], 3)} vs "
                         f"{num(diag['benign_backbone'], 3)}；"
                         f"恶意边最大 z-score {num(diag['max_malicious_score'], 2)}。\n")
        if entry.get("pgd_execution"):
            pgd = entry["pgd_execution"]
            lines.append(f"PGD 执行：第 1 轮无公开参考；第 2 轮起每轮投影 "
                         f"{int(pgd['max_projection_count'])} 次（每 epoch 一次），"
                         f"半径由 {num(pgd['first_radius'])} 收缩到 "
                         f"{num(pgd['last_radius'])}，恶意更新被裁剪到该半径。\n")

    lines.append("## 完整性与复现\n")
    lines.append(f"完整性检查：{'全部通过' if check['passed'] else '存在失败项，见 results/integrity.json'}。"
                 f"八组共享初始状态哈希 `{check['initial_state_sha256'][:16]}…`；"
                 "清洁配对与各攻击配对的第 1 轮 state hash 均一致。"
                 "每组每轮保存全局模型、防御状态和真实聚合更新范数。\n")
    lines.append("## 限制\n")
    lines.append("- 单一随机种子（42）和 IID 20 客户端全参与协议；不能外推为多种子或非 IID 结论。")
    lines.append("- Neurotoxin 使用本地 clean-gradient GradMask（非服务器 broadcast public-delta 版本）；"
                 "PGD 使用 parameter-scope + benign-update-norm 半径的 tracked-BN 适配，"
                 "不是对源实现的逐行复现。")
    lines.append("- 三种攻击共享同一清洁对照组；清洁 ASR 为清洁模型对同一参考触发器的误响应。")
    lines.append("- 最终测试集未使用；ACC/ASR 均来自 2000 张训练留出样本。")
    lines.append("")
    lines.append("## 产物\n")
    lines.append("- 运行入口：`experiments/relation_graph_mnist_paired.py`；"
                 "分析入口：`experiments/relation_graph_mnist_multi_attack_analysis.py`")
    lines.append("- 原始日志：`rounds_*.jsonl`、`graph_*.jsonl`、`status.json`、`summary.json`")
    lines.append("- 指标与曲线：`results/round_metrics.csv`、`client_weights.csv`、"
                 "`accuracy_asr_curves.png`、`accuracy_asr_<attack>.png`、"
                 "`weight_and_graph_diagnostics.png`")
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
    names = group_names(cfg)
    rows = {name: load_lines(run / f"rounds_{name}.jsonl") for name in names}
    for name in names:
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
    write_csvs(run, cfg, rows)
    plot_all(run, cfg, rows, run / "results")
    report = report_markdown(summary, cfg, check)
    (run / "results" / "report.md").write_text(report, encoding="utf-8")
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if out != run / "results":
        for name in ("accuracy_asr_curves.png",
                     *(f"accuracy_asr_{attack}.png" for attack in ATTACK_TYPES),
                     "weight_and_graph_diagnostics.png", "round_metrics.csv",
                     "client_weights.csv"):
            source = run / "results" / name
            if source.exists():
                import shutil
                shutil.copyfile(source, out / name)
    print(json.dumps(dict(passed=check["passed"], run=str(run),
                          output=str(out), report=str(out / "REPORT.md")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()