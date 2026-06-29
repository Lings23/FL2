"""
experiments/sweep.py
---------------------
Grid-sweep runner: iterate over attack × defense combinations
and collect results into a comparison table.

Usage
-----
python experiments/sweep.py --config config/config.yaml --rounds 20
python experiments/sweep.py --quick          # 5-round smoke test
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config, override_config
from main import run_simulation

logger = logging.getLogger(__name__)


# ── Sweep grid ────────────────────────────────────────────────────────────────

DEFAULT_ATTACKS = [
    "none",
    "label_flip",
    "gaussian_noise",
    "byzantine",
    "backdoor",
    "dba",
    "model_replacement",
]

DEFAULT_DEFENSES = [
    "none",
    "krum",
    "trimmed_mean",
    "median",
    "fltrust",
    "foolsgold",
    "time_consistency",
]


def run_sweep(
    config_path: str,
    attacks: List[str],
    defenses: List[str],
    num_rounds: int = 50,
    output_dir: str = "logs/sweep/",
    extra_overrides: Optional[Dict[str, Any]] = None,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Run all attack × defense combinations and return a results DataFrame.

    Columns: attack, defense, best_accuracy, final_accuracy,
             best_round, total_rounds
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []

    seeds = seeds or [42]
    combinations = list(itertools.product(attacks, defenses, seeds))
    logger.info(
        "Sweep: %d experiments (%d attacks × %d defenses × %d seeds)",
        len(combinations), len(attacks), len(defenses), len(seeds),
    )

    for idx, (attack, defense, seed) in enumerate(combinations, start=1):
        exp_name = f"sweep_{attack}_vs_{defense}_seed{seed}"
        logger.info(
            "[%d/%d] Running: attack=%s | defense=%s | seed=%d",
            idx, len(combinations), attack, defense, seed,
        )

        cfg = load_config(config_path)

        # Apply sweep overrides
        overrides: Dict[str, Any] = {
            "federation.num_rounds": num_rounds,
            "security.attack.type":    attack,
            "security.attack.enabled": attack != "none",
            "security.defense.type":   defense,
            "security.defense.enabled": defense != "none",
            "project.log_dir": str(output_path),
            "project.seed": seed,
        }
        if extra_overrides:
            overrides.update(extra_overrides)
        cfg = override_config(cfg, overrides)

        try:
            tracker = run_simulation(cfg, experiment_name=exp_name)
            best = tracker.best("val_accuracy") or tracker.best("accuracy")
            server_records = [
                record for record in tracker.to_list()
                if record.get("split") == "server"
            ]
            fit_records = [
                record for record in tracker.to_list()
                if record.get("split") == "fit"
            ]
            final = server_records[-1] if server_records else {}
            final_fit = fit_records[-1] if fit_records else {}
            rows.append({
                "attack":           attack,
                "defense":          defense,
                "seed":             seed,
                "best_accuracy":    best.get("val_accuracy", best.get("accuracy", None)) if best else None,
                "best_round":       best.get("round") if best else None,
                "final_accuracy":   final.get("val_accuracy", final.get("accuracy", None)),
                "final_asr":        final.get("asr"),
                "malicious_weight_share": final_fit.get("malicious_aggregation_weight_share"),
                "malicious_impact_share": final_fit.get("malicious_impact_share"),
                "active_attacker_impact_share": final_fit.get("active_attacker_impact_share"),
                "malicious_trust_mean": final_fit.get("malicious_trust_mean"),
                "benign_trust_mean": final_fit.get("benign_trust_mean"),
                "trust_detection_auc": final_fit.get("trust_detection_auc"),
                "clip_precision": final_fit.get("clip_precision"),
                "clip_recall_active_attackers": final_fit.get("clip_recall_active_attackers"),
                "benign_quarantine_rate": final_fit.get("benign_quarantine_rate"),
                "aggregation_time_seconds": final_fit.get("aggregation_time_seconds"),
                "total_rounds":     num_rounds,
                "status":           "ok",
            })
        except Exception as exc:
            logger.exception("Experiment failed: %s", exp_name)
            rows.append({
                "attack": attack, "defense": defense, "seed": seed,
                "status": f"error: {exc}",
            })

    df = pd.DataFrame(rows)

    # Save
    csv_out = output_path / "sweep_results.csv"
    json_out = output_path / "sweep_results.json"
    df.to_csv(csv_out, index=False)
    df.to_json(json_out, orient="records", indent=2)

    successful = df[df["status"] == "ok"].copy()
    numeric_metrics = [
        "best_accuracy",
        "final_accuracy",
        "final_asr",
        "malicious_weight_share",
        "malicious_impact_share",
        "active_attacker_impact_share",
        "trust_detection_auc",
        "clip_precision",
        "clip_recall_active_attackers",
        "benign_quarantine_rate",
        "aggregation_time_seconds",
    ]
    available_metrics = [
        metric for metric in numeric_metrics if metric in successful.columns
    ]
    if not successful.empty and available_metrics:
        for metric in available_metrics:
            successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
        summary = successful.groupby(["attack", "defense"])[available_metrics].agg(
            ["mean", "std", "count"]
        )
        summary.to_csv(output_path / "sweep_summary.csv")

    logger.info("\nSweep complete. Results saved → %s", csv_out)
    print_sweep_table(df)
    return df


def print_sweep_table(df: pd.DataFrame) -> None:
    """Pretty-print mean final accuracy across seeds."""
    try:
        pivot = df.pivot_table(
            values="final_accuracy",
            index="defense",
            columns="attack",
            aggfunc="mean",
        )
        print("\n" + "=" * 70)
        print("SWEEP RESULTS — Mean Final Accuracy (defense × attack)")
        print("=" * 70)
        print(pivot.to_string(float_format="{:.4f}".format))
        print("=" * 70 + "\n")
    except Exception:
        print(df.to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FedSec sweep runner")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--output", default="logs/sweep/")
    p.add_argument("--quick", action="store_true",
                   help="5-round smoke test with minimal grid")
    p.add_argument("--attacks",  nargs="+", default=None)
    p.add_argument("--defenses", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    attacks  = args.attacks  or (["none", "label_flip"] if args.quick else DEFAULT_ATTACKS)
    defenses = args.defenses or (["none", "krum"]       if args.quick else DEFAULT_DEFENSES)
    rounds   = 5 if args.quick else args.rounds
    extra_overrides: Dict[str, Any] = {}
    for item in args.override:
        key, _, raw_value = item.partition("=")
        try:
            value: Any = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                value = True if raw_value.lower() == "true" else (
                    False if raw_value.lower() == "false" else raw_value
                )
        extra_overrides[key] = value

    run_sweep(
        config_path=args.config,
        attacks=attacks,
        defenses=defenses,
        num_rounds=rounds,
        output_dir=args.output,
        extra_overrides=extra_overrides,
        seeds=args.seeds,
    )
