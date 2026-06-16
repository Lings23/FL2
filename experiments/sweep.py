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
import copy
import itertools
import json
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
]

DEFAULT_DEFENSES = [
    "none",
    "krum",
    "trimmed_mean",
    "median",
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
) -> pd.DataFrame:
    """
    Run all attack × defense combinations and return a results DataFrame.

    Columns: attack, defense, best_accuracy, final_accuracy,
             best_round, total_rounds
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []

    combinations = list(itertools.product(attacks, defenses))
    logger.info("Sweep: %d experiments (%d attacks × %d defenses)",
                len(combinations), len(attacks), len(defenses))

    for idx, (attack, defense) in enumerate(combinations, start=1):
        exp_name = f"sweep_{attack}_vs_{defense}"
        logger.info("[%d/%d] Running: attack=%s | defense=%s",
                    idx, len(combinations), attack, defense)

        cfg = load_config(config_path)

        # Apply sweep overrides
        overrides: Dict[str, Any] = {
            "federation.num_rounds": num_rounds,
            "security.attack.type":    attack,
            "security.attack.enabled": attack != "none",
            "security.defense.type":   defense,
            "security.defense.enabled": defense != "none",
            "project.log_dir": str(output_path),
        }
        if extra_overrides:
            overrides.update(extra_overrides)
        cfg = override_config(cfg, overrides)

        try:
            tracker = run_simulation(cfg, experiment_name=exp_name)
            best = tracker.best("val_accuracy") or tracker.best("accuracy")
            final = tracker.to_list()[-1] if tracker.to_list() else {}
            rows.append({
                "attack":           attack,
                "defense":          defense,
                "best_accuracy":    best.get("val_accuracy", best.get("accuracy", None)) if best else None,
                "best_round":       best.get("round") if best else None,
                "final_accuracy":   final.get("val_accuracy", final.get("accuracy", None)),
                "total_rounds":     num_rounds,
                "status":           "ok",
            })
        except Exception as exc:
            logger.exception("Experiment failed: %s", exp_name)
            rows.append({
                "attack": attack, "defense": defense,
                "status": f"error: {exc}",
            })

    df = pd.DataFrame(rows)

    # Save
    csv_out = output_path / "sweep_results.csv"
    json_out = output_path / "sweep_results.json"
    df.to_csv(csv_out, index=False)
    df.to_json(json_out, orient="records", indent=2)

    logger.info("\nSweep complete. Results saved → %s", csv_out)
    print_sweep_table(df)
    return df


def print_sweep_table(df: pd.DataFrame) -> None:
    """Pretty-print accuracy table: rows=defenses, cols=attacks."""
    try:
        pivot = df.pivot_table(
            values="best_accuracy",
            index="defense",
            columns="attack",
            aggfunc="max",
        )
        print("\n" + "=" * 70)
        print("SWEEP RESULTS — Best Accuracy (defense × attack)")
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    attacks  = args.attacks  or (["none", "label_flip"] if args.quick else DEFAULT_ATTACKS)
    defenses = args.defenses or (["none", "krum"]       if args.quick else DEFAULT_DEFENSES)
    rounds   = 5 if args.quick else args.rounds

    run_sweep(
        config_path=args.config,
        attacks=attacks,
        defenses=defenses,
        num_rounds=rounds,
        output_dir=args.output,
    )
