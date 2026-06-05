"""
utils/metrics.py
----------------
Metric collection, aggregation, and CSV/JSON export utilities.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricTracker:
    """
    Lightweight per-round metric store.

    Usage
    -----
    tracker = MetricTracker(log_dir="logs/", experiment_name="exp_01")
    tracker.log(round=1, split="server", loss=0.45, accuracy=0.82)
    tracker.log(round=1, split="client_avg", train_loss=0.51)
    tracker.save()               # writes CSV + JSON
    tracker.best("accuracy")     # -> {"round": 1, "accuracy": 0.82, ...}
    """

    def __init__(self, log_dir: str = "logs/", experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self._records: List[Dict[str, Any]] = []

    def log(self, round: int, split: str = "server", **kwargs: Any) -> None:
        record = {"round": round, "split": split, **kwargs}
        self._records.append(record)
        logger.debug("Metric | %s", record)

    def best(self, metric: str, split: str = "server", higher_is_better: bool = True) -> Optional[Dict]:
        candidates = [
            r for r in self._records
            if r.get("split") == split and r.get(metric) is not None
        ]
        if not candidates:
            return None
        fn = max if higher_is_better else min
        return fn(candidates, key=lambda r: r[metric])

    def to_list(self) -> List[Dict]:
        return list(self._records)

    def save(self) -> None:
        if not self._records:
            return
        base = self.log_dir / self.experiment_name

        # JSON
        json_path = base.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(self._records, f, indent=2, default=str)

        # CSV
        csv_path = base.with_suffix(".csv")
        all_keys: List[str] = list(
            dict.fromkeys(k for r in self._records for k in r)
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._records)

        logger.info("Metrics saved → %s, %s", json_path, csv_path)

    def print_summary(self) -> None:
        server_records = [r for r in self._records if r.get("split") == "server"]
        if not server_records:
            print("No server metrics recorded.")
            return
        print(f"\n{'─'*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'─'*60}")
        print(f"{'Round':>6}  {'Loss':>8}  {'Accuracy':>10}")
        for r in server_records[-10:]:
            loss = r.get("loss")
            accuracy = r.get("accuracy")
            loss = float("nan") if loss is None else loss
            accuracy = float("nan") if accuracy is None else accuracy
            print(f"{r.get('round','?'):>6}  "
                  f"{loss:>8.4f}  "
                  f"{accuracy:>10.4f}")
        best = self.best("accuracy")
        if best and best.get("accuracy") is not None:
            print(f"\nBest accuracy: {best.get('accuracy', 0):.4f} @ round {best.get('round')}")
        print(f"{'─'*60}\n")
