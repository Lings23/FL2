"""
main.py
-------
Main entry point for federated security experiments.

Runs Flower simulation using the VirtualClientEngine
(no actual network — all clients run in the same process).

Usage
-----
# Default config
python main.py

# Override specific keys
python main.py --config config/config.yaml \\
    --override federation.num_rounds=20 \\
    --override security.attack.enabled=true \\
    --override security.attack.type=label_flip \\
    --override security.defense.enabled=true \\
    --override security.defense.type=krum

# Sweep example (see experiments/ for full sweep scripts)
python main.py --experiment backdoor_vs_fltrust
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import flwr as fl

from config.config_loader import Config, load_config, override_config
from data.dataset import get_dataset, FederatedPartitioner
from models.model_factory import get_model
from client.fl_client import make_client_fn
from server.fl_server import build_server
from utils.logger import setup_logging
from utils.metrics import MetricTracker

logger = logging.getLogger(__name__)


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Setup helpers ─────────────────────────────────────────────────────────────

def build_data_pipeline(cfg: Config):
    """Returns partitioner, per-client loaders, and test loader."""
    dataset = get_dataset(cfg.dataset.name, cfg.dataset.data_dir)
    train_ds = dataset.load_train()

    partitioner = FederatedPartitioner(
        dataset=train_ds,
        num_clients=cfg.federation.num_clients,
        strategy=cfg.dataset.partition,
        dirichlet_alpha=cfg.dataset.dirichlet_alpha,
        seed=cfg.project.seed,
        val_split=cfg.dataset.val_split,
    )

    # Log partition summary
    logger.info("\n%s", partitioner.summary())

    loaders_map: Dict[int, Tuple[DataLoader, DataLoader]] = {}
    for cid in range(cfg.federation.num_clients):
        train_sub, val_sub = partitioner.get_client_data(cid)
        loaders_map[cid] = (
            DataLoader(train_sub, batch_size=cfg.client.batch_size,
                       shuffle=True, num_workers=0, pin_memory=False),
            DataLoader(val_sub, batch_size=cfg.client.batch_size,
                       shuffle=False, num_workers=0),
        )

    test_loader = dataset.get_test_loader(batch_size=128)
    return loaders_map, test_loader, dataset


def build_model_pool(cfg: Config) -> Dict[int, torch.nn.Module]:
    """Create one model instance per client (required for simulation)."""
    return {
        cid: get_model(
            architecture=cfg.model.architecture,
            num_classes=cfg.dataset.num_classes,
            pretrained=cfg.model.pretrained,
            dataset_name=cfg.dataset.name,
        )
        for cid in range(cfg.federation.num_clients)
    }


def determine_malicious_ids(cfg: Config) -> set:
    """Randomly select malicious client IDs based on configured fraction."""
    if not (cfg.security.attack.enabled and cfg.security.attack.type != "none"):
        return set()
    n_mal = max(1, int(cfg.federation.num_clients * cfg.security.attack.malicious_fraction))
    rng = np.random.default_rng(cfg.project.seed)
    mal_ids = set(rng.choice(cfg.federation.num_clients, n_mal, replace=False).tolist())
    logger.info("Malicious clients (%d/%d): %s", n_mal, cfg.federation.num_clients, mal_ids)
    return mal_ids


# ── Main ──────────────────────────────────────────────────────────────────────

def run_simulation(cfg: Config, experiment_name: str = "experiment") -> MetricTracker:
    set_seed(cfg.project.seed)
    setup_logging(cfg.project.log_dir, cfg.project.log_level, name=experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders_map, test_loader, dataset_obj = build_data_pipeline(cfg)

    # ── Models ────────────────────────────────────────────────────────────────
    models_map = build_model_pool(cfg)

    # Global model (for server-side evaluation)
    global_model = get_model(
        architecture=cfg.model.architecture,
        num_classes=cfg.dataset.num_classes,
        pretrained=cfg.model.pretrained,
        dataset_name=cfg.dataset.name,
    )

    # ── Security setup ────────────────────────────────────────────────────────
    malicious_ids = determine_malicious_ids(cfg)

    # ── Client factory ────────────────────────────────────────────────────────
    client_fn = make_client_fn(
        models_map=models_map,
        loaders_map=loaders_map,
        client_cfg=cfg.client,
        attack_cfg=cfg.security.attack if cfg.security.attack.enabled else None,
        dp_cfg=cfg.differential_privacy if cfg.differential_privacy.enabled else None,
        malicious_ids=malicious_ids,
        device=device,
    )

    # ── Server ────────────────────────────────────────────────────────────────
    server, server_config = build_server(cfg, global_model, test_loader, device)

    # ── Metric tracker ────────────────────────────────────────────────────────
    tracker = MetricTracker(log_dir=cfg.project.log_dir, experiment_name=experiment_name)

    # Monkey-patch strategy to capture metrics each round
    orig_aggregate_evaluate = server.strategy.aggregate_evaluate  # type: ignore

    def _patched_aggregate_evaluate(server_round, results, failures):
        loss, metrics = orig_aggregate_evaluate(server_round, results, failures)
        if loss is not None:
            # Avoid duplicate 'loss' key if already in metrics
            safe_metrics = {k: v for k, v in (metrics or {}).items() if k != "loss"}
            tracker.log(round=server_round, split="server",
                        loss=loss,
                        accuracy=safe_metrics.get("accuracy", safe_metrics.get("val_accuracy")),
                        **safe_metrics)
        return loss, metrics

    server.strategy.aggregate_evaluate = _patched_aggregate_evaluate  # type: ignore

    logger.info("=" * 60)
    logger.info("Experiment: %s", experiment_name)
    logger.info("Dataset: %s | Model: %s | Strategy: %s | Defense: %s",
                cfg.dataset.name, cfg.model.architecture,
                cfg.strategy.name, cfg.security.defense.type)
    logger.info("Attack: %s (enabled=%s)", cfg.security.attack.type,
                cfg.security.attack.enabled)
    logger.info("=" * 60)

    # ── Client resources for Flower simulation ────────────────────────────────
    # num_gpus=0.0: virtual client actors run on CPU. The actual training
    # device (self.device inside FedSecClient) is set separately and can
    # still be CUDA; this only controls Ray actor resource reservations.
    # Setting num_gpus > 0 here multiplies by the number of concurrent
    # actors, quickly exceeding the single GPU available on Colab and
    # causing Ray to stall waiting for resources that never free up.
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    # ── Run simulation ────────────────────────────────────────────────────────
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.federation.num_clients,
        config=server_config,
        strategy=server.strategy,
        client_resources=client_resources,
        ray_init_args={"ignore_reinit_error": True, "log_to_driver": False},
    )

    tracker.save()
    tracker.print_summary()
    return tracker


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FedSec — Federated Security Framework")
    p.add_argument("--config", default="config/config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--override", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Override config values, e.g. federation.num_rounds=20")
    p.add_argument("--experiment", default=None,
                   help="Experiment name (used for log/checkpoint naming)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    overrides = {}
    for ov in args.override:
        k, _, v = ov.partition("=")
        # Attempt type coercion
        try:
            v_typed = int(v)
        except ValueError:
            try:
                v_typed = float(v)
            except ValueError:
                v_typed = True if v.lower() == "true" else (
                    False if v.lower() == "false" else v)
        overrides[k] = v_typed

    if overrides:
        cfg = override_config(cfg, overrides)
        logger.info("Config overrides applied: %s", overrides)

    experiment_name = args.experiment or (
        f"{cfg.dataset.name}_{cfg.model.architecture}_"
        f"{cfg.strategy.name}_{cfg.security.defense.type}_"
        f"atk-{cfg.security.attack.type}"
    )

    run_simulation(cfg, experiment_name=experiment_name)


if __name__ == "__main__":
    main()
