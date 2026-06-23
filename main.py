"""
main.py
-------
Main entry point for federated security experiments.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

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


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def build_data_pipeline(cfg: Config):
    """
    Returns per-client DataLoader pairs and a test DataLoader.

    Performance note: DataLoaders are built here (in the parent process) so
    that dataset partitioning happens once.  However, the loaders_map is
    intentionally NOT closed over inside client_fn -- only the individual
    (train_loader, val_loader) pair for each cid is captured per-actor.
    See make_client_fn() in fl_client.py for how this is enforced.
    """
    dataset = get_dataset(
        cfg.dataset.name,
        cfg.dataset.data_dir,
        download_source=cfg.dataset.download_source,
    )
    train_ds = dataset.load_train()

    partitioner = FederatedPartitioner(
        dataset=train_ds,
        num_clients=cfg.federation.num_clients,
        strategy=cfg.dataset.partition,
        dirichlet_alpha=cfg.dataset.dirichlet_alpha,
        seed=cfg.project.seed,
        val_split=cfg.dataset.val_split,
    )

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


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model_factory(cfg: Config) -> Callable[[], torch.nn.Module]:
    """
    Return a callable that builds one fresh model instance on demand.

    Each Ray actor calls _factory() once after it starts; the model lives
    only inside that actor's address space and is garbage-collected when the
    actor goes idle between rounds.  The parent process holds exactly one
    model: the server-side global model for evaluation and checkpointing.
    """
    def _factory() -> torch.nn.Module:
        return get_model(
            architecture=cfg.model.architecture,
            num_classes=cfg.dataset.num_classes,
            pretrained=cfg.model.pretrained,
            dataset_name=cfg.dataset.name,
        )
    return _factory


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def determine_malicious_ids(cfg: Config) -> set:
    if not (cfg.security.attack.enabled and cfg.security.attack.type != "none"):
        return set()
    n_mal = max(1, int(cfg.federation.num_clients * cfg.security.attack.malicious_fraction))
    rng = np.random.default_rng(cfg.project.seed)
    mal_ids = set(rng.choice(cfg.federation.num_clients, n_mal, replace=False).tolist())
    logger.info("Malicious clients (%d/%d): %s", n_mal, cfg.federation.num_clients, mal_ids)
    return mal_ids


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation(cfg: Config, experiment_name: str = "experiment") -> MetricTracker:
    set_seed(cfg.project.seed)
    setup_logging(cfg.project.log_dir, cfg.project.log_level, name=experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    loaders_map, test_loader, dataset_obj = build_data_pipeline(cfg)
    model_factory = build_model_factory(cfg)

    global_model = get_model(
        architecture=cfg.model.architecture,
        num_classes=cfg.dataset.num_classes,
        pretrained=cfg.model.pretrained,
        dataset_name=cfg.dataset.name,
    )

    malicious_ids = determine_malicious_ids(cfg)

    client_fn = make_client_fn(
        model_factory=model_factory,
        loaders_map=loaders_map,
        client_cfg=cfg.client,
        attack_cfg=cfg.security.attack if cfg.security.attack.enabled else None,
        dp_cfg=cfg.differential_privacy if cfg.differential_privacy.enabled else None,
        malicious_ids=malicious_ids,
        device=device,
    )

    server, server_config = build_server(cfg, global_model, test_loader, device)

    tracker = MetricTracker(log_dir=cfg.project.log_dir, experiment_name=experiment_name)

    orig_evaluate = server.strategy.evaluate  # type: ignore
    orig_aggregate_evaluate = server.strategy.aggregate_evaluate  # type: ignore

    def _patched_evaluate(server_round, parameters):
        result = orig_evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            safe_metrics = {k: v for k, v in metrics.items() if k != "accuracy"}
            tracker.log(round=server_round, split="server",
                        loss=loss,
                        accuracy=metrics.get("accuracy"),
                        **safe_metrics)
        return result

    def _patched_aggregate_evaluate(server_round, results, failures):
        loss, metrics = orig_aggregate_evaluate(server_round, results, failures)
        if loss is not None:
            accuracy = (metrics or {}).get("accuracy", (metrics or {}).get("val_accuracy"))
            safe_metrics = {
                k: v for k, v in (metrics or {}).items()
                if k not in {"loss", "accuracy"}
            }
            tracker.log(round=server_round, split="client_avg",
                        loss=loss,
                        accuracy=accuracy,
                        **safe_metrics)
        return loss, metrics

    server.strategy.evaluate = _patched_evaluate  # type: ignore
    server.strategy.aggregate_evaluate = _patched_aggregate_evaluate  # type: ignore

    logger.info("=" * 60)
    logger.info("Experiment: %s", experiment_name)
    logger.info("Dataset: %s | Model: %s | Strategy: %s | Defense: %s",
                cfg.dataset.name, cfg.model.architecture,
                cfg.strategy.name, cfg.security.defense.type)
    logger.info("Attack: %s (enabled=%s)", cfg.security.attack.type,
                cfg.security.attack.enabled)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Ray / Flower client resource configuration
    #
    # client_num_cpus/client_num_gpus are Ray resource quotas for each
    # Flower client actor.  Multi-GPU hosts should set client_num_gpus to
    # 0.5 or 1.0 so Ray assigns CUDA_VISIBLE_DEVICES per actor instead of
    # allowing all actors to contend for cuda:0.
    #
    # RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
    #   Prevents Ray from injecting CUDA_VISIBLE_DEVICES="" into actors
    #   when client_num_gpus=0, which would hide the GPU from PyTorch.
    # ------------------------------------------------------------------
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    client_resources = {
        "num_cpus": cfg.ray.client_num_cpus,
        "num_gpus": cfg.ray.client_num_gpus,
    }
    logger.info("Ray client resources: %s", client_resources)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.federation.num_clients,
        config=server_config,
        strategy=server.strategy,
        client_resources=client_resources,
        ray_init_args={
            "ignore_reinit_error": True,
            "log_to_driver": cfg.ray.log_to_driver,
        },
    )

    tracker.save()
    tracker.print_summary()
    return tracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FedSec -- Federated Security Framework")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--experiment", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    overrides = {}
    for ov in args.override:
        k, _, v = ov.partition("=")
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
