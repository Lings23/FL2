"""
main.py
-------
Main entry point for federated security experiments.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

import flwr as fl
import ray
from ray._common.utils import get_system_memory
from ray._private.utils import get_used_memory

from config.config_loader import Config, load_config, override_config
from data.dataset import get_dataset, FederatedPartitioner
from models.model_factory import get_model
from client.fl_client import make_client_fn
from server.fl_server import build_server
from utils.logger import setup_logging
from utils.metrics import MetricTracker

logger = logging.getLogger(__name__)


class RayResourcePreflightError(RuntimeError):
    """Raised when the host cannot satisfy the configured Ray memory gate."""


def estimate_available_memory_bytes() -> int:
    """Use the same total-minus-used estimate as the installed Ray version."""
    return int(get_system_memory() - get_used_memory())


def wait_for_available_memory(
    minimum_mb: int,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
) -> int:
    """Wait until Ray can start with the configured amount of host memory."""
    minimum_bytes = max(0, int(minimum_mb)) * 1024 * 1024
    available = estimate_available_memory_bytes()
    if minimum_bytes == 0 or available >= minimum_bytes:
        return available

    timeout = max(0.0, float(timeout_seconds))
    deadline = time.monotonic() + timeout
    interval = max(0.05, float(poll_seconds))
    while time.monotonic() < deadline:
        gc.collect()
        remaining = deadline - time.monotonic()
        time.sleep(min(interval, max(0.0, remaining)))
        available = estimate_available_memory_bytes()
        if available >= minimum_bytes:
            logger.info(
                "Ray memory preflight recovered | available=%.1f MiB required=%d MiB",
                available / (1024 * 1024),
                int(minimum_mb),
            )
            return available

    raise RayResourcePreflightError(
        "Ray memory preflight timed out: "
        f"available={available / (1024 * 1024):.1f} MiB, "
        f"required={int(minimum_mb)} MiB, waited={timeout:.1f}s"
    )


def shutdown_ray_runtime() -> None:
    """Release Ray and allocator-owned memory before another specification."""
    try:
        if ray.is_initialized():
            ray.shutdown()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

def build_data_pipeline(cfg: Config, manifest_path: Optional[Path] = None):
    """
    Returns per-client DataLoader pairs, a test DataLoader, and an optional
    disjoint trusted root subset. Root data is reserved only for FLTrust.

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

    use_root = (
        cfg.security.defense.reserve_root_for_all
        or (
            cfg.security.defense.enabled
            and cfg.security.defense.type.lower() == "fltrust"
        )
    )
    root_dataset = None
    root_indices = np.asarray([], dtype=np.int64)
    federated_dataset = train_ds
    if use_root:
        root_size = min(
            max(1, int(cfg.security.defense.root_dataset_size)),
            max(1, len(train_ds) - cfg.federation.num_clients),
        )
        targets = getattr(train_ds, "targets", None)
        targets_array = None
        if targets is not None:
            targets_array = np.asarray(
                targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
            )
        root_indices = _select_root_indices(
            len(train_ds), root_size, cfg.project.seed, targets_array
        )
        root_index_set = set(root_indices.tolist())
        federated_indices = [idx for idx in range(len(train_ds)) if idx not in root_index_set]
        root_dataset = Subset(train_ds, root_indices.tolist())
        federated_dataset = Subset(train_ds, federated_indices)
        if targets_array is not None:
            federated_dataset.targets = targets_array[federated_indices]  # type: ignore[attr-defined]
        logger.info(
            "Reserved trusted FLTrust root dataset | root=%d federated=%d stratified=%s",
            len(root_dataset), len(federated_dataset), targets_array is not None,
        )

    partitioner = FederatedPartitioner(
        dataset=federated_dataset,
        num_clients=cfg.federation.num_clients,
        strategy=cfg.dataset.partition,
        dirichlet_alpha=cfg.dataset.dirichlet_alpha,
        seed=cfg.project.seed,
        val_split=cfg.dataset.val_split,
    )

    if manifest_path is not None:
        manifest = {
            "dataset": cfg.dataset.name,
            "seed": int(cfg.project.seed),
            "max_client_samples": int(cfg.federation.max_client_samples),
            "root_indices": root_indices.tolist(),
            "client_indices": {
                str(cid): [int(idx) for idx in indices]
                for cid, indices in partitioner.get_all_client_indices().items()
            },
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        import hashlib
        manifest["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    logger.info("\n%s", partitioner.summary())

    loaders_map: Dict[int, Tuple[DataLoader, DataLoader]] = {}
    for cid in range(cfg.federation.num_clients):
        train_sub, val_sub = partitioner.get_client_data(cid)
        max_client_samples = int(cfg.federation.max_client_samples)
        if max_client_samples > 0 and len(train_sub) > max_client_samples:
            train_sub = Subset(train_sub, range(max_client_samples))
        loaders_map[cid] = (
            DataLoader(train_sub, batch_size=cfg.client.batch_size,
                       shuffle=True, num_workers=0, pin_memory=False),
            DataLoader(val_sub, batch_size=cfg.client.batch_size,
                       shuffle=False, num_workers=0),
        )

    test_loader = dataset.get_test_loader(batch_size=128)
    max_test_samples = int(cfg.evaluation.max_test_samples)
    if max_test_samples > 0 and len(test_loader.dataset) > max_test_samples:
        test_loader = DataLoader(
            Subset(test_loader.dataset, range(max_test_samples)),
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
    return loaders_map, test_loader, root_dataset


def _select_root_indices(
    dataset_size: int,
    root_size: int,
    seed: int,
    targets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select a deterministic, approximately class-balanced trusted subset."""
    rng = np.random.default_rng(seed)
    if targets is None or len(targets) != dataset_size:
        return np.sort(rng.choice(dataset_size, size=root_size, replace=False))
    pools = []
    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        pools.append(indices.tolist())
    selected = []
    while len(selected) < root_size and any(pools):
        for pool in pools:
            if pool and len(selected) < root_size:
                selected.append(pool.pop())
    return np.sort(np.asarray(selected, dtype=np.int64))


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
    if cfg.federation.pairing_mode == "strict":
        from experiments.trial_plan import TrialPlanV1

        plan = TrialPlanV1.load(cfg.federation.trial_plan_path)
        if int(plan.payload["num_clients"]) != int(cfg.federation.num_clients):
            raise RuntimeError("TrialPlanV1 num_clients differs from runtime config")
        mal_ids = {int(value) for value in plan.malicious_partition_ids}
        logger.info(
            "Plan-bound malicious identities (%d/%d): %s",
            len(mal_ids), cfg.federation.num_clients, sorted(mal_ids),
        )
        return mal_ids
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
    from attacks.attack_client import validate_label_flip_config

    validate_label_flip_config(cfg.security.attack, cfg.dataset.num_classes)
    if cfg.federation.pairing_mode == "strict":
        # Required by deterministic CUDA matrix multiplication. Set it before
        # Ray actors or the first CUDA context are created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(cfg.project.seed)
    setup_logging(cfg.project.log_dir, cfg.project.log_level, name=experiment_name)
    config_path = Path(cfg.project.log_dir) / f"{experiment_name}_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)
    logger.info("Effective config saved -> %s", config_path)

    # Flower's legacy simulation API keeps Ray alive after a successful run.
    # Tear down any inherited runtime before constructing the next dataset and
    # model, then wait for enough host memory to provision the object store.
    shutdown_ray_runtime()
    available_memory = wait_for_available_memory(
        cfg.ray.min_available_memory_mb,
        cfg.ray.memory_wait_seconds,
        cfg.ray.memory_poll_seconds,
    )
    logger.info(
        "Ray memory preflight | available=%.1f MiB required=%d MiB object_store=%d MiB",
        available_memory / (1024 * 1024),
        cfg.ray.min_available_memory_mb,
        cfg.ray.object_store_memory_mb,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    data_manifest_path = Path(cfg.project.log_dir) / f"{experiment_name}_data_manifest.json"
    loaders_map, test_loader, root_dataset = build_data_pipeline(
        cfg, manifest_path=data_manifest_path
    )
    model_factory = build_model_factory(cfg)

    global_model = get_model(
        architecture=cfg.model.architecture,
        num_classes=cfg.dataset.num_classes,
        pretrained=cfg.model.pretrained,
        dataset_name=cfg.dataset.name,
    )

    malicious_ids = determine_malicious_ids(cfg)

    if cfg.federation.pairing_mode == "strict":
        initial_hasher = hashlib.sha256()
        for parameter in global_model.state_dict().values():
            array = parameter.detach().cpu().numpy()
            initial_hasher.update(str(array.dtype).encode("ascii"))
            initial_hasher.update(json.dumps(array.shape).encode("ascii"))
            initial_hasher.update(array.tobytes(order="C"))
        with data_manifest_path.open("r", encoding="utf-8") as handle:
            data_manifest = json.load(handle)
        runtime_manifest = {
            "pairing_mode": cfg.federation.pairing_mode,
            "sampling_protocol": cfg.federation.sampling_protocol,
            "trial_plan_hash": cfg.federation.trial_plan_hash,
            "trial_plan_path": cfg.federation.trial_plan_path,
            "malicious_partition_ids": sorted(int(value) for value in malicious_ids),
            "malicious_identity_sha256": hashlib.sha256(
                json.dumps(sorted(malicious_ids), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "data_manifest_sha256": str(data_manifest.get("sha256", "")),
            "initial_model_sha256": initial_hasher.hexdigest(),
            "deterministic_client_training": bool(
                cfg.federation.deterministic_client_training
            ),
        }
        runtime_manifest_path = (
            Path(cfg.project.log_dir) / f"{experiment_name}_pairing_manifest.json"
        )
        with runtime_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(runtime_manifest, handle, indent=2, ensure_ascii=False)

    client_fn = make_client_fn(
        model_factory=model_factory,
        loaders_map=loaders_map,
        client_cfg=cfg.client,
        attack_cfg=cfg.security.attack if cfg.security.attack.enabled else None,
        dp_cfg=cfg.differential_privacy if cfg.differential_privacy.enabled else None,
        malicious_ids=malicious_ids,
        device=device,
        experiment_seed=cfg.project.seed,
        num_classes=cfg.dataset.num_classes,
    )

    server, server_config = build_server(
        cfg,
        global_model,
        test_loader,
        device,
        root_dataset=root_dataset,
    )

    tracker = MetricTracker(log_dir=cfg.project.log_dir, experiment_name=experiment_name)

    orig_evaluate = server.strategy.evaluate  # type: ignore
    orig_aggregate_fit = server.strategy.aggregate_fit  # type: ignore
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

    def _patched_aggregate_fit(server_round, results, failures):
        aggregated, metrics = orig_aggregate_fit(server_round, results, failures)
        if metrics:
            tracker.log(round=server_round, split="fit", **metrics)
        for record in getattr(server.strategy, "last_client_records", []):
            safe_record = {key: value for key, value in record.items() if key != "round"}
            tracker.log(round=server_round, split="client", **safe_record)
        return aggregated, metrics

    server.strategy.evaluate = _patched_evaluate  # type: ignore
    server.strategy.aggregate_fit = _patched_aggregate_fit  # type: ignore
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

    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": cfg.ray.include_dashboard,
        "log_to_driver": cfg.ray.log_to_driver,
    }
    if int(cfg.ray.object_store_memory_mb) > 0:
        ray_init_args["object_store_memory"] = (
            int(cfg.ray.object_store_memory_mb) * 1024 * 1024
        )

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.federation.num_clients,
            config=server_config,
            strategy=server.strategy,
            client_resources=client_resources,
            ray_init_args=ray_init_args,
        )

        tracker.save()
        tracker.print_summary()
        return tracker
    finally:
        shutdown_ray_runtime()


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
