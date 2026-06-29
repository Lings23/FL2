"""
server/fl_server.py
--------------------
Flower server construction + server-side evaluation logic.
Handles checkpointing, early stopping, and metric logging.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import flwr as fl
from flwr.common import NDArrays

from config.config_loader import AttackConfig, Config
from models.model_factory import get_model, get_parameters, set_parameters
from strategies.fed_strategy import FedSecStrategy
from utils.metrics import MetricTracker
from utils.logger import setup_logging
from attacks.attack_client import get_dba_trigger_coords

logger = logging.getLogger(__name__)


def stamp_backdoor_trigger(x: torch.Tensor, attack_cfg: AttackConfig) -> torch.Tensor:
    """Return a copy of x with the standard bottom-right backdoor trigger."""
    triggered = x.clone()
    trigger_size = max(1, int(attack_cfg.trigger_size))
    triggered[..., -trigger_size:, -trigger_size:] = float(attack_cfg.trigger_value)
    return triggered


def stamp_dba_full_trigger(x: torch.Tensor, attack_cfg: AttackConfig) -> torch.Tensor:
    """Return a copy of x stamped with all DBA trigger fragments."""
    triggered = x.clone()
    image_shape = tuple(triggered.shape[1:]) if triggered.dim() >= 3 else tuple(triggered.shape)
    for fragment_index in range(max(1, int(attack_cfg.dba_trigger_num))):
        coords = get_dba_trigger_coords(
            fragment_index=fragment_index,
            image_shape=image_shape,
            trigger_size=attack_cfg.trigger_size,
            dba_trigger_num=attack_cfg.dba_trigger_num,
            gap=attack_cfg.dba_gap,
            base_row=attack_cfg.dba_base_row,
            base_col=attack_cfg.dba_base_col,
        )
        for row, col in coords:
            triggered[..., row, col] = float(attack_cfg.trigger_value)
    return triggered


def evaluate_targeted_asr(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_cfg: AttackConfig,
) -> Optional[Dict[str, Any]]:
    """Evaluate targeted ASR on non-target-label test samples."""
    attack_type = attack_cfg.type.lower()
    if not attack_cfg.enabled or attack_type not in {"backdoor", "dba", "model_replacement"}:
        return None

    target_label = int(attack_cfg.backdoor_target_label)
    success = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mask = y != target_label
            if not mask.any():
                continue

            if attack_type == "dba":
                triggered_x = stamp_dba_full_trigger(x, attack_cfg)
            else:
                triggered_x = stamp_backdoor_trigger(x, attack_cfg)

            logits = model(triggered_x)
            preds = logits.argmax(1)
            success += (preds[mask] == target_label).sum().item()
            total += mask.sum().item()

    asr = success / total if total else 0.0
    return {"asr": asr, "asr_total": total, "attack_type": attack_type}


# ── Server-side evaluator ─────────────────────────────────────────────────────

class ServerEvaluator:
    """
    Wraps the global model + test loader for server-side evaluation.
    Passed as evaluate_fn to the strategy.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        checkpoint_dir: Path,
        attack_cfg: Optional[AttackConfig] = None,
        save_best_model: bool = True,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.attack_cfg = attack_cfg or AttackConfig()
        self.save_best_model = bool(save_best_model)
        self.best_accuracy = 0.0
        self.best_round = 0

    def __call__(
        self,
        server_round: int,
        parameters: NDArrays,
        config: Dict,
    ) -> Optional[Tuple[float, Dict]]:
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_sum += criterion(logits, y).item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)

        loss = loss_sum / total
        acc = correct / total
        metrics: Dict[str, Any] = {"accuracy": acc, "server_round": server_round}

        asr_metrics = evaluate_targeted_asr(
            self.model,
            self.test_loader,
            self.device,
            self.attack_cfg,
        )
        if asr_metrics:
            metrics.update(asr_metrics)
            logger.info(
                "Server eval | round=%d loss=%.4f acc=%.4f asr=%.4f attack=%s",
                server_round,
                loss,
                acc,
                asr_metrics["asr"],
                asr_metrics["attack_type"],
            )
        else:
            logger.info("Server eval | round=%d loss=%.4f acc=%.4f", server_round, loss, acc)

        if acc > self.best_accuracy:
            self.best_accuracy = acc
            self.best_round = server_round
            if self.save_best_model:
                self._save_checkpoint(parameters, server_round, acc)

        return loss, metrics

    def _save_checkpoint(self, params: NDArrays, rnd: int, acc: float) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"best_model_round{rnd:04d}_acc{acc:.4f}.pt"
        set_parameters(self.model, params)
        torch.save(self.model.state_dict(), path)
        logger.info("Checkpoint saved → %s", path)


# ── Main server builder ───────────────────────────────────────────────────────

def build_fltrust_server_update_fn(
    cfg: Config,
    model: nn.Module,
    root_dataset: Dataset,
    device: torch.device,
) -> Callable[[NDArrays], NDArrays]:
    """Build a deterministic trusted-root training callback for FLTrust."""
    root_size = min(max(1, int(cfg.security.defense.root_dataset_size)), len(root_dataset))
    rng = np.random.default_rng(cfg.project.seed)
    indices = rng.choice(len(root_dataset), size=root_size, replace=False).tolist()
    generator = torch.Generator().manual_seed(cfg.project.seed)
    root_loader = DataLoader(
        Subset(root_dataset, indices),
        batch_size=cfg.client.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    root_model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()

    def compute_server_update(global_params: NDArrays) -> NDArrays:
        set_parameters(root_model, global_params)
        root_model.train()
        if cfg.client.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                root_model.parameters(),
                lr=cfg.client.learning_rate,
                weight_decay=cfg.client.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                root_model.parameters(),
                lr=cfg.client.learning_rate,
                momentum=cfg.client.momentum,
                weight_decay=cfg.client.weight_decay,
            )

        for batch_x, batch_y in root_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(root_model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        trained_params = get_parameters(root_model)
        server_delta: NDArrays = []
        for trained, reference in zip(trained_params, global_params):
            if np.issubdtype(reference.dtype, np.floating):
                server_delta.append(
                    (trained.astype(np.float32) - reference.astype(np.float32)).astype(
                        reference.dtype, copy=False
                    )
                )
            else:
                server_delta.append(np.zeros_like(reference))
        return server_delta

    logger.info("FLTrust root dataset prepared | samples=%d", root_size)
    return compute_server_update


def build_server(
    cfg: Config,
    global_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    root_dataset: Optional[Dataset] = None,
) -> Tuple[fl.server.Server, fl.server.ServerConfig]:
    """
    Construct and return a Flower Server + ServerConfig.

    Returns
    -------
    server : fl.server.Server
    server_config : fl.server.ServerConfig
    """
    setup_logging(cfg.project.log_dir, cfg.project.log_level)
    checkpoint_dir = Path(cfg.project.checkpoint_dir)

    # Server-side evaluator
    evaluator = ServerEvaluator(
        model=global_model,
        test_loader=test_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        attack_cfg=cfg.security.attack,
        save_best_model=cfg.evaluation.save_best_model,
    )

    # Initial parameters
    initial_params = get_parameters(global_model)

    server_update_fn = None
    if cfg.security.defense.enabled and cfg.security.defense.type.lower() == "fltrust":
        if root_dataset is None:
            raise ValueError("FLTrust requires a trusted root dataset")
        server_update_fn = build_fltrust_server_update_fn(
            cfg,
            global_model,
            root_dataset,
            device,
        )

    # Strategy
    strategy = FedSecStrategy(
        strategy_cfg=cfg.strategy,
        defense_cfg=cfg.security.defense,
        initial_params=initial_params,
        evaluate_fn=evaluator,
        num_clients=cfg.federation.num_clients,
        min_fit_clients=cfg.federation.min_fit_clients,
        min_evaluate_clients=cfg.federation.min_evaluate_clients,
        min_available_clients=cfg.federation.min_available_clients,
        clients_per_round=cfg.federation.clients_per_round,
        server_update_fn=server_update_fn,
    )

    server = fl.server.Server(
        client_manager=fl.server.SimpleClientManager(),
        strategy=strategy,
    )
    server_config = fl.server.ServerConfig(num_rounds=cfg.federation.num_rounds)

    logger.info(
        "Server built | rounds=%d clients=%d strategy=%s defense=%s",
        cfg.federation.num_rounds,
        cfg.federation.num_clients,
        cfg.strategy.name,
        cfg.security.defense.type,
    )
    return server, server_config
