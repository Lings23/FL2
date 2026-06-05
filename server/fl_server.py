"""
server/fl_server.py
--------------------
Flower server construction + server-side evaluation logic.
Handles checkpointing, early stopping, and metric logging.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays

from config.config_loader import Config
from models.model_factory import get_model, get_parameters, set_parameters
from strategies.fed_strategy import FedSecStrategy
from utils.metrics import MetricTracker
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


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
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
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

        logger.info("Server eval | round=%d loss=%.4f acc=%.4f", server_round, loss, acc)

        if acc > self.best_accuracy:
            self.best_accuracy = acc
            self.best_round = server_round
            self._save_checkpoint(parameters, server_round, acc)

        return loss, {"accuracy": acc, "server_round": server_round}

    def _save_checkpoint(self, params: NDArrays, rnd: int, acc: float) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"best_model_round{rnd:04d}_acc{acc:.4f}.pt"
        set_parameters(self.model, params)
        torch.save(self.model.state_dict(), path)
        logger.info("Checkpoint saved → %s", path)


# ── Main server builder ───────────────────────────────────────────────────────

def build_server(
    cfg: Config,
    global_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
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
    )

    # Initial parameters
    initial_params = get_parameters(global_model)

    # Strategy
    strategy = FedSecStrategy(
        strategy_cfg=cfg.strategy,
        defense_cfg=cfg.security.defense,
        initial_params=initial_params,
        evaluate_fn=evaluator if cfg.evaluation.save_best_model else None,
        num_clients=cfg.federation.num_clients,
        min_fit_clients=cfg.federation.min_fit_clients,
        min_evaluate_clients=cfg.federation.min_evaluate_clients,
        min_available_clients=cfg.federation.min_available_clients,
        clients_per_round=cfg.federation.clients_per_round,
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
