"""
client/fl_client.py
--------------------
Flower FlowerClient implementation.

Design goals
------------
- Clean separation of training logic from FL protocol
- Attack injection hooks for security experiments
- Differential privacy wrapper support
- Metrics returned in fit() / evaluate() for server-side aggregation

Extension points
----------------
- Override on_before_fit()  to inject custom local behaviour before training
- Override on_after_fit()   to intercept / poison the uploaded gradient/model
- Override on_evaluate()    to add custom local metrics
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from config.config_loader import ClientConfig, AttackConfig, DPConfig
from models.model_factory import get_parameters, set_parameters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LocalTrainer
# ---------------------------------------------------------------------------

class LocalTrainer:
    """Encapsulates a single local training loop."""

    def __init__(
        self,
        model: nn.Module,
        cfg: ClientConfig,
        device: torch.device,
        dp_cfg: Optional[DPConfig] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.dp_cfg = dp_cfg
        self.criterion = nn.CrossEntropyLoss()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.cfg.optimizer.lower() == "adam":
            return Adam(self.model.parameters(),
                        lr=self.cfg.learning_rate,
                        weight_decay=self.cfg.weight_decay)
        return SGD(self.model.parameters(),
                   lr=self.cfg.learning_rate,
                   momentum=self.cfg.momentum,
                   weight_decay=self.cfg.weight_decay)

    def _build_scheduler(self, optimizer, num_steps: int):
        s = self.cfg.lr_scheduler.lower()
        if s == "cosine":
            return CosineAnnealingLR(optimizer, T_max=num_steps)
        if s == "step":
            return StepLR(optimizer, step_size=max(1, num_steps // 3), gamma=0.1)
        return None

    def train(self, train_loader: DataLoader, epochs: int) -> Dict[str, float]:
        self.model.to(self.device)
        self.model.train()

        optimizer = self._build_optimizer()
        total_steps = epochs * len(train_loader)
        scheduler = self._build_scheduler(optimizer, total_steps)

        running_loss = correct = total = 0
        t0 = time.time()

        for _epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

                if self.dp_cfg and self.dp_cfg.enabled:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.dp_cfg.max_grad_norm)
                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.grad is not None:
                                noise = torch.randn_like(p.grad) * (
                                    self.dp_cfg.noise_multiplier * self.dp_cfg.max_grad_norm)
                                p.grad += noise
                    optimizer.step()
                else:
                    loss.backward()
                    optimizer.step()

                if scheduler:
                    scheduler.step()

                with torch.no_grad():
                    running_loss += loss.item() * batch_x.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total += batch_x.size(0)

        return {
            "train_loss": running_loss / total,
            "train_accuracy": correct / total,
            "train_time": time.time() - t0,
        }

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.to(self.device)
        self.model.eval()

        loss_sum = correct = total = 0

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss_sum += loss.item() * batch_x.size(0)
            correct += (logits.argmax(1) == batch_y).sum().item()
            total += batch_x.size(0)

        return {
            "val_loss": loss_sum / total,
            "val_accuracy": correct / total,
        }


# ---------------------------------------------------------------------------
# FedSecClient
# ---------------------------------------------------------------------------

class FedSecClient(fl.client.Client):
    """
    Flower client for federated security research.

    Parameters
    ----------
    client_id    : unique client identifier
    model        : the model to train locally
    train_loader : local training data
    val_loader   : local validation data
    client_cfg   : ClientConfig
    attack_cfg   : AttackConfig  (may be None)
    dp_cfg       : DPConfig      (may be None)
    device       : torch.device
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        client_cfg: ClientConfig,
        attack_cfg: Optional[AttackConfig] = None,
        dp_cfg: Optional[DPConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_cfg = client_cfg
        self.attack_cfg = attack_cfg or AttackConfig()
        self.dp_cfg = dp_cfg or DPConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.trainer = LocalTrainer(model, client_cfg, self.device, dp_cfg)

    # Extension hooks

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        """Called before local training. Override for data-poisoning attacks."""
        pass

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        """Called after local training. Override for model-poisoning attacks."""
        return parameters

    def on_evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Override to add custom evaluation logic."""
        return self._default_evaluate(parameters, config)

    # Flower protocol

    def fit(self, ins: FitIns) -> FitRes:
        server_params = parameters_to_ndarrays(ins.parameters)
        config = ins.config

        set_parameters(self.model, server_params)
        self.on_before_fit(server_params, config)

        metrics = self.trainer.train(
            self.train_loader,
            epochs=int(config.get("local_epochs", self.client_cfg.local_epochs)),
        )

        updated_params = get_parameters(self.model)
        updated_params = self.on_after_fit(updated_params, metrics)

        logger.debug("Client %d | loss=%.4f acc=%.4f",
                     self.client_id, metrics["train_loss"], metrics["train_accuracy"])

        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=len(self.train_loader.dataset),
            metrics={k: float(v) for k, v in metrics.items()},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        server_params = parameters_to_ndarrays(ins.parameters)
        loss, n, metrics = self.on_evaluate(server_params, ins.config)
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=loss,
            num_examples=n,
            metrics={k: float(v) for k, v in metrics.items()},
        )

    def _default_evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)
        val_metrics = self.trainer.evaluate(self.val_loader)
        return (
            val_metrics["val_loss"],
            len(self.val_loader.dataset),
            val_metrics,
        )

    @property
    def is_malicious(self) -> bool:
        return getattr(self, "_is_malicious", False)

    @is_malicious.setter
    def is_malicious(self, val: bool) -> None:
        self._is_malicious = val


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_client_fn(
    model_factory: Callable[[], nn.Module],
    loaders_map: Dict[int, Tuple[DataLoader, DataLoader]],
    client_cfg: ClientConfig,
    attack_cfg: Optional[AttackConfig] = None,
    dp_cfg: Optional[DPConfig] = None,
    malicious_ids: Optional[set] = None,
    device: Optional[torch.device] = None,
):
    """
    Return a Flower-compatible client_fn(cid: str) -> fl.client.Client.

    Memory fix: accepts a *model_factory* callable instead of a pre-built
    models_map dict.  Each time Flower asks for client `cid`, a fresh model
    is instantiated inside the actor process rather than being passed in
    (and kept alive) from the parent process.  This eliminates the
    ~200 MB x num_clients footprint that caused OOM kills on Colab.

    Parameters
    ----------
    model_factory : callable () -> nn.Module
        Returns a freshly constructed model.  Called once per client_fn
        invocation inside the Ray actor.
    loaders_map   : {client_id: (train_loader, val_loader)}
    malicious_ids : set of int client IDs that run attack behaviour
    """
    from attacks.attack_client import get_attack_client_class

    malicious_ids = malicious_ids or set()
    _device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        # Build a fresh model inside the actor -- not shared with the parent
        model = model_factory()
        train_loader, val_loader = loaders_map[cid_int]

        if cid_int in malicious_ids and attack_cfg and attack_cfg.enabled:
            ClientClass = get_attack_client_class(attack_cfg.type)
        else:
            ClientClass = FedSecClient

        client = ClientClass(
            client_id=cid_int,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            client_cfg=client_cfg,
            attack_cfg=attack_cfg,
            dp_cfg=dp_cfg,
            device=_device,
        )
        client.is_malicious = cid_int in malicious_ids
        return client

    return client_fn
