"""
client/fl_client.py
--------------------
Flower FlowerClient implementation.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import logging
import os
import random
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


def _configure_cpu_worker_threads(device: torch.device) -> None:
    """Apply an opt-in per-Ray-actor PyTorch CPU thread ceiling.

    BLAS environment variables alone do not constrain PyTorch's intra-op
    pool.  The explicit switch keeps CPU smoke tests from multiplying a full
    thread pool by every concurrent Flower actor.  GPU/formal runs are
    deliberately unaffected unless they explicitly select a CPU device.
    """

    if device.type != "cpu":
        return
    raw = os.environ.get("FEDSEC_CPU_THREADS", "").strip()
    if not raw:
        return
    try:
        threads = int(raw)
    except ValueError as exc:
        raise ValueError("FEDSEC_CPU_THREADS must be a positive integer") from exc
    if threads <= 0:
        raise ValueError("FEDSEC_CPU_THREADS must be a positive integer")
    torch.set_num_threads(threads)
    # Inter-op threads can only be initialized once in a process. Ray may
    # reuse an actor for multiple logical clients, so an already initialized
    # pool is acceptable after the intra-op ceiling has been enforced.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if "cannot set number of interop threads" not in str(exc).lower():
            raise


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
                if not bool(torch.isfinite(loss).item()):
                    from utils.numerical_failure import NumericalFailure
                    raise NumericalFailure("nonfinite_client_metric", "client_training", "Non-finite training loss")

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
    """Flower client for federated security research."""

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
        experiment_seed: int = 0,
        num_classes: int = 10,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self._clean_train_loader = train_loader
        self.val_loader = val_loader
        self.client_cfg = client_cfg
        self.attack_cfg = attack_cfg or AttackConfig()
        self.dp_cfg = dp_cfg or DPConfig()
        self.experiment_seed = int(experiment_seed)
        self.num_classes = int(num_classes)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.trainer = LocalTrainer(model, client_cfg, self.device, dp_cfg)
        self._attack_active = False
        self._current_attack_seed = int(experiment_seed)

    @staticmethod
    def _rebuild_train_loader(loader: DataLoader, seed: int) -> DataLoader:
        """Recreate shuffle and augmentation RNG for one client-round stream."""
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
            timeout=0,
            worker_init_fn=None,
            generator=generator,
        )

    @contextlib.contextmanager
    def _client_round_randomness(self, seed: int, deterministic: bool):
        """Isolate Python/NumPy/Torch/CUDA randomness from actor scheduling."""
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        if deterministic:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        try:
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                random.seed(int(seed))
                np.random.seed(int(seed) % (2**32))
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _is_attack_active(self, server_round: int) -> bool:
        if not self.is_malicious or not self.attack_cfg.enabled:
            return False
        start = max(1, int(self.attack_cfg.attack_start_round))
        end = int(self.attack_cfg.attack_end_round)
        if server_round < start or (end >= 0 and server_round > end):
            return False
        on_rounds = max(1, int(self.attack_cfg.attack_on_rounds))
        off_rounds = max(0, int(self.attack_cfg.attack_off_rounds))
        cycle = on_rounds + off_rounds
        return ((server_round - start) % cycle) < on_rounds

    def on_before_fit(self, parameters: List[np.ndarray], config: Dict) -> None:
        pass

    def on_after_fit(
        self, parameters: List[np.ndarray], metrics: Dict
    ) -> List[np.ndarray]:
        return parameters

    def on_evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        return self._default_evaluate(parameters, config)

    def fit(self, ins: FitIns) -> FitRes:
        server_params = parameters_to_ndarrays(ins.parameters)
        config = ins.config
        server_round = int(config.get("server_round", 1))
        fit_seed = int(config.get(
            "fit_seed",
            (self.experiment_seed * 1_000_003 + server_round * 1009 + self.client_id) % (2**32),
        ))
        attack_seed = int(config.get(
            "attack_seed",
            (self.experiment_seed * 1_000_003 + self.client_id) % (2**32),
        ))
        deterministic = bool(config.get("deterministic_client_training", False))
        self._current_attack_seed = attack_seed
        self.train_loader = self._clean_train_loader
        self._attack_active = self._is_attack_active(server_round)

        with self._client_round_randomness(fit_seed, deterministic):
            set_parameters(self.model, server_params)
            with self._client_round_randomness(attack_seed, deterministic=False):
                self.on_before_fit(server_params, config)
            self.train_loader = self._rebuild_train_loader(self.train_loader, fit_seed)

            local_epochs = int(config.get("local_epochs", self.client_cfg.local_epochs))
            metrics = self.trainer.train(
                self.train_loader,
                epochs=local_epochs,
            )
            metrics["local_epochs"] = local_epochs

            updated_params = get_parameters(self.model)
            with self._client_round_randomness(attack_seed, deterministic=False):
                updated_params = self.on_after_fit(updated_params, metrics)

        logger.debug("Client %d | loss=%.4f acc=%.4f",
                     self.client_id, metrics["train_loss"], metrics["train_accuracy"])

        fit_metrics = {
            key: (
                value
                if isinstance(value, (str, bytes))
                else float(value)
            )
            for key, value in metrics.items()
        }
        # These fields support stable state and offline evaluation. The server
        # must never use is_malicious as an input to a defense decision.
        fit_metrics["client_id"] = int(self.client_id)
        fit_metrics["is_malicious"] = bool(self.is_malicious)
        fit_metrics["attack_active"] = bool(self._attack_active)
        fit_metrics["fit_seed"] = int(fit_seed)
        fit_metrics["attack_seed"] = int(attack_seed)

        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=len(self.train_loader.dataset),
            metrics=fit_metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        server_params = parameters_to_ndarrays(ins.parameters)
        eval_seed = int(ins.config.get(
            "eval_seed",
            (self.experiment_seed * 1_000_003 + self.client_id * 1013) % (2**32),
        ))
        deterministic = bool(
            ins.config.get("deterministic_client_evaluation", False)
        )
        with self._client_round_randomness(eval_seed, deterministic):
            loss, n, metrics = self.on_evaluate(server_params, ins.config)
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["client_id"] = int(self.client_id)
        metrics["eval_seed"] = int(eval_seed)
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=loss,
            num_examples=n,
            metrics=metrics,
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

def assign_dba_fragments(
    malicious_ids: set[int], trigger_num: int
) -> Dict[int, int]:
    """Deterministically cover DBA fragments by malicious-client rank."""
    count = max(1, int(trigger_num))
    return {
        client_id: rank % count
        for rank, client_id in enumerate(sorted(int(value) for value in malicious_ids))
    }

def make_client_fn(
    model_factory: Callable[[], nn.Module],
    loaders_map: Dict[int, Tuple[DataLoader, DataLoader]],
    client_cfg: ClientConfig,
    attack_cfg: Optional[AttackConfig] = None,
    dp_cfg: Optional[DPConfig] = None,
    malicious_ids: Optional[set] = None,
    device: Optional[torch.device] = None,
    experiment_seed: int = 0,
    num_classes: int = 10,
):
    """
    Return a Flower-compatible client_fn(cid: str) -> fl.client.Client.

    Serialization fix: client_fn must NOT close over the full loaders_map.
    Ray pickles the entire closure of client_fn on every actor dispatch.
    If loaders_map (all clients' DataLoaders) is captured in the closure,
    Ray serializes it on every call even though each actor only needs one
    client's pair.  For MNIST with 10 clients this is ~47 MB of redundant
    pickle work per dispatch (x5 clients/round = 235 MB/round wasted).

    Fix: extract each client's loader pair into a flat lookup dict keyed by
    cid.  The closure captures this lightweight dict instead of the full
    map of DataLoader objects.  On MNIST the serialized size drops from
    ~47 MB to ~5 MB per dispatch.
    """
    from attacks.attack_client import get_attack_client_class

    malicious_ids = malicious_ids or set()
    _dba_fragments = assign_dba_fragments(
        {int(value) for value in malicious_ids},
        int(attack_cfg.dba_trigger_num) if attack_cfg is not None else 1,
    )
    _device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a plain dict of (train_loader, val_loader) tuples.
    # This is the same data as loaders_map but captured as a local variable
    # so the closure does not hold a reference to the loaders_map name,
    # preventing accidental capture of the full outer-scope dict.
    _loaders: Dict[int, Tuple[DataLoader, DataLoader]] = dict(loaders_map)

    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        _configure_cpu_worker_threads(_device)
        model = model_factory()
        train_loader, val_loader = _loaders[cid_int]

        effective_attack_cfg = attack_cfg
        if (
            cid_int in malicious_ids
            and attack_cfg
            and attack_cfg.enabled
            and attack_cfg.type.lower() == "dba"
        ):
            effective_attack_cfg = replace(
                attack_cfg,
                dba_fragment_index=_dba_fragments[cid_int],
            )

        if cid_int in malicious_ids and effective_attack_cfg and effective_attack_cfg.enabled:
            ClientClass = get_attack_client_class(effective_attack_cfg.type)
        else:
            ClientClass = FedSecClient

        client = ClientClass(
            client_id=cid_int,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            client_cfg=client_cfg,
            attack_cfg=effective_attack_cfg,
            dp_cfg=dp_cfg,
            device=_device,
            experiment_seed=experiment_seed,
            num_classes=num_classes,
        )
        client.is_malicious = cid_int in malicious_ids
        return client

    return client_fn
