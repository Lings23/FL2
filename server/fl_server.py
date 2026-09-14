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
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import flwr as fl
from flwr.common import NDArrays

from config.config_loader import AttackConfig, Config
from models.model_factory import (
    get_model,
    get_parameter_names,
    get_parameter_roles,
    get_parameters,
    get_trainable_parameter_indices,
    set_parameters,
)
from strategies.fed_strategy import FedSecStrategy
from utils.metrics import MetricTracker
from utils.logger import setup_logging
from attacks.attack_client import get_dba_trigger_coords, stamp_dba_coords_

logger = logging.getLogger(__name__)


def _model_state_is_finite(model: nn.Module) -> bool:
    """Return whether every floating parameter and buffer is finite."""
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all().item())
        for value in model.state_dict().values()
    )


def _invalid_asr_metrics(
    attack_type: str,
    reason: str,
    *,
    dba_trigger_num: int = 0,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "asr": math.nan,
        "asr_total": 0,
        "asr_valid": False,
        "asr_invalid_reason": reason,
        "attack_type": attack_type,
    }
    if attack_type == "dba":
        metrics.update({
            "dba_full_trigger_asr": math.nan,
            "dba_local_asr_mean": math.nan,
            "dba_local_asr_min": math.nan,
            "dba_local_asr_max": math.nan,
        })
        for fragment_index in range(max(1, int(dba_trigger_num))):
            metrics[f"dba_local_asr_fragment_{fragment_index}"] = math.nan
    return metrics


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
            pattern_mode=attack_cfg.dba_pattern_mode,
            gap=attack_cfg.dba_gap,
            base_row=attack_cfg.dba_base_row,
            base_col=attack_cfg.dba_base_col,
        )
        stamp_dba_coords_(
            triggered,
            coords,
            trigger_value=attack_cfg.trigger_value,
            value_mode=attack_cfg.dba_trigger_value_mode,
        )
    return triggered


def stamp_dba_local_trigger(
    x: torch.Tensor, attack_cfg: AttackConfig, fragment_index: int
) -> torch.Tensor:
    """Return a copy of x stamped with exactly one DBA local fragment."""
    triggered = x.clone()
    image_shape = tuple(triggered.shape[1:]) if triggered.dim() >= 3 else tuple(triggered.shape)
    coords = get_dba_trigger_coords(
        fragment_index=fragment_index,
        image_shape=image_shape,
        trigger_size=attack_cfg.trigger_size,
        dba_trigger_num=attack_cfg.dba_trigger_num,
        pattern_mode=attack_cfg.dba_pattern_mode,
        gap=attack_cfg.dba_gap,
        base_row=attack_cfg.dba_base_row,
        base_col=attack_cfg.dba_base_col,
    )
    stamp_dba_coords_(
        triggered,
        coords,
        trigger_value=attack_cfg.trigger_value,
        value_mode=attack_cfg.dba_trigger_value_mode,
    )
    return triggered


def evaluate_targeted_asr(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_cfg: AttackConfig,
    *,
    server_loss: float | None = None,
    model_state_valid: bool | None = None,
    clean_logits_valid: bool | None = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate targeted ASR on non-target-label test samples."""
    attack_type = attack_cfg.type.lower()
    if not attack_cfg.enabled or attack_type not in {
        "backdoor",
        "dba",
        "model_replacement",
        "scaling_backdoor",
    }:
        return None

    target_label = int(attack_cfg.backdoor_target_label)
    state_valid = (
        _model_state_is_finite(model)
        if model_state_valid is None
        else bool(model_state_valid)
    )
    if not state_valid:
        return _invalid_asr_metrics(
            attack_type,
            "nonfinite_model_state",
            dba_trigger_num=attack_cfg.dba_trigger_num,
        )
    if clean_logits_valid is False:
        return _invalid_asr_metrics(
            attack_type,
            "nonfinite_clean_logits",
            dba_trigger_num=attack_cfg.dba_trigger_num,
        )
    if server_loss is not None and not math.isfinite(float(server_loss)):
        return _invalid_asr_metrics(
            attack_type,
            "nonfinite_server_loss",
            dba_trigger_num=attack_cfg.dba_trigger_num,
        )

    success = 0
    total = 0
    local_success = [0 for _ in range(max(1, int(attack_cfg.dba_trigger_num)))]

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

            if not bool(torch.isfinite(triggered_x).all().item()):
                return _invalid_asr_metrics(
                    attack_type,
                    "nonfinite_triggered_input",
                    dba_trigger_num=attack_cfg.dba_trigger_num,
                )

            logits = model(triggered_x)
            if not bool(torch.isfinite(logits).all().item()):
                return _invalid_asr_metrics(
                    attack_type,
                    "nonfinite_logits",
                    dba_trigger_num=attack_cfg.dba_trigger_num,
                )
            preds = logits.argmax(1)
            success += (preds[mask] == target_label).sum().item()
            total += mask.sum().item()

            if attack_type == "dba":
                for fragment_index in range(len(local_success)):
                    local_logits = model(
                        stamp_dba_local_trigger(x, attack_cfg, fragment_index)
                    )
                    if not bool(torch.isfinite(local_logits).all().item()):
                        return _invalid_asr_metrics(
                            attack_type,
                            f"nonfinite_local_logits_fragment_{fragment_index}",
                            dba_trigger_num=attack_cfg.dba_trigger_num,
                        )
                    local_preds = local_logits.argmax(1)
                    local_success[fragment_index] += (
                        local_preds[mask] == target_label
                    ).sum().item()

    asr = success / total if total else 0.0
    result: Dict[str, Any] = {
        "asr": asr,
        "asr_total": total,
        "asr_valid": True,
        "asr_invalid_reason": "",
        "attack_type": attack_type,
    }
    if attack_type == "dba":
        local_asr = [value / total if total else 0.0 for value in local_success]
        result["dba_full_trigger_asr"] = asr
        result["dba_local_asr_mean"] = float(np.mean(local_asr))
        result["dba_local_asr_min"] = float(np.min(local_asr))
        result["dba_local_asr_max"] = float(np.max(local_asr))
        for fragment_index, value in enumerate(local_asr):
            result[f"dba_local_asr_fragment_{fragment_index}"] = float(value)
    return result


def label_flip_metrics_from_confusion(
    confusion: np.ndarray,
    attack_cfg: AttackConfig,
) -> Dict[str, Any]:
    """Derive objective-aware label-flip metrics from a clean confusion matrix."""
    attack_type = str(attack_cfg.type).lower()
    if (
        not attack_cfg.enabled
        or attack_type not in {"label_flip_targeted", "label_flip_all_reverse"}
    ):
        return {}

    matrix = np.asarray(confusion, dtype=np.int64)
    supports = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        supports,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=supports > 0,
    )
    supported_recalls = recalls[supports > 0]
    metrics: Dict[str, Any] = {
        "macro_recall": float(supported_recalls.mean()) if supported_recalls.size else 0.0,
        "confusion_matrix_json": json.dumps(matrix.tolist(), separators=(",", ":")),
        "attack_type": attack_type,
    }

    if attack_type == "label_flip_targeted":
        source = int(attack_cfg.source_label)
        target = int(attack_cfg.target_label)
        source_total = int(supports[source])
        if source_total == 0:
            raise RuntimeError(
                f"Targeted label-flip ASR is undefined: test set has no source label {source}"
            )
        source_to_target = int(matrix[source, target])
        target_predictions = int(matrix[:, target].sum())
        metrics.update({
            "asr": float(source_to_target / source_total),
            "asr_total": source_total,
            "source_recall": float(matrix[source, source] / source_total),
            "target_precision": (
                float(matrix[target, target] / target_predictions)
                if target_predictions else 0.0
            ),
            "source_to_target_count": source_to_target,
            "target_prediction_count": target_predictions,
            "source_label": source,
            "target_label": target,
        })
    elif attack_type == "label_flip_all_reverse":
        # The attack objective is the deterministic mapping y -> C - 1 - y.
        # Report its success rate in addition to clean accuracy/macro recall so
        # promotion gates can compare candidate and baseline on the attack's
        # actual objective.  For even-sized label spaces (for example CIFAR-10)
        # no class maps to itself; the definition also remains well-defined for
        # odd-sized label spaces.
        reverse_targets = matrix.shape[0] - 1 - np.arange(matrix.shape[0])
        reverse_success = int(matrix[np.arange(matrix.shape[0]), reverse_targets].sum())
        reverse_total = int(supports.sum())
        if reverse_total == 0:
            raise RuntimeError("All-reverse label-flip ASR is undefined: empty test set")
        metrics.update({
            "asr": float(reverse_success / reverse_total),
            "asr_total": reverse_total,
            "reverse_mapping_rate": float(reverse_success / reverse_total),
            "reverse_mapping_count": reverse_success,
        })
    return metrics


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
        num_classes: int = 10,
        save_best_model: bool = True,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.attack_cfg = attack_cfg or AttackConfig()
        self.num_classes = int(num_classes)
        self.save_best_model = bool(save_best_model)
        self.state_provider: Optional[Callable[[], Dict[str, Any]]] = None
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
        model_state_valid = _model_state_is_finite(self.model)

        criterion = nn.CrossEntropyLoss()
        loss_sum = 0.0
        correct = 0
        total = 0
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        clean_logits_valid = True
        loss_float64_recomputations = 0

        if model_state_valid:
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    if not bool(torch.isfinite(logits).all().item()):
                        clean_logits_valid = False
                        continue
                    batch_loss = float(criterion(logits, y).item())
                    if not math.isfinite(batch_loss):
                        batch_loss = float(criterion(logits.to(torch.float64), y).item())
                        loss_float64_recomputations += 1
                    if not math.isfinite(batch_loss):
                        clean_logits_valid = False
                        continue
                    preds = logits.argmax(1)
                    loss_sum += batch_loss * x.size(0)
                    correct += (preds == y).sum().item()
                    encoded = (y * self.num_classes + preds).detach().cpu().numpy()
                    confusion += np.bincount(
                        encoded, minlength=self.num_classes * self.num_classes
                    ).reshape(self.num_classes, self.num_classes)
                    total += x.size(0)
        else:
            clean_logits_valid = False

        evaluation_valid = bool(model_state_valid and clean_logits_valid and total > 0)
        loss = loss_sum / total if evaluation_valid else math.nan
        acc = correct / total if evaluation_valid else math.nan
        supports = confusion.sum(axis=1)
        class_recalls = np.divide(
            np.diag(confusion),
            supports,
            out=np.full(self.num_classes, math.nan, dtype=np.float64),
            where=supports > 0,
        )
        supported = class_recalls[np.isfinite(class_recalls)]
        metrics: Dict[str, Any] = {
            "accuracy": acc,
            "server_round": server_round,
            "loss_valid": math.isfinite(float(loss)),
            "loss_float64_recomputations": loss_float64_recomputations,
            "logits_valid": clean_logits_valid,
            "model_state_valid": model_state_valid,
            "macro_recall": (
                float(supported.mean()) if evaluation_valid and supported.size else math.nan
            ),
            "confusion_matrix_json": (
                json.dumps(confusion.tolist(), separators=(",", ":"))
                if evaluation_valid
                else ""
            ),
        }
        for label, recall in enumerate(class_recalls):
            metrics[f"class_recall_{label}"] = (
                float(recall) if evaluation_valid and math.isfinite(float(recall)) else math.nan
            )

        asr_metrics = evaluate_targeted_asr(
            self.model,
            self.test_loader,
            self.device,
            self.attack_cfg,
            server_loss=loss,
            model_state_valid=model_state_valid,
            clean_logits_valid=clean_logits_valid,
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
            attack_type = str(self.attack_cfg.type).lower()
            if evaluation_valid:
                label_flip_metrics = label_flip_metrics_from_confusion(
                    confusion, self.attack_cfg
                )
                if label_flip_metrics:
                    metrics.update(label_flip_metrics)
                    metrics["asr_valid"] = True
                    metrics["asr_invalid_reason"] = ""
            elif self.attack_cfg.enabled and attack_type in {
                "label_flip_targeted", "label_flip_all_reverse"
            }:
                reason = (
                    "nonfinite_model_state"
                    if not model_state_valid
                    else "nonfinite_clean_logits_or_loss"
                )
                metrics.update(_invalid_asr_metrics(attack_type, reason))
                metrics["source_recall"] = math.nan
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
        if self.state_provider is not None:
            rtc_path = path.with_name(f"{path.stem}_rtc_state.pt")
            torch.save(self.state_provider(), rtc_path)
            logger.info("Defense state checkpoint saved → %s", rtc_path)
        logger.info("Checkpoint saved → %s", path)


# ── Main server builder ───────────────────────────────────────────────────────

def build_fltrust_server_update_fn(
    cfg: Config,
    model: nn.Module,
    root_dataset: Dataset,
    device: torch.device,
) -> Callable[[NDArrays], NDArrays]:
    """Build a deterministic trusted-root training callback for FLTrust."""
    params = cfg.security.defense.custom_params or {}
    root_size = len(root_dataset)
    generator = torch.Generator().manual_seed(cfg.project.seed)
    root_loader = DataLoader(
        root_dataset,
        batch_size=int(params.get("root_batch_size", 32)),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    root_model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()

    def compute_server_update(global_params: NDArrays) -> NDArrays:
        set_parameters(root_model, global_params)
        root_model.train()
        root_optimizer = str(params.get("root_optimizer", "sgd")).lower()
        root_lr = float(params.get("root_learning_rate", 0.01))
        root_momentum = float(params.get("root_momentum", 0.0))
        root_weight_decay = float(params.get("root_weight_decay", 0.0))
        if root_optimizer == "adam":
            optimizer = torch.optim.Adam(
                root_model.parameters(),
                lr=root_lr,
                weight_decay=root_weight_decay,
            )
        elif root_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                root_model.parameters(),
                lr=root_lr,
                momentum=root_momentum,
                weight_decay=root_weight_decay,
            )
        else:
            raise ValueError("FLTrust root_optimizer must be 'sgd' or 'adam'")

        for _ in range(max(1, int(params.get("root_epochs", 1)))):
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
        num_classes=cfg.dataset.num_classes,
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

    trial_plan = None
    if cfg.federation.pairing_mode == "strict":
        from experiments.trial_plan import TrialPlanV1

        trial_plan = TrialPlanV1.load(cfg.federation.trial_plan_path)
        if trial_plan.trial_plan_hash != cfg.federation.trial_plan_hash:
            raise RuntimeError(
                "Configured trial-plan hash does not match the loaded plan: "
                f"{cfg.federation.trial_plan_hash} != {trial_plan.trial_plan_hash}"
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
        sampling_seed=cfg.project.seed,
        pairing_mode=cfg.federation.pairing_mode,
        sampling_protocol=cfg.federation.sampling_protocol,
        trial_plan=trial_plan,
        parameter_roles=get_parameter_roles(global_model),
        parameter_names=get_parameter_names(global_model),
        scalable_parameter_indices=get_trainable_parameter_indices(global_model),
        attack_cfg=cfg.security.attack,
    )
    state_provider = getattr(strategy.defense, "state_dict", None)
    if callable(state_provider):
        evaluator.state_provider = state_provider

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
