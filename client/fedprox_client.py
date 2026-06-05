"""
client/fedprox_client.py
-------------------------
FedProx client (Li et al. 2020).

Adds a proximal regularisation term μ/2 · ‖w − w_global‖²
to the local loss, limiting how far local updates drift from
the global model — especially important under heterogeneous data.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from client.fl_client import FedSecClient
from models.model_factory import set_parameters

logger = logging.getLogger(__name__)


class FedProxClient(FedSecClient):
    """
    FedProx local training with proximal term.

    The proximal μ is read from the per-round config dict
    sent by the server (key: "proximal_mu"), falling back
    to the client-level default if not provided.
    """

    def fit(self, ins):  # type: ignore[override]
        from flwr.common import (
            Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
        )
        from models.model_factory import get_parameters

        server_params = parameters_to_ndarrays(ins.parameters)
        config = ins.config
        mu = float(config.get("proximal_mu", 0.01))

        # Store global weights for proximal term
        set_parameters(self.model, server_params)
        global_weights = [p.data.clone() for p in self.model.parameters()]

        self.on_before_fit(server_params, config)

        # Override criterion to include proximal term
        orig_criterion = self.trainer.criterion

        def prox_criterion(logits, labels):
            base_loss = orig_criterion(logits, labels)
            prox = torch.tensor(0.0, device=self.device)
            for p, g in zip(self.model.parameters(), global_weights):
                prox += (p - g).norm() ** 2
            return base_loss + (mu / 2.0) * prox

        self.trainer.criterion = prox_criterion
        metrics = self.trainer.train(
            self.train_loader,
            epochs=int(config.get("local_epochs", self.client_cfg.local_epochs)),
        )
        self.trainer.criterion = orig_criterion  # restore

        updated_params = get_parameters(self.model)
        updated_params = self.on_after_fit(updated_params, metrics)

        logger.debug("FedProx Client %d | μ=%.4f loss=%.4f",
                     self.client_id, mu, metrics["train_loss"])

        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=len(self.train_loader.dataset),
            metrics={k: float(v) for k, v in metrics.items()},
        )
