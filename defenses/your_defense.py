"""
defenses/your_defense.py
-------------------------
Template for implementing YOUR custom defense method.

Steps
-----
1. Fill in the aggregate() method with your algorithm.
2. Add any hyper-parameters to DefenseConfig.custom_params in config.yaml.
3. Register the class in defenses/defense_base.py:

    from defenses.your_defense import YourDefense
    DEFENSE_REGISTRY["your_defense"] = YourDefense

4. Set  defense.type: your_defense  in config.yaml.

The framework will automatically instantiate and call your defense
during each aggregation round.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from defenses.defense_base import BaseDefense, UpdateList
from config.config_loader import DefenseConfig

logger = logging.getLogger(__name__)


class YourDefense(BaseDefense):
    """
    Custom defense — replace this docstring and aggregate() body.

    Config params
    -------------
    Access your custom hyper-parameters via:
        self.cfg.custom_params["your_param"]

    Example config.yaml:
        security:
          defense:
            type: your_defense
            custom_params:
              threshold: 0.5
              num_selected: 5
    """

    def __init__(self, cfg: DefenseConfig):
        super().__init__(cfg)
        # Read custom hyper-parameters
        self.threshold: float = cfg.custom_params.get("threshold", 0.5)
        self.num_selected: int = cfg.custom_params.get("num_selected", -1)
        logger.info(
            "YourDefense init | threshold=%.3f  num_selected=%d",
            self.threshold, self.num_selected,
        )

    # ── Main interface — implement this ───────────────────────────────────────

    def aggregate(self, updates: UpdateList) -> List[np.ndarray]:
        """
        Parameters
        ----------
        updates : List[Tuple[List[np.ndarray], int]]
            Each element is (client_parameters, num_samples).
            client_parameters is a list of numpy arrays matching the
            model's state_dict ordering.

        Returns
        -------
        List[np.ndarray]
            Aggregated global parameters (same shape as each client's params).

        Notes
        -----
        Useful inherited helpers:
            self._flatten(params)                 -> 1-D np.ndarray
            self._unflatten(flat, template_params) -> List[np.ndarray]
            self._weighted_average(updates)        -> List[np.ndarray]
        """

        # ── Example skeleton ──────────────────────────────────────────────────
        # Step 1: Convert each client's params to a flat vector
        vectors = np.array([self._flatten(params) for params, _ in updates])
        num_clients = len(vectors)

        # Step 2: Compute your scoring / filtering criterion
        # (Replace with your algorithm)
        scores = self._compute_scores(vectors)

        # Step 3: Select or re-weight updates
        k = self.num_selected if self.num_selected > 0 else num_clients
        selected_idx = np.argsort(scores)[:k]           # lower score = better

        selected_updates = [updates[i] for i in selected_idx]
        logger.debug("YourDefense selected %d/%d clients: %s",
                     len(selected_idx), num_clients, selected_idx.tolist())

        # Step 4: Aggregate (weighted average over selected)
        return self._weighted_average(selected_updates)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_scores(self, vectors: np.ndarray) -> np.ndarray:
        """
        Assign a suspicion score to each client update.
        Lower = more trustworthy.

        Replace this with your actual scoring logic.
        Currently returns random scores as a placeholder.
        """
        # TODO: Implement your scoring logic here.
        # Examples:
        #   - L2 distance from the mean
        #   - Cosine similarity to a reference
        #   - Gradient norm outlier detection
        #   - Any learned / heuristic criterion
        rng = np.random.default_rng(42)
        return rng.random(len(vectors))    # placeholder
