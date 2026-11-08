"""FL2 adapter for conditional client relations and sample-weight attenuation."""

import copy
from dataclasses import asdict, dataclass
import math

import numpy as np

from defenses.defense_base import BaseDefense
from . import VERSION
from .graph import EdgeObservation, RoundObservation, assess, observe_edges
from .settings import Settings


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    group: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or any(type(d) is not int or d < 1 for d in self.shape):
            raise ValueError("Invalid model tensor name or shape")
        if not isinstance(self.dtype, str) or np.dtype(self.dtype).kind not in "fiub":
            raise ValueError("Unsupported model tensor dtype")
        if self.group not in {"backbone", "task", "buffer"}:
            raise ValueError("Invalid model tensor group")
        if self.group != "buffer" and np.dtype(self.dtype).kind != "f":
            raise ValueError("Trainable parameters must be floating point")


def _validate_layout(layout):
    if len({spec.name for spec in layout}) != len(layout):
        raise ValueError("Parameter names must be unique")
    if not {"backbone", "task"} <= {spec.group for spec in layout}:
        raise ValueError("Relation graph needs both backbone and classifier parameters")


def _validate_arrays(arrays, layout):
    if len(arrays) != len(layout):
        raise ValueError("Model tensor count changed")
    for value, spec in zip(arrays, layout):
        if not isinstance(value, np.ndarray) or value.shape != spec.shape or value.dtype.str != spec.dtype:
            raise ValueError(f"Model tensor layout changed for {spec.name}")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite values in {spec.name}")


class RelationGraphDefense(BaseDefense):
    requires_parameter_roles = True
    requires_principal_first_sampling = True
    requires_server_partition_ids = True

    def __init__(self, cfg, num_clients=20):
        super().__init__(cfg)
        if type(num_clients) is not int or num_clients < 2:
            raise ValueError("Relation graph needs at least two registered clients")
        custom = dict(cfg.custom_params or {})
        unknown = set(custom) - {"settings", "principal_map"}
        if unknown:
            raise ValueError(f"Unknown relation graph options: {sorted(unknown)}")
        self.settings = Settings.from_dict(custom.get("settings", {}))
        mapping = custom.get("principal_map", {str(index): str(index) for index in range(num_clients)})
        self.principal_map = {str(key): str(value) for key, value in mapping.items()}
        if (len(self.principal_map) != num_clients or len(set(self.principal_map.values())) != num_clients
                or not all(self.principal_map) or not all(self.principal_map.values())):
            raise ValueError("Relation graph needs one stable principal per registered client")
        self._layout = None
        self._principals = ()
        self._history = ()
        self._last_round = 0
        self._last_diagnostics = {}

    def principal_id_for(self, client_id):
        try:
            return self.principal_map[str(client_id)]
        except KeyError as error:
            raise ValueError(f"Unknown server client identity: {client_id}") from error

    def set_context(self, server_round, client_ids, global_params, *, principal_ids=None,
                    server_optimizer=None, parameter_roles=None, parameter_names=None,
                    trainable_parameter_indices=None, **kwargs):
        if server_optimizer != "fedavg":
            raise ValueError("Relation graph currently supports synchronous FedAvg")
        if type(server_round) is not int or server_round <= self._last_round:
            raise ValueError("Relation graph round must advance beyond the committed state")
        clients = [str(value) for value in client_ids]
        principals = tuple(self.principal_id_for(value) for value in clients)
        if not clients or len(set(principals)) != len(principals):
            raise ValueError("A round requires nonempty, distinct principals")
        if principal_ids is not None and tuple(map(str, principal_ids)) != principals:
            raise ValueError("Server principal mapping does not match the registered identities")
        if parameter_roles is None or parameter_names is None or trainable_parameter_indices is None:
            raise ValueError("Relation graph requires authoritative parameter names, roles and indices")
        names = {int(index): name for index, name in parameter_names.items()}
        roles = {int(index): role for index, role in parameter_roles.items()}
        all_indices = set(range(len(global_params)))
        indices = tuple(trainable_parameter_indices)
        if (len(names) != len(parameter_names) or set(names) != all_indices
                or len(roles) != len(parameter_roles) or not set(roles) <= all_indices
                or any(type(index) is not int or index not in all_indices for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Invalid authoritative parameter metadata")
        trainable = set(indices)
        task = {index for index, role in roles.items() if role in ("classifier_weight", "classifier_bias")}
        if not task <= trainable:
            raise ValueError("Classifier roles must refer to trainable parameters")
        if any(not isinstance(value, np.ndarray) for value in global_params):
            raise ValueError("Model parameters must be NumPy arrays")
        layout = tuple(TensorSpec(names[index], value.shape, value.dtype.str,
                                  "task" if index in task else "backbone" if index in trainable else "buffer")
                       for index, value in enumerate(global_params))
        _validate_layout(layout)
        _validate_arrays(global_params, layout)
        if self._layout is not None and layout != self._layout:
            raise ValueError("Model layout or relation roles changed")
        super().set_context(server_round, clients, global_params, **kwargs)
        self._layout = layout
        self._principals = principals

    def aggregate(self, updates):
        if self._global_params is None or self._server_round <= self._last_round or len(updates) != len(self._principals):
            raise ValueError("A fresh context matching all uploads is required")
        for params, count in updates:
            _validate_arrays(params, self._layout)
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
                raise ValueError("Sample counts must be positive integers")
        counts = np.asarray([count for _, count in updates], dtype=np.float64)
        total = float(counts.sum())
        if not math.isfinite(total):
            raise ValueError("Sample count total must be finite")
        nominal = counts / total

        # Keep graph observations canonical, independently of the reduction order used by FedAvg.
        order = sorted(range(len(updates)), key=self._principals.__getitem__)
        identities = tuple(self._principals[index] for index in order)
        backbone = [index for index, spec in enumerate(self._layout) if spec.group == "backbone"]
        task = [index for index, spec in enumerate(self._layout) if spec.group == "task"]
        observations = observe_edges(identities, self._global_params, [updates[index][0] for index in order], backbone, task)
        cutoff = self._server_round - self.settings.history_rounds
        previous = tuple(frame for frame in self._history if frame.round_number >= cutoff)
        assessment = assess(self._server_round, identities, observations, previous, self.settings)
        risks = np.asarray([assessment.risks[principal] for principal in self._principals])
        removed = self._removal(nominal, risks, assessment)
        weights = nominal - removed
        anchor = float(removed.sum())
        if anchor:
            result = self._weighted_average_with_weights(
                [*updates, (self._global_params, 1)], [*weights, anchor])
        else:
            result = self._weighted_average(updates)
        # NumPy rounding in the shared FedAvg helper can return a scalar for a BN counter.
        result = [np.asarray(value) for value in result]
        _validate_arrays(result, self._layout)

        clients = [dict(client_id=client, principal_id=principal, risk=float(risk),
                        nominal_weight=float(share), aggregation_weight=float(weight))
                   for client, principal, risk, share, weight in zip(self._client_ids, self._principals, risks, nominal, weights)]
        metrics = dict(relation_graph_state=assessment.state,
                       relation_graph_calibrated_fraction=assessment.calibrated_fraction,
                       relation_graph_observed_edges=len(observations),
                       relation_graph_confirmed_groups=len(assessment.groups),
                       relation_graph_downweighted_clients=int(np.count_nonzero(removed)),
                       relation_graph_anchor_weight=anchor)
        self._last_diagnostics = dict(version=VERSION, round=self._server_round, metrics=metrics,
                                      clients=clients, groups=assessment.groups, edges=assessment.edges)
        self._history = tuple(frame for frame in previous if frame.round_number > cutoff) + (assessment.observation,)
        self._last_round = self._server_round
        self.last_client_aggregation_weights = dict(zip(self._client_ids, map(float, weights)))
        self.last_client_weights = dict(self.last_client_aggregation_weights)
        self.last_client_trusts = dict(zip(self._client_ids, map(float, 1.0 - self.settings.attenuation * risks)))
        self.last_round_metrics = dict(metrics)
        return result

    def _removal(self, nominal, risks, assessment):
        """Mass removed from each client before the anchor receives it.

        The default action is the soft downweighting rule
        ``nominal * attenuation * risk``.  Variants override this hook to test
        stronger action rules without changing detection or aggregation.
        """
        return nominal * self.settings.attenuation * risks

    def diagnostics(self):
        return copy.deepcopy(self._last_diagnostics)

    def state_dict(self):
        return dict(version=VERSION, settings=asdict(self.settings), principal_map=dict(self.principal_map),
                    last_round=self._last_round, layout=None if self._layout is None else [asdict(spec) for spec in self._layout],
                    history=[asdict(frame) for frame in self._history])

    def load_state_dict(self, state):
        if (state.get("version") != VERSION or state.get("settings") != asdict(self.settings)
                or state.get("principal_map") != self.principal_map):
            raise ValueError("Relation graph checkpoint contract mismatch")
        last_round = state["last_round"]
        if type(last_round) is not int or last_round < 0:
            raise ValueError("Invalid relation graph checkpoint round")
        layout = None if state["layout"] is None else tuple(
            TensorSpec(spec["name"], tuple(spec["shape"]), spec["dtype"], spec["group"]) for spec in state["layout"])
        if layout is not None:
            _validate_layout(layout)
        history = tuple(RoundObservation(frame["round_number"], tuple(EdgeObservation(**edge) for edge in frame["edges"]))
                        for frame in state["history"])
        rounds = [frame.round_number for frame in history]
        known = set(self.principal_map.values())
        if (rounds != sorted(set(rounds)) or (rounds[-1] if rounds else 0) != last_round
                or any(number <= last_round - self.settings.history_rounds for number in rounds)
                or (history and layout is None)
                or any(edge.left not in known or edge.right not in known for frame in history for edge in frame.edges)):
            raise ValueError("Invalid relation graph checkpoint history")
        self._layout, self._history, self._last_round = layout, history, last_round
        self._principals = ()
        self._global_params = None
        self._client_ids = []
        self._server_round = 0
        self._last_diagnostics = {}
        self.last_client_weights = {}
        self.last_client_aggregation_weights = {}
        self.last_client_trusts = {}
        self.last_round_metrics = {}
