"""Versioned, hash-bound contracts for federated attack implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AttackSpec:
    name: str
    version: str
    family: str
    objective: str
    domain: str
    attacker_knowledge: str
    formula: str
    source_url: str
    requires_benign_updates: bool = False
    requires_collusion: bool = False
    server_coordinated: bool = False
    canonical_name: str | None = None
    implementation_class: str = "paper_reproduction"

    @property
    def contract_hash(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_LIE_SOURCE = (
    "https://proceedings.neurips.cc/paper/2019/hash/"
    "ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html"
)
_MIN_SOURCE = (
    "https://www.ndss-symposium.org/ndss-paper/"
    "manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/"
)
_MODEL_REPLACEMENT_SOURCE = "https://proceedings.mlr.press/v108/bagdasaryan20a.html"
_DBA_SOURCE = "https://openreview.net/forum?id=rkgyS0VFvr"
_SIGN_FLIP_SOURCE = "https://proceedings.mlr.press/v119/xie20c.html"
_LABEL_FLIP_SOURCE = "https://arxiv.org/abs/2007.08432"
_RANDOM_BASELINE_SOURCE = "https://proceedings.mlr.press/v235/huang24u.html"


ATTACK_SPECS: dict[str, AttackSpec] = {
    "none": AttackSpec(
        "none", "none.v1", "clean", "none", "none", "none", "identity", "",
    ),
    "gaussian_noise": AttackSpec(
        "gaussian_noise", "gaussian_noise.delta_additive.v2", "random_outlier",
        "untargeted_availability", "model_delta", "black_box",
        "delta_mal = delta_local + Normal(mean, std)", _RANDOM_BASELINE_SOURCE,
        implementation_class="generic_byzantine_baseline",
    ),
    "random_noise": AttackSpec(
        "random_noise", "random_noise.rademacher_norm_matched.v1", "random_outlier",
        "untargeted_availability", "model_delta", "black_box",
        "delta_mal = scale * ||delta_local||_2 * rademacher / sqrt(d)", _RANDOM_BASELINE_SOURCE,
        implementation_class="generic_byzantine_baseline",
    ),
    "byzantine": AttackSpec(
        "byzantine", "byzantine.random_weights.legacy_v1", "random_outlier",
        "untargeted_availability", "model_parameters", "black_box",
        "model_mal ~ Normal(0, I)", _RANDOM_BASELINE_SOURCE,
        implementation_class="stress_test_variant",
    ),
    "sign_flip": AttackSpec(
        "sign_flip", "sign_flip.delta.v1", "direction_destruction",
        "untargeted_availability", "model_delta", "local_update",
        "delta_mal = -scale * delta_local", _SIGN_FLIP_SOURCE,
        implementation_class="paper_baseline_at_scale_1",
    ),
    "lie": AttackSpec(
        "lie", "lie.coordinate_inlier.v1", "inlier_model_poisoning",
        "untargeted_availability", "model_delta", "updates_only",
        "delta_mal = mean(reference_deltas) - z * std(reference_deltas)",
        _LIE_SOURCE, True, True, True,
    ),
    # ALIE is a widespread alias for A Little Is Enough, not independent evidence.
    "alie": AttackSpec(
        "alie", "alie.alias_of_lie.v1", "inlier_model_poisoning",
        "untargeted_availability", "model_delta", "updates_only",
        "alias of lie: delta_mal = mean - z * std", _LIE_SOURCE,
        True, True, True, "lie",
    ),
    "min_max": AttackSpec(
        "min_max", "min_max.agr_agnostic.v1", "optimization_model_poisoning",
        "untargeted_availability", "model_delta", "updates_only",
        "maximize gamma: max_i ||delta_mal-delta_i||^2 <= max_ij ||delta_i-delta_j||^2",
        _MIN_SOURCE, True, True, True,
    ),
    "min_sum": AttackSpec(
        "min_sum", "min_sum.agr_agnostic.v1", "optimization_model_poisoning",
        "untargeted_availability", "model_delta", "updates_only",
        "maximize gamma: sum_i ||delta_mal-delta_i||^2 <= max_i sum_j ||delta_i-delta_j||^2",
        _MIN_SOURCE, True, True, True,
    ),
    "backdoor": AttackSpec(
        "backdoor", "pixel_backdoor.data_poisoning.v1", "targeted_backdoor",
        "targeted_integrity", "training_data", "local_control",
        "stamp full trigger and relabel to target", _MODEL_REPLACEMENT_SOURCE,
        implementation_class="generic_data_poisoning_baseline",
    ),
    "model_replacement": AttackSpec(
        "model_replacement", "model_replacement.legacy_v1", "targeted_backdoor",
        "targeted_integrity", "training_data+model_delta", "aggregation_weight",
        "model_mal = global + scale * (model_backdoor-global)", _MODEL_REPLACEMENT_SOURCE,
        implementation_class="legacy_fixed_boost_model_replacement",
    ),
    "scaling_backdoor": AttackSpec(
        "scaling_backdoor", "scaling_backdoor.train_and_scale.v1", "targeted_backdoor",
        "targeted_integrity", "training_data+model_delta", "aggregation_weight",
        "model_mal = global + gain / malicious_weight_share * (model_backdoor-global)", _MODEL_REPLACEMENT_SOURCE,
        implementation_class="paper_inspired_train_and_scale",
    ),
    "label_flip_targeted": AttackSpec(
        "label_flip_targeted", "label_flip.source_to_target.v2", "data_poisoning",
        "targeted_integrity", "labels", "local_data_control",
        "y=source -> target for a deterministic poisoned subset", _LABEL_FLIP_SOURCE,
        implementation_class="data_poisoning_baseline",
    ),
    "label_flip_all_reverse": AttackSpec(
        "label_flip_all_reverse", "label_flip.all_reverse.v2", "data_poisoning",
        "untargeted_availability", "labels", "local_data_control",
        "y -> num_classes-1-y for a deterministic poisoned subset", _LABEL_FLIP_SOURCE,
        implementation_class="data_poisoning_baseline",
    ),
    "dba": AttackSpec(
        "dba", "dba.paper_cifar_normalized_white_train_and_scale.v4", "distributed_backdoor",
        "targeted_integrity", "training_data+model_delta", "colluding_local_control",
        "official CIFAR 4x(1x6) normalized-white fragments; train one per adversary; optionally compensate FedAvg dilution on delta",
        _DBA_SOURCE, False, True, False,
        implementation_class="paper_reproduction_adapted_scaling",
    ),
    "mpaf": AttackSpec(
        "mpaf", "mpaf.global_drift.v1", "model_poisoning",
        "untargeted_availability", "model_delta", "global_model_history",
        "delta_mal = lambda * (base-global)", "",
    ),
}


COORDINATED_ATTACKS = frozenset(
    name for name, spec in ATTACK_SPECS.items() if spec.server_coordinated
)


def get_attack_spec(name: str) -> AttackSpec:
    key = str(name).lower()
    if key not in ATTACK_SPECS:
        raise ValueError(f"Unknown attack contract {key!r}")
    return ATTACK_SPECS[key]


def attack_source_hash() -> str:
    """Hash the executable attack sources used to construct a run."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    repository = root.parent
    sources = (
        root / "spec.py",
        root / "attack_client.py",
        root / "coordinator.py",
        repository / "client" / "fl_client.py",
        repository / "strategies" / "fed_strategy.py",
        repository / "server" / "fl_server.py",
    )
    for path in sources:
        if path.is_file():
            digest.update(str(path.relative_to(repository)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def attack_contract_payload(name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    spec = get_attack_spec(name)
    payload = {
        "spec": asdict(spec),
        "contract_hash": spec.contract_hash,
        "implementation_source_sha256": attack_source_hash(),
        "parameters": dict(config),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    payload["implementation_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload
