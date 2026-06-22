"""
config/config_loader.py
-----------------------
Typed config dataclasses + YAML loader.
All experiment hyper-parameters flow through here so that
scripts, strategies, and defenses stay decoupled from YAML parsing.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class ProjectConfig:
    name: str = "FedSec"
    seed: int = 42
    log_level: str = "INFO"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"


@dataclass
class FederationConfig:
    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 5
    min_fit_clients: int = 3
    min_evaluate_clients: int = 3
    min_available_clients: int = 5


@dataclass
class DatasetConfig:
    name: str = "cifar10"           # cifar10 | mnist | femnist
    data_dir: str = "data/"
    download_source: str = "huggingface"  # huggingface | torchvision
    partition: str = "iid"          # iid | non_iid | dirichlet
    dirichlet_alpha: float = 0.5
    num_classes: int = 10
    val_split: float = 0.1


@dataclass
class ModelConfig:
    architecture: str = "resnet18"  # resnet18 | cnn | mlp
    pretrained: bool = False
    num_classes: int = 10


@dataclass
class ClientConfig:
    local_epochs: int = 5
    batch_size: int = 32
    optimizer: str = "sgd"
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"


@dataclass
class StrategyConfig:
    name: str = "fedavg"
    proximal_mu: float = 0.01
    eta: float = 0.01
    eta_l: float = 0.0316
    beta_1: float = 0.9
    beta_2: float = 0.99
    tau: float = 1e-3


@dataclass
class AttackConfig:
    enabled: bool = False
    type: str = "none"
    malicious_fraction: float = 0.2
    backdoor_target_label: int = 0
    trigger_pattern: str = "pixel"
    poison_fraction: float = 0.1
    trigger_size: int = 3
    trigger_value: float = 1.0
    dba_trigger_num: int = 4
    dba_gap: int = 3
    dba_base_row: int = 0
    dba_base_col: int = 0
    dba_scale_update: bool = True
    dba_boost_factor: float = 10.0
    source_label: int = 0
    target_label: int = 1


@dataclass
class DefenseConfig:
    enabled: bool = False
    type: str = "none"
    krum_num_to_select: int = 1
    trim_fraction: float = 0.1
    root_dataset_size: int = 100
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.001


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    eval_every_n_rounds: int = 1
    save_best_model: bool = True
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class DPConfig:
    enabled: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5


# ── Master Config ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    differential_privacy: DPConfig = field(default_factory=DPConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

def _convert_value_type(val, target_type):
    """Convert value to target type, handling scientific notation strings."""
    if val is None:
        return val
    if target_type == float:
        # Handle scientific notation like "1e-4" which YAML may parse as string
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val
        return float(val)
    if target_type == int:
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return val
        return int(val)
    if target_type == bool:
        if isinstance(val, str):
            return val.lower() in ("true", "yes", "1")
        return bool(val)
    return val


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a nested dict to a dataclass."""
    import dataclasses
    from typing import get_type_hints

    if not dataclasses.is_dataclass(cls):
        return data
    kwargs = {}
    type_hints = get_type_hints(cls) if hasattr(cls, '__annotations__') else {}

    for f in dataclasses.fields(cls):
        val = data.get(f.name, None)
        if val is None:
            kwargs[f.name] = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore
        elif dataclasses.is_dataclass(f.type):
            kwargs[f.name] = _dict_to_dataclass(f.type, val)
        else:
            # Resolve string type hints (e.g., "AttackConfig")
            origin_type = f.type
            if isinstance(origin_type, str):
                origin_type = globals().get(origin_type, None)
            if origin_type and dataclasses.is_dataclass(origin_type):
                kwargs[f.name] = _dict_to_dataclass(origin_type, val)
            else:
                # Convert value to the field's type
                target_type = type_hints.get(f.name, type(val))
                kwargs[f.name] = _convert_value_type(val, target_type)
    return cls(**kwargs)


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """Load and validate YAML config into typed Config dataclass."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return _dict_to_dataclass(Config, raw)


def override_config(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """
    Apply flat dot-notation overrides to a Config, e.g.:
        override_config(cfg, {"federation.num_rounds": 100, "client.lr": 0.001})
    Useful for sweep scripts and CLI argument parsing.
    """
    import dataclasses

    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if dataclasses.is_dataclass(obj):
            setattr(obj, parts[-1], val)
    return cfg
