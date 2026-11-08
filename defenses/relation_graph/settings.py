"""Explicit parameters for the empirical graph evidence rule."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class Settings:
    history_rounds: int = 12
    backbone_bins: int = 8
    min_reference_rounds: int = 3
    min_reference_pairs: int = 6
    residual_scale_floor: float = 0.05
    pair_shrinkage: float = 4.0
    pair_shift_limit: float = 0.15
    edge_z_threshold: float = 3.0
    edge_half_life: float = 4.0
    min_edge_confirmations: int = 2
    min_exceedance_fraction: float = 0.6
    min_group_density: float = 0.67
    min_boundary_fraction: float = 0.5
    min_calibrated_fraction: float = 0.5
    attenuation: float = 0.9

    def __post_init__(self):
        for name in ("history_rounds", "backbone_bins", "min_reference_rounds",
                     "min_reference_pairs", "min_edge_confirmations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_reference_rounds > self.history_rounds:
            raise ValueError("min_reference_rounds exceeds the history window")
        if self.min_edge_confirmations > self.history_rounds + 1:
            raise ValueError("min_edge_confirmations exceeds the available observations")
        for name in ("residual_scale_floor", "pair_shrinkage", "edge_z_threshold", "edge_half_life"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("pair_shift_limit", "min_exceedance_fraction", "min_group_density",
                     "min_boundary_fraction", "min_calibrated_fraction", "attenuation"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.attenuation == 1:
            raise ValueError("attenuation must leave positive client weight")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("relation graph settings must be a mapping")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown relation graph settings: {sorted(unknown)}")
        return cls(**values)
