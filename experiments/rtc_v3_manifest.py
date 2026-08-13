"""Create a model-bound RTC-v3 bootstrap manifest.

This command intentionally creates a runtime-smoke artifact, not a formally
calibrated defense contract.  Its permissive limits exercise the complete
RTC-v3 Phase 6 pipeline without presenting guessed clean-tail thresholds as
research results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from defenses.rtc.calibration import CalibrationManifest, build_manifest
from models.model_factory import get_model, get_parameters


BOOTSTRAP_LIMIT = 1e12


def build_bootstrap_manifest(config_path: str | Path) -> dict[str, Any]:
    """Build a fail-fast model contract with explicitly non-formal limits."""
    resolved_config = Path(config_path).expanduser().resolve()
    cfg = load_config(resolved_config)
    model = get_model(
        architecture=cfg.model.architecture,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
        dataset_name=cfg.dataset.name,
    )
    params = get_parameters(model)
    resolutions = ("full",)
    server_windows = (1, 8)
    principal_windows = (1, 4)
    server_budgets = {
        resolution: {
            str(window): {
                "anchor": BOOTSTRAP_LIMIT,
                "residual": BOOTSTRAP_LIMIT,
                "total": BOOTSTRAP_LIMIT,
            }
            for window in server_windows
        }
        for resolution in resolutions
    }
    principal_betas = {
        resolution: {
            str(window): BOOTSTRAP_LIMIT for window in principal_windows
        }
        for resolution in resolutions
    }
    cumulative = {
        "enabled": True,
        "resolutions": {
            resolution: {
                "scale_center": 1.0,
                "scale_lower": 1e-12,
                "scale_upper": BOOTSTRAP_LIMIT,
                "kappa": BOOTSTRAP_LIMIT,
                "threshold": BOOTSTRAP_LIMIT,
                "eta": 0.0,
                "q_min": 1.0,
            }
            for resolution in resolutions
        },
    }
    direction = {
        "enabled": True,
        "resolutions": {
            resolution: {
                "z_threshold": BOOTSTRAP_LIMIT,
                "history_length": 4,
                "min_matches": 3,
                "server_window": 8,
                "principal_window": 4,
                "server_usage_threshold": BOOTSTRAP_LIMIT,
                "principal_usage_threshold": BOOTSTRAP_LIMIT,
                "q_value": 1.0,
            }
            for resolution in resolutions
        },
    }
    manifest = build_manifest(
        params=params,
        profile="bootstrap_runtime_smoke",
        clip_lower=0.0,
        clip_upper=BOOTSTRAP_LIMIT,
        resolutions=resolutions,
        residual_scales={resolution: 1.0 for resolution in resolutions},
        server_windows=server_windows,
        server_budgets=server_budgets,
        principal_windows=principal_windows,
        principal_betas=principal_betas,
        stable_cones={
            "enabled": True,
            "dimension": 256,
            "seed": 42,
            "sketch_algorithm_version": "splitmix64_v1",
            "coordinate_median_block_size": 262144,
            "max_prototypes": 16,
            "match_threshold": 0.85,
            "momentum": 0.8,
            "retirement_rounds": 16,
        },
        cone_budgets={
            resolution: {
                str(window): BOOTSTRAP_LIMIT for window in server_windows
            }
            for resolution in resolutions
        },
        cumulative=cumulative,
        direction_persistence=direction,
        metadata={
            "artifact_kind": "rtc_v3_bootstrap",
            "formal_evaluation_ready": False,
            "warning": (
                "Runtime smoke only. Replace with a manifest calibrated from "
                "independent clean seeds before reporting defense results."
            ),
            "source_config": str(resolved_config),
            "dataset": cfg.dataset.name,
            "model_architecture": cfg.model.architecture,
            "num_classes": int(cfg.dataset.num_classes),
            "sketch_algorithm_version": "splitmix64_v1",
        },
    )
    CalibrationManifest.load(manifest)
    return manifest


def write_bootstrap_manifest(
    *,
    config_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing RTC-v3 manifest: {output}; use --force"
        )
    manifest = build_bootstrap_manifest(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a model-bound RTC-v3 runtime-smoke manifest"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="config/rtc_v3_manifest.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    path = write_bootstrap_manifest(
        config_path=args.config,
        output_path=args.output,
        force=args.force,
    )
    manifest = CalibrationManifest.load(path)
    print(f"RTC-v3 bootstrap manifest written: {path}")
    print(f"content_hash: {manifest.hash}")
    print("WARNING: runtime smoke only; not ready for formal evaluation.")
    return path


if __name__ == "__main__":
    main()
